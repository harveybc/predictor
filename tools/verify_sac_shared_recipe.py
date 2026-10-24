#!/usr/bin/env python3
"""Verify selected-source -> native SAC inputs without fitting or trading."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(a):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    torch.set_num_threads(1)
    sys.path.insert(0, str(a.agent_root))
    from app.frozen_recipe_inputs import FrozenRecipeInputs
    from agent_plugins.sac_agent import Plugin

    if sha(a.recipe) != a.recipe_sha or sha(a.reference) != a.reference_sha:
        raise ValueError("SAC_PROBE_REFERENCE_PIN")
    rec = json.loads(a.recipe.read_text())
    reference = json.loads(a.reference.read_text())
    cp = module("shared_probe_definition", a.hs_root / "tools/consumer_probe.py")
    data = FrozenRecipeInputs.load(rec, a.store, a.manifest)
    retirement = None
    if a.parent_recipe is not None:
        if a.parent_sha is None or sha(a.parent_recipe) != a.parent_sha:
            raise ValueError("SAC_PROBE_PARENT_PIN")
        parent = json.loads(a.parent_recipe.read_text())
        old = FrozenRecipeInputs.load(parent, a.store, a.manifest)
        idx = parent["list"]["columns"].index("px.er_1")
        keep = [j for j in range(parent["list"]["n"]) if j != idx]
        if rec["list"]["columns"] != [parent["list"]["columns"][j] for j in keep]:
            raise ValueError("SAC_PROBE_RETIREMENT_COLUMNS")
        if not np.array_equal(old.row_ts_ns, data.row_ts_ns) or not np.array_equal(old.period, data.period):
            raise ValueError("SAC_PROBE_RETIREMENT_CLOCK")
        if np.any(old.valid & ~data.valid):
            raise ValueError("SAC_PROBE_RETIREMENT_LOST_ROWS")
        for start in range(0, len(old.valid), 512):
            ids = np.flatnonzero(old.valid[start:start + 512]) + start
            if not np.array_equal(old.windows[ids][:, :, keep], data.windows[ids]):
                raise ValueError("SAC_PROBE_RETIREMENT_CHANGED_SURVIVORS")
        retirement = {"parent_sha256": a.parent_sha, "common_valid_tensors_exact": True,
                      "lost_valid_rows": 0, "new_valid_rows": {
                          p: int((data.valid & ~old.valid & (data.period == p)).sum())
                          for p in rec["split"]["periods"]},
                      "expanded_support_trading_equivalence_claimed": False}
    g, sc = rec["representation"]["resolved"], rec["scaler"]
    candidate = cp.probe_document(rec, data.row_ts_ns, data.windows, data.valid, data.period,
                                  data.slot_ts_ns, sc["features"], g, sc["mean"], sc["std"])
    errors = cp.compare(reference, candidate)
    if errors:
        raise ValueError("SAC_SOURCE_PROBE:" + ",".join(errors))
    rows = cp.probe_rows(data.valid, data.period, rec["split"]["periods"])
    obs = {"features": np.stack([data.observation(int(i))["features"] for i in rows])}
    if cp.h(obs["features"]) != reference["tensor_sha256"]:
        raise ValueError("SAC_OBSERVATION_METHOD_CHANGED_TENSOR")

    class ProbeEnv(gym.Env):
        observation_space = spaces.Dict({"features": spaces.Box(-np.inf, np.inf, obs["features"].shape[1:], dtype=np.float32)})
        action_space = spaces.Box(-1., 1., (4,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return {"features": obs["features"][0].copy()}, {}

        def step(self, action):
            raise AssertionError("NO_FINANCIAL_ENV_STEPS_IN_INPUT_PROBE")

    cfg = {"observation_encoder": "shared_recipe", "shared_recipe": rec, "device": "cpu",
           "net_arch": [rec["representation"]["model"]["head"]], "buffer_size": 1, "train_seed": 7}
    plugin = Plugin(cfg)
    env = plugin.wrap_env(ProbeEnv(), cfg)
    model = plugin.build(env, cfg)
    captured = []
    hook = model.actor.features_extractor.register_forward_pre_hook(
        lambda m, args: captured.append(args[0]["features"].detach().cpu().numpy().copy()))
    actions = model.predict(obs, deterministic=True)[0]
    hook.remove()
    if len(captured) != 1 or not np.array_equal(captured[0], obs["features"]):
        raise ValueError("SAC_POLICY_INPUT_CHANGED")
    critic_captured = []
    hook = model.critic.features_extractor.register_forward_pre_hook(
        lambda m, args: critic_captured.append(args[0]["features"].detach().cpu().numpy().copy()))
    tensor = {"features": torch.as_tensor(obs["features"])}
    model.policy.set_training_mode(False)
    with torch.no_grad():
        q = model.critic(tensor, torch.as_tensor(actions))
    hook.remove()
    if len(q) != 2 or len(critic_captured) != 1 or not np.array_equal(critic_captured[0], obs["features"]):
        raise ValueError("SAC_CRITIC_INPUT_CHANGED")

    # Independent topology/operation check against C's trunk, with identical test weights.
    temporal = module("shared_reference_temporal", a.hs_root / "app/short_horizon/expert_guided/temporal.py")
    m = rec["representation"]["model"]
    ref_model = temporal.TemporalTrunk(rec["list"]["n"], units=m["units"], core=m["core"], depth=m["depth"],
                                     dropout=rec["regularization"]["dropout"], groups=g["indices"],
                                     group_units=g["units"], group_kinds=g["kinds"]).eval()
    actual = model.actor.features_extractor.eval()
    left, right = list(ref_model.parameters()), list(actual.parameters())
    if len(left) != len(right) or any(x.shape != y.shape for x, y in zip(left, right)):
        raise ValueError("SAC_TRUNK_PARAMETER_SHAPES")
    with torch.no_grad():
        for x, y in zip(left, right):
            x.copy_(y)
        expected, obtained = ref_model(tensor["features"]), actual(tensor)
    if not torch.equal(expected, obtained):
        raise ValueError("SAC_TRUNK_FORWARD_PARITY")
    with tempfile.TemporaryDirectory(prefix="sac-recipe-probe-") as tmp:
        checkpoint = Path(tmp) / "probe.zip"
        model.save(checkpoint)
        restored = plugin.load(checkpoint, env)
        if not np.array_equal(actions, restored.predict(obs, deterministic=True)[0]):
            raise ValueError("SAC_POLICY_RELOAD_PARITY")
    return {"schema": "sac_shared_recipe_probe.v1", "state": "PASS", "recipe": rec["identity"],
            "recipe_sha256": sha(a.recipe), "reference_sha256": sha(a.reference),
            "tensor_sha256": candidate["tensor_sha256"], "probe_shape": candidate["tensor_shape"],
            "valid_rows": candidate["valid_rows"], "valid_masks_exact": True, "source_to_tensor_exact": True,
            "actor_input_exact": True, "twin_critic_input_exact": True, "trunk_same_weights_exact": True,
            "policy_reload_exact": True, "observations": "features only", "scientific_fits": 0,
            "retirement_comparison": retirement,
            "environment_steps": 0, "test_scored": False, "trading_engine_integration_verified": False,
            "inputs": {"store": sha(a.store), "manifest": sha(a.manifest)},
            "consumer_code": {name: sha(a.agent_root / name) for name in (
                "app/frozen_recipe_inputs.py", "agent_plugins/recipe_features.py",
                "agent_plugins/temporal_features.py", "agent_plugins/sac_agent.py")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("agent-root", "hs-root", "recipe", "reference", "store", "manifest", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--recipe-sha", required=True)
    p.add_argument("--reference-sha", required=True)
    p.add_argument("--parent-recipe", type=Path)
    p.add_argument("--parent-sha")
    a = p.parse_args()
    result = verify(a)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
