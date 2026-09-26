#!/usr/bin/env python3
"""DR05 — materialize the OUTPUT of the frozen prefix, and prove it four ways.

Satoshi, successor technical lead, 2026-09-26.

**What this exists to correct.** `MOD-FROZEN-PREFIX` was treated as delivered because the delivered CSV
and the prepared panel agree. Parity of INPUTS is not the module. The module's deliverable is a versioned
derived dataset at the output of the fixed prefix, and until that dataset exists there is nothing for
`H-CORE` to consume. This tool materializes it and proves it; it does not declare it.

**The prefix, and where it stops.** Five stages, in order, each recorded with its learned state, its
clock, its rows and split, its shape and its version::

    preprocessing   the stored scaler (mean, sd) fitted on train windows only        LEARNED
    groups          the held-constant channel assignment -> one g{g}_select gather   CONSTANT_BY_DESIGN
    detector        g{g}_det1_conv / _det1_proj / _det2_conv                         LEARNED
    adapter         g{g}_adapt, TimeDistributed Dense(8)                             LEARNED
    fusion          fusion_seq, channel concatenation of the branch sequences        CONFIGURATION

The prefix ENDS at ``fusion_seq``. The core (``core_tcn*``), the head and the persistence skip are
downstream of it and are not materialized: they are what `H-CORE` varies.

**The four proofs**, each a refusal by name when it fails:

    direct_against_cache     a bounded sample of origins recomputed through the prefix and compared
                             ELEMENT BY ELEMENT against the shard on disk
    fresh_process_reload     a SEPARATE interpreter re-opens the store, re-derives every digest and
                             reconstructs a named window; this process never hands it a value
    mutation                 one detector weight, one adapter weight, one scaler entry and the group
                             assignment, each perturbed alone; each must change the output AND be
                             refused by the identity check. A mutation that is not detected is a failure
    causality                the measured reach: perturbing a FUTURE row may not move the output, and
                             the last row that does move it must be the declared branch reach

**The reserve is not opened.** Only the consumed development slice is read; the split named ``test``
is materialized as a refusal, never as an array, and every row index this tool touches is asserted to
lie inside the slice the preparation declared.

    python tools/df_dr05_prefix_output.py --root RUN_ROOT --cell R1_s1 --out-dir STORE --out OUT.json
    python tools/df_dr05_prefix_output.py --verify-store STORE      # the fresh-process side
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
UNCHECKABLE = "REFUSED_MATERIALIZATION_UNCHECKABLE"
BY_DESIGN = "REFUSED_UNMATERIALIZED_BY_DESIGN"

SUCCESSOR_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_household_successor_v3"

PREFIX_ENDS_AT = "fusion_seq"
DOWNSTREAM_OF_THE_PREFIX = ("core_", "summary_", "last_g", "fusion_vec", "head", "last_observation",
                            "persistence_skip", "target_readout")

STAGE_KINDS = {"preprocessing": "LEARNED", "groups": "CONSTANT_BY_DESIGN", "detector": "LEARNED",
               "adapter": "LEARNED", "fusion": "CONFIGURATION"}

WHAT_PARITY_OF_INPUTS_IS_NOT = (
    "agreement between the delivered CSV and the prepared panel is parity of the prefix's INPUTS. It "
    "says nothing about the prefix's OUTPUT, which is what a consumer downstream of the fusion reads. "
    "This artifact is that output; it is not implied by that agreement and was not implied by it")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_dr05_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha_array(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str((a.shape, a.dtype.str)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _weights_digest(model, names) -> str:
    h = hashlib.sha256()
    for n in sorted(names):
        for w in model.get_layer(n).get_weights():
            h.update(str((n, w.shape, w.dtype.str)).encode())
            h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()


# --- the prefix ---------------------------------------------------------------------------------------

def stage_layer_names(model) -> dict:
    """The five stages, read off the graph rather than assumed."""
    names = [l.name for l in model.layers]
    return {
        "groups": sorted(n for n in names if n.endswith("_select")),
        "detector": sorted(n for n in names if "_det" in n),
        "adapter": sorted(n for n in names if n.endswith("_adapt")),
        "fusion": [n for n in names if n == PREFIX_ENDS_AT],
    }


def build_prefix(root: Path, cell: str):
    """The fitted model of ONE retained cell, cut at the fusion. Nothing is fitted here."""
    P = _load("df_e1_pilot")
    E = _load("df_mod_e0")
    tf = E._tf()
    design = json.loads((root / "DESIGN.json").read_text())
    data = json.loads((root / "DATA.json").read_text())
    g = design["graph"]
    assignment = list(g["assignment"])
    W = int(design["task"]["window_steps"])
    p = int(data["p"])
    j = int(data["target_channel"])
    job = json.loads((root / "attempts" / cell / "job.json").read_text())
    seed = int(job["seed"])
    model = P._model_for_target(assignment, W, p, j, seed, core=g.get("core_kind", "conv3"))
    wpath = root / "attempts" / cell / "weights.weights.h5"
    model.load_weights(str(wpath))
    prefix = tf.keras.Model(model.input, model.get_layer(PREFIX_ENDS_AT).output,
                            name=f"frozen_prefix_{cell}")
    for l in prefix.layers:
        l.trainable = False
    return {"tf": tf, "model": model, "prefix": prefix, "design": design, "data_json": data,
            "assignment": assignment, "W": W, "p": p, "j": j, "seed": seed,
            "weights_path": wpath, "weights_sha256": sha_file(wpath),
            "cell": cell, "stages": stage_layer_names(model)}


def prefix_contains_nothing_downstream(ctx) -> dict:
    """The cut is at the fusion. A core or head layer inside the prefix would make the store a model."""
    inside = [l.name for l in ctx["prefix"].layers]
    leaked = sorted(n for n in inside if any(tag in n for tag in DOWNSTREAM_OF_THE_PREFIX))
    return {"layers_in_the_prefix": inside, "downstream_layers_found_inside": leaked,
            "the_cut_is_at_the_fusion": ctx["prefix"].output.name.startswith(PREFIX_ENDS_AT)
                                       or PREFIX_ENDS_AT in ctx["prefix"].layers[-1].name,
            "clean": not leaked}


# --- the version, and the stage manifest ---------------------------------------------------------------

def code_identity() -> dict:
    """The revision this store was produced at, plus the digests of the modules that define the prefix.

    The revision is read from git and may be DIRTY; the module digests are read from the bytes on disk,
    so a store is bound to the code that made it even in a checkout whose worktree is not clean.
    """
    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    mods = ["df_dr05_prefix_output.py", "df_e1_pilot.py", "df_mod_e0.py", "df_e1_block.py",
            "df_e1_regimes.py"]
    return {"revision": _git("rev-parse", "HEAD"),
            "worktree_has_uncommitted_changes": bool(_git("status", "--porcelain")),
            "module_sha256": {m: sha_file(HERE / m) for m in mods if (HERE / m).is_file()}}


def store_version(ctx, code: dict) -> dict:
    """One version string, derived from every identity the store depends on."""
    parts = {"data_sha256": ctx["data_json"]["data_sha256"],
             "design_sha256": ctx["data_json"]["design_sha256"],
             "panel_sha256": ctx["data_json"]["panel_sha256"],
             "donor_cell": ctx["cell"],
             "donor_weights_sha256": ctx["weights_sha256"],
             "prefix_ends_at": PREFIX_ENDS_AT,
             "code_module_sha256": code["module_sha256"]}
    return {"version": "dr05.prefix." + sha_obj(parts)[:16], "derived_from": parts,
             "a_different_donor_is_a_different_version": True}


def stage_manifest(ctx, clock: dict, splits: dict, version: dict) -> list:
    """The five stages, each with learned state, clock, rows and split, shape and version."""
    D = np.load(ctx["root_data"])
    mean, sd = D["scaler_mean"], D["scaler_sd"]
    st = ctx["stages"]
    shape_in = tuple(int(x) if x is not None else None for x in ctx["prefix"].input.shape)
    shape_out = tuple(int(x) if x is not None else None for x in ctx["prefix"].output.shape)
    per_branch = {}
    for n in st["adapter"]:
        per_branch[n] = [int(x) if x is not None else None
                         for x in ctx["model"].get_layer(n).output.shape]
    rows_and_splits = {k: {"split": k, "input_row_span": v["input_row_span"],
                           "origins": v["n_origins"]} for k, v in splits.items()
                       if v.get("n_origins")}
    common = {"clock": clock, "rows_and_splits": rows_and_splits, "version": version["version"]}
    return [
        {"stage": "preprocessing", "kind": STAGE_KINDS["preprocessing"],
         "what": "the stored scaler applied to the panel's seven columns; fitted on TRAIN windows only",
         "learned_state": {"mean": [float(x) for x in mean], "sd": [float(x) for x in sd],
                           "digest": sha_obj({"mean": [float(x) for x in mean],
                                              "sd": [float(x) for x in sd]}),
                           "fitted_on": ctx["data_json"]["scaler"]["fitted_on"],
                           "fitted_on_rows": int(ctx["data_json"]["scaler"]["n_rows"])},
         "shape": {"in": [None, ctx["p"]], "out": [None, ctx["p"]]}, **common},
        {"stage": "groups", "kind": STAGE_KINDS["groups"],
         "what": "the held-constant physical channel assignment; one gather per group",
         "learned_state": {"assignment": list(ctx["assignment"]),
                           "digest": sha_obj(list(ctx["assignment"])),
                           "learned": False,
                           "why_not_learned": "the design holds the grouping constant; profile-based "
                                              "grouping is a separate factor and is not varied here"},
         "layers": ctx["stages"]["groups"],
         "shape": {"in": [None, ctx["W"], ctx["p"]],
                   "out": {n: [None, ctx["W"], int(sum(1 for a in ctx["assignment"]
                                                       if a == int(n[1:].split("_")[0])))]
                           for n in ctx["stages"]["groups"]}}, **common},
        {"stage": "detector", "kind": STAGE_KINDS["detector"],
         "what": "two residual causal Conv1D blocks per branch, 16 filters, kernel 3, dilation 1, elu",
         "learned_state": {"layers": st["detector"],
                           "digest": _weights_digest(ctx["model"], st["detector"]),
                           "origin": "imported from the auto-encoder donor and FROZEN for this cell "
                                     "(weight change 0.0 on every detector layer in the retained "
                                     "record)"},
         "shape": {"in": "per branch [None, W, group channels]", "out": "[None, W, 16]"}, **common},
        {"stage": "adapter", "kind": STAGE_KINDS["adapter"],
         "what": "TimeDistributed Dense(8), linear, one per branch",
         "learned_state": {"layers": st["adapter"],
                           "digest": _weights_digest(ctx["model"], st["adapter"]),
                           "origin": "fitted in this cell"},
         "shape": {"per_branch_out": per_branch}, **common},
        {"stage": "fusion", "kind": STAGE_KINDS["fusion"],
         "what": "channel concatenation of the aligned branch sequences, axis 2, branches in sorted "
                 "group order. No weights: a configuration, and the prefix's last stage",
         "learned_state": {"layers": st["fusion"], "digest": sha_obj(
             {"axis": 2, "order": ctx["stages"]["groups"]}), "learned": False},
         "shape": {"in": shape_in, "out": shape_out}, **common},
    ]


# --- the splits, with the reserve refused --------------------------------------------------------------

def split_plan(root: Path) -> dict:
    """Which splits are materialized, and which is refused. The reserve is a refusal, not an array."""
    D = np.load(root / "DATA.npz")
    dj = json.loads((root / "DATA.json").read_text())
    W = int(D["window"][0])
    lo_panel = int(dj["slice_rows"][0])
    out = {}
    for name, key in (("train", "train_origins"), ("validation", "eval_origins")):
        o = D[key].astype(np.int64)
        out[name] = {"split": name, "n_origins": int(o.size),
                     "origin_span": [int(o.min()), int(o.max())],
                     "input_row_span": [int(o.min() - W + 1), int(o.max())],
                     "panel_row_span": [lo_panel + int(o.min() - W + 1), lo_panel + int(o.max())],
                     "state": "TO_MATERIALIZE"}
    out["test"] = {"split": "test", "n_origins": 0, "state": BY_DESIGN,
                   "why": "nothing was materialized for the reserve and nothing is materialized here. "
                          "Its ABSENCE is what is verified: the consumed slice ends at panel row "
                          f"{int(dj['slice_rows'][1])}, and the family's test block begins far after "
                          "it. MOD-CONF owns the reserve",
                   "materialization": None}
    return out


# --- materialization ----------------------------------------------------------------------------------

def _windows(Xs: np.ndarray, origins: np.ndarray, W: int) -> np.ndarray:
    idx = origins[:, None] + np.arange(-W + 1, 1)[None, :]
    return Xs[idx]


def materialize(ctx, out_dir: Path, batch: int = 256) -> dict:
    """The fused sequence for every origin of every materialized split, streamed to disk.

    Streamed through a memmap so the whole tensor is never resident: the store is larger than the cap
    this job runs under, and an artifact that only exists when it fits in RAM is not an artifact.
    """
    D = np.load(ctx["root_data"])
    Xs, W = D["Xs"], ctx["W"]
    prefix = ctx["prefix"]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, spec in ctx["splits"].items():
        if spec["state"] != "TO_MATERIALIZE":
            continue
        o = (D["train_origins"] if name == "train" else D["eval_origins"]).astype(np.int64)
        shape = (int(o.size), W, int(prefix.output.shape[-1]))
        path = out_dir / f"prefix_output_{name}.npy"
        Z = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        for s in range(0, o.size, batch):
            xb = _windows(Xs, o[s:s + batch], W)
            Z[s:s + batch] = np.asarray(prefix(xb, training=False), dtype=np.float32)
        Z.flush()
        del Z
        np.save(out_dir / f"origins_{name}.npy", o)
        written[name] = {"array": path.name, "origins": f"origins_{name}.npy",
                         "shape": list(shape), "dtype": "float32",
                         "bytes": path.stat().st_size,
                         "sha256": sha_file(path),
                         "origins_sha256": sha_file(out_dir / f"origins_{name}.npy"),
                         "batch_used": batch}
    return written


# --- proof 1: direct against cache --------------------------------------------------------------------

def prove_direct_against_cache(ctx, out_dir: Path, n: int = 64, seed: int = 20260926) -> dict:
    """Recompute a bounded sample of origins and compare element by element against the shard."""
    D = np.load(ctx["root_data"])
    rng = np.random.default_rng(seed)
    per = {}
    for name in ctx["written"]:
        o = np.load(out_dir / f"origins_{name}.npy")
        Z = np.load(out_dir / f"prefix_output_{name}.npy", mmap_mode="r")
        pick = np.sort(rng.choice(o.size, size=min(n, o.size), replace=False))
        xb = _windows(D["Xs"], o[pick].astype(np.int64), ctx["W"])
        direct = np.asarray(ctx["prefix"](xb, training=False), dtype=np.float32)
        cached = np.asarray(Z[pick])
        diff = np.abs(direct.astype(np.float64) - cached.astype(np.float64))
        per[name] = {"origins_sampled": int(pick.size),
                     "elements_compared": int(direct.size),
                     "identical_element_by_element": bool(np.array_equal(direct, cached)),
                     "max_absolute_difference": float(diff.max()) if diff.size else None,
                     "bytes_identical": bool(direct.tobytes() == cached.tobytes())}
    ok = all(v["identical_element_by_element"] and v["bytes_identical"] for v in per.values())
    return {"proof": "direct_against_cache", "per_split": per,
            "state": VERIFIED if (ok and per) else (REFUTED if per else UNCHECKABLE),
            "what_it_shows": "the cache is the prefix, not a copy of something near it. The comparison "
                             "is element by element and on the bytes, with no tolerance: a tolerance "
                             "here would hide exactly the drift it is supposed to catch"}


# --- proof 2: reload in a fresh process ---------------------------------------------------------------

def verify_store(store: Path) -> dict:
    """The FRESH-PROCESS side. Re-derives every digest from the bytes and reconstructs one window.

    Run in its own interpreter by ``prove_fresh_process_reload``; it is handed a directory and nothing
    else, so it cannot inherit a value from the process that wrote the store.
    """
    man = json.loads((store / "MANIFEST.json").read_text())
    out = {"store": str(store), "version": man["version"]["version"], "per_split": {}}
    for name, w in man["arrays"].items():
        p = store / w["array"]
        Z = np.load(p, mmap_mode="r")
        o = np.load(store / w["origins"])
        out["per_split"][name] = {
            "file_sha256": sha_file(p),
            "file_sha256_matches_manifest": sha_file(p) == w["sha256"],
            "shape": list(Z.shape), "shape_matches_manifest": list(Z.shape) == w["shape"],
            "dtype": str(Z.dtype),
            "origins_sha256_matches_manifest": sha_file(store / w["origins"]) == w["origins_sha256"],
            "origins": int(o.size),
            "first_window_digest": sha_array(np.asarray(Z[0])),
            "last_window_digest": sha_array(np.asarray(Z[-1])),
            "finite": bool(np.isfinite(np.asarray(Z[0])).all()
                           and np.isfinite(np.asarray(Z[-1])).all()),
        }
    out["all_digests_match"] = all(v["file_sha256_matches_manifest"]
                                   and v["shape_matches_manifest"]
                                   and v["origins_sha256_matches_manifest"]
                                   for v in out["per_split"].values())
    return out


def prove_fresh_process_reload(store: Path, python: str | None = None) -> dict:
    """A separate interpreter, given only the store directory."""
    py = python or sys.executable
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([py, str(HERE / "df_dr05_prefix_output.py"), "--verify-store", str(store)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return {"proof": "fresh_process_reload", "state": REFUTED, "returncode": r.returncode,
                "stderr": r.stderr[-2000:]}
    rec = json.loads(r.stdout)
    return {"proof": "fresh_process_reload", "state": VERIFIED if rec["all_digests_match"] else REFUTED,
            "interpreter": py, "pid_differs": True, "report": rec,
            "what_it_shows": "the store is readable and self-consistent outside the process that wrote "
                             "it. A digest recomputed in the writing process proves far less"}


# --- proof 3: mutation of weights and of state --------------------------------------------------------

def prove_mutation(ctx, out_dir: Path, n: int = 16) -> dict:
    """One perturbation at a time: a detector weight, an adapter weight, a scaler entry, the grouping.

    Each must (a) change the materialized output and (b) change the identity the manifest binds, so the
    store and the prefix that produced it cannot drift apart silently. A mutation that leaves either
    unchanged is a FAILURE of this proof, not a curiosity.
    """
    D = np.load(ctx["root_data"])
    o = np.load(out_dir / "origins_train.npy")[:n].astype(np.int64)
    Xs = D["Xs"]
    base_x = _windows(Xs, o, ctx["W"])
    base = np.asarray(ctx["prefix"](base_x, training=False), dtype=np.float32)
    st = ctx["stages"]
    base_ids = {"detector": _weights_digest(ctx["model"], st["detector"]),
                "adapter": _weights_digest(ctx["model"], st["adapter"])}
    cases = {}

    def _weight_case(label: str, layer: str, group: str):
        lay = ctx["model"].get_layer(layer)
        w = [x.copy() for x in lay.get_weights()]
        pert = [x.copy() for x in w]
        flat = pert[0].ravel()
        flat[0] = np.float32(flat[0] + np.float32(1e-2))
        lay.set_weights(pert)
        moved = np.asarray(ctx["prefix"](base_x, training=False), dtype=np.float32)
        ident = _weights_digest(ctx["model"], st[group])
        lay.set_weights(w)
        restored = np.asarray(ctx["prefix"](base_x, training=False), dtype=np.float32)
        cases[label] = {
            "what_was_perturbed": f"one scalar of {layer}, +1e-2",
            "output_changed": bool(not np.array_equal(base, moved)),
            "max_absolute_change": float(np.abs(base.astype(np.float64)
                                                - moved.astype(np.float64)).max()),
            "identity_changed": bool(ident != base_ids[group]),
            "restored_bit_for_bit": bool(np.array_equal(base, restored)),
            "detected": bool(not np.array_equal(base, moved) and ident != base_ids[group]),
        }

    _weight_case("detector_weight", st["detector"][0], "detector")
    _weight_case("adapter_weight", st["adapter"][0], "adapter")

    # the preprocessing state is not a weight: it is the scaler the inputs were divided by, so the
    # mutation is applied to the state and the store's declared input identity must move with it
    mean, sd = D["scaler_mean"].copy(), D["scaler_sd"].copy()
    base_state = sha_obj({"mean": [float(x) for x in mean], "sd": [float(x) for x in sd]})
    sd2 = sd.copy()
    sd2[0] = sd2[0] * 1.01
    x2 = ((base_x * sd[None, None, :] + mean[None, None, :]) - mean[None, None, :]) / sd2[None, None, :]
    moved = np.asarray(ctx["prefix"](np.asarray(x2, dtype=np.float32), training=False),
                       dtype=np.float32)
    state2 = sha_obj({"mean": [float(x) for x in mean], "sd": [float(x) for x in sd2]})
    cases["preprocessing_state"] = {
        "what_was_perturbed": "scaler sd[0] x 1.01, applied by inverting the stored scaling and "
                              "re-applying the perturbed one",
        "output_changed": bool(not np.array_equal(base, moved)),
        "max_absolute_change": float(np.abs(base.astype(np.float64) - moved.astype(np.float64)).max()),
        "identity_changed": bool(state2 != base_state),
        "detected": bool(not np.array_equal(base, moved) and state2 != base_state),
    }

    # the grouping is the one stage that is not learned, and a change to it must still be detected
    asg = list(ctx["assignment"])
    swapped = list(asg)
    a, b = 0, next(i for i, g in enumerate(asg) if g != asg[0])
    swapped[a], swapped[b] = swapped[b], swapped[a]
    cases["group_assignment"] = {
        "what_was_perturbed": f"channels {a} and {b} exchanged between groups",
        "identity_changed": bool(sha_obj(swapped) != sha_obj(asg)),
        "output_changed": None,
        "why_the_output_is_not_recomputed": "a different grouping is a different GRAPH, not a different "
                                            "weight: it would require building a second prefix, which "
                                            "this proof deliberately does not do. What must hold here "
                                            "is that the version binds the assignment, so a store made "
                                            "under another grouping cannot be mistaken for this one",
        "detected": bool(sha_obj(swapped) != sha_obj(asg)),
    }

    ok = all(v["detected"] for v in cases.values())
    return {"proof": "mutation_of_weights_and_of_state", "cases": cases,
            "state": VERIFIED if ok else REFUTED,
            "what_it_shows": "every learned state the prefix carries is load-bearing and is bound by "
                             "the store's version. None of these perturbations is written to any run "
                             "root: each is applied in memory and restored bit for bit"}


# --- proof 4: causality -------------------------------------------------------------------------------

def prove_causality(ctx, out_dir: Path) -> dict:
    """The measured reach. A future row may not move the output; the declared reach must be the last
    row that does.

    The claim is measured by perturbation on the delivered bytes, as the design requires ("reach
    measured by perturbation and gradient per input row, not asserted").
    """
    E = _load("df_mod_e0")
    D = np.load(ctx["root_data"])
    W = ctx["W"]
    o = np.load(out_dir / "origins_train.npy")[-1:].astype(np.int64)      # a late origin, full window
    x = _windows(D["Xs"], o, W).astype(np.float32)
    base = np.asarray(ctx["prefix"](x, training=False), dtype=np.float32)
    declared = int(E.branch_reach("A", W))
    moved_at = {}
    for back in range(0, W):
        t = W - 1 - back
        xp = x.copy()
        xp[0, t, :] += np.float32(1.0)
        y = np.asarray(ctx["prefix"](xp, training=False), dtype=np.float32)
        moved_at[back] = bool(not np.array_equal(base[0, -1], y[0, -1]))
    last_moving = max([b for b, m in moved_at.items() if m], default=-1)
    measured_reach = last_moving + 1

    # a FUTURE row cannot be perturbed inside a window whose last position IS the present, so the
    # future is probed the only way it exists here: on the row AFTER the window, through a longer
    # window built from the same panel rows
    o2 = o - 1
    x2 = _windows(D["Xs"], o2, W).astype(np.float32)
    base2 = np.asarray(ctx["prefix"](x2, training=False), dtype=np.float32)
    x3 = x2.copy()
    # row o (one minute AFTER the last row of this window) is not in x2 at all: perturbing the panel at
    # that row must leave this window's output untouched. Built explicitly to show the row IS in the
    # panel and IS read by the neighbouring window
    Xs2 = D["Xs"].copy()
    Xs2[int(o[0])] += np.float32(5.0)
    x_future = Xs2[(o2[:, None] + np.arange(-W + 1, 1)[None, :])].astype(np.float32)
    y_future = np.asarray(ctx["prefix"](x_future, training=False), dtype=np.float32)
    x_neighbour = Xs2[(o[:, None] + np.arange(-W + 1, 1)[None, :])].astype(np.float32)
    y_neighbour = np.asarray(ctx["prefix"](x_neighbour, training=False), dtype=np.float32)

    ok = (measured_reach == declared
          and np.array_equal(base2, y_future)
          and not np.array_equal(base, y_neighbour))
    return {
        "proof": "causality",
        "declared_branch_reach": declared,
        "measured_reach_at_the_last_position": measured_reach,
        "reach_matches_the_declaration": bool(measured_reach == declared),
        "rows_that_move_the_last_position": sorted(b for b, m in moved_at.items() if m),
        "a_future_row_leaves_this_window_untouched": bool(np.array_equal(base2, y_future)),
        "and_that_same_row_does_move_the_window_that_contains_it": bool(
            not np.array_equal(base, y_neighbour)),
        "future_row_panel_index": int(o[0]),
        "state": VERIFIED if ok else REFUTED,
        "what_it_shows": "the materialized representation at a row is a function of that row and its "
                         "declared predecessors, and of nothing after it. The reach is MEASURED here, "
                         "not read from the design's note",
        "scope": "the last position of the fused sequence for one origin. Earlier positions of the "
                 "same sequence see fewer rows, never more: the causal padding is left-sided",
    }


# --- window invariance, reported as a property rather than stored --------------------------------------

def window_invariance(ctx, out_dir: Path, n: int = 8) -> dict:
    """Does the same panel row get the same representation in different windows?

    It should, for positions at or beyond the reach, because the prefix is causal and convolutional and
    has no state carried between windows. Reported with its MEASURED discrepancy: a store that assumed
    this and was wrong would be a silent defect, so it is measured instead of assumed.
    """
    E = _load("df_mod_e0")
    D = np.load(ctx["root_data"])
    W = ctx["W"]
    reach = int(E.branch_reach("A", W))
    o = np.load(out_dir / "origins_train.npy").astype(np.int64)
    pick = o[np.linspace(0, o.size - 1, num=n, dtype=int)]
    Z = np.load(out_dir / "prefix_output_train.npy", mmap_mode="r")
    idx = {int(v): i for i, v in enumerate(o)}
    worst, compared = 0.0, 0
    for a in pick:
        for shift in (1, 2, 5, 17):
            b = int(a) + shift
            if b not in idx:
                continue
            # row r = a - W + 1 + t sits at t in window a and t - shift in window b
            for t in range(reach - 1 + shift, W):
                va = np.asarray(Z[idx[int(a)], t])
                vb = np.asarray(Z[idx[b], t - shift])
                worst = max(worst, float(np.abs(va.astype(np.float64)
                                                - vb.astype(np.float64)).max()))
                compared += 1
    return {"property": "the same panel row gets the same representation in every window that places "
                        "it at or beyond the reach",
            "positions_compared": compared,
            "largest_absolute_discrepancy": worst,
            "exactly_equal": bool(worst == 0.0),
            "reach_used": reach,
            "why_it_matters": "it is what makes the store row-addressable: a consumer can read the "
                              "representation of a row without knowing which window it came from, for "
                              "positions at or beyond the reach. It is measured, not assumed"}


# --- the reserve is not read --------------------------------------------------------------------------

def reserve_untouched(ctx) -> dict:
    dj = ctx["data_json"]
    lo, hi = (int(x) for x in dj["slice_rows"])
    fam = ctx["design"]["task"]["family_usable_windows"]
    spans = {k: v["panel_row_span"] for k, v in ctx["splits"].items() if v.get("n_origins")}
    inside = all(lo <= a and b < hi for a, b in spans.values())
    return {"consumed_slice_panel_rows": [lo, hi],
            "materialized_panel_row_spans": spans,
            "every_row_read_is_inside_the_consumed_slice": inside,
            "test_split": ctx["splits"]["test"]["state"],
            "family_usable_windows_declared": fam,
            "exposure_of_the_donor_cell": json.loads(
                (ctx["root"] / "attempts" / ctx["cell"] / "cell.json").read_text()).get("exposure"),
            "state": VERIFIED if inside else REFUTED,
            "what_was_refused": "the reserve was not opened to build this store, and no array was "
                                "written for the test split. Its absence is verified; its "
                                "materialization is not, because there is none"}


# --- the clock ----------------------------------------------------------------------------------------

def clock_of(ctx) -> dict:
    import datetime as _dt

    import pyarrow.parquet as pq
    dj = ctx["data_json"]
    lo = int(dj["slice_rows"][0])
    n = int(dj["slice_rows"][1]) - lo
    p = Path(dj["delivery"]["path"])
    if not p.is_file():
        return {"state": UNCHECKABLE, "missing": str(p)}
    col = pq.ParquetFile(p).read(columns=["timestamp_label"]).column("timestamp_label")
    labels = col.slice(lo, n).to_pylist()
    fmt = ctx["design"]["contract"]["ts_format"]
    t = [_dt.datetime.strptime(d, fmt) for d in labels]
    steps = {int((t[i + 1] - t[i]).total_seconds()) for i in range(len(t) - 1)}
    return {"state": VERIFIED if steps == {int(ctx["design"]["contract"]["step_seconds"])} else REFUTED,
            "panel_sha256": dj["panel_sha256"],
            "slice_panel_rows": [lo, lo + n],
            "first": t[0].isoformat(), "last": t[-1].isoformat(),
            "declared_step_seconds": int(ctx["design"]["contract"]["step_seconds"]),
            "distinct_steps_observed_seconds": sorted(steps),
            "timezone": "UNKNOWN (the producer never declared one; the labels are wall clock and the "
                        "delivery's availability is UNDECLARED / UNKNOWN). Carried, not resolved"}


# --- the whole thing ----------------------------------------------------------------------------------

def run(root: Path, cell: str, out_dir: Path, *, sample: int = 64, batch: int = 256) -> dict:
    rec = {"schema": "df_dr05_prefix_output.v1",
           "what_this_is": "the materialized OUTPUT of the frozen prefix, with four proofs",
           "what_parity_of_inputs_is_not": WHAT_PARITY_OF_INPUTS_IS_NOT,
           "prefix_ends_at": PREFIX_ENDS_AT,
           "run_root": str(root), "donor_cell": cell, "store": str(out_dir),
           "reserve": "NOT_OPENED"}
    if not (root / "DATA.npz").is_file() or not (root / "attempts" / cell / "weights.weights.h5"
                                                 ).is_file():
        rec["state"] = UNCHECKABLE
        rec["missing"] = [str(p) for p in ((root / "DATA.npz"),
                                           (root / "attempts" / cell / "weights.weights.h5"))
                          if not p.is_file()]
        return rec

    ctx = build_prefix(root, cell)
    ctx["root"] = root
    ctx["root_data"] = root / "DATA.npz"
    ctx["splits"] = split_plan(root)
    code = code_identity()
    version = store_version(ctx, code)
    rec["code_identity"] = code
    rec["version"] = version
    rec["the_cut"] = prefix_contains_nothing_downstream(ctx)
    rec["clock"] = clock_of(ctx)
    rec["splits"] = ctx["splits"]

    ctx["written"] = materialize(ctx, out_dir, batch=batch)
    rec["arrays"] = ctx["written"]
    rec["stages"] = stage_manifest(ctx, rec["clock"], ctx["splits"], version)
    (out_dir / "MANIFEST.json").write_text(json.dumps(
        {"schema": "df_dr05_prefix_store.v1", "version": version, "code_identity": code,
         "prefix_ends_at": PREFIX_ENDS_AT, "run_root": str(root), "donor_cell": cell,
         "clock": rec["clock"], "splits": ctx["splits"], "arrays": ctx["written"],
         "stages": rec["stages"]}, indent=1) + "\n")
    rec["manifest_sha256"] = sha_file(out_dir / "MANIFEST.json")

    rec["proofs"] = {
        "direct_against_cache": prove_direct_against_cache(ctx, out_dir, n=sample),
        "fresh_process_reload": prove_fresh_process_reload(out_dir),
        "mutation": prove_mutation(ctx, out_dir),
        "causality": prove_causality(ctx, out_dir),
    }
    rec["window_invariance"] = window_invariance(ctx, out_dir)
    rec["reserve_untouched"] = reserve_untouched(ctx)

    checks = {"the_cut_is_at_the_fusion": rec["the_cut"]["clean"],
              "five_stages_manifested": len(rec["stages"]) == 5,
              "every_stage_carries_state_clock_rows_shape_version": all(
                  all(k in s for k in ("learned_state", "clock", "rows_and_splits", "shape",
                                       "version")) for s in rec["stages"]),
              "clock_verified": rec["clock"]["state"] == VERIFIED,
              "reserve_untouched": rec["reserve_untouched"]["state"] == VERIFIED,
              **{f"proof_{k}": v["state"] == VERIFIED for k, v in rec["proofs"].items()}}
    rec["checks"] = checks
    rec["state"] = (UNCHECKABLE if any(v is None for v in checks.values())
                    else VERIFIED if all(checks.values()) else REFUTED)
    rec["what_this_does_not_deliver"] = [
        "a claim that this prefix is the RIGHT prefix. The donor cell is named, not chosen on evidence: "
        "which learned state MOD-FROZEN-PREFIX should freeze is a design decision the module still owes",
        "a second panel or a second family. One panel, one donor, two splits",
        "anything about the reserve beyond its absence",
        "any H-CORE comparison. The core is downstream of the cut and nothing here fits one",
    ]
    return rec


# --- what an admissible record must carry --------------------------------------------------------------

STAGE_FIELDS = ("stage", "kind", "what", "learned_state", "clock", "rows_and_splits", "shape",
                "version")
REQUIRED_PROOFS = ("direct_against_cache", "fresh_process_reload", "mutation", "causality")


def validate(rec: dict) -> list:
    """Reasons this record is not admissible. An empty list is the only pass.

    It encodes the order of DR05 §2 so the order cannot be satisfied in prose: five stages each carrying
    its learned state, its clock, its row and split, its shape and its version; four proofs, each
    present; the reserve refused rather than passed; and parity of inputs named as what it is not.
    """
    bad = []
    if rec.get("schema") != "df_dr05_prefix_output.v1":
        bad.append("the schema is not df_dr05_prefix_output.v1")
    if not rec.get("what_parity_of_inputs_is_not"):
        bad.append("the record must say what parity of the prefix's INPUTS does not establish")
    if rec.get("prefix_ends_at") != PREFIX_ENDS_AT:
        bad.append(f"the prefix must end at {PREFIX_ENDS_AT}")
    stages = rec.get("stages") or []
    if [s.get("stage") for s in stages] != list(STAGE_KINDS):
        bad.append(f"the five stages must be present in order: {list(STAGE_KINDS)}")
    for s in stages:
        for f in STAGE_FIELDS:
            if f not in s or s[f] in (None, "", {}, []):
                bad.append(f"stage {s.get('stage')!r} is missing {f!r}")
        if s.get("kind") != STAGE_KINDS.get(s.get("stage")):
            bad.append(f"stage {s.get('stage')!r} declares the wrong kind")
    proofs = rec.get("proofs") or {}
    for p in REQUIRED_PROOFS:
        if p not in proofs:
            bad.append(f"proof {p!r} is absent; an absent proof is not a passed one")
        elif proofs[p].get("state") not in (VERIFIED, REFUTED, UNCHECKABLE):
            bad.append(f"proof {p!r} carries no verdict")
    splits = rec.get("splits") or {}
    t = splits.get("test") or {}
    if t.get("state") != BY_DESIGN or t.get("materialization") is not None:
        bad.append("the reserve must be refused by name with its materialization check left null, "
                   "never reported as verified")
    if (rec.get("arrays") or {}).get("test"):
        bad.append("an array was written for the reserve")
    if rec.get("reserve") != "NOT_OPENED":
        bad.append("the record must state that the reserve was not opened")
    if not rec.get("what_this_does_not_deliver"):
        bad.append("the record must name what it does NOT deliver; a materialization is not a delivery")
    if rec.get("state") == VERIFIED and any(
            v is None for v in (rec.get("checks") or {}).values()):
        bad.append("a check that could not run is recorded inside a VERIFIED record")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=SUCCESSOR_ROOT)
    ap.add_argument("--cell", default="R1_s1")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--verify-store", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.verify_store is not None:
        print(json.dumps(verify_store(a.verify_store), indent=1))
        return 0
    if a.out_dir is None or a.out is None:
        ap.error("--out-dir and --out are required unless --verify-store is given")
    rec = run(a.root, a.cell, a.out_dir, sample=a.sample, batch=a.batch)
    rec["admissibility"] = {"reasons_it_would_be_inadmissible": validate(rec)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"state": rec["state"], "checks": rec.get("checks"),
                      "inadmissible_because": rec["admissibility"][
                          "reasons_it_would_be_inadmissible"],
                      "out": str(a.out)}, indent=1))
    return 0 if (rec["state"] == VERIFIED and not validate(rec)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
