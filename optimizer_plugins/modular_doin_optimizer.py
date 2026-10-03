"""Opt-in ``optimizer.plugins`` entry for the modular temporal predictor (M04).

Same plugin surface as ``default_optimizer`` (``plugin_params``, ``set_params``,
``get_debug_info``, ``add_debug_info``, ``optimize(predictor_plugin,
preprocessor_plugin, config)``) and the same flat ``hyperparameter_bounds``
vocabulary, but candidates are *not* evaluated by the legacy pipeline: each flat
parameter set is mapped reversibly (``tools.modular_search_space``) to a
versioned nested ``modular.candidate.v1``, persisted in the campaign queue
before execution, trained by the real modular evaluator through the DOIN
predictor bridge, and independently rescored from its saved checkpoint.

``optimize`` returns the incumbent's flat hyperparameters (seed excluded) plus
nothing else; the incumbent is the paired-seed verified configuration with the
best mean declared validation objective. The legacy ``default_optimizer`` and
existing configs are untouched; selecting this plugin is explicit.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from tools import modular_doin_campaign as campaign_mod
from tools import modular_search_space as ss


class Plugin:
    plugin_params = {
        "modular_campaign_root": None,
        "modular_campaign_declaration": None,
        "modular_proposal_draws": 0,
        "modular_proposal_seed": 0,
        "modular_max_candidates": None,
        "modular_proposal_strategy": "random",
        "modular_neat_population_size": 12,
        "modular_neat_generations": 1,
        "modular_neat_seed": 0,
        "modular_neat_restrict": None,
        "hyperparameter_bounds": None,
    }
    plugin_debug_vars = ["modular_campaign_root", "modular_proposal_draws", "modular_max_candidates",
                         "modular_proposal_strategy", "modular_neat_population_size",
                         "modular_neat_generations", "modular_neat_seed", "modular_neat_restrict"]

    def __init__(self, executor=None):
        self.params = dict(self.plugin_params)
        self.executor = executor

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.params[key] = value

    def get_debug_info(self):
        return {var: self.params.get(var) for var in self.plugin_debug_vars}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    # The explicit reversible mapping, exposed for callers of the plugin API.
    def to_nested(self, flat, declaration):
        return ss.from_flat(flat, declaration["base"], declaration["search_space"])

    def to_flat(self, nested, declaration):
        return ss.to_flat(nested, declaration["search_space"])

    def _campaign(self):
        root = Path(self.params["modular_campaign_root"])
        if (root / "CAMPAIGN.json").exists():
            return campaign_mod.Campaign(root)
        declaration = self.params["modular_campaign_declaration"]
        if isinstance(declaration, (str, Path)):
            declaration = json.loads(Path(declaration).read_text())
        return campaign_mod.Campaign.create(root, declaration)

    @staticmethod
    def _write_json(path, payload):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        with tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _verified_objective(campaign, flat):
        declaration = campaign.declaration
        seeds = list(declaration["paired_seeds"])
        config_ids = set()
        for seed in seeds:
            nested = ss.from_flat({**flat, "train.seed": seed}, declaration["base"], campaign.space)
            config_ids.add(ss.config_identity(nested))
        if len(config_ids) != 1:
            raise RuntimeError("one NEAT candidate produced multiple configuration identities across seeds")
        rows = campaign.db.execute(
            "SELECT seed,status,objective FROM candidates WHERE config_id=? ORDER BY seed",
            (config_ids.pop(),)).fetchall()
        by_seed = {int(row["seed"]): row for row in rows}
        if set(by_seed) != set(seeds) or any(by_seed[s]["status"] != "verified" for s in seeds):
            raise RuntimeError("NEAT generation lacks complete paired-seed verified fitness")
        return sum(float(by_seed[s]["objective"]) for s in seeds) / len(seeds)

    def _run_neat(self, campaign, executor):
        from tools.modular_neat_policy import ModularNeatProposalPolicy

        population_size = int(self.params["modular_neat_population_size"])
        generations = int(self.params["modular_neat_generations"])
        if generations < 1:
            raise ValueError("modular_neat_generations must be >= 1")
        seeds = campaign.declaration["paired_seeds"]
        if not 1 <= len(seeds) <= 3:
            raise ValueError("modular NEAT requires one to three paired seeds")
        required = population_size * generations * len(seeds)
        cap = self.params.get("modular_max_candidates")
        if cap is not None and int(cap) < required:
            raise ValueError("modular_max_candidates cannot split a NEAT generation")
        state_path = campaign.root / "NEAT_STATE.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            policy = ModularNeatProposalPolicy.from_state(
                campaign.space, campaign.declaration["default_candidate"],
                campaign.declaration["base"], state, restrict=self.params.get("modular_neat_restrict"))
        else:
            policy = ModularNeatProposalPolicy(
                campaign.space, campaign.declaration["default_candidate"],
                campaign.declaration["base"], population_size=population_size,
                seed=int(self.params["modular_neat_seed"]),
                default_huber_delta=campaign.declaration["default_huber_delta"],
                restrict=self.params.get("modular_neat_restrict"))
        target_generation = policy.generation + generations
        while policy.generation < target_generation:
            candidates = policy.ask()
            generation = policy.generation
            for index, flat in enumerate(candidates):
                campaign.enqueue(flat, f"neat_g{generation:04d}_i{index:04d}")
            campaign.run(executor)
            fitness = {ss.digest(flat): self._verified_objective(campaign, flat) for flat in candidates}
            policy.tell(fitness)
            self._write_json(state_path, policy.to_state())

    def optimize(self, predictor_plugin, preprocessor_plugin, config):
        del predictor_plugin, preprocessor_plugin  # the modular evaluator owns model and windows
        self.set_params(**{k: v for k, v in config.items() if k in self.plugin_params})
        campaign = self._campaign()
        bounds = self.params.get("hyperparameter_bounds")
        if bounds is not None and bounds != campaign.space["bounds"]:
            raise ValueError("hyperparameter_bounds differ from the campaign's declared search space")
        strategy = self.params.get("modular_proposal_strategy", "random")
        if strategy not in ("random", "neat"):
            raise ValueError("modular_proposal_strategy must be random or neat")
        executor = self.executor or campaign_mod.DoinBridgeExecutor(campaign.declaration)
        if strategy == "neat":
            self._run_neat(campaign, executor)
        elif campaign.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0:
            decl = campaign.declaration
            h, m = campaign_mod.paired_loss_arms(decl["default_candidate"], decl["default_huber_delta"])
            campaign.enqueue(h, "default_R0_huber")
            campaign.enqueue(m, "default_R0_mae")
            rng = random.Random(self.params["modular_proposal_seed"])
            for k in range(self.params["modular_proposal_draws"]):
                draw = campaign_mod.propose(campaign.space, rng, fixed=decl.get("fixed_in_batch"))
                delta = campaign_mod._sample(campaign.space["bounds"]["train.huber_delta"], rng)
                h, m = campaign_mod.paired_loss_arms(draw, delta)
                campaign.enqueue(h, f"draw{k}_huber")
                campaign.enqueue(m, f"draw{k}_mae")
        if strategy == "random":
            campaign.run(executor, self.params.get("modular_max_candidates"))
        incumbent = campaign.update_incumbent()
        if incumbent is None:
            raise RuntimeError("no paired-seed verified configuration yet; queue state is persisted")
        cid = json.loads(incumbent["cids"])[0]
        flat = json.loads(campaign.db.execute("SELECT flat FROM candidates WHERE cid=?", (cid,)).fetchone()[0])
        flat.pop("train.seed")
        return flat
