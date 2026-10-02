"""N2 coverage for ``Plugin._run_neat``: guards, atomic-state resume, fail-closed identity."""
import ast
import inspect
import json

import pytest

from optimizer_plugins.modular_doin_optimizer import Plugin
from tools import modular_neat_policy as neat_policy
from tools import modular_search_space as ss
from tests.test_modular_doin_m04 import FakeExecutor, make_campaign


class ScoreExecutor(FakeExecutor):
    """Deterministic verified objective over real candidate fields; counts trainings."""

    def __init__(self, campaign):
        super().__init__(campaign, {})

    def train(self, nested, output_root, declaration):
        value = (nested["model"]["core"]["params"]["d_model"] / 1000.0
                 + nested["evaluator"]["learning_rate"] + nested["evaluator"]["seed"] / 1e9)
        self.values = {(nested["evaluator"]["loss"], nested["evaluator"]["seed"]): value}
        return FakeExecutor.train(self, nested, output_root, declaration)

    def verify(self, receipt, output_root, declaration):
        return {"status": "completed", "verdict": "VERIFIED"}


def neat_plugin(campaign, executor, pop=3, gens=1, **extra):
    plugin = Plugin(executor=executor)
    plugin.set_params(modular_campaign_root=str(campaign.root), modular_proposal_strategy="neat",
                      modular_neat_population_size=pop, modular_neat_generations=gens,
                      modular_neat_seed=5, **extra)
    plugin._campaign = lambda: campaign
    return plugin


def count(campaign):
    return campaign.db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]


def test_run_neat_directly_trains_every_seed_and_writes_state(tmp_path):
    campaign = make_campaign(tmp_path)
    executor = ScoreExecutor(campaign)
    plugin = neat_plugin(campaign, executor, pop=3, gens=2)
    plugin._run_neat(campaign, executor)
    state = json.loads((campaign.root / "NEAT_STATE.json").read_text())
    assert state["generation"] == 2 and len(state["population"]) == 3
    assert not (campaign.root / "NEAT_STATE.json.tmp").exists()
    counts = campaign.status()["counts"]
    assert set(counts) == {"verified"} and 6 <= counts["verified"] <= 12
    # every trained candidate was trained exactly once
    assert len(executor.trained) == len(set(executor.trained)) == counts["verified"]


def test_three_seeds_accepted(tmp_path):
    campaign = make_campaign(tmp_path, seeds=(2021, 2022, 2023))
    executor = ScoreExecutor(campaign)
    neat_plugin(campaign, executor, pop=2)._run_neat(campaign, executor)
    assert campaign.status()["counts"]["verified"] >= 6


def test_four_seeds_raise_from_run_neat_before_any_enqueue(tmp_path):
    campaign = make_campaign(tmp_path, seeds=(2021, 2022, 2023, 2024))
    plugin = neat_plugin(campaign, object())
    with pytest.raises(ValueError, match="one to three"):
        plugin._run_neat(campaign, object())
    assert count(campaign) == 0
    assert not (campaign.root / "NEAT_STATE.json").exists()


def test_candidate_cap_below_population_generations_seeds_raises(tmp_path):
    campaign = make_campaign(tmp_path)  # 2 seeds; 3 x 2 x 2 = 12 required
    plugin = neat_plugin(campaign, object(), pop=3, gens=2, modular_max_candidates=11)
    with pytest.raises(ValueError, match="cannot split"):
        plugin._run_neat(campaign, object())
    assert count(campaign) == 0
    # exactly the required count is allowed
    executor = ScoreExecutor(campaign)
    neat_plugin(campaign, executor, pop=3, gens=2, modular_max_candidates=12)._run_neat(campaign, executor)
    assert json.loads((campaign.root / "NEAT_STATE.json").read_text())["generation"] == 2


def test_zero_generations_raises(tmp_path):
    campaign = make_campaign(tmp_path)
    with pytest.raises(ValueError, match="generations"):
        neat_plugin(campaign, object(), gens=0)._run_neat(campaign, object())


def test_crash_between_enqueue_and_state_write_resumes_idempotently(tmp_path, monkeypatch):
    campaign = make_campaign(tmp_path)
    executor = ScoreExecutor(campaign)
    plugin = neat_plugin(campaign, executor, pop=3)

    def die(path, payload):
        raise OSError("crash before NEAT_STATE is durable")

    monkeypatch.setattr(Plugin, "_write_json", staticmethod(die))
    with pytest.raises(OSError):
        plugin._run_neat(campaign, executor)
    assert not (campaign.root / "NEAT_STATE.json").exists()
    queued = count(campaign)
    trained = list(executor.trained)
    assert queued == len(trained) > 0
    monkeypatch.undo()

    resumed = ScoreExecutor(campaign)
    reopened = neat_plugin(camp_reopen(campaign), resumed, pop=3)
    reopened._run_neat(reopened._campaign(), resumed)
    assert resumed.trained == []                       # nothing retrained
    assert count(campaign) == queued                   # cid dedup: no duplicate rows
    state = json.loads((campaign.root / "NEAT_STATE.json").read_text())
    assert state["generation"] == 1


def camp_reopen(campaign):
    from tools import modular_doin_campaign as camp
    return camp.Campaign(campaign.root)


def test_crash_during_training_resumes_and_finishes(tmp_path):
    campaign = make_campaign(tmp_path)
    executor = ScoreExecutor(campaign)
    first = neat_plugin(campaign, executor, pop=3)
    victim = campaign.enqueue  # keep reference; crash on the third candidate trained
    order = []
    original = executor.train

    def flaky(nested, output_root, declaration):
        order.append(ss.digest(nested))
        if len(order) == 3:
            raise KeyboardInterrupt
        return original(nested, output_root, declaration)

    executor.train = flaky
    with pytest.raises(KeyboardInterrupt):
        first._run_neat(campaign, executor)
    assert not (campaign.root / "NEAT_STATE.json").exists()
    del victim
    reopened = camp_reopen(campaign)
    resumed = ScoreExecutor(reopened)
    plugin = neat_plugin(reopened, resumed, pop=3)
    plugin._run_neat(reopened, resumed)
    assert json.loads((reopened.root / "NEAT_STATE.json").read_text())["generation"] == 1
    assert set(reopened.status()["counts"]) == {"verified"}


def test_second_invocation_continues_from_persisted_generation(tmp_path):
    campaign = make_campaign(tmp_path)
    executor = ScoreExecutor(campaign)
    neat_plugin(campaign, executor, pop=3)._run_neat(campaign, executor)
    neat_plugin(campaign, executor, pop=3)._run_neat(campaign, executor)
    assert json.loads((campaign.root / "NEAT_STATE.json").read_text())["generation"] == 2


def test_reused_identity_with_a_different_seed_set_fails_closed(tmp_path):
    campaign = make_campaign(tmp_path)  # declared seeds 2021, 2022
    executor = ScoreExecutor(campaign)
    plugin = neat_plugin(campaign, executor, pop=3)
    policy = neat_policy.ModularNeatProposalPolicy(
        campaign.space, campaign.declaration["default_candidate"], campaign.declaration["base"],
        population_size=3, seed=5, default_huber_delta=campaign.declaration["default_huber_delta"])
    first = policy.ask()[1]
    # the same configuration identity was earlier queued on a foreign seed (2023) only
    campaign.enqueue(first, "foreign_seed_set", seeds=[2023])
    with pytest.raises(RuntimeError, match="paired-seed verified fitness"):
        plugin._run_neat(campaign, executor)
    assert not (campaign.root / "NEAT_STATE.json").exists()


def test_decode_has_no_dead_local_variables():
    source = inspect.getsource(neat_policy.ModularNeatProposalPolicy._decode)
    import textwrap
    func = ast.parse(textwrap.dedent(source)).body[0]
    stored = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loaded = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert stored - loaded == set()
