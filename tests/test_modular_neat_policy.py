"""Contract tests for NEAT proposals over the modular conditional search space."""
import json
from pathlib import Path

from tools.modular_neat_policy import ModularNeatProposalPolicy
from tools import modular_search_space as ss


ROOT = Path(__file__).resolve().parents[1]
SPACE = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_search_space_v2_full_grid.json").read_text())
DEFAULT = json.loads((ROOT / "examples/config/modular_doin/ecl_l24_h24_default_r0_v2_full_grid.json").read_text())
BASE = {
    "feature_names": [f"c{i}" for i in range(321)],
    "window": 24,
    "sample_hours": 1,
    "horizons": list(range(1, 25)),
    "target_feature_indices": list(range(321)),
    "objective": {"metric": "MAE", "split": "validation", "higher_is_better": False},
    "evaluator_fixed": {"max_updates": 100, "max_seconds": 30.0},
    "donors": {},
}


def policy(seed=17):
    return ModularNeatProposalPolicy(SPACE, DEFAULT, BASE, population_size=6, seed=seed,
                                     default_huber_delta=1.0)


def test_initial_population_is_deterministic_unique_and_engine_valid():
    left, right = policy(), policy()
    a, b = left.ask(), right.ask()
    assert a == b and len(a) == 6
    assert a[0] == {**DEFAULT, "train.huber_delta": 1.0}
    assert len({ss.digest(row) for row in a}) == len(a)
    for row in a:
        assert "train.seed" not in row
        with_seed = {**row, "train.seed": 2021}
        ss.validate_flat(with_seed, SPACE)
        ss.from_flat(with_seed, BASE, SPACE)


def test_feedback_advances_generation_and_round_trips_persistent_state():
    first = policy()
    candidates = first.ask()
    scores = {ss.digest(row): float(i + 1) for i, row in enumerate(candidates)}
    first.tell(scores)
    assert first.generation == 1
    state = first.to_state()
    restored = ModularNeatProposalPolicy.from_state(SPACE, DEFAULT, BASE, state)
    assert restored.to_state() == state
    assert restored.ask() == first.ask()


def test_missing_or_foreign_fitness_is_rejected_before_reproduction():
    p = policy()
    candidates = p.ask()
    scores = {ss.digest(row): 1.0 for row in candidates[:-1]}
    try:
        p.tell(scores)
    except ValueError as exc:
        assert "fitness" in str(exc)
    else:
        raise AssertionError("partial fitness must not advance a generation")
    assert p.generation == 0
