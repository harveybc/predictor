import copy
import json
from pathlib import Path

from tools.test_led_state import load_state, validate_state


REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "docs/metodologias/PROJECT_METHOD_STATE.example.json"


def test_example_state_is_valid():
    assert validate_state(load_state(EXAMPLE), REPO) == []


def test_future_stage_cannot_start_early():
    state = copy.deepcopy(load_state(EXAMPLE))
    state["stages"][2]["status"] = "IN_PROGRESS"
    assert any("future stage started" in item for item in validate_state(state, REPO))


def test_passed_stage_requires_review_and_evidence():
    state = copy.deepcopy(load_state(EXAMPLE))
    state["stages"][0]["evidence"] = []
    state["stages"][0]["review"] = None
    errors = validate_state(state, REPO)
    assert any("PASSED requires evidence and review" in item for item in errors)


def test_duplicate_json_key_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    try:
        load_state(path)
    except ValueError as exc:
        assert "duplicate JSON key" in str(exc)
    else:
        raise AssertionError("duplicate key accepted")


def test_missing_passed_evidence_is_rejected():
    state = copy.deepcopy(load_state(EXAMPLE))
    state["stages"][0]["evidence"] = ["docs/does-not-exist"]
    errors = validate_state(state, REPO)
    assert any("missing or unsafe reference" in item for item in errors)
