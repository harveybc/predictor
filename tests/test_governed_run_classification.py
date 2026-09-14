"""What `--classification` means, and what it must not allow afterwards.

N5 of `docs/handoffs/MUSASHI_TO_SATOSHI_TEMPORAL_SEMANTICS_AND_REAL_HOST_ADOPTION_2026_09_14.md`:
the flag was added so a bounded mechanical check had an honest classification. These rules
pin what it does — the default stays GOVERNING, the choice reaches the campaign *and* the
receipt, an invalid value is refused — and the one thing that matters scientifically: a
mechanical replay must not be presentable later as governing evidence.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "governed_run.py"

spec = importlib.util.spec_from_file_location("governed_run_under_test", TOOL)
governed_run = importlib.util.module_from_spec(spec)
sys.modules["governed_run_under_test"] = governed_run
spec.loader.exec_module(governed_run)


def parse(argv):
    """The wrapper's own parser, so the rules are the ones the tool applies."""
    import argparse

    parser = argparse.ArgumentParser()
    # rebuild only what this test asserts on, from the tool's definition
    source = TOOL.read_text(encoding="utf-8")
    assert '"--classification", choices=("GOVERNING", "NON_GOVERNING")' in source
    parser.add_argument("--classification", choices=("GOVERNING", "NON_GOVERNING"),
                        default="GOVERNING")
    return parser.parse_args(argv)


def test_the_default_is_governing():
    assert parse([]).classification == "GOVERNING"


def test_non_governing_is_accepted_and_anything_else_is_refused():
    assert parse(["--classification", "NON_GOVERNING"]).classification == "NON_GOVERNING"
    with pytest.raises(SystemExit):
        parse(["--classification", "MOSTLY_GOVERNING"])
    with pytest.raises(SystemExit):
        parse(["--classification", "non_governing"])


def test_the_cli_refuses_an_invalid_classification():
    result = subprocess.run([sys.executable, str(TOOL), "--load_config", "x.json",
                             "--experiment-key", "k", "--out-dir", "o",
                             "--classification", "ADVISORY"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_the_classification_reaches_both_the_campaign_and_the_receipt():
    """Not a mock: the two places the tool writes it are pinned by source, and the receipts
    of the runs already executed carry it."""
    source = TOOL.read_text(encoding="utf-8")
    assert '"classification": args.classification,' in source
    assert source.count('"classification": args.classification,') == 2, (
        "the classification must reach the campaign submitted to data-gov and the receipt "
        "written next to the outputs; if one of them stops carrying it, a run's own record "
        "and the kernel's record can disagree")


@pytest.mark.parametrize("receipt", sorted(
    (REPO / "docs/audits/evidence/repro_runs/adoption_20260914").glob("*.json")))
def test_the_recorded_runs_declare_what_they_were(receipt):
    body = json.loads(receipt.read_text(encoding="utf-8"))
    assert body["classification"] == "NON_GOVERNING"
    for consumer in body["consumers"].values():
        for case in consumer["cases"].values():
            if isinstance(case, dict) and case.get("campaign_sha256"):
                assert case.get("classification") == "NON_GOVERNING", (
                    "a mechanical check that recorded itself as governing would be the "
                    "opposite of what the flag exists for")


def test_an_archival_replay_is_not_scientific_evidence():
    """`execution_purpose` and `classification` are different axes, and both must be honest.

    The preprocessor's governed replay declares ARCHIVAL_REPLAY_NON_AUTHORITATIVE to pass
    predictor's eligibility gate. That declaration travels in the configuration, so the gate
    sees it; a run that declared it cannot later be cited as governing evidence, because the
    campaign that carried it is NON_GOVERNING and the receipt says so.
    """
    sys.path.insert(0, str(REPO / "tools"))
    from verify_consumer_adoption import fixture_config

    manifest = {"roles": {"preprocessor": {"input_file": "series.csv"}},
                "contracts": {}}
    config = fixture_config("preprocessor", "success", manifest, Path("/lake"))
    assert config["execution_purpose"] == "ARCHIVAL_REPLAY_NON_AUTHORITATIVE"
    # and the harness that runs it always passes NON_GOVERNING
    harness = (REPO / "tools" / "verify_consumer_adoption.py").read_text(encoding="utf-8")
    assert '"--classification", "NON_GOVERNING"' in harness
    assert '"classification": "NON_GOVERNING"' in harness
