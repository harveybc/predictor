"""Additive reporting only: source preservation, arithmetic and policy gates."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import c2_reconcile as report


@pytest.fixture
def evidence(tmp_path):
    source = ROOT / "docs/audits/evidence/lane_c2_eth_20261001"
    for name in report.SOURCES:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    return tmp_path


def mutate(root, name, fn):
    path = root / name
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_successors_preserve_sources_and_are_reproducible(evidence):
    before = {name: (evidence / name).read_bytes() for name in report.SOURCES}
    expected = report.build_successors(evidence)
    output = report.write_successors(evidence)
    for name, content in expected.items():
        assert (output / name).read_text() == content
    assert report.build_successors(evidence) == expected
    for name, raw in before.items():
        assert (evidence / name).read_bytes() == raw
    assert "TBD_D4" in before["RETURN.md"].decode()
    assert "TBD_D4" not in expected["RETURN.md"]
    status = json.loads(expected["STATUS.json"])
    results = json.loads(expected["RESULTS.json"])
    d4 = next(d for d in status["deliverables"] if d["id"] == "D4")
    assert (d4["completed"], d4["planned"], d4["remaining"]) == (136, 240, 104)
    assert results["D4_ps3r_rerun"] == d4
    assert status["eta"] is None
    assert status["scheduled_actions"] == []
    assert status["seed_policy"]["seeds"] == [2021, 2022, 2023, 2024]
    assert status["seed_policy"]["disposition"] == "HISTORICAL_PRESERVED_NONCOMPLIANT"
    assert status["seed_policy"]["repetitions_scheduled"] == 0
    assert results["D2_dossiers"]["battery"]["battery_verdict"] == "CONTROLS_FAIL_AS_REQUIRED"
    assert results["D2_dossiers"]["schema_errors_total"] == 0
    assert results["D2_dossiers"]["identification"] == "NOT_IDENTIFIED"
    assert results["D1"] == json.loads(before["RESULTS.json"])["D1"]
    for name, digest in status["provenance"]["source_sha256"].items():
        assert digest == hashlib.sha256(before[name]).hexdigest()


def test_refuses_to_overwrite_a_successor(evidence):
    output = report.write_successors(evidence)
    before = (output / "STATUS.json").read_bytes()
    with pytest.raises(FileExistsError):
        report.write_successors(evidence)
    assert (output / "STATUS.json").read_bytes() == before


@pytest.mark.parametrize("completed", [-1, 241, True, 136.5])
def test_invalid_progress_fails_closed(evidence, completed):
    mutate(evidence, "STATUS.json", lambda d: d["deliverables"][-1].update(completed=completed))
    with pytest.raises(ValueError, match="progress"):
        report.write_successors(evidence)
    assert not (evidence / report.DESTINATION).exists()


def test_progress_is_computed_not_hardcoded(evidence):
    mutate(evidence, "STATUS.json", lambda d: d["deliverables"][-1].update(completed=140))
    result = json.loads(report.build_successors(evidence)["RESULTS.json"])
    assert result["D4_ps3r_rerun"]["remaining"] == 100


def test_calibration_disagreement_refused(evidence):
    mutate(evidence, "dossiers/DOSSIER_INDEX.json",
           lambda d: d["battery_calibration"].update(cells=497))
    with pytest.raises(ValueError, match="calibration"):
        report.build_successors(evidence)


def test_seed_cells_must_match_declared_seeds(evidence):
    mutate(evidence, "paired_inference/multiseed/MULTISEED.json", lambda d: d["seeds"].pop())
    with pytest.raises(ValueError, match="seed"):
        report.build_successors(evidence)


def test_committed_successors_match_generator():
    source = ROOT / "docs/audits/evidence/lane_c2_eth_20261001"
    for name, content in report.build_successors(source).items():
        assert (source / report.DESTINATION / name).read_text() == content
