import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.feature_selection_status import CampaignError, summarize_campaign


TOOL = Path(__file__).parents[2] / "tools" / "feature_selection_status.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_plan(path: Path, rows: list[tuple[str, str]]) -> None:
    body = "cell_id\tresult_dir\n" + "".join(
        f"{cell_id}\t{result_dir}\n" for cell_id, result_dir in rows
    )
    path.write_text(body, encoding="utf-8")


def _write_manifest(
    root: Path,
    cell_id: str,
    status: str,
    *,
    wall_seconds: float | None = None,
    artifact: bytes = b"result\n",
) -> Path:
    result_dir = root / cell_id
    result_dir.mkdir(parents=True)
    manifest = {"cell_id": cell_id, "status": status}
    if status == "COMPLETED":
        result = result_dir / "results.jsonl"
        result.write_bytes(artifact)
        manifest["results_sha256"] = _sha256(result)
        manifest["wall_seconds"] = wall_seconds
    path = result_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_summarizes_states_durations_and_parallel_eta(tmp_path: Path) -> None:
    plan = tmp_path / "plan.tsv"
    results = tmp_path / "results"
    _write_plan(plan, [(name, name) for name in ("done-a", "done-b", "live", "bad", "new")])
    _write_manifest(results, "done-a", "COMPLETED", wall_seconds=10.0)
    _write_manifest(results, "done-b", "COMPLETED", wall_seconds=20.0)
    (results / "live").mkdir(parents=True)
    (results / "live" / "RUNNING").write_text("", encoding="utf-8")
    _write_manifest(results, "bad", "FAILED")

    report = summarize_campaign(plan, results, workers=2)

    assert report["counts"] == {
        "total": 5,
        "completed": 2,
        "running": 1,
        "failed": 1,
        "pending": 1,
    }
    assert report["durations_seconds"] == {
        "sample_size": 2,
        "median": 15.0,
        "p90": 20.0,
    }
    assert report["eta_seconds"] == {"median": 15.0, "p90": 20.0}
    assert report["integrity_errors"] == []


def test_tampered_completed_artifact_is_failed_not_completed(tmp_path: Path) -> None:
    plan = tmp_path / "plan.tsv"
    results = tmp_path / "results"
    _write_plan(plan, [("cell-a", "cell-a")])
    _write_manifest(results, "cell-a", "COMPLETED", wall_seconds=12.0)
    (results / "cell-a" / "results.jsonl").write_bytes(b"tampered\n")

    report = summarize_campaign(plan, results)

    assert report["counts"]["completed"] == 0
    assert report["counts"]["failed"] == 1
    assert report["durations_seconds"]["sample_size"] == 0
    assert report["eta_seconds"] == {"median": None, "p90": None}
    assert report["integrity_errors"][0]["code"] == "ARTIFACT_DIGEST_MISMATCH"


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("wrong_cell", "CELL_ID_MISMATCH"),
        ("non_finite_duration", "INVALID_WALL_SECONDS"),
    ],
)
def test_manifest_identity_and_duration_are_validated(
    tmp_path: Path, mutation: str, code: str
) -> None:
    plan = tmp_path / "plan.tsv"
    results = tmp_path / "results"
    _write_plan(plan, [("cell-a", "cell-a")])
    manifest_path = _write_manifest(
        results, "cell-a", "COMPLETED", wall_seconds=12.0
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "wrong_cell":
        manifest["cell_id"] = "cell-b"
    else:
        manifest["wall_seconds"] = "NaN"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = summarize_campaign(plan, results)

    assert report["counts"]["failed"] == 1
    assert report["integrity_errors"][0]["code"] == code


def test_plan_rejects_duplicate_ids_and_escaping_paths(tmp_path: Path) -> None:
    results = tmp_path / "results"
    duplicate = tmp_path / "duplicate.tsv"
    escaping = tmp_path / "escaping.tsv"
    _write_plan(duplicate, [("cell-a", "a"), ("cell-a", "b")])
    _write_plan(escaping, [("cell-a", "../outside")])

    with pytest.raises(CampaignError, match="duplicate cell_id"):
        summarize_campaign(duplicate, results)
    with pytest.raises(CampaignError, match="escapes results root"):
        summarize_campaign(escaping, results)


def test_cli_emits_machine_readable_json(tmp_path: Path) -> None:
    plan = tmp_path / "plan.tsv"
    results = tmp_path / "results"
    _write_plan(plan, [("cell-a", "cell-a")])
    _write_manifest(results, "cell-a", "COMPLETED", wall_seconds=8.5)

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--plan",
            str(plan),
            "--results-root",
            str(results),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["counts"]["completed"] == 1
    assert report["durations_seconds"]["median"] == 8.5
