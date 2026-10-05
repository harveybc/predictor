"""Tests for the closure matrix states and the planned-only staging of terminals."""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ps3r_closure_matrix as M  # noqa: E402


def _terminal(directory: Path, families, wall=100.0):
    directory.mkdir(parents=True, exist_ok=True)
    body = f'{{"d":"{directory}"}}\n'.encode()
    (directory / "results.jsonl").write_bytes(body)
    (directory / "run_manifest.json").write_text(json.dumps({"status": "COMPLETED", "results_sha256": hashlib.sha256(body).hexdigest(), "wall_seconds": wall, "families": families}))


def _plan(path: Path, cells):
    path.write_text("cell_id\tresult_dir\n" + "".join(f"{c}\t{d}\n" for c, d in cells))


def _config(tmp_path: Path):
    base_plan = tmp_path / "baseline.tsv"
    alt_plan = tmp_path / "alternative.tsv"
    _plan(base_plan, [("batch_002::f.a::baseline", "batch_002/f.a"), ("batch_002::f.b::baseline", "batch_002/f.b"), ("batch_001::f.c::baseline", "batch_001/f.c")])
    _plan(alt_plan, [
        ("batch_002::f.a::masked_temporal_ae", "batch_002/f.a"), ("batch_002::f.a::past_to_current_siamese", "batch_002/f.a/pass_p2c"),
        ("batch_002::f.b::masked_temporal_ae", "batch_002/f.b"), ("batch_002::f.b::past_to_current_siamese", "batch_002/f.b/pass_p2c"),
        ("batch_001::f.c::masked_temporal_ae", "batch_001/f.c"),
    ])
    return {
        "denominator": 3,
        "plans": {"baseline": [str(base_plan)], "alternative": [str(alt_plan)]},
        "roots": {
            "baseline": [{"role": "worker_b", "path": str(tmp_path / "b")}, {"role": "worker_a", "path": str(tmp_path / "a_base")}],
            "alternative": [{"role": "worker_a", "path": str(tmp_path / "a_alt")}],
        },
        "logs": [{"role": "worker_b", "path": str(tmp_path / "b.log")}],
        "claims_ledger": str(tmp_path / "ledger.json"),
        "ingest": {"root": str(tmp_path / "ingest"), "roles": [
            {"name": "baseline", "plan_kind": "baseline", "family": "baseline", "sources": [{"path": str(tmp_path / "b")}, {"path": str(tmp_path / "a_base")}]},
            {"name": "alternative", "plan_kind": "alternative", "family": "past_to_current_siamese", "sources": [{"path": str(tmp_path / "a_alt")}]},
            {"name": "alternative_mtae", "plan_kind": "alternative", "family": "masked_temporal_ae", "sources": [{"path": str(tmp_path / "a_alt")}]},
        ]},
    }


def test_matrix_states_and_closure(tmp_path):
    cfg = _config(tmp_path)
    base4 = ["identity", "random", "ae", "dae"]
    _terminal(tmp_path / "b/batch_002/f.a", base4)
    _terminal(tmp_path / "a_alt/batch_002/f.a", ["identity", "random", "masked_temporal_ae"])
    _terminal(tmp_path / "a_alt/batch_002/f.a/pass_p2c", ["identity", "random", "past_to_current_siamese"])
    (tmp_path / "a_alt/batch_002/f.b").mkdir(parents=True)
    (tmp_path / "a_alt/batch_002/f.b/FAILED.codex.json").write_text('{"status":"FAILED","rc":1}')
    (tmp_path / "a_alt/batch_002/f.b/FAILED.reason.txt").write_text("Traceback\napp.univariate_temporal.ContractError: no observed TRAIN values\n")
    (tmp_path / "b.log").write_text("2026-10-05T04:00:00Z BEGIN cap=8000M tier_1 batch_002 f.b\n")
    (tmp_path / "ledger.json").write_text(json.dumps({"cells": {"batch_001::f.c": {"feature_id": "f.c", "winner": "worker_a", "claims": {"worker_a": {"state": "CLAIMED", "claimed_at_utc": "2026-10-05T03:00:00Z"}}}}}))
    matrix = M.build_matrix(cfg, tmp_path, M._dt.datetime(2026, 10, 5, 4, 5, tzinfo=M.UTC))
    rows = {r["feature_id"]: r for r in matrix["rows"]}
    assert matrix["heavy_candidates"] == 3 and matrix["counts"] == {"closed": 1, "open": 2}
    fa = rows["f.a"]["families"]
    assert fa["identity"]["state"] == fa["dae"]["state"] == "DONE" and fa["identity"]["role"] == "worker_b"
    assert fa["masked_temporal_ae"]["state"] == "DONE" and fa["past_to_current_siamese"]["state"] == "DONE" and rows["f.a"]["closed"]
    fb = rows["f.b"]["families"]
    assert fb["ae"]["state"] == "RUNNING" and fb["ae"]["role"] == "worker_b"
    assert fb["masked_temporal_ae"]["state"] == "FAILED" and "no observed TRAIN values" in fb["masked_temporal_ae"]["reason"]
    assert fb["masked_temporal_ae"]["receipt"].endswith("FAILED.codex.json")
    assert fb["past_to_current_siamese"]["state"] == "PENDING"
    fc = rows["f.c"]["families"]
    assert fc["past_to_current_siamese"]["state"] == "NOT_APPLICABLE"
    assert fc["identity"]["state"] == "CLAIMED" and fc["identity"]["role"] == "worker_a"
    assert matrix["family_counts"]["past_to_current_siamese"] == {"DONE": 1, "FAILED": 0, "RUNNING": 0, "CLAIMED": 0, "PENDING": 1, "NOT_APPLICABLE": 1}


def test_duplicate_terminals_are_reported_on_the_cell(tmp_path):
    cfg = _config(tmp_path)
    base4 = ["identity", "random", "ae", "dae"]
    _terminal(tmp_path / "b/batch_002/f.a", base4)
    _terminal(tmp_path / "a_base/batch_002/f.a", base4, wall=50.0)
    matrix = M.build_matrix(cfg, tmp_path)
    cell = {r["feature_id"]: r for r in matrix["rows"]}["f.a"]["families"]["ae"]
    assert cell["state"] == "DONE" and cell["duplicate_roles"] == ["worker_b", "worker_a"] and cell["digests_equal"] is False


def test_staging_only_adopts_planned_cells_and_is_idempotent(tmp_path):
    cfg = _config(tmp_path)
    base4 = ["identity", "random", "ae", "dae"]
    _terminal(tmp_path / "b/batch_002/f.a", base4)
    _terminal(tmp_path / "a_alt/batch_002/f.a", ["identity", "random", "masked_temporal_ae"])
    _terminal(tmp_path / "a_alt/batch_002/f.a/pass_p2c", ["identity", "random", "past_to_current_siamese"])
    _terminal(tmp_path / "a_alt/measure_mtae/batch_001/f.c", ["identity", "random", "masked_temporal_ae"])  # probe dir, not a plan cell
    _terminal(tmp_path / "a_alt/sealed_px_pass_p2c_deadbeef", ["identity", "random", "past_to_current_siamese"])
    root = tmp_path / "ingest"
    first = M.stage_terminals(cfg, tmp_path, root)
    assert first["copied"] == 3 and first["roles"] == {"baseline": 1, "alternative": 1, "alternative_mtae": 1}
    assert sorted(p.name for p in (root / "alternative").iterdir()) == ["f.a"]
    assert sorted(p.name for p in (root / "alternative_mtae").iterdir()) == ["f.a"]
    assert json.loads((root / "alternative" / "f.a" / "run_manifest.json").read_text())["families"][-1] == "past_to_current_siamese"
    second = M.stage_terminals(cfg, tmp_path, root)
    assert second["copied"] == 0 and second["unchanged"] == 3


def test_csv_has_one_row_per_candidate(tmp_path):
    cfg = _config(tmp_path)
    matrix = M.build_matrix(cfg, tmp_path)
    out = tmp_path / "m.csv"
    M.write_matrix_csv(out, matrix)
    lines = out.read_text().splitlines()
    assert len(lines) == 4 and lines[0].startswith("feature_id,batch_id,closed,identity_state")
