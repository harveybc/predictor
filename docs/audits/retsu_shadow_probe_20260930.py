"""Independent, disposable checks against the first DOIN shadow archive."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--core", required=True)
    p.add_argument("--node", required=True)
    p.add_argument("--micro", type=Path, required=True)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    sys.path[:0] = [a.core, a.node]
    from doin_core.archive.body import build_archive, canonical_bytes, ArchiveRefusal
    from doin_core.models.block import Block
    from doin_node.archive.file_adapter import FileArchiveAdapter
    from doin_node.archive.warehouse import DisposableWarehouse, project_metrics

    row = dict(record_id="r1", candidate_id="c1", attempt=1, won=False,
               domain_id="test", peer_id="test-peer", performance=0.1,
               parameters={"depth": 2}, metrics={"MAE": 0.1})
    envelope = build_archive(block=None, candidates=[row])
    original = next(s for s in envelope.sections if s.name == "candidates")
    forged_content = canonical_bytes([{**row, "kind": "candidate", "performance": 999.0}])
    forged = replace(envelope, sections=(replace(original, content=forged_content),)
                     + tuple(s for s in envelope.sections if s.name != "candidates"))
    db = DisposableWarehouse(":memory:")
    db.create_experiment(domain_id="test", node_id="n", experiment_id="e1")
    inserted = project_metrics(db, forged, experiment_id="e1")
    stored = db.get_rounds("e1")[0]
    assert inserted == 1 and stored["performance"] == 999.0
    assert json.loads(stored["metrics"])["manifest_digest"] == envelope.manifest_digest
    metrics_lost = "MAE" not in stored["metrics"]
    db.create_experiment(domain_id="test", node_id="n", experiment_id="e2")
    reused = db.record_round(experiment_id="e2", domain_id="test", round_number=9,
                             performance=123.0, round_id="r1",
                             detail_metrics=json.loads(stored["metrics"]))
    assert reused == "r1" and db.get_rounds("e2") == []
    db.close()

    block = Block.genesis()
    first = build_archive(block=block, candidates=[row])
    second = build_archive(block=block, candidates=[{**row, "performance": 0.2}])
    assert first.body_digest == second.body_digest
    assert first.manifest_digest != second.manifest_digest
    with tempfile.TemporaryDirectory(prefix="musashi-shadow-") as d:
        archive = FileArchiveAdapter(d)
        archive.put(first)
        try:
            archive.put(second)
        except ArchiveRefusal as exc:
            conflict = str(exc)
        else:
            raise AssertionError("Expected body-key/manifest collision")
        assert conflict == "CONFLICT"

    data = json.loads(a.micro.read_text())
    cash = {}
    for name in ("ideal/ideal", "persistence/ideal"):
        arm = data["arms"][name]
        gap = arm["cash"] - arm["initial_cash"] - arm["realized_pnl"]
        swap = arm["costs"]["swap_measured"]
        assert abs(gap - swap) < 1e-7 and arm["end_exposure_units"] == 0
        cash[name] = {"cash_change_minus_reported_pnl": gap, "reported_swap": swap}

    result = {
        "scope": "Disposable archive probes and retained synthetic metrics; no real trading or GPU",
        "forged_section_projects_performance": stored["performance"],
        "retains_original_manifest_digest": True,
        "input_MAE_absent_from_projection": metrics_lost,
        "cross_experiment_round_id_reuse_silently_returns_old_id": True,
        "same_block_different_candidates_body_digest_equal": True,
        "same_block_different_candidates_manifest_digest_equal": False,
        "second_manifest_storage_result": conflict,
        "cash_reconciliation": cash,
    }
    print(json.dumps(result, indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
