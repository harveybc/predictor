"""Two terminals for the boundary gate: a declared test, and the same one promoted.

Every value is fabricated. No model ran, no store is contacted, nothing is written
outside the directory given on the command line.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from app import classification_receipt as cr
from app import classification_provenance as cp
from test_classification_provenance import document, non_model_path, H

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
receipt = cr.build_receipt(document(path=non_model_path(),
                                    evidence_class="TRANSPORT_TEST_NOT_SCIENCE"))
body = dict(schema="governed_terminal.v1", campaign_sha256="c" * 64,
            campaign_key="boundary-demonstration", unit_id="declared", generation=1,
            actor="nightly-eval", project="predictor", classification="NON_GOVERNING",
            status="COMPLETED", reason=None, started_at="2026-09-29T00:00:00Z",
            finished_at="2026-09-29T00:00:01Z", terminal_lake="olap_cube",
            config_sha256=cr.sha256_of(cr.CONTRACT),
            code_identity={"kind": "git_commit", "value": "d" * 40},
            costs={"wall_seconds": 1.0}, tags=cr.terminal_tags(receipt),
            synthetic_spec_sha256=None, deliveries=["0" * 32],
            metrics=cr.terminal_metrics(receipt), verified_datasets=[], artifacts=[])
body["terminal_sha256"] = cr.sha256_of(body)
(out / "declared_test.json").write_text(json.dumps(body, indent=2, sort_keys=True))

promoted = json.loads(json.dumps(body))
promoted["unit_id"] = "promoted"
promoted["tags"].update(evidence_class="MEASUREMENT", evidence_role=cp.MODEL_RESULT,
                        provenance_sha256=cp.provenance_sha256(
                            dict(receipt["answering_path"])))
promoted["terminal_sha256"] = cr.sha256_of(
    {k: v for k, v in promoted.items() if k != "terminal_sha256"})
(out / "promoted_test.json").write_text(json.dumps(promoted, indent=2, sort_keys=True))
print("wrote", out / "declared_test.json", "and", out / "promoted_test.json")
