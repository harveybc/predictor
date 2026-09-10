import json

import pytest

from olap.financial_data_reconciliation import reconcile_manifest


def test_reconciliation_preserves_declared_and_physical_facts(tmp_path):
    feature = tmp_path / "features" / "trading_asset_data" / "asset"
    feature.mkdir(parents=True)
    (feature / "1h.csv").write_text("timestamp,close\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")
    manifest = {
        "trading_assets": {
            "asset": {"timeframes": {"1h": {
                "path": "features/trading_asset_data/asset/1h.csv",
                "status": "ok",
                "rows": 1,
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T00:00:00Z",
                "columns": ["timestamp", "close"],
            }}},
        },
        "cross_source_features": {
            "1h": {"missing": {
                "path": "features/cross_source_features/1h/missing.csv",
                "status": "ok",
                "rows": 2,
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T01:00:00Z",
                "columns": ["timestamp", "value"],
            }},
        },
    }
    manifest_path = tmp_path / "features" / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = reconcile_manifest(manifest_path, tmp_path, "2026-09-10T00:00:00Z")

    assert result["summary"]["dataset_slice_count"] == 2
    assert result["summary"]["physical_present_count"] == 1
    assert result["summary"]["physical_missing_count"] == 1
    present = next(record for record in result["records"] if record["physical_exists"])
    assert present["relative_path"] == "features/trading_asset_data/asset/1h.csv"
    assert present["reconciliation_status"] == "DECLARED_PHYSICAL_PRESENT_METADATA_JOIN_REQUIRED"
    assert "CONTENT_DIGEST_NOT_PRESENT_IN_FEATURE_MANIFEST" in present["metadata_gaps"]


def test_reconciliation_rejects_boolean_row_count(tmp_path):
    feature = tmp_path / "features"
    feature.mkdir()
    manifest = {
        "trading_assets": {
            "asset": {"timeframes": {"1h": {
                "path": "features/asset.csv",
                "status": "ok",
                "rows": True,
                "start": "2026-01-01T00:00:00Z",
                "end": "2026-01-01T01:00:00Z",
                "columns": ["timestamp", "value"],
            }}},
        },
        "cross_source_features": {},
    }
    manifest_path = feature / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative integer"):
        reconcile_manifest(manifest_path, tmp_path, "2026-09-10T00:00:00Z")
