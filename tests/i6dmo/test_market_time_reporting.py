"""Report the sealed MG2/MG12 bundles without a fit or an injected arrays loader."""
import json
from pathlib import Path

import numpy as np
import pytest

from tests.i6dmo.test_market_time_fit_path import SEALED, market  # noqa: F401
from tools.i6dmo import market_grid, provenance, reporting


def _sha(path):
    return provenance.file_sha256(Path(path))


def _spec(market, tmp_path, key):
    family = "short" if key == "mg2" else "long"
    design = json.loads(SEALED[key].read_text())
    bundle = market[key]
    manifest = market_grid.load_market_manifest(bundle)
    arrays = market_grid.load_market_split(bundle, "validation")
    fit = tmp_path / "fit"
    out = fit / "out"
    out.mkdir(parents=True)
    # Known error = 0.1 * naive; the reporter must recover it on every valid head.
    pred = arrays["target_price"] + .1 * (arrays["target_price"] - arrays["origin_price"][:, None])
    np.savez(out / "validation.npz", row_id=arrays["row_id"],
             horizons_hours=np.asarray(design["horizons_hours"]),
             predicted_price=pred, mask=arrays["mask"])
    stored = {"stored_at": 1760000000, "identity_sha256": "f" * 64, "terminal": {"artifacts": {
        "predictions/validation.npz": {"path": "validation.npz", "sha256": _sha(out / "validation.npz")}}}}
    (fit / "TERMINAL.json").write_text(json.dumps(stored))
    effective = {"design": design, "long": {"design": design}, "code_commit": "c" * 40}
    spec = {"unit_id": f"{family}.report", "kind": "report", "family": family,
            "identity_sha256": "r" * 64, "campaign_id": "market-report-regression",
            "cadence": "NONE", "test_open": False, "effective_config": effective,
            "upstream": {f"{family}.fit": {"path": str(fit / "TERMINAL.json"),
                                            "sha256": _sha(fit / "TERMINAL.json")}},
            "pins": {f"{family}_bundle": {"path": str(bundle), "sha256": _sha(bundle / "BUNDLE.json"),
                                           "bundle_sha256": manifest["bundle_sha256"]},
                     f"{family}_input": {"path": str(market["input"]),
                                          "sha256": _sha(market["input"] / "INPUT.npz")}}}
    return spec, arrays, pred


@pytest.mark.parametrize("key", ["mg2", "mg12"])
def test_report_real_market_bundle_without_training_or_arrays_override(market, tmp_path, key, monkeypatch):
    from tools.i6dmo import model
    monkeypatch.setattr(model, "build_dmo", lambda *a, **kw: pytest.fail("report attempted model construction"))
    spec, arrays, pred = _spec(market, tmp_path, key)
    result = reporting.run_unit(spec, tmp_path / "report", tmp_path / "progress.json")
    assert result["status"] == "COMPLETE", result
    for j, h in enumerate(spec["effective_config"]["design"]["horizons_hours"]):
        mask = arrays["mask"][:, j]
        model_mae = np.abs(pred[mask, j] - arrays["target_price"][mask, j]).mean()
        naive = np.abs(arrays["target_price"][mask, j] - arrays["origin_price"][mask]).mean()
        row = result["eligibility"][str(h)]
        assert row["n"] == int(mask.sum())
        assert row["mae"] == pytest.approx(model_mae, rel=1e-10)
        assert row["naive"] == pytest.approx(naive, rel=1e-10)
        assert row["relative_error"] == pytest.approx(-.9, abs=1e-8)
    assert result["costs"]["fit_seconds"] == 0
    assert all(r["tags"]["sampling"] == ("MG2_MARKET_EXACT" if key == "mg2" else "MG12_MARKET_EXACT")
               for r in result["reports"])
    metrics = json.loads((tmp_path / "report" / "METRICS.json").read_text())
    assert metrics["agreement"]["status"] == "AGREE"


@pytest.mark.parametrize("key", ["mg2", "mg12"])
def test_report_rejects_wrong_market_bundle(market, tmp_path, key):
    spec, _, _ = _spec(market, tmp_path, key)
    wrong = market["mg12" if key == "mg2" else "mg2"]
    family = spec["family"]
    spec["pins"][f"{family}_bundle"] = {"path": str(wrong), "sha256": _sha(wrong / "BUNDLE.json")}
    result = reporting.run_unit(spec, tmp_path / "report", tmp_path / "progress.json")
    assert result["status"] == "FAILED"
    assert result["disposition"] == "BUNDLE_SAMPLING"
    assert not result["reports"]
