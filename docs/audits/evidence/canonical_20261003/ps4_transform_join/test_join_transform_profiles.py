"""Focused refusals for the PS4 transform join. The generator is a separate file."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from join_transform_profiles import (  # noqa: E402
    JoinRefusal,
    build,
    join_loaded,
    load,
    render,
)

ROOT = Path(__file__).resolve().parents[5]


def _loaded():
    return load(ROOT)


def _raise(loaded):
    with pytest.raises(JoinRefusal) as caught:
        join_loaded(loaded)
    return caught.value


class TestRetainedDenominators:
    @classmethod
    def setup_class(cls):
        cls.report, cls.rows = build(ROOT)

    def test_exact_variant_and_feature_counts(self):
        den = self.report["denominators"]
        self.assert_counts(den)
        emitted = {row["emitted_feature_id"] for row in self.rows if row["emitted_feature_id"]}
        assert len(emitted) == 10
        assert den["ps1_metric_cells"] == 110
        assert self.report["ps1_cell_states"]["MEASURED"] == 110
        assert self.report["measurement_flag"] == "NO_NEW_MODEL_MEASUREMENT"
        assert self.report["selection_status"] == "PENDING"
        assert self.report["ps4_expanded_state"] == "PENDING"
        assert all(row["ps4_expanded_state"] == "PENDING" for row in self.rows)

    def assert_counts(self, den):
        assert den["variants"] == 9
        assert den["admissible_causal_computability"] == 5
        assert den["not_admissible_variants"] == 4
        assert den["emitted_features"] == 10

    def test_rejected_variants_have_explicit_zero_output_rows(self):
        zeros = [row for row in self.rows if row["emitted_feature_id"] == ""]
        assert len(zeros) == 4
        assert {row["variant_id"] for row in zeros} == {
            "tv.wavelet_dwt_db4_global",
            "tv.hilbert_global",
            "tv.stl_global",
            "tv.kalman_local_level_smoother",
        }
        for row in zeros:
            assert row["metric"] == ""
            assert row["ps1_metric_state"] == "NO_EMITTED_FEATURE"
            assert row["ps1_metric_state"] != "0"
            assert row["variant_state"] == "NOT_ADMISSIBLE"
            assert "SELECTED" not in row["ps2_status_by_target_horizon"]

    def test_ps1_cells_are_not_shortened(self):
        measured = [row for row in self.rows if row["ps1_metric_state"] == "MEASURED"]
        assert len(measured) == 110
        assert len(self.rows) == 114
        versions = {row["ps1_metric_version"] for row in measured}
        assert versions == {"laneA_ps1_metrics.v1"}
        assert self.report["ps1_cell_states"] == {
            "MEASURED": 110,
            "FAILED": 0,
            "PENDING": 0,
            "NOT_APPLICABLE": 0,
        }
        assert self.report["ps1_profile_coverage"] == "COMPLETE"
        assert any("PENDING_PROFILE" in item for item in self.report["findings"])
        rendered = render(self.report, self.rows)
        out = ROOT / "docs/audits/evidence/canonical_20261003/ps4_transform_join"
        for name, payload in rendered.items():
            path = out / name
            if path.is_file():
                assert path.read_bytes() == payload


class TestMutationsRefuse:
    def test_missing_metric_cell(self):
        loaded = _loaded()
        loaded["cells"] = [row for row in loaded["cells"] if not (
            row["feature_id"] == "tv.wav_d1" and row["metric"] == "acf")]
        refusal = _raise(loaded)
        assert refusal.code == "MISSING_METRIC_CELL"
        assert "tv.wav_d1" in refusal.detail
        assert "acf" in refusal.detail

    def test_duplicate_feature_id(self):
        loaded = _loaded()
        duplicate = copy.deepcopy(loaded["features"][0])
        for feature in loaded["features"]:
            if feature.get("feature_id") == "tv.wav_d1":
                duplicate = copy.deepcopy(feature)
                break
        duplicate["transform"] = "tv.multitaper_trailing applied to ln(close); output tv.wav_d1"
        loaded["features"].append(duplicate)
        refusal = _raise(loaded)
        assert refusal.code == "DUPLICATE_FEATURE_ID"

    def test_admitted_feature_mapped_to_rejected_variant(self):
        loaded = _loaded()
        for feature in loaded["features"]:
            if feature.get("feature_id") == "tv.wav_d1":
                feature["transform"] = "tv.wavelet_dwt_db4_global applied to ln(close); output tv.wav_d1"
                break
        refusal = _raise(loaded)
        assert refusal.code == "REJECTED_VARIANT_EMITTED"

    def test_altered_digest(self):
        loaded = _loaded()
        loaded["digests"]["artifacts_sha256"]["profile_cells.csv"] = "0" * 64
        refusal = _raise(loaded)
        assert refusal.code == "DIGEST_MISMATCH"
        assert "profile_cells.csv" in refusal.detail

    def test_unrecognized_variant(self):
        loaded = _loaded()
        loaded["variants"].append({
            "variant_id": "tv.not_in_probe",
            "admissibility": "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY",
            "prefix_causality_result": "PREFIX_INVARIANT_MEASURED",
        })
        refusal = _raise(loaded)
        assert refusal.code == "UNRECOGNIZED_VARIANT"
        assert "tv.not_in_probe" in refusal.detail

    def test_absent_ps2_status(self):
        loaded = _loaded()
        loaded["ps2_rows"] = [row for row in loaded["ps2_rows"] if row["feature"] != "tv.kalman_dev"]
        refusal = _raise(loaded)
        assert refusal.code == "ABSENT_PS2_STATUS"
        assert "tv.kalman_dev" in refusal.detail
