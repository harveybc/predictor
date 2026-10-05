import unittest
from pathlib import Path

from generate_source_transform_coverage import (
    AuditError,
    BASE,
    build,
    classify_source,
    classify_transform,
    clock_kind,
    column_evidence,
    overlaps_train,
    parse_span_blob,
    render,
    require_measured_clocks,
    require_source_completeness,
    require_train_labels,
    require_transform_rules,
    to_comparable,
    parse_dt,
)

ROOT = Path(__file__).resolve().parents[5]
TRAIN = (
    to_comparable(parse_dt("2012-05-01 00:00:00+00:00")),
    to_comparable(parse_dt("2024-01-01 00:00:00+00:00")),
)


def source_row(**overrides):
    row = {
        "path": "fixture/resource",
        "provider": "Fixture",
        "frequency": "1d",
        "event_time": "trading date D",
        "availability_time": "conservative: (D+1) 00:00 UTC",
        "state": "QUEUED_BATCH_002 (declared clock only)",
    }
    row.update(overrides)
    return row


def empty_evidence():
    return {"matched": False, "span_conflict": False, "span": None, "rows_in_train": None, "absent_columns": []}


class LedgerInvariantTests(unittest.TestCase):
    def test_omitted_source_is_rejected(self):
        inputs = [source_row(path="a"), source_row(path="b", provider="Other")]
        outputs = [
            {"resource_identity": "a", "provider": "Fixture", "contract_byte_status": "DECLARED_ONLY"},
        ]
        with self.assertRaises(AuditError):
            require_source_completeness(inputs, outputs)

    def test_conflicting_duplicate_identities_stay_separate(self):
        report, rows, _ = build(ROOT)
        self.assertEqual(len(rows), report["denominators"]["inventory_source_rows"])
        by_path = {}
        for row in rows:
            by_path.setdefault(row["resource_identity"], []).append(row["inventory_state"])
        for states in by_path.values():
            if len(set(states)) > 1:
                self.fail("conflicting states were collapsed")
        left = classify_source(source_row(path="same", state="QUEUED one"), empty_evidence(), TRAIN, "")
        right = classify_source(source_row(path="same", state="QUEUED two"), empty_evidence(), TRAIN, "")
        left["row_index"] = "0"
        right["row_index"] = "1"
        self.assertEqual(left["resource_identity"], right["resource_identity"])
        self.assertNotEqual(left["inventory_state"], right["inventory_state"])
        self.assertEqual(len({left["inventory_state"], right["inventory_state"]}), 2)

    def test_missing_clock_is_not_measured(self):
        row = classify_source(
            source_row(availability_time="UNDEFINED", state="NOT_ADMITTED: NO_PUBLICATION_INSTANT"),
            empty_evidence(),
            TRAIN,
            "",
        )
        self.assertNotEqual(row["contract_byte_status"], "MEASURED")
        self.assertEqual(row["contract_byte_status"], "UNKNOWN")
        self.assertEqual(clock_kind("UNDEFINED", row["inventory_state"]), "MISSING")
        forged = dict(row, contract_byte_status="MEASURED", train_intersection="MEASURED_ROWS_IN_TRAIN_1")
        with self.assertRaises(AuditError):
            require_measured_clocks([forged])

    def test_prefix_violation_is_not_admissible(self):
        item = classify_transform({
            "variant_id": "tv.wavelet_dwt_db4_global",
            "family": "wavelet",
            "status": "PREFIX_VIOLATION_MEASURED",
            "admissible_as_feature": "True",
            "ps1_profile": "PENDING (profile later)",
            "causal_by_construction": "False",
            "agrees_with_declaration": "True",
            "note": "",
        }, 4)
        self.assertEqual(item["admissibility"], "NOT_ADMISSIBLE")
        self.assertEqual(item["variant_class"], "GLOBAL_OR_SMOOTHER")
        self.assertEqual(item["ps1_ps4_profile_state"], "PENDING_PROFILE")
        self.assertIn("PENDING_PROFILE", item["pending_action"])
        bad = dict(item, admissibility="ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY", denominator="1", ps1_ps4_profile_state="PENDING_PROFILE")
        with self.assertRaises(AuditError):
            require_transform_rules([
                {"status": "PREFIX_VIOLATION_MEASURED"},
            ], [bad])

    def test_fake_train_overlap_is_rejected(self):
        row = classify_source(
            source_row(
                state="INVENTORIED; span 2024-06-01..2025-01-01 lies outside this TRAIN",
                availability_time="bar end",
            ),
            empty_evidence(),
            TRAIN,
            "",
        )
        self.assertNotIn("OVERLAPS", row["train_intersection"])
        self.assertEqual(row["contract_byte_status"], "NOT_AVAILABLE_FOR_TRAIN")
        forged = dict(row, train_intersection="OBSERVATION_SPAN_OVERLAPS_TRAIN_NOT_AVAILABILITY")
        with self.assertRaises(AuditError):
            require_train_labels([forged], TRAIN)
        validation = parse_span_blob("2024-06-01..2025-01-01")
        self.assertIs(overlaps_train(validation, TRAIN), False)

    def test_unknown_is_not_zero_coverage(self):
        row = classify_source(
            source_row(availability_time="UNDEFINED_UNTIL_EVIDENCED", state="QUEUED_BATCH_002: holiday flags need publication-in-advance evidence"),
            empty_evidence(),
            TRAIN,
            "",
        )
        self.assertEqual(row["contract_byte_status"], "UNKNOWN")
        self.assertNotIn("0", row["train_intersection"])
        self.assertEqual(row["train_intersection"], "NOT_ESTABLISHED")


class RetainedLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report, cls.sources, cls.transforms = build(ROOT)

    def test_denominators_stay_distinct(self):
        den = self.report["denominators"]
        self.assertEqual(den["inventory_source_rows"], 388)
        self.assertEqual(len(self.sources), 388)
        self.assertEqual(den["feature_population_admissible"], 366)
        self.assertEqual(den["feature_population_ps3c_join"], 279)
        self.assertEqual(den["feature_population_low_priority_outside_join"], 87)
        self.assertNotEqual(den["inventory_source_rows"], den["feature_population_admissible"])
        self.assertEqual(den["transform_variant_rows"], 9)
        self.assertEqual(len(self.transforms), 9)
        self.assertTrue(self.report["measurement_flag"] == "NO_NEW_MODEL_MEASUREMENT")

    def test_named_providers_keep_their_states(self):
        by_provider = {}
        for row in self.sources:
            by_provider.setdefault(row["provider"], []).append(row)
        yahoo = by_provider["Yahoo Finance"]
        self.assertEqual(len(yahoo), 78)
        self.assertTrue(all(row["contract_byte_status"] == "DECLARED_ONLY" for row in yahoo))
        self.assertTrue(all(row["contract_byte_status"] != "MEASURED" for row in yahoo))
        fx = by_provider["FXMacroData"]
        self.assertEqual(len(fx), 2)
        self.assertTrue(all(row["contract_byte_status"] == "NOT_AVAILABLE_FOR_TRAIN" for row in fx))
        self.assertTrue(all("ZERO_TRAIN_ROWS" in row["train_intersection"] for row in fx))
        alpaca = by_provider["Alpaca"]
        self.assertEqual(len(alpaca), 1)
        self.assertEqual(alpaca[0]["contract_byte_status"], "NOT_INGESTED")
        self.assertEqual(alpaca[0]["train_intersection"], "NOT_ESTABLISHED")
        providers = {row["provider"] for row in self.sources}
        self.assertEqual(providers, set(self.report["source_status_by_provider"]))

    def test_transform_classes_and_profiles(self):
        classes = {row["variant_id"]: row["variant_class"] for row in self.transforms}
        self.assertEqual(classes["tv.wavelet_modwt_haar_causal"], "TRAILING_OR_FILTER")
        self.assertEqual(classes["tv.kalman_local_level_filter"], "TRAILING_OR_FILTER")
        self.assertEqual(classes["tv.wavelet_dwt_db4_global"], "GLOBAL_OR_SMOOTHER")
        self.assertEqual(classes["tv.kalman_local_level_smoother"], "GLOBAL_OR_SMOOTHER")
        self.assertEqual(classes["tv.hilbert_global"], "GLOBAL_OR_SMOOTHER")
        self.assertEqual(classes["tv.stl_global"], "GLOBAL_OR_SMOOTHER")
        for row in self.transforms:
            self.assertEqual(row["ps1_ps4_profile_state"], "PENDING_PROFILE")
            self.assertEqual(row["denominator"], "9")
            if row["prefix_causality_result"] == "PREFIX_VIOLATION_MEASURED":
                self.assertEqual(row["admissibility"], "NOT_ADMISSIBLE")
            if row["prefix_causality_result"] == "PREFIX_INVARIANT_MEASURED":
                self.assertEqual(row["admissibility"], "ADMISSIBLE_CAUSAL_COMPUTABILITY_ONLY")

    def test_contract_train_and_registry_are_not_byte_claims(self):
        self.assertEqual(self.report["train_interval"]["start"], "2012-05-01 00:00:00+00:00")
        self.assertEqual(self.report["train_interval"]["end"], "2024-01-01 00:00:00+00:00")
        registry = self.report["resource_contracts"]
        self.assertEqual(registry["status"], "LISTING_ONLY")
        self.assertEqual(registry["alpaca_mentions"], 0)
        self.assertEqual(registry["yahoo_mentions"], 0)
        self.assertEqual(len(registry["fxmacro_declared_rows"]), 2)
        self.assertIn("not ingested", registry["interpretation"])
        joined = column_evidence(
            "features/trading_asset_data/eurusd",
            {"lake_eurusd_5m": [{"source": "lake_eurusd_5m", "rows_in_train": "1", "non_null_in_train": "1", "column": "close", "span": "['2012-05-01 00:00:00+00:00', '2023-12-01 00:00:00+00:00']"}]},
        )
        self.assertTrue(joined["matched"])

    def test_render_is_deterministic(self):
        first = render(self.report, self.sources, self.transforms)
        again_report, again_sources, again_transforms = build(ROOT)
        second = render(again_report, again_sources, again_transforms)
        self.assertEqual(first, second)
        committed = ROOT / BASE / "source_transform_coverage"
        for name, payload in first.items():
            path = committed / name
            if path.is_file():
                self.assertEqual(path.read_bytes(), payload)

    def test_every_input_file_has_a_sha256(self):
        hashes = self.report["inputs_sha256"]
        self.assertGreaterEqual(len(hashes), 12)
        self.assertTrue(all(len(value) == 64 for value in hashes.values()))
        self.assertIn(str(BASE / "laneA" / "batch_001" / "inventory_sources.csv"), hashes)


if __name__ == "__main__":
    unittest.main()
