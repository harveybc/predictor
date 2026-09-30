"""The row-identity defect and its repair, at the far end of a DISPOSABLE store.

The claim under test is not about a function's return value: it is about what a
reader holding nothing but stored warehouse rows can conclude. So the store here
is the installed `PredictorDuckdbStore`, the same distribution the warehouse host
runs, on a temporary DuckDB file, and every identity is recomputed from the stored
`tags_json` and `gov_terminal_metric` rows - never from an in-memory receipt.

Two terminals are written that reproduce the live shape: one accuracy-primary and
one macro-F1-primary carrying the same measurement, exactly as `native-accuracy`
and `native-macro-f1` do in production. First with the legacy tags, where the
double count is visible from the store. Then with the successor tags, where it is
counted once.

Nothing is mocked, patched or wrapped, and nothing is written outside the
temporary directory.

Run with the store host's environment, which has the provider installed:
  /home/harveybc/.venvs/store-hosts-duckdb-prod/bin/python \
      tools/test_classification_row_identity_warehouse.py
"""
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from sqlalchemy import text                                       # noqa: E402
from predictor_duckdb_store.provider import PredictorDuckdbStore  # noqa: E402

from app import classification_receipt as cr                      # noqa: E402
from app import classification_row_identity as ri                 # noqa: E402

from test_classification_row_identity import (                    # noqa: E402
    VALUE, document, accuracy_primary, macro_f1_primary)

PROBABILITY = {"nll": 0.3, "brier": 0.12, "ece": 0.08, "ece_bins": 15}


def terminal(receipt, unit_id, tags):
    body = dict(
        schema="governed_terminal.v1", campaign_sha256="e" * 64,
        campaign_key="cb04-row-identity-disposable-fixture",
        unit_id=unit_id, generation=1, actor="fixture", project="predictor",
        classification="NON_GOVERNING", status="COMPLETED", reason=None,
        started_at="2026-09-29T00:00:00Z", finished_at="2026-09-29T00:00:01Z",
        terminal_lake="olap_cube", config_sha256=cr.sha256_of(ri.CONTRACT),
        code_identity={"kind": "git_commit", "value": "f" * 40},
        costs={"wall_seconds": 1.0}, tags=tags, synthetic_spec_sha256=None,
        deliveries=["0" * 32], metrics=cr.terminal_metrics(receipt),
        verified_datasets=[], artifacts=[])
    body["terminal_sha256"] = cr.sha256_of(body)
    return body


class RowIdentityInADisposableWarehouse(unittest.TestCase):
    """Real provider, real builders, temporary database."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cb04-row-identity-disposable-")
        self.store = PredictorDuckdbStore()
        self.store.set_params(duckdb_path=str(Path(self.temp.name) / "cube.duckdb"),
                              schema="main", memory_limit="256MB", threads=1,
                              min_free_bytes=1)
        self.store.engine()
        self.left = accuracy_primary(probability=PROBABILITY,
                                     evidence_class="MEASUREMENT")
        self.right = macro_f1_primary(probability=PROBABILITY,
                                      evidence_class="MEASUREMENT")

    def tearDown(self):
        self.store.engine().dispose()
        self.temp.cleanup()

    # -- the substitution guard ------------------------------------------------
    def test_nothing_under_test_was_substituted(self):
        self.assertIn("site-packages", inspect.getfile(PredictorDuckdbStore))
        self.assertEqual(PredictorDuckdbStore.__module__,
                         "predictor_duckdb_store.provider")
        self.assertIs(type(self.store), PredictorDuckdbStore)
        self.assertEqual(self.store.write_terminal.__func__,
                         PredictorDuckdbStore.write_terminal)
        for function in (cr.terminal_tags, cr.terminal_metrics):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_receipt")
        for function in (ri.terminal_tags_with_row_identity, ri.recompute_from_tags,
                         ri.deduplicate, ri.aggregation_groups):
            self.assertEqual(inspect.getmodule(function).__name__,
                             "app.classification_row_identity")

    # -- helpers that read ONLY the store -------------------------------------
    def _write(self, tagger):
        for receipt, unit in ((self.left, "accuracy-terminal"),
                              (self.right, "macro-f1-terminal")):
            body = terminal(receipt, unit, tagger(receipt))
            self.assertTrue(self.store.write_terminal(body)["stored"])

    def _stored_rows(self):
        with self.store.engine().connect() as con:
            terminals = con.execute(text(
                "SELECT terminal_sha256, unit_id, tags_json FROM gov_terminal"
                " ORDER BY unit_id")).fetchall()
            out = []
            for digest, unit, tags_json in terminals:
                tags = json.loads(tags_json)
                rows = con.execute(text(
                    "SELECT metric, value, unit FROM gov_terminal_metric"
                    " WHERE terminal_sha256=:d ORDER BY metric"), {"d": digest}).fetchall()
                out.append({"unit_id": unit, "tags": tags,
                            "rows": [{"metric": m, "value": v, "unit": u}
                                     for m, v, u in rows]})
        return out

    # -- the defect, from the store ------------------------------------------
    def test_the_legacy_tags_let_the_store_double_count_one_measurement(self):
        self._write(cr.terminal_tags)
        stored = self._stored_rows()
        self.assertEqual(len(stored), 2)

        # the rule classification_metrics.v1 prescribes: group by the identity tag
        groups = {}
        for stored_terminal in stored:
            identity = stored_terminal["tags"]["metric_identity_sha256"]
            for row in stored_terminal["rows"]:
                if row["metric"] == "classification.macro_f1":
                    groups.setdefault(identity, []).append(
                        (stored_terminal["unit_id"], row["value"]))
        self.assertEqual(len(groups), 2, "one macro-F1 measurement, two identity groups")
        self.assertEqual({round(value, 12) for rows in groups.values()
                          for _unit, value in rows}, {round(VALUE, 12)})
        # and a mean over those groups counts the same number twice
        values = [value for rows in groups.values() for _unit, value in rows]
        self.assertEqual(len(values), 2)

        # the accuracy identity group contains a macro-F1 row that is not its own
        accuracy_identity = next(t["tags"]["metric_identity_sha256"] for t in stored
                                 if t["tags"]["author_primary_metric_family"] == "ACCURACY")
        riding_along = [row["metric"] for t in stored
                        if t["tags"]["metric_identity_sha256"] == accuracy_identity
                        for row in t["rows"]
                        if row["metric"] == "classification.macro_f1"]
        self.assertEqual(riding_along, ["classification.macro_f1"])

        with self.assertRaises(ri.RowIdentityRefused) as caught:
            ri.assert_row_identity_tags(stored[0]["tags"])
        self.assertEqual(caught.exception.refusal, "ROW_IDENTITY_TAGS_MISSING")

    # -- the repair, from the store ------------------------------------------
    def test_the_successor_tags_let_the_store_count_it_once(self):
        self._write(ri.terminal_tags_with_row_identity)
        stored = self._stored_rows()
        rebuilt = []
        for stored_terminal in stored:
            tags = stored_terminal["tags"]
            ri.assert_row_identity_tags(tags)
            for row in stored_terminal["rows"]:
                identified = ri.recompute_from_tags(tags, row["metric"], unit=row["unit"])
                rebuilt.append({**row, **identified, "unit_id": stored_terminal["unit_id"],
                                "evidence_class": tags["evidence_class"]})

        self.assertEqual(len(rebuilt), 40)
        report = ri.aggregation_groups(rebuilt)
        self.assertEqual(report["rows_in"], 40)
        self.assertEqual(report["rows_counted"], 21)
        self.assertEqual(len(report["duplicates_dropped"]), 19)
        self.assertEqual(report["conflicts"], [])
        macro = [g for g in report["groups"] if g["metric"] == "classification.macro_f1"]
        self.assertEqual(len(macro), 1)
        self.assertEqual(macro[0]["n_occurrences"], 1)
        accuracy = [g for g in report["groups"] if g["metric"] == "classification.accuracy"]
        self.assertEqual(len(accuracy), 1)
        self.assertEqual(accuracy[0]["n_occurrences"], 1)
        self.assertNotEqual(macro[0]["row_identity_sha256"],
                            accuracy[0]["row_identity_sha256"])

    def test_the_stored_map_digest_binds_every_row_the_store_holds(self):
        self._write(ri.terminal_tags_with_row_identity)
        for stored_terminal in self._stored_rows():
            tags = stored_terminal["tags"]
            self.assertEqual(int(tags["metric_row_count"]), len(stored_terminal["rows"]))
            recomputed = {row["metric"]:
                          ri.recompute_from_tags(tags, row["metric"],
                                                 unit=row["unit"])["row_identity_sha256"]
                          for row in stored_terminal["rows"]}
            self.assertEqual(cr.sha256_of(recomputed),
                             tags["metric_row_identity_map_sha256"])

    def test_the_nine_stored_metric_fields_are_the_same_rows_as_before(self):
        """The successor changes tags only: the metric rows are byte-for-byte the old ones."""
        legacy = [dict(row) for row in cr.terminal_metrics(self.left)]
        self._write(ri.terminal_tags_with_row_identity)
        with self.store.engine().connect() as con:
            digest = con.execute(text(
                "SELECT terminal_sha256 FROM gov_terminal WHERE unit_id='accuracy-terminal'"
            )).scalar()
            rows = con.execute(text(
                "SELECT metric, value, unit, split FROM gov_terminal_metric"
                " WHERE terminal_sha256=:d ORDER BY metric"), {"d": digest}).fetchall()
        self.assertEqual([tuple(r) for r in rows],
                         sorted((m["metric"], m["value"], m["unit"], m["split"])
                                for m in legacy))

    def test_a_published_reference_row_never_joins_a_measured_group(self):
        """Why CB-C kept the author's 0.9525 out of the warehouse - now storable safely."""
        published = cr.build_receipt(document("ACCURACY", "accuracy", VALUE,
                                              evidence_class="PUBLISHED_REFERENCE"))
        for receipt, unit in ((self.left, "measured"), (published, "published")):
            body = terminal(receipt, unit, ri.terminal_tags_with_row_identity(receipt))
            self.assertTrue(self.store.write_terminal(body)["stored"])
        rebuilt = []
        for stored_terminal in self._stored_rows():
            tags = stored_terminal["tags"]
            for row in stored_terminal["rows"]:
                if row["metric"] != "classification.accuracy":
                    continue
                rebuilt.append({**row,
                                **ri.recompute_from_tags(tags, row["metric"],
                                                         unit=row["unit"]),
                                "unit_id": stored_terminal["unit_id"],
                                "evidence_class": tags["evidence_class"]})
        self.assertEqual(len(rebuilt), 2)
        self.assertEqual(len({r["row_identity_sha256"] for r in rebuilt}), 1,
                         "one identity: they are comparable")
        report = ri.aggregation_groups(rebuilt)
        self.assertEqual(len(report["groups"]), 2, "two groups: they are never averaged")
        self.assertEqual({g["evidence_class"] for g in report["groups"]},
                         {"MEASUREMENT", "PUBLISHED_REFERENCE"})
        self.assertEqual(report["duplicates_dropped"], [])
        self.assertTrue(report["evidence_class_separations"])

    def test_seventy_seven_classes_round_trip_through_the_store(self):
        """A BANKING77-sized receipt: 313 rows, no tag over budget, every row recoverable."""
        receipt = cr.build_receipt(document("ACCURACY", "accuracy", VALUE, classes=77,
                                            secondary={"MACRO_F1": VALUE},
                                            evidence_class="MEASUREMENT"))
        tags = ri.terminal_tags_with_row_identity(receipt)
        body = terminal(receipt, "banking77-shaped", tags)
        self.assertTrue(self.store.write_terminal(body)["stored"])
        with self.store.engine().connect() as con:
            stored_tags = json.loads(con.execute(text(
                "SELECT tags_json FROM gov_terminal WHERE unit_id='banking77-shaped'"
            )).scalar())
            rows = con.execute(text(
                "SELECT metric, value, unit FROM gov_terminal_metric"
                " WHERE terminal_sha256=:d"), {"d": body["terminal_sha256"]}).fetchall()
        self.assertEqual(len(rows), 77 * 4 + 5)
        self.assertEqual(int(stored_tags["metric_row_count"]), len(rows))
        self.assertEqual(stored_tags["metric_row_identity_json"],
                         cr.canonical_text(json.loads(
                             stored_tags["metric_row_identity_json"])))
        recomputed = {metric: ri.recompute_from_tags(stored_tags, metric,
                                                     unit=unit)["row_identity_sha256"]
                      for metric, _value, unit in rows}
        self.assertEqual(cr.sha256_of(recomputed),
                         stored_tags["metric_row_identity_map_sha256"])
        for value in stored_tags.values():
            self.assertLessEqual(len(str(value).encode()), ri.TAG_BUDGET_BYTES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
