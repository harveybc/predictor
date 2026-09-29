"""RB02: the PRODUCER path, proved by refusal.

`tools/test_tsl_warehouse_receipt.py` already proved the deployed DuckDB provider transports the governed metric schema and
refuses a nonfinite value. It also showed, by filling every identity tag with the literal string `fixture`, that the
general-purpose warehouse does NOT check whether a run's protocol, scaler and evaluation population are identified. That is
correct for a generic warehouse and unacceptable for a literature comparison, so the check lives in the producer.

These tests exercise `df_tsl_repro.validate_receipt` against a receipt built by `df_tsl_repro.build_receipt` from the REAL
sealed Weather design and the REAL characterization of the delivered bytes, and then round-trip the accepted receipt through a
TEMPORARY DuckDB store. The production cube is never opened. Every metric value here is a declared transport fixture: this
file measures no model.

    ~/.venvs/store-hosts-duckdb-prod/bin/python -m pytest tools/test_tsl_producer_contract.py
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import df_tsl_repro as T                                            # noqa: E402

RUN_ROOT = Path.home() / ".local/state/crispdm-data-foundation/tsl_weather_rb02_20260928"
DESIGN = RUN_ROOT / "DESIGN.weather.L96.json"
CHARACTERIZATION = RUN_ROOT / "CHARACTERIZATION.weather.json"

FIXTURE_NOTE = "TRANSPORT_TEST_NOT_SCIENCE"


def _receipt(horizon: int = 96, seed: int = 2021) -> dict:
    design = json.loads(DESIGN.read_text())
    char = json.loads(CHARACTERIZATION.read_text())
    sets = char["sets"][f"L{design['seq_len']}_h{horizon}"]
    population = {"sha256": "b" * 64, "windows": sets["test"]["windows"],
                  "target_channels": sets["test"]["channels"], "elements": sets["elements_test"]}
    return T.build_receipt(
        design=design, characterization=char, horizon=horizon, seed=seed,
        # declared fabricated numbers: this file proves the producer gate, it does not score a model
        metrics={"mse": 0.25, "mae": 0.30}, naive={"mse": 0.75, "mae": 0.55},
        population=population, model_commit=T.PINNED_COMMIT, scorer_sha256="c" * 64,
        campaign_key="tsl-weather-producer-contract-disposable", campaign_sha256="a" * 64,
        unit_id=f"weather-h{horizon}-s{seed}", delivery_id="0" * 32, availability_contract_sha256="e" * 64,
        costs={"wall_seconds": 1.0, "cpu_seconds": 1.0}, started_at="2026-09-28T00:00:00Z",
        finished_at="2026-09-28T00:00:01Z", comparison_class=FIXTURE_NOTE)


def _reseal(body: dict) -> dict:
    body = copy.deepcopy(body)
    body.pop("terminal_sha256", None)
    body["terminal_sha256"] = T.S.sha_obj(body)
    return body


class ProducerGate(unittest.TestCase):
    """Each test names ONE defect and asserts the producer refuses it and says which field."""

    def setUp(self):
        self.assertTrue(DESIGN.is_file(), f"the sealed design is missing: {DESIGN}")
        self.assertTrue(CHARACTERIZATION.is_file(), f"the characterization is missing: {CHARACTERIZATION}")
        self.body = _receipt()

    def test_a_complete_receipt_is_accepted_and_carries_its_identities(self):
        ok = T.validate_receipt(self.body)
        self.assertEqual(ok["dataset"], "weather")
        self.assertEqual(ok["metric_rows"], 4)
        design = json.loads(DESIGN.read_text())
        self.assertEqual(ok["protocol_sha256"], design["lock"]["protocol_sha256"])
        self.assertTrue(T.is_digest(ok["scaler_sha256"]))
        self.assertTrue(T.is_digest(ok["evaluation_population_sha256"]))

    def test_missing_protocol_identity_is_refused(self):
        for mutate in (lambda t: t.pop("protocol_sha256"), lambda t: t.update(protocol_sha256=""),
                       lambda t: t.update(protocol_sha256="fixture"), lambda t: t.update(protocol_sha256="not-a-digest")):
            body = copy.deepcopy(self.body)
            mutate(body["tags"])
            with self.assertRaises(T.TslRefusal) as cm:
                T.validate_receipt(_reseal(body))
            self.assertIn("protocol_sha256", str(cm.exception))

    def test_missing_scaler_identity_is_refused(self):
        for field in ("scaler_sha256", "scaler_fit_population_sha256"):
            for mutate in (lambda t, f=field: t.pop(f), lambda t, f=field: t.update({f: "fixture"})):
                body = copy.deepcopy(self.body)
                mutate(body["tags"])
                with self.assertRaises(T.TslRefusal) as cm:
                    T.validate_receipt(_reseal(body))
                self.assertIn(field, str(cm.exception))

    def test_missing_population_identity_is_refused(self):
        body = copy.deepcopy(self.body)
        body["tags"].pop("evaluation_population_sha256")
        with self.assertRaises(T.TslRefusal) as cm:
            T.validate_receipt(_reseal(body))
        self.assertIn("evaluation_population_sha256", str(cm.exception))
        # a population that is merely UNCOUNTED is refused too: the element count must reconcile
        body = copy.deepcopy(self.body)
        body["tags"]["elements"] = str(int(body["tags"]["elements"]) + 1)
        with self.assertRaises(T.TslRefusal) as cm:
            T.validate_receipt(_reseal(body))
        self.assertIn("POPULATION", str(cm.exception))

    def test_nonfinite_metric_is_refused(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for index in range(4):
                body = copy.deepcopy(self.body)
                body["metrics"][index]["value"] = value
                with self.assertRaises(T.TslRefusal) as cm:
                    T.validate_receipt(_reseal(body))
                self.assertIn("NONFINITE", str(cm.exception))

    def test_a_missing_paired_naive_is_refused(self):
        body = copy.deepcopy(self.body)
        body["metrics"] = [m for m in body["metrics"] if "naive" not in m["metric"]]
        with self.assertRaises(T.TslRefusal) as cm:
            T.validate_receipt(_reseal(body))
        self.assertIn("MISSING_METRIC", str(cm.exception))

    def test_the_wrong_clock_is_refused(self):
        """Weather steps are 600 s. Labelling 96 Weather steps as 96 hours is the error that would invalidate every
        cross-dataset comparison, so the producer refuses it arithmetically rather than by convention."""
        body = copy.deepcopy(self.body)
        body["tags"]["horizon_seconds"] = str(96 * 3600)             # the hourly clock of Electricity/Traffic
        with self.assertRaises(T.TslRefusal) as cm:
            T.validate_receipt(_reseal(body))
        self.assertIn("CLOCK", str(cm.exception))
        self.assertEqual(T.horizon_seconds("weather", 96), 96 * 600)
        self.assertEqual(T.horizon_seconds("traffic", 96), 96 * 3600)
        self.assertEqual(T.horizon_seconds("electricity", 96), 96 * 3600)

    def test_foreign_bytes_and_foreign_resource_are_refused(self):
        for field, value in (("dataset_sha256", "f" * 64), ("resource", "thuml_tsl_traffic/traffic.csv")):
            body = copy.deepcopy(self.body)
            body["tags"][field] = value
            with self.assertRaises(T.TslRefusal):
                T.validate_receipt(_reseal(body))

    def test_the_warehouse_transport_fixture_shape_is_refused_by_the_producer(self):
        """The shape `test_tsl_warehouse_receipt.py` legitimately stores — every identity tag the literal 'fixture' — is
        exactly what a literature comparison may not contain. The producer refuses it and names each field."""
        body = copy.deepcopy(self.body)
        for field in ("protocol_sha256", "scaler_sha256", "scaler_fit_population_sha256",
                      "evaluation_population_sha256", "scorer_sha256", "configuration_sha256"):
            body["tags"][field] = "fixture"
        with self.assertRaises(T.TslRefusal) as cm:
            T.validate_receipt(_reseal(body))
        message = str(cm.exception)
        for field in ("protocol_sha256", "scaler_sha256", "scaler_fit_population_sha256", "evaluation_population_sha256"):
            self.assertIn(field, message)

    def test_a_tampered_terminal_digest_is_refused(self):
        body = copy.deepcopy(self.body)
        body["metrics"][0]["value"] = 0.0001                         # a number changed after the digest was taken
        with self.assertRaises(T.TslRefusal) as cm:
            T.validate_receipt(body)
        self.assertIn("TERMINAL_DIGEST", str(cm.exception))

    def test_every_sealed_horizon_produces_an_accepted_receipt_with_its_own_clock(self):
        design = json.loads(DESIGN.read_text())
        seen = {}
        for horizon in design["horizons"]:
            ok = T.validate_receipt(_receipt(horizon=horizon))
            seen[horizon] = int(ok["horizon_seconds"])
        self.assertEqual(seen, {h: h * 600 for h in design["horizons"]})


class PinnedRecipe(unittest.TestCase):
    """What the seal claims about the author's recipe, asserted against the author's own files and the published tables."""

    def setUp(self):
        self.design = json.loads(DESIGN.read_text())
        self.char = json.loads(CHARACTERIZATION.read_text())

    def test_the_design_digest_recomputes(self):
        T.validate_design(self.design)
        broken = copy.deepcopy(self.design)
        broken["lock"]["training"]["loss"] = "MAE"
        with self.assertRaises(T.TslRefusal):
            T.validate_design(broken)

    def test_the_published_electricity_row_read_here_agrees_with_the_row_pinned_since_rp92(self):
        """The same arXiv table was read twice, months apart, by two different tools. If this fails, one reading is wrong."""
        mine = T.PAPER["electricity_crosscheck"]["L96"]
        theirs = T.S.PAPER["L96"]["per_horizon"]
        self.assertEqual(mine, theirs)

    def test_the_agreement_margin_is_this_datasets_own_dispersion(self):
        weather = T.agreement("weather", "L96")
        self.assertEqual(weather["std_paper"], {"mse": 0.006, "mae": 0.004})
        self.assertNotEqual(weather["std_paper"], T.S.AGREEMENT["std_paper"])        # Electricity's margin is 0.005 / 0.006
        self.assertEqual(T.agreement("traffic", "L96")["std_paper"], {"mse": 0.008, "mae": 0.004})

    def test_every_sealed_cell_carries_the_authors_own_argv_and_the_papers_hyperparameters(self):
        table6 = T.PAPER["weather"]["table6"]
        for cell in self.design["cells"]:
            eff = cell["effective_args"]
            self.assertEqual(eff["data_path"], "weather.csv")
            self.assertEqual(eff["patch_len"], table6["patch_len"])
            self.assertEqual(eff["e_layers"], table6["e_layers"])
            self.assertEqual(eff["learning_rate"], table6["learning_rate"])
            self.assertEqual(eff["d_model"], table6["d_model"])
            self.assertEqual(eff["d_ff"], table6["d_ff"])
            self.assertEqual(eff["train_epochs"], table6["train_epochs"])
            self.assertEqual(eff["batch_size"], table6["batch_size"])
            self.assertEqual(eff["enc_in"], eff["c_out"], 21)
            self.assertEqual(eff["features"], "M")
            self.assertEqual(eff["label_len"], 48)
            self.assertEqual(eff["patience"], 3)
            self.assertEqual(eff["lradj"], "cosine")
            self.assertEqual(eff["loss"], "MSE")
            self.assertFalse(eff["inverse"])
            self.assertEqual(cell["horizon_seconds"], cell["horizon_steps"] * 600)

    def test_is_training_two_is_recorded_verbatim_and_named_as_a_no_op(self):
        self.assertTrue(all(c["effective_args"]["is_training"] == 2 for c in self.design["cells"]))
        ids = {d["id"]: d for d in self.design["lock"]["paper_code_disagreements"]}
        self.assertIn("WEATHER-IS-TRAINING-2", ids)
        self.assertIn("run.py:153", ids["WEATHER-IS-TRAINING-2"]["resolution"])
        source = (T.AUTHOR_REPO / "run.py").read_text()
        self.assertIn("if args.is_training:", source)                # a truth test, not a comparison against 1
        # `args.is_training` is READ in exactly two places in the whole pinned clone: run.py's truth test, and
        # utils/print_args.py, which only formats it into the banner. Neither compares it to a number, so 2 and 1 are the
        # same branch. Everything else named `is_training` is the model's own train/eval boolean, which never receives it.
        reads = {p.relative_to(T.AUTHOR_REPO).as_posix(): p.read_text().count("args.is_training")
                 for p in sorted(T.AUTHOR_REPO.rglob("*.py")) if "args.is_training" in p.read_text()}
        self.assertEqual(reads, {"run.py": 1, "utils/print_args.py": 1})
        self.assertIn("Is Training:", (T.AUTHOR_REPO / "utils/print_args.py").read_text())

    def test_the_sealed_split_arithmetic_reproduces_the_loaders_own_window_counts(self):
        for horizon in self.design["horizons"]:
            sealed = T.borders(52696, 96, horizon)["windows"]
            observed = self.char["sets"][f"L96_h{horizon}"]
            for split in ("train", "vali", "test"):
                self.assertEqual(sealed[split], observed[split]["windows"], f"h{horizon}/{split}")

    def test_a_seq_len_the_author_script_does_not_offer_is_refused(self):
        with self.assertRaises(T.TslRefusal):
            T.seal(dataset="weather", seq_len=512, protocol="Lsearched")
        with self.assertRaises(T.TslRefusal):
            T.seal(dataset="weather", seq_len=96, protocol="Lsearched")

    def test_the_delivered_bytes_are_the_registered_bytes_and_have_no_missing_values(self):
        facts = T.dataset_facts("weather")
        self.assertEqual(self.char["file_sha256"], facts["sha256"])
        self.assertEqual(self.char["rows"], facts["rows"])
        self.assertEqual(self.char["missing_values"], 0)
        self.assertEqual(self.char["columns"]["count_including_date"], facts["channels"] + 1)
        self.assertTrue(self.char["columns"]["target_moved_last"])
        self.assertTrue(self.char["columns"]["reordering_is_a_noop"])

    def test_traffic_is_sealable_and_carries_the_hourly_clock(self):
        design = T.seal(dataset="traffic", seq_len=96, protocol="L96")
        self.assertEqual(len(design["cells"]), 12)
        for cell in design["cells"]:
            self.assertEqual(cell["horizon_seconds"], cell["horizon_steps"] * 3600)
            self.assertEqual(cell["effective_args"]["enc_in"], 862)
        self.assertEqual(design["lock"]["published"]["average"], {"mse": 0.407, "mae": 0.268})


class TemporaryStoreRoundTrip(unittest.TestCase):
    """The accepted receipt goes through the DEPLOYED provider on a disposable database file. Production is never touched."""

    def setUp(self):
        from predictor_duckdb_store.provider import PredictorDuckdbStore
        self.temp = tempfile.TemporaryDirectory(prefix="tsl-producer-disposable-")
        self.store = PredictorDuckdbStore()
        self.store.set_params(duckdb_path=str(Path(self.temp.name) / "cube.duckdb"), schema="main",
                              memory_limit="256MB", threads=1, min_free_bytes=1)
        self.store.engine()

    def tearDown(self):
        self.store.engine().dispose()
        self.temp.cleanup()

    def test_accepted_receipts_round_trip_and_refused_ones_never_reach_the_store(self):
        from sqlalchemy import text
        design = json.loads(DESIGN.read_text())
        stored = 0
        for horizon in design["horizons"]:
            for seed in design["seeds"]:
                body = _receipt(horizon=horizon, seed=seed)
                T.validate_receipt(body)                             # the producer gate runs BEFORE the store is opened
                self.assertTrue(self.store.write_terminal(body)["stored"])
                self.store.write_terminal(body)                      # idempotent
                stored += 1
                with self.store.engine().connect() as con:
                    tags = con.execute(text("SELECT tags_json FROM gov_terminal WHERE terminal_sha256=:d"),
                                       {"d": body["terminal_sha256"]}).scalar()
                    self.assertEqual(json.loads(tags), body["tags"])
                    rows = con.execute(text("SELECT metric,value,unit,split,horizon FROM gov_terminal_metric "
                                            "WHERE terminal_sha256=:d ORDER BY metric"),
                                       {"d": body["terminal_sha256"]}).fetchall()
                    self.assertEqual([tuple(r) for r in rows],
                                     sorted((m["metric"], m["value"], m["unit"], m["split"], m["horizon"])
                                            for m in body["metrics"]))
                    resource = con.execute(text("SELECT resource_id FROM gov_terminal_dataset WHERE terminal_sha256=:d"),
                                           {"d": body["terminal_sha256"]}).scalar()
                    self.assertEqual(resource, T.dataset_facts("weather")["resource"])
        self.store.engine().dispose()
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), stored)
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal_metric")).scalar(), stored * 4)
            # the clock is queryable, per dataset, from the stored tags
            seconds = con.execute(text("SELECT DISTINCT json_extract_string(tags_json,'$.horizon_steps'), "
                                       "json_extract_string(tags_json,'$.horizon_seconds') FROM gov_terminal "
                                       "ORDER BY 1")).fetchall()
            self.assertEqual({int(s): int(sec) for s, sec in seconds}, {h: h * 600 for h in design["horizons"]})

        refused = copy.deepcopy(_receipt())
        refused["tags"].pop("protocol_sha256")
        refused = _reseal(refused)
        with self.assertRaises(T.TslRefusal):
            T.validate_receipt(refused)
        with self.store.engine().connect() as con:                   # nothing was written by the refusal
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal WHERE terminal_sha256=:d"),
                                         {"d": refused["terminal_sha256"]}).scalar(), 0)

    def test_the_store_itself_still_refuses_a_nonfinite_value(self):
        from sqlalchemy import text
        body = _receipt()
        body["metrics"][0]["value"] = math.inf
        body = _reseal(body)
        with self.assertRaises(Exception):
            self.store.write_terminal(body)
        with self.store.engine().connect() as con:
            self.assertEqual(con.execute(text("SELECT count(*) FROM gov_terminal")).scalar(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
