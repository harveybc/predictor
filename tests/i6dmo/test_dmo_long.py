"""I6-DMO long family: sealed long design, seal-long-design command, long-head training.

PLAN section 5: heads [18, 24, 36, 48, 60, 72] h, 144-hour inclusive span,
12 h spacing, 13 observations, same small modular family, hourly issuance.
Invented CPU fixtures only; no validation/TEST outcome is read.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tools.i6dmo import design as D
from tests.i6dmo.dmo_model_fixtures import (DESIGN, LONG_HORIZONS, LONG_OFFSETS, short_terminal,
                                           split_arrays, write_bundle)

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


@pytest.fixture()
def sealed(tmp_path):
    out = tmp_path / "EXPERIMENT.long.json"
    result = D.seal_long_design(DESIGN, short_terminal(tmp_path / "T.json"), out)
    return out, result


def _body(doc):
    return {k: v for k, v in doc.items() if k not in ("derived_from", "seal_sha256")}


# --------------------------------------------------------------------- design + seal

def test_long_design_is_sealed_and_mirrors_short_except_plan_fields(sealed):
    out, result = sealed
    doc = D.load_design(out)
    assert result["design_sha256"] == D.design_sha256(doc)
    assert doc["experiment_id"] == "I6-DMO-LONG-DS12-STATIC-R0"
    assert doc["horizons_hours"] == LONG_HORIZONS
    assert doc["input"]["history_span_hours"] == 144 and doc["input"]["spacing_hours"] == 12
    assert doc["input"]["observations"] == 13 and doc["input"]["issuance_hours"] == 1
    short = json.loads(DESIGN.read_text())
    differing = {k for k in _body(doc) if doc[k] != short.get(k)}
    assert differing == {"experiment_id", "horizons_hours", "input"}
    assert {k for k in doc["input"] if doc["input"][k] != short["input"][k]} == {
        "history_span_hours", "spacing_hours"}
    eff = D.effective_config(doc)
    assert eff["family"] == "long" and eff["offsets_hours"] == LONG_OFFSETS
    assert eff["model"]["offsets_hours"] == LONG_OFFSETS and eff["model"]["horizons_hours"] == LONG_HORIZONS
    assert eff["splits"] == D.effective_config(D.load_design(DESIGN))["splits"]
    assert eff["design_sha256"] != D.effective_config(D.load_design(DESIGN))["design_sha256"]
    assert doc["derived_from"]["short_design_sha256"] == D.design_sha256(short)
    report = D.validate_design_report(out)
    assert report["status"] == "PASS" and report["family"] == "long"


def test_seal_is_deterministic_and_reads_no_short_outcome(tmp_path):
    a = D.seal_long_design(DESIGN, short_terminal(tmp_path / "a.json"), tmp_path / "a" / "L.json")
    b = D.seal_long_design(DESIGN, short_terminal(tmp_path / "b.json"), tmp_path / "b" / "L.json")
    assert (tmp_path / "a" / "L.json").read_bytes() == (tmp_path / "b" / "L.json").read_bytes()
    assert a["design_sha256"] == b["design_sha256"]
    # Different short outcome (status, eligibility, metrics): identical long hyperparameters.
    c = D.seal_long_design(DESIGN, short_terminal(tmp_path / "c.json", status="NOT_ELIGIBLE",
                                                  eligibility={"2": {"eligible": False, "mae": 9.0}}),
                           tmp_path / "c" / "L.json")
    docs = [json.loads((tmp_path / d / "L.json").read_text()) for d in "ac"]
    assert _body(docs[0]) == _body(docs[1])
    assert set(docs[1]["derived_from"]["short_terminal"]) == {"unit_id", "identity_sha256",
                                                             "terminal_file_sha256"}
    assert c["design_sha256"] != a["design_sha256"]  # bound to a different terminal file
    # Idempotent re-run onto an identical file; refusal onto a different one.
    D.seal_long_design(DESIGN, tmp_path / "a.json", tmp_path / "a" / "L.json")
    with pytest.raises(D.DesignError, match="LONG_DESIGN_EXISTS"):
        D.seal_long_design(DESIGN, tmp_path / "c.json", tmp_path / "a" / "L.json")


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"status": "FAILED"}, "SHORT_NOT_TERMINAL_VALID"),
    ({"status": "RUNNING"}, "SHORT_NOT_TERMINAL_VALID"),
    ({"unit_id": "short.fit"}, "SHORT_NOT_TERMINAL_VALID"),
    ({"unit_id": "final.pair_test"}, "SHORT_NOT_TERMINAL_VALID"),
    ({"schema": "other"}, "SHORT_NOT_TERMINAL_VALID"),
])
def test_seal_requires_a_valid_short_closure_terminal(tmp_path, kwargs, code):
    with pytest.raises(D.DesignError, match=code):
        D.seal_long_design(DESIGN, short_terminal(tmp_path / "t.json", **kwargs), tmp_path / "L.json")
    assert not (tmp_path / "L.json").exists()


def test_seal_refuses_a_non_short_or_invalid_source_design(tmp_path, sealed):
    out, _ = sealed
    with pytest.raises(D.DesignError, match="SHORT_DESIGN_REQUIRED"):
        D.seal_long_design(out, short_terminal(tmp_path / "t.json"), tmp_path / "L2.json")
    bad = json.loads(DESIGN.read_text())
    bad["seed"] = 1
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(D.DesignError, match="FIXED_VALUE"):
        D.seal_long_design(tmp_path / "bad.json", short_terminal(tmp_path / "t.json"), tmp_path / "L3.json")


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda d: d["input"].update(spacing_hours=6), "FIXED_VALUE"),
    (lambda d: d.update(horizons_hours=[18, 24, 36, 48, 96, 120]), "FIXED_VALUE"),
    (lambda d: d["splits"]["test"].update(open=True), "TEST_FORBIDDEN"),
    (lambda d: d["model"]["core"].update(transformer_blocks=1), "FIXED_VALUE"),
    (lambda d: d.update(surprise=1), "UNKNOWN_FIELD"),
    (lambda d: d["derived_from"].update(short_design_sha256="0" * 64), "LONG_DESIGN_SEAL"),
    (lambda d: d["input"].update(history_span_hours="UNDECIDED"), "LONG_DESIGN_UNDECIDED"),
])
def test_long_design_tamper_and_undecided_are_typed(sealed, tmp_path, mutate, code):
    out, _ = sealed
    doc = json.loads(out.read_text())
    mutate(doc)
    with pytest.raises(D.DesignError, match=code) as err:
        D.validate_design(doc)
    if code == "LONG_DESIGN_UNDECIDED":
        assert "input.history_span_hours" in str(err.value)


def test_plan_open_fields_block_the_seal_with_the_exact_field(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "LONG_PLAN_OPEN_FIELDS", (("input", "history_span_hours"),))
    with pytest.raises(D.DesignError, match=r"LONG_DESIGN_UNDECIDED.*input\.history_span_hours"):
        D.seal_long_design(DESIGN, short_terminal(tmp_path / "t.json"), tmp_path / "L.json")
    assert not (tmp_path / "L.json").exists()
    assert D.LONG_PLAN_OPEN_FIELDS_DEFAULT == ()


def test_cli_seal_long_design_and_validate(tmp_path):
    term = short_terminal(tmp_path / "t.json")
    out = tmp_path / "EXPERIMENT.long.json"
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "seal-long-design", "--short-design",
                          str(DESIGN), "--short-terminal", str(term), "--out", str(out), "--json"],
                         capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["design_sha256"] == D.design_sha256(json.loads(out.read_text()))
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "validate-design", "--design", str(out),
                          "--json"], capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 0 and json.loads(run.stdout)["family"] == "long"
    bad = short_terminal(tmp_path / "bad.json", status="FAILED")
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "seal-long-design", "--short-design",
                          str(DESIGN), "--short-terminal", str(bad), "--out", str(tmp_path / "x.json"), "--json"],
                         capture_output=True, text=True, cwd=REPO, env=ENV, check=False)
    assert run.returncode == 2 and json.loads(run.stdout)["code"] == "SHORT_NOT_TERMINAL_VALID"


def test_add_cli_coexists_with_the_ops_seal_long_command():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("seal-long")  # owned by the ops lane (campaign sealing)
    D.add_cli(sub)
    args = parser.parse_args(["seal-long-design", "--short-design", "s", "--short-terminal", "t", "--out", "o"])
    assert args.command == "seal-long-design"


# --------------------------------------------------------------------- model + trainer

tf = pytest.importorskip("tensorflow")


def test_long_model_heads_and_physical_offsets():
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    cfg = M.model_config(list(MANIFEST_NO_STLFSI4), family="long")
    bundle = M.build_dmo(cfg)
    assert list(bundle.forecast_model.output_names) == ["h18", "h24", "h36", "h48", "h60", "h72"]
    assert bundle.forecast_model.get_layer("positional_encoding").offsets_hours == tuple(LONG_OFFSETS)
    assert bundle.forecast_model.count_params() == 4834
    mixed = dict(cfg, horizons_hours=[2, 3, 4, 5, 6, 12])
    with pytest.raises(ValueError, match="FIXED_VALUE"):
        M.build_dmo(mixed)


def test_long_training_uses_the_same_masked_loss_end_to_end(sealed, tmp_path):
    from tools.i6dmo import train as T
    out, _ = sealed
    bundle = write_bundle(tmp_path / "lb", horizons=LONG_HORIZONS, offsets=LONG_OFFSETS)
    receipt = T.train_unit(out, bundle, tmp_path / "u", device="cpu", training_overrides={"max_epochs": 3})
    assert receipt["status"] == "COMPLETED", receipt
    assert receipt["experiment_id"] == "I6-DMO-LONG-DS12-STATIC-R0"
    assert receipt["model_side_summary"]["horizons_hours"] == LONG_HORIZONS
    with np.load(tmp_path / "u" / "predictions" / "validation.npz") as pz:
        assert pz["horizons_hours"].tolist() == LONG_HORIZONS
        assert pz["predicted_price"].shape == (40, 6)
    training = json.loads((tmp_path / "u" / "TRAINING.json").read_text())
    assert training["settings"]["loss"] == "equal_head_masked_mae_over_fit_only_naive"
    # A short bundle cannot feed the long design.
    short_bundle = write_bundle(tmp_path / "sb")
    receipt = T.train_unit(out, short_bundle, tmp_path / "u2", device="cpu")
    assert receipt["status"] == "FAILED_TYPED" and receipt["code"] in ("BUNDLE_HORIZONS", "BUNDLE_OFFSETS")


def test_long_fit_labels_crossing_inner_are_purge_violations(sealed, tmp_path):
    from tools.i6dmo import train as T
    out, _ = sealed
    eff = T.effective_from_design(out)
    kw = dict(horizons=LONG_HORIZONS, steps=13)
    splits = {"fit": split_arrays("2023-11-03T00:00:00", 4, 1, invalid_rate=0.0, **kw),   # +72 h reaches inner
              "inner": split_arrays("2023-11-13T00:00:00", 8, 2, **kw),
              "validation": split_arrays("2024-01-08T00:00:00", 8, 3, **kw)}
    root = write_bundle(tmp_path / "p", splits=splits, horizons=LONG_HORIZONS, offsets=LONG_OFFSETS)
    with pytest.raises(T.TrainDisposition, match="PURGE_VIOLATION"):
        T.load_data_bundle(root, eff)
