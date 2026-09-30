"""Consumidor por filas de una representación ya materializada.

El fixture sintético no es el prefijo anclado. Si el almacén nombrado por el
registro DR05 no está, la prueba nombra la ausencia y no afirma paridad.

    CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_hcore_row_consumer.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "hcore_row_consumer"
STORE = FIXTURE / "store"
PINNED = REPO / "docs" / "audits" / "evidence" / "DR05_DR06_20260926" / "PREFIX_OUTPUT_MATERIALIZATION.json"

TRAIN_SHA = "8a24071667bfedbeaf882b2869f3482b289e42f06ff3d3adb8d531745c382522"
VALIDATION_SHA = "0e6dd5bb871553621fa1f5b12343dfce7fff6425c5bf3d5d203defc894e17432"
TRAIN_ORIGINS_SHA = "6f76b30bbc4cef3a40df066bf3e7453bb203ffd084ce85c91bbfcc3f843c504a"
VALIDATION_ORIGINS_SHA = "ee2e01ce8793b059727bf468159c3b19bc79d9eeeabff59b447c43cecbcfbd1f"
MANIFEST_SHA = "589a9e935f2aaa62db8da95b1d0b97bd6c1aa742673cdfd618f6ae6f379c9276"
DATA_SHA = "70485ac9d1a41ac86d0910d23928a15c1aa88737c6261343e4c7dc248e30c194"
DESIGN_SHA = "143abb57d97daa07e3f5228eadf4e3f1a0deb5fe30eb95ca065663f76ff888a7"
PANEL_SHA = "b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db"
DONOR_WEIGHTS_SHA = "c374076c49c9631803bd16f85b4a577a67466a38e60371040af7d4ebe4ae3845"
PINNED_VERSION = "dr05.prefix.fd0fde9b2a913257"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "_t_hcore_row_consumer", REPO / "tools" / "df_hcore_row_consumer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _tool()
SYNTHETIC = json.loads((FIXTURE / "SYNTHETIC_RECORD.json").read_text())


def _values():
    return np.load(STORE / "prefix_output_train.npy")


def _origins():
    return np.load(STORE / "origins_train.npy")


def _refuses(code, fn):
    with pytest.raises(C.RowConsumerRefusal) as caught:
        fn()
    assert caught.value.code == code
    return caught.value


def test_alcance_is_not_defined_and_five_row_cases_exist():
    assert C.ALCANCE == "NOT_DEFINED_IN_REPO"
    assert C.ROW_CASES == (
        "EXACT_MATCH", "VALUE_MISMATCH", "ROW_COUNT_MISMATCH", "WRONG_IDENTITY", "MISSING_ARTIFACT")
    assert len(C.ROW_CASES) >= 5


def test_the_consumer_source_names_no_home_and_imports_no_tensorflow():
    text = (REPO / "tools" / "df_hcore_row_consumer.py").read_text()
    home_prefix = "/" + "home" + "/"
    assert "import tensorflow" not in text
    assert "trust_remote_code" not in text
    assert home_prefix not in text
    assert "Path.home" not in text
    fixture = (FIXTURE / "SYNTHETIC_RECORD.json").read_text() + (STORE / "MANIFEST.json").read_text()
    assert home_prefix not in fixture


def test_exact_match_on_the_declared_synthetic_fixture():
    values, origins = _values(), _origins()
    accepted = C.accept_rows(values, values.copy(), origins, origins.copy())
    assert accepted == {"code": "EXACT_MATCH", "row_equal": True, "value_equal": True,
                        "rows": 4, "elements": 24, "alcance": "NOT_DEFINED_IN_REPO"}
    window = C.read_origin(STORE, "train", 100)
    assert C.accept_window(window, window.copy())["rows"] == 1
    report = C.equality_report(C.open_store(STORE), SYNTHETIC)
    assert report["bytes"] == "MATCH"
    assert report["parity"] == "ROW_AND_VALUE_EQUAL"
    assert report["version"] == "synthetic.hcore.not-the-pinned-prefix"
    assert report["version"] != PINNED_VERSION
    assert report["splits"]["train"]["rows"] == 4
    assert report["splits"]["train"]["endpoint_digests"] == "NOT_IN_RECORD"
    assert "NO_ES_EL_PREFIJO" in SYNTHETIC["what_this_is"]


def test_value_mismatch_when_two_rows_change():
    values = _values()
    changed = values.copy()
    changed[0, 0, 0] = np.float32(changed[0, 0, 0] + 1)
    changed[1, 0, 0] = np.float32(changed[1, 0, 0] + 1)
    assert C.judge_rows(values, changed, _origins(), _origins()) == "VALUE_MISMATCH"
    _refuses("VALUE_MISMATCH", lambda: C.accept_rows(values, changed))


def test_one_scalar_is_a_mutated_value_and_one_replaced_row_is_a_mutated_row():
    values = _values()
    one = values.copy()
    one[2, 1, 1] = np.float32(one[2, 1, 1] + 1)
    assert C.judge_rows(values, one) == "MUTATED_VALUE"
    replaced = values.copy()
    replaced[3] = np.float32(-7)
    assert C.judge_rows(values, replaced) == "MUTATED_ROW"


def test_row_count_mismatch():
    values, origins = _values(), _origins()
    assert C.judge_rows(values, values[:3]) == "ROW_COUNT_MISMATCH"
    assert C.judge_rows(values, values, origins, origins[:3]) == "ROW_COUNT_MISMATCH"


def test_wrong_origin_identity():
    origins = _origins()
    moved = origins.copy()
    moved[-1] = np.int64(999)
    assert C.judge_rows(_values(), _values(), origins, moved) == "WRONG_IDENTITY"
    _refuses("WRONG_IDENTITY", lambda: C.read_origin(STORE, "train", 999))


def test_wrong_version_wrong_digest_and_wrong_identity_on_the_fixture():
    measured = C.open_store(STORE)
    bad_version = copy.deepcopy(SYNTHETIC)
    bad_version["version"]["version"] = "synthetic.hcore.other"
    _refuses("WRONG_VERSION", lambda: C.require_identity(measured, bad_version))
    bad_digest = copy.deepcopy(SYNTHETIC)
    bad_digest["arrays"]["train"]["sha256"] = "0" * 64
    _refuses("WRONG_DIGEST", lambda: C.require_identity(measured, bad_digest))
    bad_identity = copy.deepcopy(SYNTHETIC)
    bad_identity["donor_cell"] = "OTHER"
    bad_identity["version"]["derived_from"]["donor_cell"] = "OTHER"
    _refuses("WRONG_IDENTITY", lambda: C.require_identity(measured, bad_identity))


def test_missing_artifact():
    _refuses("MISSING_ARTIFACT", lambda: C.open_store(STORE / "no-such-directory"))


def test_a_store_missing_its_array_is_a_missing_artifact(tmp_path):
    broken = tmp_path / "store"
    broken.mkdir()
    (broken / "MANIFEST.json").write_text((STORE / "MANIFEST.json").read_text())
    _refuses("MISSING_ARTIFACT", lambda: C.open_store(broken))


def test_the_synthetic_fixture_is_not_the_pinned_prefix():
    pinned = json.loads(PINNED.read_text())
    _refuses("WRONG_VERSION", lambda: C.require_identity(C.open_store(STORE), pinned))


def test_pinned_bytes_match_recorded_digests_or_the_absence_is_named():
    record = json.loads(PINNED.read_text())
    derived = record["version"]["derived_from"]
    assert record["schema"] == "df_dr05_prefix_output.v1"
    assert record["version"]["version"] == PINNED_VERSION
    assert derived["data_sha256"] == DATA_SHA
    assert derived["design_sha256"] == DESIGN_SHA
    assert derived["panel_sha256"] == PANEL_SHA
    assert derived["donor_weights_sha256"] == DONOR_WEIGHTS_SHA
    assert record["arrays"]["train"]["sha256"] == TRAIN_SHA
    assert record["arrays"]["validation"]["sha256"] == VALIDATION_SHA
    assert record["arrays"]["train"]["origins_sha256"] == TRAIN_ORIGINS_SHA
    assert record["arrays"]["validation"]["origins_sha256"] == VALIDATION_ORIGINS_SHA
    assert record["manifest_sha256"] == MANIFEST_SHA
    assert record["splits"]["train"]["n_origins"] == 40080
    assert record["splits"]["validation"]["n_origins"] == 10020
    store = Path(record["store"])
    if not store.is_dir() or not (store / "MANIFEST.json").is_file():
        _refuses("MISSING_ARTIFACT", lambda: C.open_store(store))
        print("PINNED bytes=ABSENT parity=NOT_CLAIMED")
        return
    measured = C.open_store(store)
    report = C.equality_report(measured, record)
    assert report["bytes"] == "MATCH"
    assert report["parity"] == "ROW_AND_VALUE_EQUAL"
    assert report["alcance"] == "NOT_DEFINED_IN_REPO"
    assert report["version"] == PINNED_VERSION
    assert report["splits"]["train"]["rows"] == 40080
    assert report["splits"]["validation"]["rows"] == 10020
    assert report["splits"]["train"]["sha256"] == TRAIN_SHA
    assert report["splits"]["validation"]["sha256"] == VALIDATION_SHA
    assert report["splits"]["train"]["origins_sha256"] == TRAIN_ORIGINS_SHA
    assert report["splits"]["validation"]["origins_sha256"] == VALIDATION_ORIGINS_SHA
    assert report["splits"]["train"]["row_count_equal"] is True
    assert report["splits"]["validation"]["row_count_equal"] is True
    assert report["splits"]["train"]["endpoint_digests"] == "MATCH"
    assert report["splits"]["validation"]["endpoint_digests"] == "MATCH"
    assert report["splits"]["train"]["second_read_byte_equal"] is True
    assert report["splits"]["validation"]["second_read_byte_equal"] is True
    train_shape = record["arrays"]["train"]["shape"]
    validation_shape = record["arrays"]["validation"]["shape"]
    assert report["splits"]["train"]["endpoint_elements"] == 2 * int(np.prod(train_shape[1:]))
    assert report["splits"]["validation"]["endpoint_elements"] == 2 * int(np.prod(validation_shape[1:]))
    window = C.read_origin(store, "train", record["splits"]["train"]["origin_span"][0])
    assert C.accept_window(window, window.copy())["value_equal"] is True
    mutated_value = window.copy()
    mutated_value.reshape(-1)[0] = np.float32(mutated_value.reshape(-1)[0] + np.float32(1))
    assert C.judge_window(window, mutated_value) == "MUTATED_VALUE"
    mutated_row = np.full_like(window, np.float32(-3))
    if np.array_equal(mutated_row, window):
        mutated_row = np.full_like(window, np.float32(-4))
    assert int(np.count_nonzero(window != mutated_row)) > 1
    assert C.judge_window(window, mutated_row) == "MUTATED_ROW"
    pair = np.stack([window, C.read_row_index(store, "train", 1)])
    both = pair.copy()
    both[0, 0, 0] = np.float32(both[0, 0, 0] + 1)
    both[1, 0, 0] = np.float32(both[1, 0, 0] + 1)
    assert C.judge_rows(pair, both) == "VALUE_MISMATCH"
    assert C.judge_rows(pair, pair[:1]) == "ROW_COUNT_MISMATCH"
    bad_digest = copy.deepcopy(record)
    bad_digest["arrays"]["train"]["sha256"] = "0" * 64
    _refuses("WRONG_DIGEST", lambda: C.require_identity(measured, bad_digest))
    bad_version = copy.deepcopy(record)
    bad_version["version"]["version"] = PINNED_VERSION + ".no"
    _refuses("WRONG_VERSION", lambda: C.require_identity(measured, bad_version))
    bad_identity = copy.deepcopy(record)
    bad_identity["version"]["derived_from"]["donor_weights_sha256"] = "ab" * 32
    _refuses("WRONG_IDENTITY", lambda: C.require_identity(measured, bad_identity))
    _refuses("MISSING_ARTIFACT", lambda: C.open_store(store / "no-such-directory"))
    print(
        "PINNED bytes=MATCH parity=ROW_AND_VALUE_EQUAL"
        f" train_rows={report['splits']['train']['rows']}"
        f" validation_rows={report['splits']['validation']['rows']}"
        f" train_sha256={report['splits']['train']['sha256']}"
        f" validation_sha256={report['splits']['validation']['sha256']}"
        f" train_endpoint_elements={report['splits']['train']['endpoint_elements']}"
        f" validation_endpoint_elements={report['splits']['validation']['endpoint_elements']}"
        " rejections=EXACT_MATCH,VALUE_MISMATCH,ROW_COUNT_MISMATCH,MUTATED_VALUE,"
        "MUTATED_ROW,WRONG_DIGEST,WRONG_VERSION,WRONG_IDENTITY,MISSING_ARTIFACT"
    )
