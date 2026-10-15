"""Musashi point 5: ONE bounded element-level isolation diagnostic (CPU logic tests).

No production fit, no new rounds: freeze the first monitored batch + checkpoint
+ preprocessing identity from a retained unit, compare predictions element by
element before any reduction, reproduce the fsum sums independently, and run a
controlled flags-on/off comparison in fresh processes.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tools.i6dmo import dmo10_isolation as I  # noqa: E402
from tools.i6dmo import train as T  # noqa: E402
from tests.i6dmo.dmo_model_fixtures import DESIGN, write_bundle  # noqa: E402


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    root = tmp_path_factory.mktemp("iso")
    bundle = write_bundle(root / "b")
    receipt = T.train_unit(DESIGN, bundle, root / "u", device="cpu",
                           training_overrides={"max_epochs": 2})
    assert receipt["status"] == "COMPLETED"
    return root, bundle, root / "u"


def test_freeze_records_every_identity_and_keeps_arrays_local(unit, tmp_path):
    root, bundle, unit_dir = unit
    frozen = I.freeze(DESIGN, bundle, unit_dir, tmp_path / "frozen", split="fit", batch=0)
    doc = json.loads((tmp_path / "frozen" / "FROZEN.json").read_text())
    for key in ("x_sha256", "y_sha256", "mask_sha256", "row_ids_sha256", "scales_sha256",
                "weights_sha256", "model_archive_sha256", "determinism_flags",
                "monitor_forward", "input_materialization", "runtime", "device",
                "recorded_partial_sums", "shape", "dtype"):
        assert doc[key] is not None, key
    assert doc["dtype"] == "float32" and doc["shape"][1:] == [13, 19]
    assert (tmp_path / "frozen" / "batch.npz").exists()  # arrays stay local; Git gets digests
    assert frozen["frozen_sha256"] == doc["frozen_sha256"]


def test_forward_and_diff_element_level_before_any_reduction(unit, tmp_path):
    root, bundle, unit_dir = unit
    frozen = tmp_path / "frozen"
    I.freeze(DESIGN, bundle, unit_dir, frozen, split="fit", batch=0)
    a = I.forward(frozen, tmp_path / "a.json")
    b = I.forward(frozen, tmp_path / "b.json")
    diff = I.diff(frozen, tmp_path / "a.json", tmp_path / "b.json", tmp_path / "DIFF.json")
    assert diff["identical"] is True and diff["tensor_sha256_equal"] is True
    assert diff["first_differing_element"] is None
    assert diff["max_abs_error"] == 0.0 and diff["mean_abs_error"] == 0.0
    # Independent fsum reproduction against the retained witness partial sums.
    assert diff["fsum_reproduction"]["matches_recorded"] is True
    assert diff["comparison_environment"] == "same_environment"  # never requires 5090==4090
    # A corrupted prediction file is localized to its first differing element.
    doc = json.loads((tmp_path / "b.json").read_text())
    values = np.frombuffer(bytes.fromhex(doc["predictions_hex"]), dtype=np.float64).copy()
    values[7] += 1e-3
    doc["predictions_hex"] = values.tobytes().hex()
    (tmp_path / "b.json").write_text(json.dumps(doc))
    diff = I.diff(frozen, tmp_path / "a.json", tmp_path / "b.json", tmp_path / "DIFF2.json")
    assert diff["identical"] is False
    assert diff["first_differing_element"]["flat_index"] == 7
    assert diff["max_abs_error"] > 0 and diff["n_differing"] == 1


def test_flags_comparison_is_controlled_and_honest_on_cpu(unit, tmp_path):
    root, bundle, unit_dir = unit
    frozen = tmp_path / "frozen"
    I.freeze(DESIGN, bundle, unit_dir, frozen, split="fit", batch=0)
    report = I.flags_comparison(frozen, tmp_path / "FLAGS.json")
    assert report["flags_on"]["predictions_sha256"] == report["flags_off"]["predictions_sha256"]
    assert report["verdict"] == "NOT_INFORMATIVE_ON_CPU"  # honest: CPU cannot test GPU autotune
    assert report["fits_run"] == 0 and report["test_rows_read"] == 0
