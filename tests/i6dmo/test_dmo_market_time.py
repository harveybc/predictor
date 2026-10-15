"""Market-time pair (order 1.2): typed MG2/MG12 designs and the dynamic time encoder.

Plan sections 2/3/6 are frozen. Tests first, CPU only, no fit, no GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tools.i6dmo import design as D
from tests.i6dmo.dmo_model_fixtures import DESIGN, load_design_doc

REPO = DESIGN.parents[3]
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
CANDIDATES = DESIGN.parent / "market_grid"
MG2_ID = "I6-DMO-SHORT-MG2-MARKET-EXACT-R0"
MG12_ID = "I6-DMO-LONG-MG12-MARKET-EXACT-R0"


@pytest.fixture()
def sealed(tmp_path):
    result = D.seal_market_time(DESIGN, CANDIDATES, tmp_path)
    return tmp_path, result


# --------------------------------------------------------------- designs (order 1.2.1)

def test_mg2_and_mg12_designs_are_sealed_with_the_frozen_scope(sealed):
    out, result = sealed
    short = D.load_design(out / "EXPERIMENT.mg2.json")
    long_ = D.load_design(out / "EXPERIMENT.mg12.json")
    assert (short["experiment_id"], long_["experiment_id"]) == (MG2_ID, MG12_ID)
    assert short["market_time"] == {
        "rule": "MG2_MARKET_EXACT", "observations": 13, "spacing_open_market_hours": 2,
        "span_open_market_hours": 24, "include_origin": True, "elapsed_span_fixed": False,
        "require_exact_market_slots": True}
    assert long_["market_time"] == {
        "rule": "MG12_MARKET_EXACT", "observations": 13, "spacing_open_market_hours": 12,
        "span_open_market_hours": 144, "include_origin": True, "elapsed_span_fixed": False,
        "require_exact_market_slots": True}
    assert short["horizons_hours"] == [2, 3, 4, 5, 6, 12]
    assert long_["horizons_hours"] == [18, 24, 36, 48, 60, 72]
    assert short["schema"] == "i6dmo.market_time_design.v1"
    # The OLD schema's history_span_hours keeps its elapsed meaning: it is absent here.
    assert "input" not in short and "history_span_hours" not in json.dumps(short["market_time"])
    for doc in (short, long_):
        assert doc["seed"] == 6102026 and doc["repetitions"] == 1
        assert doc["retraining_frequency"] == "NONE"
        assert doc["splits"]["train"] == load_design_doc()["splits"]["train"]
        assert doc["splits"]["test"]["open"] is False
        assert doc["model"] == load_design_doc()["model"]          # frozen recipe
        assert doc["training"] == load_design_doc()["training"]    # frozen optimizer/stopping
        assert doc["feature_policy"] == load_design_doc()["feature_policy"]
    assert result["mg2"]["design_sha256"] == D.design_sha256(short)
    assert result["mg12"]["design_sha256"] == D.design_sha256(long_)
    assert result["mg2"]["design_sha256"] != result["mg12"]["design_sha256"]


def test_seal_binds_every_required_identity(sealed):
    out, _ = sealed
    for name in ("EXPERIMENT.mg2.json", "EXPERIMENT.mg12.json"):
        doc = json.loads((out / name).read_text())
        seal = doc["sealed_identities"]
        for key in ("time_encoder", "feature_manifest_sha256", "availability_contract_sha256",
                    "calendar", "candidate_sha256", "support_study_sha256", "seed", "runtime",
                    "base_short_design_sha256"):
            assert seal[key], key
        assert seal["calendar"]["status"] == "ASSUMED_NOT_BROKER"
        assert seal["time_encoder"]["layer"] == "i6dmo.DynamicPhysicalTimeEncoder"
        assert seal["time_encoder"]["version"] == "1.0.0"
        assert seal["time_encoder"]["context_channels"] == ["adjacent_elapsed_hours_over_24",
                                                            "closure_crossed"]
        assert seal["time_encoder"]["unit_conversion_hours"] == 24
        assert seal["time_encoder"]["fitted"] is False
        assert seal["time_encoder"]["dtype"] == "float32"
        assert seal["runtime"]["tensorflow"] and seal["runtime"]["keras"]
        assert doc["seal_sha256"] == D._sha({k: v for k, v in doc.items() if k != "seal_sha256"})
        report = D.validate_design_report(out / name)
        assert report["status"] == "PASS"
        assert report["family"] in ("short_mg2", "long_mg12")


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda d: d["market_time"].update(spacing_open_market_hours=1), "FIXED_VALUE"),
    (lambda d: d["market_time"].update(observations=25), "FIXED_VALUE"),
    (lambda d: d["market_time"].update(elapsed_span_fixed=True), "FIXED_VALUE"),
    (lambda d: d["market_time"].update(require_exact_market_slots=False), "FIXED_VALUE"),
    (lambda d: d.update(horizons_hours=[2, 3, 4, 5, 6, 24]), "FIXED_VALUE"),
    (lambda d: d["splits"]["test"].update(open=True), "TEST_FORBIDDEN"),
    (lambda d: d["training"].update(min_delta=1e-6), "FIXED_VALUE"),
    (lambda d: d["model"]["core"].update(transformer_blocks=1), "FIXED_VALUE"),
    (lambda d: d.update(surprise=1), "UNKNOWN_FIELD"),
    (lambda d: d["sealed_identities"]["time_encoder"].update(unit_conversion_hours=12), "FIXED_VALUE"),
    (lambda d: d["sealed_identities"].update(seed=1), "FIXED_VALUE"),
    (lambda d: d["market_time"].update(span_open_market_hours="UNDECIDED"), "MARKET_TIME_UNDECIDED"),
])
def test_market_time_typed_refusals(sealed, mutate, code):
    out, _ = sealed
    doc = json.loads((out / "EXPERIMENT.mg2.json").read_text())
    mutate(doc)
    with pytest.raises(D.DesignError, match=code + "|MARKET_TIME_SEAL"):
        D.validate_design(doc)


@pytest.mark.parametrize("contradiction", [
    {"input": {"history_span_hours": 24, "spacing_hours": 2, "observations": 13,
               "include_origin": True, "value_mode": "RAW", "issuance_hours": 1,
               "smoothing": False, "require_exact_timestamps": True}},
    {"history_span_hours": 24},
    {"spacing_hours": 2},
])
def test_fixed_elapsed_fields_alongside_market_time_are_refused(sealed, contradiction):
    out, _ = sealed
    doc = json.loads((out / "EXPERIMENT.mg2.json").read_text())
    doc.update(contradiction)
    with pytest.raises(D.DesignError, match="MARKET_TIME_CONTRADICTION|UNKNOWN_FIELD"):
        D.validate_design(doc)


def test_seal_is_deterministic_idempotent_and_identity_bound(tmp_path):
    a = D.seal_market_time(DESIGN, CANDIDATES, tmp_path / "a")
    b = D.seal_market_time(DESIGN, CANDIDATES, tmp_path / "b")
    assert (tmp_path / "a" / "EXPERIMENT.mg2.json").read_bytes() == \
        (tmp_path / "b" / "EXPERIMENT.mg2.json").read_bytes()
    assert a["mg2"]["design_sha256"] == b["mg2"]["design_sha256"]
    D.seal_market_time(DESIGN, CANDIDATES, tmp_path / "a")  # idempotent
    # A market-time design is not a base for the elapsed derivations.
    mt = D.load_design(tmp_path / "a" / "EXPERIMENT.mg2.json")
    for fn in (D.final_refit_design, D.hourly_control_design):
        with pytest.raises(D.DesignError, match="SHORT_DESIGN_REQUIRED|FINAL_FROM_VALIDATION_ONLY"):
            fn(mt)
    assert D.family(mt) == "short_mg2" and D.mode(mt) == "VALIDATION"


def test_cli_seal_market_time_and_validate(tmp_path):
    run = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "seal-market-time",
                          "--short-design", str(DESIGN), "--candidate-dir", str(CANDIDATES),
                          "--out-dir", str(tmp_path), "--json"], capture_output=True, text=True,
                         cwd=REPO, env=ENV, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    out = json.loads(run.stdout)
    assert out["status"] == "SEALED" and out["mg2"]["experiment_id"] == MG2_ID
    for name, family in (("EXPERIMENT.mg2.json", "short_mg2"), ("EXPERIMENT.mg12.json", "long_mg12")):
        got = subprocess.run([sys.executable, "-m", "tools.i6dmo.design", "validate-design",
                              "--design", str(tmp_path / name), "--json"], capture_output=True,
                             text=True, cwd=REPO, env=ENV, check=False)
        assert got.returncode == 0 and json.loads(got.stdout)["family"] == family


# --------------------------------------------------------------- encoder (order 1.2.2)

tf = pytest.importorskip("tensorflow")


def _offsets(spacing=2, slots=13, closure_at=None, closure_gap=50.0):
    """Signed UTC hour offsets ending exactly at 0, optionally with one closure jump."""
    deltas = [spacing] * (slots - 1)
    if closure_at is not None:
        deltas[closure_at - 1] = closure_gap
    offsets, flags = [0.0], [0.0]
    for d in deltas:
        offsets.append(offsets[-1] + d)
        flags.append(1.0 if d > spacing else 0.0)
    total = offsets[-1]
    return (np.asarray([v - total for v in offsets], np.float32),
            np.asarray(flags, np.float32))


def test_encoder_uses_actual_offsets_not_slot_indices_and_appends_two_context_channels():
    from tools.i6dmo.model import DynamicPhysicalTimeEncoder
    layer = DynamicPhysicalTimeEncoder()
    values = tf.zeros((2, 13, 76))
    regular, flags_regular = _offsets()
    weekend, flags_weekend = _offsets(closure_at=6)
    offsets = tf.constant(np.stack([regular, weekend]))
    flags = tf.constant(np.stack([flags_regular, flags_weekend]))
    out = np.asarray(layer([values, offsets, flags]))
    assert out.shape == (2, 13, 78)  # 76 encoded + exactly 2 context channels
    assert out.dtype == np.float32
    # Last offset exactly 0; first slot adjacent delta 0 and closure flag 0.
    assert float(regular[-1]) == 0.0 and float(weekend[-1]) == 0.0
    assert out[0, 0, 76] == 0.0 and out[0, 0, 77] == 0.0
    assert out[1, 0, 76] == 0.0 and out[1, 0, 77] == 0.0
    # Adjacent elapsed hours / 24, the fixed unit conversion.
    assert np.allclose(out[0, 1:, 76], 2.0 / 24.0, atol=1e-6)
    assert np.isclose(out[1, 6, 76], 50.0 / 24.0, atol=1e-6)
    assert out[1, 6, 77] == 1.0 and out[0, 6, 77] == 0.0
    # The sinusoidal code is the deterministic rule on ACTUAL offsets: two windows whose slot
    # ORDER is identical but whose physical offsets differ must encode differently.
    assert not np.allclose(out[0, :, :76], out[1, :, :76])
    # And it matches the plugin rule evaluated at those offsets, not at ordinal indices.
    width = 76
    channel = np.arange(width)
    rates = np.power(10000.0, -2.0 * (channel // 2) / width)
    angle = regular[:, None] * rates[None, :]
    expected = np.where(channel % 2 == 0, np.sin(angle), np.cos(angle))
    assert np.allclose(out[0, :, :76], expected, atol=1e-5)
    ordinal = np.arange(13, dtype=np.float32)[:, None] * rates[None, :]
    ordinal_code = np.where(channel % 2 == 0, np.sin(ordinal), np.cos(ordinal))
    assert not np.allclose(out[0, :, :76], ordinal_code, atol=1e-4)


def test_encoder_rejects_fixed_grid_substitution_and_bad_shapes():
    from tools.i6dmo.model import DynamicPhysicalTimeEncoder
    layer = DynamicPhysicalTimeEncoder()
    values = tf.zeros((2, 13, 76))
    regular, flags = _offsets()
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        layer([values, tf.constant(regular), tf.constant(flags)])      # no batch axis
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        layer([values, tf.constant(np.stack([regular[:12]] * 2)),
               tf.constant(np.stack([flags[:12]] * 2))])               # slot mismatch
    cfg = layer.get_config()
    assert cfg["unit_conversion_hours"] == 24.0 and cfg["version"] == "1.0.0"
    assert DynamicPhysicalTimeEncoder.from_config(cfg).unit_conversion_hours == 24.0


@pytest.mark.parametrize(("family", "spacing", "heads"), [
    ("short_mg2", 2, ["h2", "h3", "h4", "h5", "h6", "h12"]),
    ("long_mg12", 12, ["h18", "h24", "h36", "h48", "h60", "h72"]),
])
def test_market_time_model_wiring_and_frozen_recipe(family, spacing, heads):
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    bundle = M.build_dmo(M.model_config(list(MANIFEST_NO_STLFSI4), family=family))
    model = bundle.forecast_model
    assert [t.shape[1:] for t in model.inputs] == [(13, 19), (13,), (13,)]
    assert [t.name.split("/")[0].split(":")[0] for t in model.inputs][:1] == ["observations"]
    assert list(model.output_names) == heads
    # Encoder sits immediately after fusion and the core consumes its 78 channels.
    parents = [n.operation.name for node in model.get_layer("physical_time_encoder")._inbound_nodes
               for n in node.parent_nodes]
    assert "sequence_fusion" in parents
    assert [n.operation.name for node in model.get_layer("temporal_core")._inbound_nodes
            for n in node.parent_nodes] == ["physical_time_encoder"]
    assert tuple(bundle.core_model.input_shape) == (None, 13, 78)
    assert tuple(bundle.core_model.output_shape) == (None, 13, 4)
    # Frozen recipe outside the encoder.
    for branch in bundle.branch_models.values():
        conv = [l for l in branch.layers if isinstance(l, tf.keras.layers.Conv1D)][0].get_config()
        assert (conv["filters"], conv["kernel_size"], conv["strides"]) == (4, (3,), (1,))
    for head in bundle.head_models.values():
        dense = [l for l in head.layers if isinstance(l, tf.keras.layers.Dense)]
        assert [(d.units, d.get_config()["activation"]) for d in dense] == [(8, "relu"), (1, "linear")]
    forbidden = (tf.keras.layers.LSTM, tf.keras.layers.MultiHeadAttention, tf.keras.layers.Dropout)
    assert not [l for l in model.layers if isinstance(l, forbidden)]
    assert bundle.config["market_time"]["spacing_open_market_hours"] == spacing


def test_controlled_weights_prove_actual_time_inputs_reach_the_model():
    """A controlled-weight model whose only signal is the time context channels."""
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    bundle = M.build_dmo(M.model_config(list(MANIFEST_NO_STLFSI4), family="short_mg2"))
    model = bundle.forecast_model
    # Zero every branch so the encoded value channels are the pure positional code, then make
    # the core and one head read ONLY the two appended context channels.
    for branch in bundle.branch_models.values():
        branch.set_weights([np.zeros_like(w) for w in branch.get_weights()])
    core_w = bundle.core_model.get_weights()
    k0 = np.zeros_like(core_w[0])
    k0[-1, 76, 0] = 1.0   # current tap, adjacent-hours/24 channel
    k0[-1, 77, 1] = 1.0   # current tap, closure flag channel
    core_w[0] = k0
    core_w[1] = np.zeros_like(core_w[1])
    k1 = np.zeros_like(core_w[2])
    k1[-1, 0, 0] = 1.0
    k1[-1, 1, 1] = 1.0
    core_w[2] = k1
    core_w[3] = np.zeros_like(core_w[3])
    bundle.core_model.set_weights(core_w)
    head = bundle.head_models["h2"]
    hw = head.get_weights()
    hidden = np.zeros_like(hw[0])
    hidden[-4, 0] = 1.0   # last slot, channel 0 (adjacent hours / 24)
    hidden[-3, 1] = 1.0   # last slot, channel 1 (closure flag)
    hw[0], hw[1] = hidden, np.zeros_like(hw[1])
    out = np.zeros_like(hw[2])
    out[0, 0], out[1, 0] = 1.0, 1.0
    hw[2], hw[3] = out, np.zeros_like(hw[3])
    head.set_weights(hw)
    x = np.zeros((2, 13, 19), np.float32)
    regular, flags_regular = _offsets()
    weekend, flags_weekend = _offsets(closure_at=12, closure_gap=50.0)
    pred = M.predict_matrix(model, [x, np.stack([regular, weekend]),
                                    np.stack([flags_regular, flags_weekend])])
    # Last slot: regular window contributes 2/24 and no closure; the weekend window 50/24 + 1.
    assert np.isclose(pred[0, 0], 2.0 / 24.0, atol=1e-5)
    assert np.isclose(pred[1, 0], 50.0 / 24.0 + 1.0, atol=1e-5)
    assert pred[1, 0] > pred[0, 0]  # actual physical time changed the output


def test_serialization_reloads_all_inputs_including_time_and_closure(tmp_path):
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    bundle = M.build_dmo(M.model_config(list(MANIFEST_NO_STLFSI4), family="short_mg2"))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(5, 13, 19)).astype(np.float32)
    offsets = np.stack([_offsets(closure_at=4)[0] for _ in range(5)])
    flags = np.stack([_offsets(closure_at=4)[1] for _ in range(5)])
    inputs = [x, offsets, flags]
    doc = M.save_dmo(bundle, tmp_path / "m")
    assert doc["market_time"]["slot_order_is_not_wall_clock"] is True
    assert doc["market_time"]["rule"] == "MG2_MARKET_EXACT"
    assert doc["components"]["time_encoder"]["version"] == "1.0.0"
    loaded, document = M.load_dmo(tmp_path / "m")
    assert len(loaded.forecast_model.inputs) == 3
    assert np.array_equal(M.predict_matrix(bundle.forecast_model, inputs),
                          M.predict_matrix(loaded.forecast_model, inputs))
    # Changing only the time inputs changes the reloaded model's predictions too.
    other = [x, np.stack([_offsets()[0] for _ in range(5)]),
             np.stack([_offsets()[1] for _ in range(5)])]
    assert not np.allclose(M.predict_matrix(loaded.forecast_model, inputs),
                           M.predict_matrix(loaded.forecast_model, other))


def test_no_donor_is_compatible_by_shape_alone(tmp_path):
    """A DS2-era artifact never loads as a market-time model and vice versa."""
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    features = list(MANIFEST_NO_STLFSI4)
    mt = M.build_dmo(M.model_config(features, family="short_mg2"))
    M.save_dmo(mt, tmp_path / "mt")
    doc = json.loads((tmp_path / "mt" / "MODEL.json").read_text())
    doc["config"] = M.model_config(features)          # same horizons/slots, elapsed family
    (tmp_path / "mt" / "MODEL.json").write_text(json.dumps(doc, sort_keys=True, indent=1))
    with pytest.raises(ValueError, match="WEIGHTS_DIGEST_MISMATCH|MARKET_TIME_INCOMPATIBLE|weights"):
        M.load_dmo(tmp_path / "mt")


# --------------------------------------------------------------- DS2 preserved (order 1.2.4)

def test_ds2_and_other_elapsed_families_are_untouched():
    from tools.i6dmo import model as M
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    features = list(MANIFEST_NO_STLFSI4)
    for family, slots, head0 in (("short", 13, "h2"), ("long", 13, "h18"),
                                 ("short_hourly", 25, "h2")):
        bundle = M.build_dmo(M.model_config(features, family=family))
        model = bundle.forecast_model
        assert len(model.inputs) == 1                       # single observations input
        assert tuple(model.input_shape) == (None, slots, 19)
        assert tuple(bundle.core_model.input_shape) == (None, slots, 76)  # no context channels
        assert isinstance(model.get_layer("positional_encoding"), M.PhysicalOffsetEncoding)
        assert list(model.output_names)[0] == head0
        assert "physical_time_encoder" not in [l.name for l in model.layers]
    assert M.build_dmo(M.model_config(features)).forecast_model.count_params() == 4834


def test_numerical_policy_is_unchanged():
    from tools.i6dmo import train as T
    assert T.MONITOR_TOLERANCE["rtol"] == 5e-5 and T.MONITOR_TOLERANCE["atol"] == 1e-9
    assert T.MONITOR_TOLERANCE["version"] == "v2"
    assert T.floor_protocol_identity()["id"] == "i6dmo.monitor_numeric_equivalence.v2"
    assert T.MONITOR_EVAL == "i6dmo.monitor_eval.v2"
