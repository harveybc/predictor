"""CB01/CB02/CB04: context bridge seals, identity, R0 parity and zero capacity delta.

Synthetic model inputs only; no fit, services, real market data or TEST input.
Legacy fingerprints were captured before adding the bridge families.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import common_population as CP, design as D, model as M, provenance

DOCS = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static"
SHORT = DOCS / "EXPERIMENT.short.json"
ARMS = ("mg12c", "mg2l")
FAMILIES = ("long_mg12_bridge_control", "long_mg2_bridge")
HEADS = [18, 24, 36, 48, 60, 72]
CONTRAST = "I6-DMO-LONG-SHORT-INPUT-CONTEXT-BRIDGE-R0"


def bridge_designs():
    short = D.load_design(SHORT)
    data = {arm: CP.matched_data_design(short, arm) for arm in ARMS}
    return {arm: D.matched_market_time_design(short, arm, data) for arm in ARMS}, data


def test_cb01_explicit_scope_and_isolated_common_population_rules():
    assert tuple(D.CONTEXT_BRIDGE_SCOPE) == ARMS
    assert set(D.ALL_MATCHED_SCOPE) == set(D.MATCHED_SCOPE) | set(D.LONG_MATCHED_SCOPE) | set(ARMS)
    assert D.matched_arm_set(reversed(ARMS)) == ARMS
    assert D.CONTEXT_BRIDGE_CONTRAST_ID == CONTRAST
    rules = D.matched_population_rules(ARMS[0])
    assert rules == D.CONTEXT_BRIDGE_POPULATION_RULES == D.matched_population_rules(ARMS[1])
    assert rules["contrast_id"] == CONTRAST
    assert rules["rules_sha256"] == D._sha({k: v for k, v in rules.items() if k != "rules_sha256"})
    assert "subset" not in rules["population"].lower()
    assert "intersection" in rules["population"] and "UNION" in rules["scaler"]
    assert "MG12c/MG2l" in rules["scaler"]
    assert rules["full_population_fit_substitutes"] is False
    rules["splits"].clear()
    assert D.matched_population_rules(ARMS[0])["splits"]
    for arm, family, spacing, span, role, partner in (
        ("mg12c", FAMILIES[0], 12, 144, "CONTROL", "mg2l"),
        ("mg2l", FAMILIES[1], 2, 24, "TREATMENT", "mg12c"),
    ):
        scope = D.CONTEXT_BRIDGE_SCOPE[arm]
        assert scope["family"] == family and scope["observations"] == 13
        assert scope["spacing_open_market_hours"] == spacing
        assert scope["span_open_market_hours"] == span
        assert scope["role"] == role and scope["partner"] == partner
        assert scope["horizons_hours"] == HEADS


@pytest.mark.parametrize("arms", [(), ("mg12c",), ("mg2l",), ("mg12c", "mg12m"),
                                 ("mg1l", "mg2l"), ("mg2m", "mg2l"),
                                 ("mg12c", "mg2l", "mg1l")])
def test_cb01_mixed_partial_and_combined_pairs_refused(arms):
    with pytest.raises(D.DesignError, match="ARMS_REQUIRED"):
        D.matched_arm_set(arms)


def test_cb01_seal_shape_idempotency_and_write_once(tmp_path):
    out = tmp_path / "bridge"
    result = D.seal_context_bridge(SHORT, out)
    assert result["schema"] == "i6dmo.seal_context_bridge.v1"
    assert result["status"] == "SEALED" and result["fits_run"] == 0
    assert result["contrast_id"] == CONTRAST
    assert set(result) == {"schema", "status", "fits_run", "contrast_id",
                           "matched_population_rules_sha256", *ARMS}
    assert {p.name for p in out.iterdir()} == {
        "MATCHED_POPULATION_RULES.json", *(f"{kind}.{arm}.json" for arm in ARMS
                                         for kind in ("DESIGN", "EXPERIMENT"))}
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert D.seal_context_bridge(SHORT, out) == result
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    for arm, family in zip(ARMS, FAMILIES):
        doc = D.load_design(out / f"EXPERIMENT.{arm}.json")
        assert D.matched_arm(doc) == arm and D.family(doc) == family
        effective = D.effective_config(doc)
        assert result[arm]["effective_config_sha256"] == effective["effective_config_sha256"]
        assert result[arm]["experiment_file_sha256"] == D._file_sha(out / f"EXPERIMENT.{arm}.json")
    path = out / "EXPERIMENT.mg2l.json"
    path.write_text("{}\n")
    with pytest.raises(D.DesignError, match="MATCHED_DESIGN_EXISTS"):
        D.seal_context_bridge(SHORT, out)
    assert path.read_text() == "{}\n"


def test_cb01_schema_and_builder_bind_bridge_contrast_and_partner():
    docs, data = bridge_designs()
    for arm, family in zip(ARMS, FAMILIES):
        doc = docs[arm]
        schema = D.MATCHED_SCHEMAS[family]
        assert schema["matched_population"]["contrast_id"] == ("fixed", CONTRAST)
        assert doc["matched_population"]["contrast_id"] == CONTRAST
        assert doc["matched_population"]["partner_arm"] == D.CONTEXT_BRIDGE_SCOPE[arm]["partner"]
        effective = D.effective_config(doc)
        assert effective["matched_population_rules"] == D.CONTEXT_BRIDGE_POPULATION_RULES
        assert effective["model"]["family"] == family
        assert effective["model"]["slots"] == 13 and effective["horizons_hours"] == HEADS
        assert effective["seed"] == 6102026 and effective["offsets_hours"] is None
        assert M.normalize_model_config(effective["model"]) == M.model_config(provenance.MANIFEST, family=family)
        other = D.CONTEXT_BRIDGE_SCOPE[arm]["partner"]
        assert doc["sealed_identities"]["partner_data_identity_sha256"] == data[other]["data_identity_sha256"]
    assert docs[ARMS[0]]["model"] == docs[ARMS[1]]["model"]
    assert docs[ARMS[0]]["training"] == docs[ARMS[1]]["training"]
    legacy = CP.matched_data_design(D.load_design(SHORT), "mg12m")
    with pytest.raises(D.DesignError, match="ARMS_REQUIRED"):
        D.matched_market_time_design(D.load_design(SHORT), "mg12c", {"mg12c": data["mg12c"], "mg12m": legacy})


@pytest.mark.parametrize("field,value", [("contrast_id", D.LONG_MATCHED_CONTRAST_ID),
                                        ("partner_arm", "mg12m"), ("role", "MATCHED_REFERENCE")])
def test_cb01_resealed_wrong_contrast_partner_or_role_refused(field, value):
    docs, _ = bridge_designs()
    bad = copy.deepcopy(docs["mg2l"])
    bad["matched_population"][field] = value
    bad["seal_sha256"] = D._sha(D._seal_body(bad))
    with pytest.raises(D.DesignError, match="FIXED_VALUE"):
        D.validate_design(bad)


def test_cb02_real_cpu_graph_capacity_and_initial_weight_parity(tmp_path):
    records, weights, predictions = [], [], []
    x = np.random.default_rng(41).normal(size=(2, 13, 19)).astype(np.float32)
    for arm, family, spacing in zip(ARMS, FAMILIES, (12, 2)):
        bundle = M.build_dmo(M.model_config(provenance.MANIFEST, family=family))
        record = M.capacity_record(bundle)
        records.append(record)
        assert record["arm"] == arm and record["family"] == family
        assert record["counts"] == M.DECLARED_CAPACITY[family]
        assert record["counts"]["total"] == 4882
        assert list(bundle.head_models) == [f"h{h}" for h in HEADS]
        assert tuple(bundle.forecast_model.input_shape[0][1:]) == (13, 19)
        weights.append([w.copy() for w in bundle.forecast_model.get_weights()])
        offsets = np.tile(np.arange(-12 * spacing, 1, spacing, dtype=np.float32), (2, 1))
        closure = np.zeros_like(offsets)
        output = bundle.forecast_model([x, offsets, closure], training=False)
        predictions.append(np.concatenate([y.numpy() for y in output], axis=1))
        artifact = M.save_dmo(bundle, tmp_path / arm)
        restored, metadata = M.load_dmo(tmp_path / arm)
        assert artifact["config"]["family"] == metadata["config"]["family"] == family
        assert M.capacity_record(restored)["arm"] == arm
        replay = restored.forecast_model([x, offsets, closure], training=False)
        np.testing.assert_array_equal(predictions[-1], np.concatenate([y.numpy() for y in replay], axis=1))
        encoder = bundle.forecast_model.get_layer("physical_time_encoder")
        assert encoder.min_adjacent_hours == spacing
        with pytest.raises((ValueError, M.tf.errors.InvalidArgumentError), match="ORDINAL_OR_INVALID_OFFSETS"):
            bundle.forecast_model([x, np.tile(np.arange(-12, 1, dtype=np.float32), (2, 1)), closure], training=False)
    assert len(weights[0]) == len(weights[1])
    for control, treatment in zip(*weights):
        np.testing.assert_array_equal(control, treatment)
    assert all(np.isfinite(p).all() for p in predictions)
    assert not np.array_equal(*predictions)
    delta = M.matched_capacity_delta(*records)
    assert delta["declared_head_delta"] == 0
    assert delta["delta"] == dict(branches=0, fusion=0, time_encoder=0, core=0, heads=0, total=0)
    assert delta["formula"] == "(13-13)*4*8*6"


@pytest.mark.parametrize("family,field,value", [(FAMILIES[0], "slots", 25),
                                                (FAMILIES[1], "horizons_hours", [2, 3, 4, 5, 6, 12]),
                                                (FAMILIES[1], "core", {"filters": [16, 4]}),
                                                (FAMILIES[0], "family", FAMILIES[1])])
def test_cb02_undeclared_model_changes_refused(family, field, value):
    config = M.model_config(provenance.MANIFEST, family=family)
    config[field] = value
    with pytest.raises(ValueError, match="FIXED_VALUE"):
        M.normalize_model_config(config)


def capacity_records():
    return [{"schema": M.CAPACITY_SCHEMA, "arm": arm,
             "counts": copy.deepcopy(M.DECLARED_CAPACITY[family])}
            for arm, family in zip(ARMS, FAMILIES)]


def test_cb04_capacity_accepts_only_ordered_bridge_pair():
    a, b = capacity_records()
    assert M.matched_capacity_delta(a, b)["delta"]["total"] == 0
    for left, right in ((b, a), (a, {**b, "arm": "mg12m"}),
                        ({**a, "arm": "mg1l"}, b)):
        with pytest.raises(M.CapacityError):
            M.matched_capacity_delta(left, right)


@pytest.mark.parametrize("field,value", [("family", "long_mg12"), ("counts", None),
                                        ("counts", {}), ("schema", "not-a-capacity-record")])
def test_cb04_malformed_or_relabelled_capacity_refused(field, value):
    a, b = capacity_records()
    a[field] = value
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(a, b)


@pytest.mark.parametrize("field", ["slots", "features", "core_input_channels", "branches", "per_branch",
                                   "fusion", "time_encoder", "core", "per_head", "per_head_hidden",
                                   "per_head_output", "heads", "total", "core_layers"])
def test_cb04_capacity_refuses_even_shared_undeclared_changes(field):
    a, b = capacity_records()
    for record in (a, b):
        if field == "core_layers":
            record["counts"][field][0] += 1
        else:
            record["counts"][field] += 1
    with pytest.raises(M.CapacityError):
        M.matched_capacity_delta(a, b)


def test_cb01_legacy_schema_effective_and_model_config_fingerprints_unchanged():
    schemas = {"short": D.SCHEMA, "long": D.LONG_SCHEMA, **D.MATCHED_SCHEMAS}
    expected_schemas = {
        "short": "275279c7ef02316e6b97bd0f3349f77479d77f247354068816a4123bd7cf041f",
        "long": "b9610e2e3a38bbeb0244ce9a7e9946bb6d7c2487641ab749a786dd8d09839357",
        "short_mg1_matched": "efe12c50fd830f7e4ab8e43cd647369bf97543f617809f8d3d9b195f07158818",
        "short_mg2_matched": "957d65b72d15d499c6d37b710f30afbaf3b89b6f3505589981170571ac592edd",
        "long_mg1_matched": "38d4b7bafce24a7ec99f9e80cedc7d28d9989700b1924cd62b5d894d9dd5aa69",
        "long_mg12_matched": "2fdf4167b3e7f2044b90231a1dc5663ce80dc4e471f2e0a1ea137351f6c542eb",
    }
    for name, expected in expected_schemas.items():
        encoded = json.dumps(schemas[name], sort_keys=True, default=lambda t: t.__name__).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected
    expected_effective = {
        "EXPERIMENT.short.json": "287e4f11e4ef385234becf02d9ed31864d8a3a58fc0b4c7f3e4c6a765a76e097",
        "matched_short/EXPERIMENT.mg1.json": "894cda3b67c80fe60ccd94eaf71d99cd8aee7fab8174c553702f119626920140",
        "matched_short/EXPERIMENT.mg2m.json": "ad949872cd5c430598369049dc82dd6063707080f55ee8f906323ab7f47218c7",
        "matched_long/EXPERIMENT.mg1l.json": "b5fd88a3ee5162bb9f69ae53f90b2acfad5001e58c1981acc7b170ebe1100486",
        "matched_long/EXPERIMENT.mg12m.json": "309354bac63f187c40e790169b294619880c915cbedb8f2c958b98a0496d1755",
    }
    for rel, expected in expected_effective.items():
        assert D.effective_config(D.load_design(DOCS / rel))["effective_config_sha256"] == expected
    expected_models = {
        "short": "4c810b5cf8e9524a959bfdc269cfd9d6ceb2a2950d7239ea6b632de8cf42547c",
        "long": "d63cbd774e271a744ee6054f0e849bcf650e6a0c4ca9a2e265c41f9991b294a1",
        "short_hourly": "0706883949e98c75d44e1ff98445abe75612e2e3d4c67be84ac132473cf5e9e0",
        "short_mg2": "8c29a338a7312edbba15269666cf20fce4e5f31ee0a251c664727ca6875b5136",
        "long_mg12": "fa7e19b47e7ca923d24151c6c7afa7bfd03aa61e75748edc7a022da3dd5a6d0b",
        "short_mg1": "e816ae066191b5298b1261414f60171089dc63bb6560de05d1558c53fe6a2344",
        "long_mg1": "639aa4dbe5aeb076f1c5b24a48f80efe2688f28b053b4ee75fc7b99361ba5d57",
    }
    for family, expected in expected_models.items():
        config = M.model_config(provenance.MANIFEST, family=family)
        assert hashlib.sha256(M._canonical(config).encode()).hexdigest() == expected


@pytest.mark.parametrize("scope,seal,directory", [(D.MATCHED_SCOPE, D.seal_matched_short, "matched_short"),
                                                (D.LONG_MATCHED_SCOPE, D.seal_matched_long, "matched_long")])
def test_cb01_old_seal_outputs_still_byte_identical(tmp_path, scope, seal, directory):
    seal(SHORT, tmp_path)
    for name in ("MATCHED_POPULATION_RULES.json", *(f"{kind}.{arm}.json" for arm in scope
                                                for kind in ("DESIGN", "EXPERIMENT"))):
        assert (tmp_path / name).read_bytes() == (DOCS / directory / name).read_bytes()
