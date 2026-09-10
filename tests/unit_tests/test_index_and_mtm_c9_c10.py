"""C9-C10: a consumable index and a total MTM boundary.

C9 — the index carried 214 rows and ZERO variables, so it could
not be the source of a selection universe. C10 — the MTM function
returned a bare array on a short partition while its caller
unpacked a tuple, and nothing stopped validation or test from
fitting a scaler that training had failed to produce.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import bank_index as bi  # noqa: E402

INDEX = REPO / "examples/research/crispdm_bank_index.v1.json"


@pytest.fixture(scope="module")
def index() -> dict:
    if not INDEX.is_file():
        pytest.skip("bank index not built")
    return json.loads(INDEX.read_text())


# ==============================================================
# C9: the index is consumable
# ==============================================================

def test_the_index_carries_the_conceptual_variables(index):
    kinds = {}
    for row in index["common_rows"]:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    assert kinds.get("variable") == 1965, (
        "the 1,965 conceptual variables are still a count, not "
        "rows")
    assert kinds.get("physical_appearance") == 1680
    assert kinds.get("series") == 4650
    assert kinds.get("generator") == 202
    assert kinds.get("dataset") == 10
    assert kinds.get("operator") == 4


def test_cardinalities_are_recounted_from_the_rows(index):
    recount = bi._recount(index["common_rows"])
    assert recount == index["cardinality_by_kind_and_authority"]
    total = sum(n for kinds in recount.values()
                for n in kinds.values())
    assert total == index["row_count"] == len(
        index["common_rows"])
    assert "derived FROM" in index["cardinality_rule"]


def test_every_row_carries_authority_source_and_identity(index):
    for row in index["common_rows"]:
        assert row["authority"] in (bi.PUBLIC, bi.SYNTHETIC,
                                    bi.FINANCIAL)
        assert row["bank"] in index["banks"]
        assert row["id"]
        assert row["family"]


def test_appearance_rows_carry_a_real_digest(index):
    apps = [r for r in index["common_rows"]
            if r["kind"] == "physical_appearance"]
    assert apps
    for r in apps[:50]:
        assert len(r["digest"]) == 64, (
            "an appearance row must carry the digest the census "
            "computed")
        assert r["identity_state"] in (
            "PHYSICALLY_DIGESTED",
            "REUSED_FROM_PREVIOUS_VERIFIED_CENSUS")


def test_variable_rows_never_claim_a_byte_identity(index):
    """A conceptual variable has no bytes of its own; claiming a
    digest for it would be the two grains collapsing again."""
    for r in index["common_rows"]:
        if r["kind"] == "variable":
            assert r["digest"] == "UNAVAILABLE"
            assert r["identity_state"] == "CONCEPTUAL_IDENTITY"


def test_a_join_still_cannot_promote_any_row(index):
    for row in index["common_rows"][::200]:
        if row["authority"] != bi.PUBLIC:
            with pytest.raises(SystemExit,
                               match="never promotes"):
                bi.assert_no_promotion(row["authority"],
                                       "PUBLICLY_ELIGIBLE",
                                       subject=row["id"])


def test_known_operators_carry_their_evidence_state(index):
    ops = {o["operator_id"]: o for o in
           index["banks"]["financial_domain"]["operators"]}
    assert ops["op.denoise.D"]["state"] == \
        "PUBLICLY_EVALUATED_VERDICT_DOES_NOT_ADVANCE"
    assert ops["op.selector.acf_pacf_pre"]["state"] == \
        "LEGACY_NON_AUTHORITATIVE"
    assert ops["op.selector.embedded_post"]["state"] == \
        "LEGACY_NON_AUTHORITATIVE"
    for o in ops.values():
        assert o["evidence"]


def test_the_financial_bank_binds_the_census_it_carries(index):
    b = index["banks"]["financial_domain"]
    assert len(b["variables"]) == 1965
    assert len(b["appearances"]) == 1680
    assert len(b["binding"]["census_document_sha256"]) == 64


def test_variables_keep_event_and_available_time_separate(index):
    b = index["banks"]["financial_domain"]
    for v in b["variables"][:100]:
        assert "event_time" in v and "available_time" in v
        assert v["available_time"] == "UNAVAILABLE"


# ==============================================================
# C10: the MTM boundary is total
# ==============================================================

@pytest.fixture(scope="module")
def plugin():
    mod = pytest.importorskip(
        "preprocessor_plugins.phase2_6_preprocessor")
    cls = getattr(mod, "PreprocessorPlugin", None) or \
        getattr(mod, "Plugin", None)
    if cls is None:
        pytest.skip("phase 2.6 plugin not found")
    return cls()


def _series(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(0, 1, n)) * scale).astype(
        np.float32)


def test_a_short_partition_keeps_the_return_contract(plugin):
    """The audit's exact reproduction: a partition shorter than
    window_size+1 must still return (components, scaler)."""
    out = plugin._apply_causal_mtm_decomposition(
        np.zeros(6, dtype=np.float32), 32, 2, "short")
    assert isinstance(out, tuple) and len(out) == 2
    components, scaler = out
    assert isinstance(components, np.ndarray)
    assert components.shape == (6, 2)
    assert scaler is None, (
        "a partition too short to decompose must not hand back "
        "a scaler")


def test_every_length_returns_the_same_shape_of_answer(plugin):
    for n in (1, 5, 32, 33, 64, 200):
        out = plugin._apply_causal_mtm_decomposition(
            _series(n, seed=n), 32, 3, f"len{n}")
        assert isinstance(out, tuple) and len(out) == 2
        comp, _ = out
        assert comp.shape == (n, 3)


def test_a_constant_train_still_returns_the_contract(plugin):
    """A constant series is not the no-scaler case: its MTM
    components still vary at the causal boundary, so a scaler IS
    fitted. What matters is that the contract holds and the ONE
    scaler travels — the no-scaler case is the short partition
    above."""
    constant = np.full(200, 3.0, dtype=np.float32)
    out = plugin._apply_causal_mtm_decomposition(
        constant, 32, 3, "constant")
    assert isinstance(out, tuple) and len(out) == 2
    comp, scaler = out
    assert comp.shape == (200, 3)
    assert scaler is not None, (
        "a constant series still decomposes; if this ever "
        "returns None the NOT_EVALUABLE rule must cover it")


def test_validation_may_not_fit_when_train_produced_none():
    """The rule, read from the code that enforces it."""
    src = (REPO / "preprocessor_plugins/"
                  "phase2_6_preprocessor.py").read_text()
    flat = " ".join(src.split())
    assert "if mtm_scaler is None and not legacy_per_split" in \
        flat
    assert "NOT_EVALUABLE" in src
    assert "are NOT_EVALUABLE rather than fitted" in flat
    assert "mtm_not_evaluable_units" in src


def test_a_valid_train_scaler_still_travels(plugin):
    train = _series(400, seed=7)
    tail = _series(200, seed=8, scale=30.0)
    comp, scaler = plugin._apply_causal_mtm_decomposition(
        train, 32, 3, "train")
    assert scaler is not None
    mean_before = np.array(scaler.mean_, copy=True)
    out, _ = plugin._apply_causal_mtm_decomposition(
        tail, 32, 3, "test", scaler=scaler, fit_scaler=False)
    assert out.shape == (200, 3)
    assert np.array_equal(scaler.mean_, mean_before)
