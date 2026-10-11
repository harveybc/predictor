"""Exact plan-domain tests for the I6-N v2 search space (lane E).

Plan: af0c3570 I6N_SWARM_WEEKLY_OFFCHAIN_EXECUTION.md section 3. CPU only,
no TensorFlow, no data, no fits. Every value here is a contract, not a score.
"""

import json
import random
import sys

import pytest

from tools import i6n_search_space as space


H = (2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72)


def raw(h=12, **changes):
    g = space.seed_genotype(h)
    g = dict(g)
    g.update(changes)
    return g


def codes(exc_or_result):
    reasons = (exc_or_result.reasons if isinstance(exc_or_result, space.Infeasible)
               else exc_or_result["reasons"])
    return {r["code"] for r in reasons}


# --- horizons, history, input spacing, samples ---------------------------

def test_exactly_twelve_business_horizons():
    assert space.HORIZONS == H
    assert space.SHORT_HORIZONS == (2, 3, 4, 5, 6, 12)
    assert space.LONG_HORIZONS == (18, 24, 36, 48, 60, 72)


@pytest.mark.parametrize("h", H)
def test_history_is_max_24_m_h_deduplicated(h):
    expected = tuple(sorted({max(24, m * h) for m in (1, 2, 4, 6)}))
    assert space.history_domain(h) == expected
    assert min(space.history_domain(h)) >= 24
    assert len(set(space.history_domain(h))) == len(space.history_domain(h))


def test_history_examples():
    assert space.history_domain(2) == (24,)
    assert space.history_domain(5) == (24, 30)
    assert space.history_domain(12) == (24, 48, 72)
    assert space.history_domain(72) == (72, 144, 288, 432)


@pytest.mark.parametrize("h", H)
def test_delta_x_is_every_common_integer_divisor_including_one(h):
    for L in space.history_domain(h):
        dom = space.delta_x_domain(h, L)
        assert 1 in dom
        assert dom == tuple(d for d in range(1, h + 1) if h % d == 0 and L % d == 0)


@pytest.mark.parametrize("h", H)
def test_samples_include_t(h):
    for L in space.history_domain(h):
        for dx in space.delta_x_domain(h, L):
            g = space.canonicalize(raw(h, history_hours=L, delta_x_hours=dx,
                                       reducer_filters=[4], stride_pattern="none"))
            d = space.derived(g)
            assert d["n_samples"] == 1 + L // dx
            assert d["input_offsets_hours"][0] == -L
            assert d["input_offsets_hours"][-1] == 0  # t itself
            assert len(d["input_offsets_hours"]) == d["n_samples"]
            assert all(b - a == dx for a, b in zip(d["input_offsets_hours"],
                                                  d["input_offsets_hours"][1:]))


def test_max_samples_is_433_not_truncated_to_241():
    g = space.canonicalize(raw(72, history_hours=432, delta_x_hours=1, width=128,
                               reducer_filters=[128], stride_pattern="none"))
    assert space.derived(g)["n_samples"] == 433
    assert space.class_of(g) == "S3"


def test_history_not_in_domain_typed():
    with pytest.raises(space.Infeasible) as e:
        space.canonicalize(raw(12, history_hours=36))
    assert "HISTORY_NOT_IN_DOMAIN" in codes(e.value)


def test_delta_x_not_common_divisor_typed_not_repaired():
    result = space.check(raw(12, history_hours=24, delta_x_hours=5))
    assert result["status"] == "INFEASIBLE"
    assert "DELTA_X_NOT_COMMON_DIVISOR" in codes(result)
    # h=5 L=24: only dx=1 is common; dx=5 does not divide 24.
    assert "DELTA_X_NOT_COMMON_DIVISOR" in codes(space.check(raw(5, history_hours=24, delta_x_hours=5)))


def test_horizon_outside_domain_typed():
    assert "HORIZON_NOT_IN_DOMAIN" in codes(space.check(raw(12, horizon_hours=96)))
    assert "HORIZON_NOT_IN_DOMAIN" in codes(space.check(raw(12, horizon_hours=1)))


@pytest.mark.parametrize("bad", [True, 12.0, "12", None])
def test_integer_genes_reject_bool_float_str(bad):
    assert space.check(raw(12, delta_x_hours=bad))["status"] == "INFEASIBLE"


def test_schema_keys_exact():
    g = raw(12)
    g["extra"] = 1
    assert "SCHEMA_MISMATCH" in codes(space.check(g))
    g = raw(12)
    del g["adapter_width"]
    assert "SCHEMA_MISMATCH" in codes(space.check(g))
    g = raw(12, schema="i6n.genotype.v1")
    assert "SCHEMA_MISMATCH" in codes(space.check(g))


# --- input transform ----------------------------------------------------

def test_input_transform_domain():
    assert space.INPUT_TRANSFORMS == ("identity", "difference")
    assert space.check(raw(12, input_transform="difference"))["status"] == "FEASIBLE"
    assert "INPUT_TRANSFORM_NOT_IN_DOMAIN" in codes(space.check(raw(12, input_transform="log")))


# --- supervision: endpoint vs real trajectory ---------------------------

@pytest.mark.parametrize("h", H)
def test_trajectory_k_domain_is_integer_h_over_k(h):
    assert space.trajectory_k_domain(h) == tuple(k for k in (2, 3, 4) if h % k == 0)


def test_trajectory_examples():
    assert space.trajectory_k_domain(5) == ()
    assert space.trajectory_k_domain(2) == (2,)
    assert space.trajectory_k_domain(12) == (2, 3, 4)
    assert space.trajectory_k_domain(18) == (2, 3)


def test_trajectory_output_grid_dy_independent_of_dx():
    g = space.canonicalize(raw(12, history_hours=24, delta_x_hours=3,
                               supervision="trajectory", trajectory_k=2))
    d = space.derived(g)
    assert d["delta_y_hours"] == 6 and d["delta_x_hours"] == 3
    assert d["output_offsets_hours"] == [6, 12]
    assert d["n_heads"] == 2
    assert d["output_offsets_hours"][-1] == 12  # endpoint is always the last head
    g = space.canonicalize(raw(72, history_hours=72, delta_x_hours=36,
                               supervision="trajectory", trajectory_k=3))
    assert space.derived(g)["output_offsets_hours"] == [24, 48, 72]


def test_endpoint_alias_k1_is_null_not_another_fit():
    endpoint = space.canonicalize(raw(12))
    alias = space.canonicalize(raw(12, supervision="trajectory", trajectory_k=1))
    assert alias == endpoint
    assert alias["trajectory_k"] is None and alias["supervision"] == "endpoint"
    assert space.genotype_sha256(alias) == space.genotype_sha256(endpoint)
    assert space.derived(endpoint)["delta_y_hours"] is None
    assert space.derived(endpoint)["output_offsets_hours"] == [12]
    # endpoint with a stray k is an inactive gene: canonicalized to null.
    assert space.canonicalize(raw(12, trajectory_k=3)) == endpoint


def test_trajectory_k_outside_domain_typed():
    assert "TRAJECTORY_K_NOT_IN_DOMAIN" in codes(space.check(raw(12, supervision="trajectory", trajectory_k=6)))
    assert "TRAJECTORY_DY_NOT_INTEGER" in codes(space.check(raw(5, supervision="trajectory", trajectory_k=2)))
    assert "TRAJECTORY_K_NOT_IN_DOMAIN" in codes(space.check(raw(12, supervision="trajectory", trajectory_k=None)))
    assert "SUPERVISION_NOT_IN_DOMAIN" in codes(space.check(raw(12, supervision="quantile")))


# --- core entry, widths, transformer blocks -----------------------------

def test_core_entry_and_width_domains():
    assert space.ENTRANCES == ("conv", "transformer", "conv_transformer")
    assert space.ENTRANCE_WIDTHS == (8, 16, 32, 64, 128)
    assert "WIDTH_NOT_IN_DOMAIN" in codes(space.check(raw(12, width=24)))
    assert "ENTRANCE_NOT_IN_DOMAIN" in codes(space.check(raw(12, entrance="lstm")))


def test_transformer_blocks_zero_for_conv_one_or_two_for_transformer_routes():
    conv = space.canonicalize(raw(12, entrance="conv", transformer_blocks=2))
    assert conv["transformer_blocks"] == 0  # inactive gene canonicalized
    assert conv == space.canonicalize(raw(12))
    for route in ("transformer", "conv_transformer"):
        for blocks in (1, 2):
            assert space.canonicalize(raw(12, entrance=route, transformer_blocks=blocks))["transformer_blocks"] == blocks
        assert "TRANSFORMER_BLOCKS_NOT_IN_DOMAIN" in codes(
            space.check(raw(12, entrance=route, transformer_blocks=0)))
        assert "TRANSFORMER_BLOCKS_NOT_IN_DOMAIN" in codes(
            space.check(raw(12, entrance=route, transformer_blocks=3)))


# --- reducers and strides ---------------------------------------------------

def test_reducers_one_to_three_nonincreasing_in_domain():
    assert space.REDUCER_FILTERS == (4, 8, 16, 32, 64, 128)
    ok = space.canonicalize(raw(12, width=64, reducer_filters=[64, 16, 4]))
    assert ok["reducer_filters"] == [64, 16, 4]
    assert "REDUCER_COUNT_NOT_IN_DOMAIN" in codes(space.check(raw(12, reducer_filters=[])))
    assert "REDUCER_COUNT_NOT_IN_DOMAIN" in codes(space.check(raw(12, width=32, reducer_filters=[8, 8, 4, 4])))
    assert "REDUCER_FILTERS_INCREASING" in codes(space.check(raw(12, width=32, reducer_filters=[8, 16])))
    assert "REDUCER_FILTERS_INCREASING" in codes(space.check(raw(12, width=8, reducer_filters=[16])))
    assert "REDUCER_FILTER_NOT_IN_DOMAIN" in codes(space.check(raw(12, width=32, reducer_filters=[12])))


def test_stride_patterns_right_anchored():
    assert space.STRIDE_PATTERNS == ("none", "progressive", "final")
    assert space.stride_list("none", 3) == [1, 1, 1]
    assert space.stride_list("progressive", 3) == [2, 2, 2]
    assert space.stride_list("final", 3) == [1, 1, 2]
    assert space.stride_list("final", 1) == [2]


def test_single_reducer_progressive_equals_final_alias():
    a = space.canonicalize(raw(12, reducer_filters=[4], stride_pattern="progressive"))
    b = space.canonicalize(raw(12, reducer_filters=[4], stride_pattern="final"))
    assert a == b and a["stride_pattern"] == "final"


def test_right_anchored_steps_and_t_survives():
    g = space.canonicalize(raw(12, history_hours=24, delta_x_hours=2, width=16,
                               reducer_filters=[16, 8, 4], stride_pattern="progressive"))
    d = space.derived(g)
    # 13 -> 7 -> 4 -> 2, ceil division keeps the right-most (t) slot.
    assert d["steps_per_stage"] == [13, 7, 4, 2]
    assert d["final_steps"] == 2
    assert d["t_retained"] is True


def test_reduction_that_empties_temporal_core_is_typed():
    # h72 L72 dx72 -> 2 samples; a stride-2 reduction leaves one step.
    r = space.check(raw(72, history_hours=72, delta_x_hours=72, reducer_filters=[4],
                        stride_pattern="final"))
    assert r["status"] == "INFEASIBLE"
    assert "FINAL_STEPS_BELOW_MINIMUM" in codes(r)
    assert space.check(raw(72, history_hours=72, delta_x_hours=72, reducer_filters=[4],
                           stride_pattern="none"))["status"] == "FEASIBLE"


def test_stride_pattern_unknown_typed():
    assert "STRIDE_PATTERN_NOT_IN_DOMAIN" in codes(space.check(raw(12, stride_pattern="middle")))


# --- adapter ---------------------------------------------------------------

def test_adapter_domain_and_inactive_width():
    assert space.ADAPTERS == ("none", "causalconv", "lstm")
    assert space.ADAPTER_WIDTHS == (8, 16, 32, 64)
    none = space.canonicalize(raw(12, adapter="none", adapter_width=64))
    assert none["adapter_width"] is None and none == space.canonicalize(raw(12))
    lstm = space.canonicalize(raw(12, adapter="lstm", adapter_width=16))
    assert space.model_core(lstm)["adapter"] == "lstm"
    assert "ADAPTER_WIDTH_NOT_IN_DOMAIN" in codes(space.check(raw(12, adapter="lstm", adapter_width=128)))
    assert "ADAPTER_WIDTH_NOT_IN_DOMAIN" in codes(space.check(raw(12, adapter="causalconv", adapter_width=None)))
    assert "ADAPTER_NOT_IN_DOMAIN" in codes(space.check(raw(12, adapter="gru", adapter_width=8)))


# --- multiple typed reasons; never repaired --------------------------------

def test_all_reasons_reported_and_input_never_mutated():
    bad = raw(12, history_hours=36, width=24, adapter="lstm", adapter_width=128)
    before = json.dumps(bad, sort_keys=True)
    r = space.check(bad)
    assert {"HISTORY_NOT_IN_DOMAIN", "WIDTH_NOT_IN_DOMAIN", "ADAPTER_WIDTH_NOT_IN_DOMAIN"} <= codes(r)
    assert json.dumps(bad, sort_keys=True) == before
    assert "genotype" not in r


# --- identity ----------------------------------------------------------------

def test_identity_is_stable_and_covers_every_active_gene():
    base = space.canonicalize(raw(12, history_hours=48, delta_x_hours=3, entrance="conv_transformer",
                                  transformer_blocks=1, width=32, reducer_filters=[16, 8],
                                  stride_pattern="final", adapter="lstm", adapter_width=16,
                                  supervision="trajectory", trajectory_k=3,
                                  input_transform="difference"))
    ident = space.genotype_sha256(base)
    assert len(ident) == 64
    changes = dict(history_hours=72, delta_x_hours=4, entrance="transformer", transformer_blocks=2,
                   width=64, reducer_filters=[16, 4], stride_pattern="none", adapter="causalconv",
                   adapter_width=32, trajectory_k=2, input_transform="identity")
    for key, value in changes.items():
        other = space.canonicalize(dict(base, **{key: value}))
        assert space.genotype_sha256(other) != ident, key
    assert space.genotype_sha256(dict(base)) == ident


def test_active_and_inactive_genes_recorded_in_effective_config():
    cfg = space.effective_config(space.canonicalize(raw(12)))
    assert cfg["schema"] == "i6n.effective_config.v2"
    assert set(cfg["inactive_genes"]) == {"transformer_blocks", "trajectory_k", "adapter_width"}
    assert set(cfg["active_genes"]) | set(cfg["inactive_genes"]) == set(space.GENE_KEYS) - {"schema"}
    assert cfg["model_core"] == {
        "entrance": "conv", "width": 8, "transformer_blocks": 0, "reducer_filters": [4],
        "reducer_strides": [1], "adapter": "none", "adapter_width": None, "n_horizons": 1}
    json.dumps(cfg, allow_nan=False)
    traj = space.effective_config(space.canonicalize(raw(12, supervision="trajectory", trajectory_k=4,
                                                         entrance="transformer", transformer_blocks=1)))
    assert "trajectory_k" in traj["active_genes"] and "transformer_blocks" in traj["active_genes"]
    assert traj["model_core"]["n_horizons"] == 4


def test_model_core_accepted_by_real_model_validator():
    from tools import i6n_model
    rng = random.Random(1)
    for h in H:
        for _ in range(20):
            g = space.sample_genotype(h, rng, "S3")
            assert i6n_model.normalize_config(space.model_core(g)) == space.model_core(g)


def test_fit_id_is_b_formula_derived_from_e_identity():
    import hashlib
    g = space.canonicalize(raw(12))
    binding = dict(experiment="i6n-swarm-weekly-20261011", fold_id="2023Q1",
                   context_sha256="c" * 64, support_sha256="b" * 64)
    a = space.fit_id(g, binding)
    payload = dict(experiment=binding["experiment"], fold="2023Q1",
                   effective_config_sha256=space.effective_config_sha256(g),
                   context_sha256="c" * 64, support_sha256="b" * 64)
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert a == hashlib.sha256(canon.encode("utf-8")).hexdigest()
    assert a != space.fit_id(g, dict(binding, fold_id="2023Q2"))
    assert a != space.fit_id(g, dict(binding, context_sha256="d" * 64))
    assert a != space.fit_id(space.alternative_genotype(12), binding)
    with pytest.raises(ValueError):
        space.fit_id(g, {k: v for k, v in binding.items() if k != "support_sha256"})
    with pytest.raises(ValueError):
        space.fit_id(g, dict(binding, context_sha256="xyz"))


def test_effective_config_sha256_is_canonical_hash_of_evaluator_request_config():
    import hashlib
    from tools import i6n_search as v1
    g = space.canonicalize(raw(24))
    cfg = v1.effective_config(space.to_evaluator_genome(g))
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert space.effective_config_sha256(g) == hashlib.sha256(canon.encode("utf-8")).hexdigest()


def test_search_candidate_sha256_binds_plan_and_horizon():
    g = space.canonicalize(raw(12))
    a = space.search_candidate_sha256(g, "a" * 64)
    assert a != space.search_candidate_sha256(g, "b" * 64)
    assert len(a) == 64


# --- size classes --------------------------------------------------------------

def test_size_classes_staged():
    names = [c["class_id"] for c in space.SIZE_CLASSES]
    assert names == ["S1", "S2", "S3"]
    s1, s2, s3 = space.SIZE_CLASSES
    assert (s1["max_samples"], s1["max_width"], s1["min_delta_x_hours"]) == (25, 32, 2)
    assert (s2["max_samples"], s2["max_width"]) == (55, 64)
    assert (s3["max_samples"], s3["max_width"]) == (433, 128)
    assert s3["requires_admitted_cost_pilot"] and s2["requires_admitted_cost_pilot"]


def test_class_of_examples():
    assert space.class_of(space.seed_genotype(12)) == "S1"
    assert space.class_of(space.canonicalize(raw(12, width=64, reducer_filters=[4]))) == "S2"
    # hourly input is not part of the first downsampled class even if short.
    assert space.class_of(space.canonicalize(raw(2, history_hours=24, delta_x_hours=1))) == "S2"
    assert space.class_of(space.canonicalize(raw(12, adapter="lstm", adapter_width=64))) == "S2"
    assert space.class_of(space.canonicalize(raw(12, width=128, reducer_filters=[4]))) == "S3"


# --- seeds -----------------------------------------------------------------------

@pytest.mark.parametrize("h", H)
def test_minimal_seed_conv8_conv4(h):
    g = space.seed_genotype(h)
    assert space.canonicalize(g) == g
    assert (g["entrance"], g["width"], g["reducer_filters"], g["stride_pattern"]) == ("conv", 8, [4], "none")
    assert g["adapter"] == "none" and g["transformer_blocks"] == 0
    assert g["supervision"] == "endpoint" and g["input_transform"] == "identity"
    assert space.class_of(g) == "S1"
    assert space.derived(g)["reducer_strides"] == [1]


def test_seed_windows_are_smallest_downsampled_history():
    assert {h: (space.seed_genotype(h)["history_hours"], space.seed_genotype(h)["delta_x_hours"]) for h in H} == {
        2: (24, 2), 3: (24, 3), 4: (24, 2), 5: (30, 5), 6: (24, 2), 12: (24, 2),
        18: (24, 2), 24: (24, 2), 36: (36, 2), 48: (48, 2), 60: (60, 3), 72: (72, 3)}


@pytest.mark.parametrize("h", H)
def test_alternative_is_small_and_not_equivalent(h):
    alt = space.alternative_genotype(h)
    assert space.genotype_sha256(alt) != space.genotype_sha256(space.seed_genotype(h))
    assert space.class_of(alt) == "S1"
    assert alt["entrance"] == "conv"  # Transformer not forced


# --- sampling and mutation ----------------------------------------------------------

@pytest.mark.parametrize("h", H)
def test_sampling_respects_class_ceiling_and_domain(h):
    rng = random.Random(f"t{h}")
    order = {"S1": 0, "S2": 1, "S3": 2}
    for ceiling in ("S1", "S2", "S3"):
        for _ in range(40):
            g = space.sample_genotype(h, rng, ceiling)
            assert space.canonicalize(g) == g
            assert order[space.class_of(g)] <= order[ceiling]


def test_sampling_reaches_larger_classes_and_keeps_small_genes_free():
    rng = random.Random(7)
    seen = {space.class_of(space.sample_genotype(72, rng, "S3")) for _ in range(300)}
    assert seen == {"S1", "S2", "S3"}
    samples = {space.derived(space.sample_genotype(12, rng, "S3"))["n_samples"] for _ in range(300)}
    assert min(samples) <= 13 and max(samples) > 55


def test_sampling_does_not_force_transformer():
    rng = random.Random(3)
    entrances = {space.sample_genotype(24, rng, "S1")["entrance"] for _ in range(200)}
    assert entrances == set(space.ENTRANCES)


def test_mutation_never_returns_parent_or_infeasible_and_types_filtered():
    rng = random.Random(11)
    parent = space.seed_genotype(12)
    filtered = []
    for _ in range(50):
        child = space.mutate(parent, rng, max_class="S1", filtered=filtered)
        assert child is not None
        assert space.canonicalize(child) == child
        assert space.genotype_sha256(child) != space.genotype_sha256(parent)
        assert space.class_of(child) == "S1"
    assert all(set(f) >= {"group", "reasons"} for f in filtered)


def test_mutation_groups_cover_all_gene_families():
    assert set(space.MUTATION_GROUPS) == {"window", "input_transform", "supervision", "family",
                                         "width", "reducers", "strides", "adapter"}
    rng = random.Random(5)
    parent = space.canonicalize(raw(12, width=32, reducer_filters=[16, 8]))
    for group in space.MUTATION_GROUPS:
        child = space.mutate(parent, rng, max_class="S3", group=group)
        assert child is not None, group


def test_mutation_exhausted_returns_none_not_duplicate():
    rng = random.Random(0)
    parent = space.seed_genotype(12)
    assert space.mutate(parent, rng, max_class="S1", group="input_transform",
                        exclude={space.genotype_sha256(space.canonicalize(dict(parent, input_transform="difference")))}) is None


def test_no_tensorflow_import():
    assert "tensorflow" not in sys.modules or True  # module import below must not pull TF
    import importlib
    import subprocess
    out = subprocess.run([sys.executable, "-c",
                          "import sys; import tools.i6n_search_space; print('tensorflow' in sys.modules)"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    importlib.import_module("tools.i6n_search_space")


# --- legacy v1 mapping (48 completed endpoint candidates) ----------------------------

LEGACY_H12_BEST = {"horizon": 12, "multiplier": 4, "delta_x": 3, "input_transform": "difference",
                   "output_mode": "endpoint", "delta_y": None,
                   "core": {"entrance": "transformer", "width": 16, "transformer_blocks": 1,
                            "reducer_filters": [16, 16, 8], "reducer_strides": [1, 2, 1],
                            "adapter": "causalconv", "adapter_width": 8, "n_horizons": 1}}
LEGACY_H72_BEST = {"horizon": 72, "multiplier": 1, "delta_x": 72, "input_transform": "identity",
                   "output_mode": "endpoint", "delta_y": None,
                   "core": {"entrance": "conv_transformer", "width": 16, "transformer_blocks": 2,
                            "reducer_filters": [16], "reducer_strides": [1],
                            "adapter": "causalconv", "adapter_width": 32, "n_horizons": 1}}


def test_legacy_identity_recomputed_with_frozen_v1_code():
    from tools import i6n_search as v1
    assert v1.effective_identity(LEGACY_H12_BEST) == "b17caa522cd45a5c4454657b80aa4d5da99880c3cec740f010eea08caeea9b6a"
    assert v1.effective_identity(LEGACY_H72_BEST) == "0d0d98b60032004eced32f36ea703eae73510f18eddba042dced8c49254fad34"


def test_legacy_mapping_types_out_of_domain_stride_pattern():
    rec = space.legacy_map(LEGACY_H12_BEST)
    assert rec["v1_identity"] == "b17caa522cd45a5c4454657b80aa4d5da99880c3cec740f010eea08caeea9b6a"
    assert rec["genotype_status"] == "OUT_OF_V2_DOMAIN"
    assert "STRIDE_PATTERN_NOT_IN_DOMAIN" in {r["code"] for r in rec["reasons"]}
    assert rec["genotype_canonical_sha256"] is None
    assert rec["fold_score_reuse"] == "NONE"


def test_legacy_mapping_in_domain_genotype_but_fold_context_non_equivalent():
    rec = space.legacy_map(LEGACY_H72_BEST)
    assert rec["genotype_status"] == "MAPPED"
    g = rec["v2_genotype"]
    assert (g["history_hours"], g["delta_x_hours"], g["stride_pattern"]) == (72, 72, "none")
    # In-domain legacy genotypes keep the SAME canonical identity (single source).
    assert rec["genotype_canonical_sha256"] == space.genotype_sha256(g) == rec["v1_identity"]
    assert rec["fold_score_reuse"] == "NONE"
    assert "EVALUATION_CONTEXT_NOT_A_2023_QUARTER_FOLD" in rec["fold_reuse_reasons"]
    assert rec["refit_of_legacy_context"] == "FORBIDDEN"


# --- lane A evaluator boundary (INTERFACE_A A.1 consumes v1 effective_config) -------

def test_every_v2_genotype_maps_to_a_valid_v1_evaluator_genome_exactly():
    from tools import i6n_search as v1
    rng = random.Random(42)
    for h in H:
        seen_v2, seen_v1 = set(), set()
        for _ in range(60):
            g = space.sample_genotype(h, rng, "S3")
            genome = space.to_evaluator_genome(g)
            assert v1.normalize_genome(genome) == genome  # already canonical v1
            cfg = v1.effective_config(genome)
            d = space.derived(g)
            assert cfg["history_hours"] == d["history_hours"]
            assert cfg["samples"] == d["n_samples"]
            assert cfg["final_steps"] == d["final_steps"]
            assert cfg["input_offsets"] == d["input_offsets_hours"]
            assert cfg["output_offsets"] == d["output_offsets_hours"]
            assert cfg["core"] == space.model_core(g)
            seen_v2.add(space.genotype_sha256(g))
            seen_v1.add(v1.effective_identity(genome))
        assert len(seen_v2) == len(seen_v1)  # injective on the sample


def test_effective_config_carries_evaluator_request_config_and_v1_identity():
    from tools import i6n_search as v1
    g = space.canonicalize(raw(12, supervision="trajectory", trajectory_k=2))
    cfg = space.effective_config(g)
    assert cfg["evaluator_config"] == v1.effective_config(space.to_evaluator_genome(g))
    # Single-sourced canonical identity == the v1-compatible evaluator identity (A, B, legacy).
    assert cfg["genotype_canonical_sha256"] == v1.effective_identity(space.to_evaluator_genome(g))
    assert space.genotype_sha256(g) == cfg["genotype_canonical_sha256"]
    assert cfg["evaluator_config"]["output_mode"] == "trajectory"
    assert cfg["evaluator_config"]["delta_y"] == 6


def test_from_evaluator_genome_roundtrip_and_typed_outside():
    g = space.canonicalize(raw(24, history_hours=96, delta_x_hours=4, width=32,
                               reducer_filters=[16, 8], stride_pattern="final"))
    assert space.canonicalize(space.from_evaluator_genome(space.to_evaluator_genome(g))) == g
