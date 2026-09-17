#!/usr/bin/env python3
"""The sealed D3 temporal amendment (J1), machine-readable and bound by its own digest.

`docs/integracion_workplan_2026_09_10/07A_ENMIENDA_TEMPORAL_D3_2026_09_16.md` is the prose;
this is the same amendment as data. It cites the ORIGINAL design's bytes by digest and never
edits them, names the nine operators and their twins exactly as the original §2 lists them,
and fixes the required-test matrix that J2 says review-ready means: every mandatory test ran
and passed, a skip or an unidentified case is scoped, never silently accepted.

Sealing follows `df_d2_design.seal_design`: the digest is over the canonical body without the
digest field, and `validate_amendment` re-derives it. A candidate is measured against
`design_sha256` of THIS document, and every row it writes carries it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ORIGINAL = REPO / "docs" / "integracion_workplan_2026_09_10" / \
    "07_DISENO_D3_CUANTIZACION_TIEMPO_FRECUENCIA_DETECTORES_2026_09_14.md"
AMENDMENT_PROSE = REPO / "docs" / "integracion_workplan_2026_09_10" / \
    "07A_ENMIENDA_TEMPORAL_D3_2026_09_16.md"
#: The bytes the amendment supersedes, as they were when it was written.
ORIGINAL_SHA256 = "45959063e01895c6f9669a21f2784c38ffb60b6f7b4198129ca3f0fea22ffa86"

SCHEMA = "d3_temporal_amendment.v1"

#: Every test the battery must run for an operator to be review-ready. `twin` may be scoped
#: NOT_APPLICABLE by the operator's own declaration with a design reason; nothing else may.
REQUIRED_TESTS = (
    "prefix_all_available", "future_perturbation", "warm_up_edge", "fit_scope_train_only",
    "fresh_state_per_branch", "chunk_restart", "response_probe", "non_causal_twin",
    "availability_emission", "cost_pilot", "applicability", "raw_branch",
)
SCOPEABLE_TESTS = ("non_causal_twin",)

#: The nine operators of the original §2, their twins, and the probe that identifies each
#: one's response. `twin: None` means the twin is NOT_APPLICABLE by design, with the reason.
OPERATORS = (
    {"kind": "uniform_decile_quantizer", "group": "quantization_compression",
     "fit_scope": "TRAIN_PREFIX_ONLY", "twin": None,
     "twin_not_applicable_reason": "a stateless pointwise codec has no window to centre",
     "probe": "step"},
    {"kind": "sax_paa_trailing", "group": "quantization_compression",
     "fit_scope": "TRAIN_PREFIX_ONLY", "twin": "sax_paa_centred", "probe": "step"},
    {"kind": "delta_run_length", "group": "quantization_compression",
     "fit_scope": "NONE", "twin": None,
     "twin_not_applicable_reason": "delta of one sample and its run length have no window to "
                                   "centre; the only past consumed is x[t-1]",
     "probe": "impulse"},
    {"kind": "stft_trailing", "group": "time_frequency", "fit_scope": "NONE",
     "twin": "stft_centred", "probe": "impulse"},
    {"kind": "wavelet_trailing", "group": "time_frequency", "fit_scope": "NONE",
     "twin": "wavelet_centred", "probe": "impulse"},
    {"kind": "butterworth_causal", "group": "time_frequency", "fit_scope": "NONE",
     "twin": "butterworth_filtfilt", "probe": "impulse"},
    {"kind": "cusum_causal", "group": "detectors", "fit_scope": "TRAIN_PREFIX_ONLY",
     "twin": "cusum_lookahead", "probe": "level_shift"},
    {"kind": "mad_extremes_trailing", "group": "detectors", "fit_scope": "TRAIN_PREFIX_ONLY",
     "twin": "mad_extremes_centred", "probe": "impulse"},
    {"kind": "variance_regime_trailing", "group": "detectors", "fit_scope": "NONE",
     "twin": "variance_regime_centred", "probe": "variance_shift"},
)

#: The sealed cut menu (§4): predetermined, seeded, repeated at two lengths and two
#: missingness regimes. Nothing here is chosen after seeing a result.
CUT_MENU = {"powers_of_two": "2^j-1, 2^j, 2^j+1 for every j with 2^j <= n",
            "warm_up": "w-1, w, w+1", "window_multiples": "k*window for k in 1..3",
            "edges": "0, n-2, n-1", "random": {"count": 16, "seed": 20260916}}
LENGTHS = (512, 2048)
MISSINGNESS_REGIMES = ("none", "mcar", "blocks")
FUTURE_PERTURBATIONS = ("zeros", "large_constant", "other_seed_noise", "reversed", "nan_blocks",
                        "impulse_at_t_plus_1", "step", "chirp", "regime_change")

#: Readiness is three separate words, machine-readable (§8).
READINESS_STATES = ("INFRASTRUCTURE_PRESENT", "TEMPORAL_BATTERY_ACCEPTED", "SCIENTIFIC_UTILITY")


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha_obj(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def build_amendment() -> dict:
    doc = {
        "schema": SCHEMA,
        "design_id": "D3_TEMPORAL_AMENDMENT_V1_2026_09_16",
        "supersedes": {"file": str(ORIGINAL.relative_to(REPO)), "sha256": ORIGINAL_SHA256,
                       "sections_superseded": ["1", "3"],
                       "note": "original bytes preserved; this document prevails where they cross"},
        "prose": str(AMENDMENT_PROSE.relative_to(REPO)),
        "authorised_by": "docs/handoffs/MUSASHI_R6_ACCEPTANCE_AND_D3_J1_J3_2026_09_16.md#J1",
        "instants": ["event_index", "input_available_at", "output_emitted_at",
                     "response_delay"],
        "availability_rule": "output_emitted_at >= max(input_available_at over consumed inputs)"
                             " + declared emission delay; UNKNOWN stays UNKNOWN",
        "duration_semantics": {"parser": "pandas.Timedelta, as data-gov files_lake",
                               "to_samples": "exact division by the declared sampling period; "
                                             "a fractional offset is refused, never truncated",
                               "no_sampling_contract": "evaluate in seconds or undecided; holes "
                                                       "are never equally spaced observations"},
        "prefix_rule": "every output available at the cutoff: value, mask and emitted_at; "
                       "no exemption for lookback; delayed outputs compared after emission; "
                       "empty tested population is INSUFFICIENT_TEST",
        "cut_menu": CUT_MENU, "lengths": list(LENGTHS),
        "missingness_regimes": list(MISSINGNESS_REGIMES),
        "future_perturbations": list(FUTURE_PERTURBATIONS),
        "state_rules": {"fit_scope": "TRAIN_PREFIX_ONLY or NONE; never the evaluated future",
                        "fresh_state_per_branch": True,
                        "restart_tested_separately_from_fit": True},
        "twins": {"required_where_meaningful": True,
                  "not_applicable_needs_design_reason": True,
                  "absence_is_refusal": True},
        "delay_rule": "impulse onset is not group delay; each operator declares its probe; "
                      "an unidentified response is UNIDENTIFIED with a reason, never zero",
        "filter_support": {"wavelet": "(dec_len-1)*(2^L-1)+1 from the library, plus boundary "
                                      "mode", "recursive": "state dependency via zi, not "
                                                              "finite memory of order p"},
        "operators": list(OPERATORS),
        "required_tests": list(REQUIRED_TESTS), "scopeable_tests": list(SCOPEABLE_TESTS),
        "readiness_states": list(READINESS_STATES),
        "memory_ceiling_bytes": 2 * 1024 ** 3,
        "classification": "NON_GOVERNING",
        "unchanged": ["scientific hypotheses", "utility margins", "data splits",
                      "accepted D2 outcomes", "the nine operators of the original §2"],
        "design_sha256": "",
    }
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    doc["design_sha256"] = sha_obj(body)
    return doc


D3_AMENDMENT_V1 = build_amendment()


def build_probe_amendment() -> dict:
    """K2/K3 successor amendment: the probe is built from the training fit and the operator's
    declared resolution, three facts are recorded apart (identifiable excitation, first change
    observed, agreement with the declared onset), a twin without observable comparisons is
    INSUFFICIENT_TEST, and emission coverage is reported apart from causality. Sealed before
    any measurement with it; every row of a run under it carries this digest."""
    doc = {
        "schema": "d3_probe_amendment.v1",
        "supersedes_amendment": {"schema": D3_AMENDMENT_V1["schema"],
                                 "design_sha256": D3_AMENDMENT_V1["design_sha256"]},
        "original_design": dict(D3_AMENDMENT_V1["supersedes"]),
        "probe_construction": {
            "baseline": "train quantile 0.10 over finite values",
            "scale": "train quantile 0.90 minus the baseline; zero scale is UNIDENTIFIED",
            "noise": "baseline + 0.01 * scale * N(0,1), seeded, identical in both branches",
            "amplitude": "the operator's declared probe_resolution(state, baseline, scale, sigma): "
                         "the smallest excitation it guarantees moves the impact-sample output, "
                         "from its training fit only; UNIDENTIFIED with a reason otherwise",
            "never": ["validation or test data", "amplitudes searched until a pass",
                      "operator thresholds or parameters changed to pass"],
            "cases_defined": ["constant training fit", "extreme ranges", "saturation",
                              "known-domain quantizer at several scales",
                              "a deliberately delayed operator", "STFT declaring onset 1"]},
        "probe_facts": {
            "identifiable": "declared by the operator from its fit; false is UNIDENTIFIED",
            "first_change_observed": "the first available output after the impact whose value "
                                     "differs between the quiet and the excited branch",
            "matches_declared": "first change minus impact equals expected_onset_samples"},
        "probe_policy": {
            "declared_identifiable_but_nothing_moved": "FAILED (the declaration is contradicted)",
            "first_change_differs_from_declared": "FAILED (a real delay never becomes a pass)",
            "unidentified": "undecided; the verdict is INCONCLUSIVE, never ACCEPTED"},
        "twin_policy": {
            "detection": "a demonstrated causality failure of the twin",
            "no_observable_comparison": "INSUFFICIENT_TEST; absence of evidence is not evidence",
            "recorded": ["twin_emissions", "twin_comparisons"],
            "complete_data_control": "a centred twin over complete data must be detected"},
        "coverage": "available outputs over samples, reported apart from causality and from "
                    "inapplicability by missingness; no future interpolation; support unchanged",
        "spec_schema": "d3_operator_spec.v3",
        "unchanged": ["the twelve required tests", "the nine operators and their parameters",
                      "cut menu, lengths, missingness regimes", "NON_GOVERNING classification"],
        "design_sha256": "",
    }
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    doc["design_sha256"] = sha_obj(body)
    return doc


D3_PROBE_AMENDMENT_V1 = build_probe_amendment()


def build_twin_sensitivity_amendment() -> dict:
    """L2 successor amendment: a twin comparison is SENSITIVE only when the twin's declared
    right reach crosses the cut (prefix) or reaches a perturbed position (future perturbation);
    zero sensitive comparisons is INSUFFICIENT_TEST; an observed violation is always a
    detection, sensitive or not; sensitive comparisons that never move are the declaration's
    failure. The candidate's own causal tests are untouched. Sealed before any measurement."""
    doc = {
        "schema": "d3_twin_sensitivity_amendment.v1",
        "supersedes_amendment": {"schema": D3_PROBE_AMENDMENT_V1["schema"],
                                 "design_sha256": D3_PROBE_AMENDMENT_V1["design_sha256"]},
        "original_design": dict(D3_AMENDMENT_V1["supersedes"]),
        "twin_reach": "each twin declares reach_right: how many samples after i its output i "
                      "consumes (centred window: w - w//2 - 1; zero-phase filter: the whole "
                      "series; lookahead statistic: its declared lookahead)",
        "sensitive_comparison": {
            "prefix": "output i compared at cut c is sensitive iff i + reach_right > c",
            "future_perturbation": "output i at cut c is sensitive iff i + reach_right > c "
                                   "(the perturbation replaces every sample after c)",
            "necessary_not_sufficient": "geometric crossing is required; a zero coefficient, "
                                        "saturation or missing data can still leave no effect"},
        "policy": {
            "any_observed_violation": "PASSED (detection); never discarded by a support "
                                      "declaration of the same operator",
            "zero_sensitive_no_violation": "INSUFFICIENT_TEST (undecided; verdict INCONCLUSIVE)",
            "sensitive_without_violation": "FAILED (the twin declared non-causal shows no "
                                           "effect where it must: the declaration is wrong)"},
        "recorded": ["twin_emissions", "twin_comparisons", "twin_sensitive_comparisons",
                     "twin_detections", "twin_nearest_output_to_cut"],
        "candidate_tests": "prefix_all_available and future_perturbation of the candidate are "
                           "NOT restricted to the twin's sensitivity mask",
        "controls": ["known centre", "future impulse just after the cut", "extreme zero weights",
                     "missingness", "edges", "restart", "deliberately non-causal controls "
                     "failing by value, by availability mask and by emission time"],
        "unchanged": ["support 50", "imputation: none", "thresholds and parameters",
                      "the complete-data control", "the three warm-up refusals",
                      "the twelve required tests", "NON_GOVERNING"],
        "replay_scope": "non_causal_twin for the operators that declare a twin, over the whole "
                        "frozen population; the other eleven tests inherit the successor run's "
                        "rows by digest, verifiably",
        "design_sha256": "",
    }
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    doc["design_sha256"] = sha_obj(body)
    return doc


D3_TWIN_SENSITIVITY_AMENDMENT_V1 = build_twin_sensitivity_amendment()
#: The amendment a run measures under today; rows carry its digest.
D3_DESIGN_CURRENT = D3_TWIN_SENSITIVITY_AMENDMENT_V1


def validate_probe_amendment(doc: dict) -> list:
    """The successor amendment is sealed and names the amendment it supersedes."""
    problems = []
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    if sha_obj(body) != doc.get("design_sha256"):
        problems.append("design_sha256 does not seal the document")
    if doc.get("supersedes_amendment", {}).get("design_sha256") != D3_AMENDMENT_V1["design_sha256"]:
        problems.append("the successor does not name the amendment it supersedes")
    problems += validate_amendment(D3_AMENDMENT_V1)
    return problems


def validate_twin_sensitivity_amendment(doc: dict) -> list:
    problems = []
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    if sha_obj(body) != doc.get("design_sha256"):
        problems.append("design_sha256 does not seal the document")
    if doc.get("supersedes_amendment", {}).get("design_sha256") != D3_PROBE_AMENDMENT_V1["design_sha256"]:
        problems.append("the successor does not name the amendment it supersedes")
    problems += validate_probe_amendment(D3_PROBE_AMENDMENT_V1)
    return problems


def validate_amendment(doc: dict) -> list:
    problems = []
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return ["not a d3_temporal_amendment.v1"]
    body = {k: v for k, v in doc.items() if k != "design_sha256"}
    if sha_obj(body) != doc.get("design_sha256"):
        problems.append("design digest does not re-derive")
    if doc.get("supersedes", {}).get("sha256") != ORIGINAL_SHA256:
        problems.append("the amendment does not cite the original design digest")
    if ORIGINAL.is_file():
        actual = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
        if actual != ORIGINAL_SHA256:
            problems.append(f"the original design bytes changed: {actual[:16]}... — the "
                            "amendment supersedes a document that no longer exists")
    kinds = [op["kind"] for op in doc.get("operators", [])]
    if len(kinds) != 9 or len(set(kinds)) != 9:
        problems.append("the amendment must name exactly the nine operators of the original §2")
    for op in doc.get("operators", []):
        if op.get("twin") is None and not op.get("twin_not_applicable_reason"):
            problems.append(f"{op.get('kind')}: a twin is NOT_APPLICABLE only with a reason")
    if set(doc.get("scopeable_tests", [])) - set(doc.get("required_tests", [])):
        problems.append("a scopeable test must be one of the required tests")
    return problems


def operator_entry(kind: str) -> dict:
    for op in OPERATORS:
        if op["kind"] == kind:
            return op
    raise KeyError(kind)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write the sealed amendment here")
    args = parser.parse_args(argv)
    problems = validate_amendment(D3_AMENDMENT_V1)
    if args.out:
        args.out.write_text(json.dumps(D3_AMENDMENT_V1, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"design_sha256": D3_AMENDMENT_V1["design_sha256"],
                      "supersedes_sha256": ORIGINAL_SHA256, "problems": problems}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
