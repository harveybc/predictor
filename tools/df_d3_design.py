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
