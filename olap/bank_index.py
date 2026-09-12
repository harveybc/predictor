"""Common index of the three evidence banks — joined, never
merged.

The work plan builds its universe from three banks with three
different kinds of authority:

  * the **public forecasting bank** (T2) can support a claim that
    a method generalizes outside the domain it was invented in;
  * the **synthetic known-mechanism bank** (E0/T1/M4) can
    calibrate a diagnostic against a truth that is known by
    construction, and can support nothing else;
  * the **financial domain bank** (`financial-data`, via the
    incremental census) is where a method is revalidated for a
    domain of use — never where it earns a general claim.

A common index makes the three addressable from one place. It
must not make them interchangeable, so authority is an attribute
of the BANK, carried into every row, and `assert_no_promotion`
refuses any claim that would let a join launder synthetic or
financial evidence into public eligibility.

Nothing here re-downloads, re-scores or re-profiles: each bank is
bound by the digests its own producer published.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "crispdm.bank_index.v1"

# --------------------------------------------------------------
# authority classes
# --------------------------------------------------------------

PUBLIC = "PUBLIC_FORECASTING_EVIDENCE"
SYNTHETIC = "SYNTHETIC_KNOWN_MECHANISM_CALIBRATION_ONLY"
FINANCIAL = "FINANCIAL_DOMAIN_DEVELOPMENT_ONLY"

# The states of the eligibility contract (doc 01 §4) each bank may
# ever support. A state absent from a bank's list can never be
# reached with that bank's evidence, whatever the join says.
MAY_SUPPORT = {
    PUBLIC: (
        "DISCOVERED", "PROFILED_WITH_METADATA_GAPS",
        "MECHANICALLY_ADMISSIBLE", "PUBLICLY_EVALUATED",
        "PUBLICLY_ELIGIBLE", "REJECTED", "INCONCLUSIVE",
    ),
    SYNTHETIC: (
        "DISCOVERED", "MECHANICALLY_ADMISSIBLE",
        "LAB_CALIBRATED", "REJECTED", "INCONCLUSIVE",
    ),
    FINANCIAL: (
        "DISCOVERED", "PROFILED_WITH_METADATA_GAPS",
        "MECHANICALLY_ADMISSIBLE", "DOMAIN_REVALIDATED",
        "REJECTED", "INCONCLUSIVE", "UNAVAILABLE",
    ),
}

# States that only external review may grant, never a join.
EXTERNALLY_GRANTED = ("PUBLICLY_ELIGIBLE", "LIVE_ELIGIBLE")


class BankIndexRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def assert_no_promotion(bank_authority: str, claimed_state: str,
                        *, subject: str = "subject") -> None:
    """The one rule a join must never break.

    Raises unless `claimed_state` is a state the bank's authority
    can actually support. Being indexed beside public evidence is
    not evidence.
    """
    allowed = MAY_SUPPORT.get(bank_authority)
    if allowed is None:
        raise BankIndexRefusal(
            f"{subject}: unknown bank authority "
            f"{bank_authority!r}")
    if claimed_state not in allowed:
        raise BankIndexRefusal(
            f"{subject}: a {bank_authority} bank cannot support "
            f"the state {claimed_state!r} — a join never "
            "promotes evidence to an authority its bank does "
            "not have")


def _sha(doc: dict, key: str) -> str:
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------
# P2.1 public bank (T2) — reconciled, never re-acquired
# --------------------------------------------------------------

def public_bank(manifest_path: Path, census_path: Path,
                adjudication_path: Path | None = None) -> dict:
    manifest = _read(manifest_path)
    census = _read(census_path)
    pop = census.get("population", {})
    admissible = census.get("admissible_series_by_family", {})
    datasets = []
    for logical_id, m in sorted(
            manifest.get("datasets", {}).items()):
        p = pop.get(logical_id, {})
        datasets.append({
            "dataset_id": logical_id,
            "family": m.get("family", "UNKNOWN"),
            "source_url": m.get("final_url", "UNKNOWN"),
            "archival_record": m.get("archival_record",
                                     "UNKNOWN"),
            "license_id": m.get("license_id", "UNKNOWN"),
            "license_id_sha256": m.get("license_id_sha256",
                                       "UNAVAILABLE"),
            "retrieved_at_utc": m.get("retrieved_at_utc",
                                      "UNKNOWN"),
            "byte_size": m.get("byte_size"),
            "bytes_sha256": m.get("sha256", "UNAVAILABLE"),
            "admission": m.get("admission", "UNKNOWN"),
            "frequency_declared": p.get("frequency_declared",
                                        "UNKNOWN"),
            "seasonal_period": p.get("seasonal_period"),
            "series_parsed": p.get("series_parsed"),
            "series_admissible": p.get("series_admissible"),
            "series_excluded": p.get("series_excluded"),
            "exposure_status": (
                "SCREEN_ADJUDICATED"
                if adjudication_path else "NOT_YET_ADJUDICATED"),
        })
    bank = {
        "authority_class": PUBLIC,
        "may_support": list(MAY_SUPPORT[PUBLIC]),
        "producer": "T2 public forecasting bank",
        "binding": {
            "manifest_sha256": census.get("manifest_sha256",
                                          "UNAVAILABLE"),
            "census_schema": census.get("schema", "UNKNOWN"),
            "acquired_at_utc": manifest.get("acquired_at_utc",
                                            "UNKNOWN"),
            "bytes_downloaded_total": manifest.get(
                "bytes_downloaded_total"),
        },
        "datasets": datasets,
        "admissible_series_by_family": admissible,
        "series_admissible_total": sum(admissible.values()),
        "reacquisition": "NONE — this index reconciles the "
                         "manifest and census T2 already "
                         "produced; it never downloads or "
                         "scores",
        # C9: the admissible series are consumable rows, bound to
        # the panel that admitted them and the dataset digest
        # they came from.
        "series": [
            {"series_id": uid,
             "family": doc.get("family", "UNKNOWN"),
             "dataset_id": logical_id,
             "digest": doc.get("tsf_member_sha256",
                               doc.get("zip_sha256",
                                       "UNAVAILABLE")),
             "frequency": doc.get("frequency_declared",
                                  "UNKNOWN"),
             "seasonal_period": doc.get("seasonal_period")}
            for logical_id, doc in sorted(pop.items())
            for uid in doc.get("admissible_unit_ids", [])],
    }
    if adjudication_path:
        adj = _read(adjudication_path)
        screen = adj.get("screen_adjudication", {})
        bank["screen_adjudication"] = {
            "verdict": screen.get("verdict", "UNKNOWN"),
            "reason": screen.get("reason", "UNKNOWN"),
            "scope": screen.get("scope", "UNKNOWN"),
            "primary_estimand": screen.get(
                "primary_estimand_unweighted_mean_of_panel_"
                "effects"),
            "panels": screen.get("panels", {}),
            "record_sha256": adj.get("record_sha256",
                                     "UNAVAILABLE"),
            "authority_note": "the verdict governs the operator "
                              "it tested, on the panels it "
                              "names — nothing else",
        }
    return bank


# --------------------------------------------------------------
# P2.2 synthetic bank (E0/T1/M4) — CALIBRATION_ONLY
# --------------------------------------------------------------

def _parse_t1_unit_id(unit_id: str) -> dict:
    """`family__noise__snr__homogeneity__seed` — the generator
    parameters are IN the id by construction."""
    parts = unit_id.split("__")
    out = {"family": parts[0] if parts else "UNKNOWN"}
    if len(parts) > 1:
        out["noise_type"] = parts[1]
    if len(parts) > 2:
        out["snr_level"] = parts[2]
    if len(parts) > 3:
        out["homogeneity"] = parts[3]
    if len(parts) > 4:
        out["seed"] = parts[4]
    return out


def synthetic_bank(t1_inventory_path: Path | None,
                   m4_design_path: Path | None) -> dict:
    generators = []
    binding: dict[str, str] = {}
    if t1_inventory_path and Path(t1_inventory_path).is_file():
        inv = _read(t1_inventory_path)
        binding["t1_inventory_schema"] = inv.get("schema",
                                                 "UNKNOWN")
        binding["t1_units_total"] = inv.get("units_total")
        binding["t1_seeds"] = inv.get("seeds", [])
        binding["t1_predeclared_cells"] = inv.get(
            "predeclared_cells")
        units = inv.get("units", {})
        for uid in sorted(inv.get("unit_ids", [])):
            u = units.get(uid, {})
            generators.append({
                "generator_id": f"t1::{uid}",
                "producer": "T1 known-truth bank",
                "mechanism": _parse_t1_unit_id(uid),
                "reconstruction": {
                    "clean_signal_sha256": u.get(
                        "arrays", {}).get("clean_signal",
                                          "UNAVAILABLE"),
                    "additive_noise_sha256": u.get(
                        "arrays", {}).get("additive_noise",
                                          "UNAVAILABLE"),
                    "observed_signal_sha256": u.get(
                        "arrays", {}).get("observed_signal",
                                          "UNAVAILABLE"),
                    "unit_record_sha256": u.get(
                        "unit_json_sha256", "UNAVAILABLE"),
                },
                "role": "CALIBRATION_ONLY",
            })
    if m4_design_path and Path(m4_design_path).is_file():
        d = _read(m4_design_path)
        pop = d.get("candidate_population", {})
        fams = d.get("task_families", {})
        binding["m4_design_sha256"] = d.get("design_sha256",
                                            "UNAVAILABLE")
        binding["m4_seed_derivation"] = d.get("seeds", {}).get(
            "derivation", "UNKNOWN")
        binding["m4_seed_phrase"] = d.get("seeds", {}).get(
            "master_phrase", "UNKNOWN")
        for kind in ("boolean", "temporal"):
            for fam in fams.get(kind, []):
                generators.append({
                    "generator_id": f"m4::{kind}::{fam}",
                    "producer": "M4 residual-capacity bank",
                    "mechanism": {
                        "family": fam,
                        "task_kind": kind,
                        "noise_regimes": fams.get(
                            "noise_regimes", []),
                    },
                    "reconstruction": {
                        "seed_rule": binding["m4_seed_derivation"],
                        "design_sha256": binding[
                            "m4_design_sha256"],
                    },
                    "role": "CALIBRATION_ONLY",
                })
        binding["m4_population_note"] = pop.get(
            "note", "see sealed design")
    return {
        "authority_class": SYNTHETIC,
        "may_support": list(MAY_SUPPORT[SYNTHETIC]),
        "producer": "E0/T1/M4 synthetic known-mechanism banks",
        "binding": binding,
        "generators": generators,
        "generator_count": len(generators),
        "confirmatory_status": "CALIBRATION_ONLY — no synthetic "
                               "generator is part of any "
                               "confirmatory bank; a mechanism "
                               "known by construction calibrates "
                               "a diagnostic and confirms "
                               "nothing",
    }


# --------------------------------------------------------------
# P2.3 financial bank — from the P1 census
# --------------------------------------------------------------

def financial_bank(census_summary_path: Path,
                   census_receipt_path: Path,
                   local_inventory_path: Path | None = None,
                   census_document_path: Path | None = None
                   ) -> dict:
    s = _read(census_summary_path)
    r = _read(census_receipt_path)
    bank = {
        "authority_class": FINANCIAL,
        "may_support": list(MAY_SUPPORT[FINANCIAL]),
        "producer": "financial-data incremental census",
        "binding": {
            "census_sha256": s.get("census_sha256",
                                   "UNAVAILABLE"),
            "receipt_sha256": r.get("receipt_sha256",
                                    "UNAVAILABLE"),
            "censused_at": s.get("censused_at", "UNKNOWN"),
            "manifest_sha256": r.get("inputs", {}).get(
                "manifest_sha256", "UNAVAILABLE"),
        },
        "coverage": s.get("coverage", {}),
        "gap_counts": s.get("gap_counts", {}),
        "dictionary_coverage": s.get("dictionary_coverage", {}),
        "availability_contract": s.get("availability_contract",
                                       {}),
        "conflicts": s.get("conflicts", []),
        "exposed_views_note":
            "the already-exposed ETH H4 model-ready views are "
            "DEVELOPMENT material; they are never a substitute "
            "for the public bank and cannot carry a general "
            "claim",
    }
    if local_inventory_path and Path(
            local_inventory_path).is_file():
        inv = _read(local_inventory_path)
        bank["exposed_model_ready_views"] = [{
            "dataset_id": d.get("dataset_id"),
            "relative_path": d.get("relative_path"),
            "row_count": d.get("row_count"),
            "variable_count": d.get("variable_count"),
            "profile_status": d.get("profile_status"),
            "physical_sha256": d.get("physical_sha256",
                                     "UNAVAILABLE"),
            "authority": FINANCIAL,
            "state": "PROFILED_WITH_METADATA_GAPS",
        } for d in inv.get("datasets", [])]
        bank["binding"]["local_inventory_sha256"] = inv.get(
            "inventory_sha256", "UNAVAILABLE")
    # C9: the index must CARRY the variables, not merely count
    # them. A count inside a summary is not a consumable row, and
    # a selection universe cannot be built from a number.
    if census_document_path and Path(
            census_document_path).is_file():
        full = _read(census_document_path)
        bank["appearances"] = [{
            "appearance_id": a["appearance_id"],
            "entity": a["entity"],
            "source_class": a["source_class"],
            "frequency": a["frequency"],
            "period_start": a["period_start"],
            "period_end": a["period_end"],
            "physical_sha256": a["physical_sha256"],
            "digest_state": a["digest_state"],
        } for a in full.get("appearances", [])]
        bank["variables"] = [{
            "variable_id": v["variable_id"],
            "entity": v["entity"],
            "concept_name": v["concept_name"],
            "source_class": v["source_class"],
            "unit": v["unit"],
            "event_time": v["event_time"],
            "available_time": v["available_time"],
            "appearance_count": v["appearance_count"],
            "semantics_declared":
                v["semantics"] != "UNKNOWN",
        } for v in full.get("variables", [])]
        bank["binding"]["census_document_sha256"] = full.get(
            "census_sha256", "UNAVAILABLE")
    return bank


# --------------------------------------------------------------
# the common index
# --------------------------------------------------------------

def _recount(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(r["authority"], {})
        out[r["authority"]][r["kind"]] = \
            out[r["authority"]].get(r["kind"], 0) + 1
    return {a: dict(sorted(k.items()))
            for a, k in sorted(out.items())}


KNOWN_OPERATORS = (
    {"operator_id": "op.denoise.D",
     "family": "denoising",
     "digest": "UNAVAILABLE",
     "state": "PUBLICLY_EVALUATED_VERDICT_DOES_NOT_ADVANCE",
     "evidence": "T2 six-panel screen"},
    {"operator_id": "op.mtm.causal_decomposition",
     "family": "spectral",
     "digest": "UNAVAILABLE",
     "state": "DEVELOPMENT_ONLY",
     "evidence": "predictor phase 2.6, train-only scaler"},
    {"operator_id": "op.selector.acf_pacf_pre",
     "family": "selection",
     "digest": "UNAVAILABLE",
     "state": "LEGACY_NON_AUTHORITATIVE",
     "evidence": "preprocessor pre-selector, no split boundary"},
    {"operator_id": "op.selector.embedded_post",
     "family": "selection",
     "digest": "UNAVAILABLE",
     "state": "LEGACY_NON_AUTHORITATIVE",
     "evidence": "preprocessor post-selector, no split boundary"},
)


def build_index(indexed_at: str, public: dict | None,
                synthetic: dict | None,
                financial: dict | None) -> dict:
    banks = {}
    if public:
        banks["public_forecasting"] = public
    if synthetic:
        banks["synthetic_known_mechanism"] = synthetic
    if financial:
        # known transformation operators travel with the bank
        # whose evidence describes them, and carry that bank's
        # authority like every other row
        financial = {**financial,
                     "operators": list(KNOWN_OPERATORS)}
        banks["financial_domain"] = financial
    # Every row carries its bank's authority, so a consumer that
    # reads one row can never lose the provenance of its claim.
    rows = []
    for bank_key, bank in sorted(banks.items()):
        auth = bank["authority_class"]
        for d in bank.get("datasets", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "dataset",
                         "id": d["dataset_id"],
                         "family": d.get("family", "UNKNOWN")})
        for g in bank.get("generators", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "generator",
                         "id": g["generator_id"],
                         "family": g["mechanism"].get(
                             "family", "UNKNOWN")})
        for v in bank.get("exposed_model_ready_views", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "model_ready_view",
                         "id": v["dataset_id"],
                         "family": "financial"})
        # C9: appearances, conceptual variables, public series
        # and known operators are consumable ROWS, each carrying
        # its authority, source, identity and digest.
        for a in bank.get("appearances", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "physical_appearance",
                         "id": a["appearance_id"],
                         "family": a["entity"],
                         "digest": a["physical_sha256"],
                         "identity_state": a["digest_state"]})
        for v in bank.get("variables", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "variable",
                         "id": v["variable_id"],
                         "family": v["entity"],
                         "digest": "UNAVAILABLE",
                         "identity_state":
                             "CONCEPTUAL_IDENTITY"})
        for sr in bank.get("series", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "series",
                         "id": sr["series_id"],
                         "family": sr["family"],
                         "digest": sr.get("digest",
                                          "UNAVAILABLE"),
                         "identity_state":
                             "PUBLIC_BANK_SERIES"})
        for op in bank.get("operators", []):
            rows.append({"bank": bank_key, "authority": auth,
                         "kind": "operator",
                         "id": op["operator_id"],
                         "family": op.get("family",
                                          "UNKNOWN"),
                         "digest": op.get("digest",
                                          "UNAVAILABLE"),
                         "identity_state":
                             op.get("state", "DECLARED")})
    doc = {
        "schema": SCHEMA,
        "indexed_at": indexed_at,
        "banks": banks,
        "common_rows": rows,
        "row_count": len(rows),
        "cardinality_by_kind_and_authority": _recount(rows),
        "cardinality_rule": "every count here is derived FROM "
                            "the rows, so a summary and an "
                            "index can never disagree",
        "authority_rule": {
            "statement": "a join never promotes evidence; the "
                         "authority class of the bank that "
                         "produced a row travels with the row",
            "may_support": {k: list(v)
                            for k, v in MAY_SUPPORT.items()},
            "externally_granted_states": list(
                EXTERNALLY_GRANTED),
            "enforced_by": "olap.bank_index.assert_no_promotion",
        },
    }
    doc["index_sha256"] = _sha(doc, "index_sha256")
    return doc
