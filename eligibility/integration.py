"""Consumer-side integration of the eligibility gate.

Every repository asks the gate the same way and stamps the same
answer into its run record, so a reader can always tell which
reviewed manifest governed a result — or that none did.

Two outcomes, no third:

  * a manifest is configured — every subject the run is about to
    use must pass `require_eligible` for the declared scope, or
    the run refuses BEFORE any window is built, any operator is
    materialized and any model is fitted;
  * no manifest is configured — the run proceeds and is stamped
    `LEGACY_NON_AUTHORITATIVE`. It still produces numbers; it can
    never be cited as gated evidence.

The second outcome exists because the program has 137 runnable
historical configurations that predate the gate. It is not a
default-allow: the absence of review is recorded in the effective
config and travels with the results.
"""
from __future__ import annotations

from pathlib import Path

from eligibility import gate

STATUS_GATED = "ELIGIBILITY_GATED"
STATUS_LEGACY = "LEGACY_NON_AUTHORITATIVE"

# config keys every consumer honours
KEY_MANIFEST = "eligibility_manifest"
KEY_SHA = "eligibility_manifest_sha256"
KEY_MAX_AGE = "eligibility_max_age_days"
KEY_SCOPE = "eligibility_scope"
KEY_STAMP = "eligibility_stamp"


def gate_subjects(config: dict, *, scope: str | None = None,
                  subject_ids: list[str] | None = None,
                  subject_kind: str = "variable",
                  consumer: str = "unknown") -> dict:
    """Enforce the gate for one run and return its stamp.

    Refuses (never returns) when a manifest is configured and any
    requested subject is not reviewed-eligible for the scope.
    """
    manifest_path = config.get(KEY_MANIFEST)
    scope = scope or config.get(KEY_SCOPE)
    if not manifest_path:
        stamp = {
            "eligibility_status": STATUS_LEGACY,
            "consumer": consumer,
            "scope": scope,
            "manifest": None,
            "reason": "no reviewed eligibility manifest was "
                      "configured for this run; results are "
                      "not gated evidence",
        }
        config[KEY_STAMP] = stamp
        return stamp
    if not scope:
        raise gate.EligibilityRefusal(
            f"{consumer}: an eligibility manifest is configured "
            f"but no scope was declared — eligibility is never "
            "global")
    manifest = gate.load_manifest(
        Path(manifest_path),
        expected_sha256=config.get(KEY_SHA),
        max_age_days=config.get(KEY_MAX_AGE))
    universe = gate.eligible_universe(
        manifest, scope=scope, subject_kind=subject_kind)
    used = sorted(subject_ids) if subject_ids else []
    for sid in used:
        gate.require_eligible(manifest, sid, scope=scope,
                              subject_kind=subject_kind)
    stamp = {
        "eligibility_status": STATUS_GATED,
        "consumer": consumer,
        "scope": scope,
        "manifest": str(manifest_path),
        "manifest_sha256": gate.manifest_fingerprint(manifest),
        "subject_kind": subject_kind,
        "universe_size": len(universe),
        "subjects_used": used,
        "subjects_used_count": len(used),
    }
    config[KEY_STAMP] = stamp
    return stamp


def gate_operator(config: dict, *, operator_id: str,
                  version: str, code_digest: str,
                  scope: str | None = None,
                  plugin_name: str | None = None,
                  consumer: str = "unknown") -> dict:
    """Enforce the gate for ONE operator before it materializes a
    transformation."""
    manifest_path = config.get(KEY_MANIFEST)
    scope = scope or config.get(KEY_SCOPE)
    if not manifest_path:
        return {"eligibility_status": STATUS_LEGACY,
                "consumer": consumer,
                "operator_id": operator_id,
                "reason": "no reviewed eligibility manifest was "
                          "configured; this transformation is "
                          "experimental, not licensed"}
    if not scope:
        raise gate.EligibilityRefusal(
            f"{consumer}: an eligibility manifest is configured "
            "but no scope was declared")
    manifest = gate.load_manifest(
        Path(manifest_path),
        expected_sha256=config.get(KEY_SHA),
        max_age_days=config.get(KEY_MAX_AGE))
    entry = gate.require_operator(
        manifest, operator_id=operator_id, version=version,
        code_digest=code_digest, scope=scope,
        plugin_name=plugin_name)
    return {"eligibility_status": STATUS_GATED,
            "consumer": consumer,
            "operator_id": operator_id,
            "version": entry["version"],
            "fit_scope": entry["fit_scope"],
            "manifest_sha256": gate.manifest_fingerprint(
                manifest),
            "scope": scope}


def describe(stamp: dict) -> str:
    """One line for a log or a report header."""
    if stamp.get("eligibility_status") == STATUS_GATED:
        return (f"eligibility: GATED by "
                f"{stamp['manifest_sha256'][:12]} "
                f"scope={stamp['scope']} "
                f"universe={stamp.get('universe_size', '?')}")
    return ("eligibility: LEGACY_NON_AUTHORITATIVE (no reviewed "
            "manifest configured)")
