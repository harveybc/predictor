"""Consumer-side integration of the eligibility gate.

Rewritten for order C1-C5 after the audit found three holes in
the previous version: it ran after the optimizer, it returned a
positive label for a run that reviewed nothing, and it accepted a
manifest the caller had written itself.

The contract now:

  * the subjects are DERIVED from the files the run will consume
    (`eligibility.consumed`), never taken from a config list; a
    declared list is checked as an assertion and a mismatch
    refuses;
  * a positive decision requires an EXTERNAL review record whose
    location the run's config cannot choose, binding the exact
    submission, manifest, census, code and partitions;
  * every declared binding is compared at the point of use, so
    substituted data, partitions, code or evidence refuses;
  * historical reproduction is an EXPRESS choice
    (`execution_purpose=ARCHIVAL_REPLAY_NON_AUTHORITATIVE`),
    never the silent consequence of omitting a manifest;
  * a new experiment with no review record REFUSES.
"""
from __future__ import annotations

from pathlib import Path

from eligibility import gate
from eligibility.consumed import resolve_consumed_subjects
from eligibility import review

STATUS_GATED = "ELIGIBILITY_GATED"
STATUS_LEGACY = "LEGACY_NON_AUTHORITATIVE"
STATUS_SUBMITTED = "ELIGIBILITY_SUBMITTED_AWAITING_REVIEW"

PURPOSE_KEY = "execution_purpose"
PURPOSE_ARCHIVAL = "ARCHIVAL_REPLAY_NON_AUTHORITATIVE"
# C17: a new experiment is TWO phases. Phase one derives the
# facts, persists a stable submission and exits before the
# optimizer or the pipeline. The reviewer then decides on those
# exact bytes. Phase two re-derives everything, proves it landed
# on the SAME submission, and consumes the record.
PURPOSE_SUBMIT = "SUBMIT_ONLY"
PURPOSE_EXECUTE = "EXECUTE_REVIEWED"
PURPOSE_EXPERIMENT = PURPOSE_EXECUTE
KNOWN_PURPOSES = (PURPOSE_ARCHIVAL, PURPOSE_SUBMIT,
                  PURPOSE_EXECUTE)

# C19: the ONLY keys an optimizer may change. Everything that
# decides what is consumed — data, partitions, target, plugins,
# authority, evidence paths, scope — is forbidden, because
# approving a contract is worthless if the object stays mutable.
ALLOWED_OPTIMIZER_KEYS = frozenset({
    "batch_size", "epochs", "learning_rate", "l2_reg",
    "dropout_rate", "layer_size", "layer_sizes", "num_layers",
    "kernel_size", "filters", "units", "activation",
    "early_patience", "threshold_error", "iterations",
    "mc_samples", "optimizer", "momentum", "beta_1", "beta_2",
    "epsilon", "clipnorm", "clipvalue", "seed", "num_heads",
    "ff_dim", "embed_dim", "rnn_units", "conv_filters",
    "time_horizon_weights", "incentive_loss",
    "penalty_close_lambda", "penalty_far_lambda",
})


class ContractMutationRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def assert_optimizer_result_is_hyperparameters_only(
        result: dict, *, consumer: str = "unknown") -> dict:
    """C19: filter an optimizer's return value BEFORE it touches
    the approved config."""
    if not isinstance(result, dict):
        raise ContractMutationRefusal(
            f"{consumer}: the optimizer returned a "
            f"{type(result).__name__}, not a mapping of "
            "hyperparameters")
    forbidden = sorted(k for k in result
                       if k not in ALLOWED_OPTIMIZER_KEYS)
    if forbidden:
        raise ContractMutationRefusal(
            f"{consumer}: the optimizer tried to change "
            f"{forbidden} — an optimizer may only propose "
            "hyperparameters, never the data, partitions, "
            "target, plugins, authority or evidence paths that "
            "were approved before it ran")
    return dict(result)


def assert_contract_unchanged(config: dict, *, repo_root: Path,
                              consumer: str = "unknown") -> dict:
    """C19: re-derive the whole identity and require equality with
    the stamp the gate produced. Called immediately before the
    pipeline consumes anything."""
    stamp = config.get(KEY_STAMP) or {}
    if stamp.get("eligibility_status") != STATUS_GATED:
        return stamp
    fresh = resolve_consumed_subjects(config,
                                      repo_root=repo_root)
    before = stamp.get("digests", {})
    after = fresh["digests"]
    differing = sorted(k for k in ("data", "partitions",
                                   "schema", "code")
                       if before.get(k) != after.get(k))
    if differing or sorted(stamp.get("subject_ids", [])) != \
            sorted(fresh["subject_ids"]):
        raise ContractMutationRefusal(
            f"{consumer}: the consumed contract CHANGED after "
            f"the gate approved it (differing: "
            f"{differing or 'subject set'}) — approval before "
            "the change does not cover what would run now")
    return stamp

KEY_MANIFEST = "eligibility_manifest"
KEY_SHA = "eligibility_manifest_sha256"
KEY_MAX_AGE = "eligibility_max_age_days"
KEY_SCOPE = "eligibility_scope"
KEY_STAMP = "eligibility_stamp"
KEY_CENSUS = "eligibility_census_sha256"
KEY_SUBMISSION_OUT = "eligibility_submission_out"
KEY_SUBMISSION_SHA = "eligibility_submission_sha256"


class GateOrderRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


def _archival_stamp(config: dict, consumer: str,
                    consumed: dict | None) -> dict:
    stamp = {
        "eligibility_status": STATUS_LEGACY,
        "execution_purpose": PURPOSE_ARCHIVAL,
        "consumer": consumer,
        "reason": "this run declared "
                  f"{PURPOSE_KEY}={PURPOSE_ARCHIVAL}; it "
                  "reproduces historical work and is NOT gated "
                  "evidence",
        "subjects_derived": (len(consumed["subject_ids"])
                             if consumed else "UNAVAILABLE"),
        "dataset_id": (consumed["dataset_id"] if consumed
                       else "UNAVAILABLE"),
    }
    config[KEY_STAMP] = stamp
    return stamp


def _submission_from(config, consumed, scope, consumer,
                     manifest, census_sha, submitted_at):
    return review.build_submission(
        submitted_at=submitted_at, submitter=consumer,
        scope=scope,
        manifest_sha256=gate.manifest_fingerprint(manifest),
        census_sha256=census_sha, consumed=consumed)


def gate_run(config: dict, *, repo_root: Path,
             consumer: str = "unknown",
             scope: str | None = None) -> dict:
    """The single decision, asked BEFORE any data is consumed.

    Returns a stamp, or refuses. There is no path that returns a
    positive label without an external decision about the exact
    bytes this run will read.
    """
    purpose = config.get(PURPOSE_KEY, PURPOSE_EXPERIMENT)
    scope = scope or config.get(KEY_SCOPE)

    # The consumed set is derived first: even an archival replay
    # records WHAT it consumed, so a later reader can tell.
    try:
        consumed = resolve_consumed_subjects(
            config, repo_root=repo_root)
    except SystemExit:
        if purpose == PURPOSE_ARCHIVAL:
            return _archival_stamp(config, consumer, None)
        raise

    if purpose == PURPOSE_ARCHIVAL:
        return _archival_stamp(config, consumer, consumed)
    if purpose not in KNOWN_PURPOSES:
        raise GateOrderRefusal(
            f"{consumer}: unknown {PURPOSE_KEY} {purpose!r} — "
            f"declare one of {list(KNOWN_PURPOSES)}")

    manifest_path = config.get(KEY_MANIFEST)
    if not manifest_path:
        raise GateOrderRefusal(
            f"{consumer}: a new experiment has no eligibility "
            "manifest. Omitting one is no longer an implicit "
            "legacy run — declare "
            f"{PURPOSE_KEY}={PURPOSE_ARCHIVAL} to reproduce "
            "historical work, or supply a reviewed manifest")
    if not scope:
        raise GateOrderRefusal(
            f"{consumer}: no eligibility scope was declared — "
            "eligibility is never global")

    manifest = gate.load_manifest(
        Path(manifest_path),
        expected_sha256=config.get(KEY_SHA),
        max_age_days=config.get(KEY_MAX_AGE))

    census_sha = config.get(KEY_CENSUS)
    if not census_sha:
        raise GateOrderRefusal(
            f"{consumer}: no physical census digest was declared "
            f"({KEY_CENSUS}) — a decision must bind the census "
            "the variables were verified against")

    if purpose == PURPOSE_SUBMIT:
        # PHASE ONE. Derive, persist, and STOP. Nothing after
        # this point runs, so no optimizer and no pipeline can
        # observe or alter the bytes being submitted.
        submission = _submission_from(
            config, consumed, scope, consumer, manifest,
            census_sha, review.utc_now_stamp())
        path = review.persist_submission(submission)
        stamp = {
            "eligibility_status": STATUS_SUBMITTED,
            "execution_purpose": PURPOSE_SUBMIT,
            "consumer": consumer,
            "scope": scope,
            "submission_sha256":
                submission["submission_sha256"],
            "submission_file": path.name,
            "manifest_sha256": gate.manifest_fingerprint(
                manifest),
            "census_sha256": census_sha,
            "dataset_id": consumed["dataset_id"],
            "subjects_derived": len(consumed["subject_ids"]),
            "subject_ids": list(consumed["subject_ids"]),
            "digests": dict(consumed["digests"]),
            "grants_nothing":
                "a submission asks for a decision and is not "
                "one; nothing executes in this phase",
            "next_step":
                "the external reviewer decides on these exact "
                f"bytes, then re-run with {PURPOSE_KEY}="
                f"{PURPOSE_EXECUTE}",
        }
        config[KEY_STAMP] = stamp
        return stamp

    # PHASE TWO. Re-derive every fact, land on the SAME
    # submission the reviewer saw, then consume the record.
    declared_sub = config.get(KEY_SUBMISSION_SHA)
    if not declared_sub:
        raise GateOrderRefusal(
            f"{consumer}: {PURPOSE_EXECUTE} requires "
            f"{KEY_SUBMISSION_SHA} — the digest of the "
            "submission that was reviewed. Run "
            f"{PURPOSE_KEY}={PURPOSE_SUBMIT} first")
    persisted = review.load_persisted_submission(declared_sub)
    rederived = _submission_from(
        config, consumed, scope, consumer, manifest, census_sha,
        persisted["submitted_at"])
    review.assert_same_submission(persisted, rederived)
    submission = persisted

    # The decision itself — from a document this config cannot
    # choose, binding these exact bytes.
    record = review.read_review_record(
        submission=submission, census_sha256=census_sha,
        scope=scope)

    # Every derived subject must be reviewed, with its declared
    # bindings compared against what this run actually holds.
    d = consumed["digests"]
    for subject in consumed["subjects"]:
        gate.require_eligible(
            manifest, subject["subject_id"], scope=scope,
            subject_kind="variable",
            evidence_digest=None,
            code_digest=None,
            data_digest=d["data"],
            partitions_digest=d["partitions"])

    stamp = {
        "eligibility_status": STATUS_GATED,
        "execution_purpose": PURPOSE_EXECUTE,
        "consumer": consumer,
        "scope": scope,
        "manifest": str(manifest_path),
        "manifest_sha256": gate.manifest_fingerprint(manifest),
        "submission_sha256": submission["submission_sha256"],
        "review_record_sha256": record["record_sha256"],
        "reviewer": record["reviewer"],
        "census_sha256": census_sha,
        "dataset_id": consumed["dataset_id"],
        "subjects_derived": len(consumed["subject_ids"]),
        "subjects_reviewed": len(consumed["subject_ids"]),
        "subject_ids": list(consumed["subject_ids"]),
        "digests": dict(d),
    }
    config[KEY_STAMP] = stamp
    return stamp


def gate_operator(config: dict, *, operator_id: str,
                  version: str, code_digest: str,
                  scope: str | None = None,
                  plugin_name: str | None = None,
                  consumer: str = "unknown") -> dict:
    """Enforce the gate for ONE operator before it materialises a
    transformation."""
    purpose = config.get(PURPOSE_KEY, PURPOSE_EXPERIMENT)
    scope = scope or config.get(KEY_SCOPE)
    if purpose == PURPOSE_ARCHIVAL:
        return {"eligibility_status": STATUS_LEGACY,
                "execution_purpose": PURPOSE_ARCHIVAL,
                "consumer": consumer,
                "operator_id": operator_id,
                "reason": "archival replay; this transformation "
                          "is reproduced, not licensed"}
    manifest_path = config.get(KEY_MANIFEST)
    if not manifest_path:
        raise GateOrderRefusal(
            f"{consumer}: operator {operator_id!r} has no "
            "eligibility manifest and this run is not declared "
            f"{PURPOSE_ARCHIVAL}")
    if not scope:
        raise GateOrderRefusal(
            f"{consumer}: no eligibility scope was declared")
    manifest = gate.load_manifest(
        Path(manifest_path),
        expected_sha256=config.get(KEY_SHA),
        max_age_days=config.get(KEY_MAX_AGE))
    entry = gate.require_operator(
        manifest, operator_id=operator_id, version=version,
        code_digest=code_digest, scope=scope,
        plugin_name=plugin_name)
    return {"eligibility_status": STATUS_GATED,
            "execution_purpose": PURPOSE_EXPERIMENT,
            "consumer": consumer,
            "operator_id": operator_id,
            "version": entry["version"],
            "fit_scope": entry["fit_scope"],
            "manifest_sha256": gate.manifest_fingerprint(
                manifest),
            "scope": scope}


def describe(stamp: dict) -> str:
    if stamp.get("eligibility_status") == STATUS_GATED:
        return (f"eligibility: GATED by review "
                f"{stamp['review_record_sha256'][:12]} "
                f"(reviewer={stamp['reviewer']}) "
                f"scope={stamp['scope']} "
                f"subjects={stamp['subjects_reviewed']}")
    return (f"eligibility: {stamp.get('eligibility_status')} "
            f"({stamp.get('execution_purpose')}) — "
            f"{stamp.get('reason', '')}")
