"""Offline recovery of original R2 donor custody, not whole-campaign adoption.

No source mutation, mirror generation, SQLite writes, network, TensorFlow or fits.
Two explicitly supplied, locally readable mirror views are required. Physical
host separation and remote staging proof are parent-owned, not inferred here.
The original failed R2 remains failed; nothing here confers CLOSE on its source.
"""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from tools.i6dmo import campaign as C, donor_trust as DT, matched_runner as MR
from tools.i6dmo import outbox as O, r2_design as R2
from tools.i6n_swarm.common import canonical


SCHEMA = "i6dmo.r2_recovery.v1"
SEED_RULE = ("keras.utils.set_random_seed(6102026 + feature_index) immediately before building "
             "each autoencoder; applied seed recorded in each donor terminal")
PURGE_RULE = "INNER selection windows whose earliest slot instant precedes the INNER start are purged"
BASE_FILES = ("EFFECTIVE_CONFIG.json", "PLAN.json", "REPORTS.json", "PUBLICATION.json",
              "backup/queue.sqlite3", "backup/outbox.sqlite3")


class RecoveryError(ValueError):
    """Typed authentication refusal. No failure schedules replacement training."""

    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (":" + detail if detail else ""))


def _require(ok, code, detail=""):
    if not ok:
        raise RecoveryError(code, detail)


def _json(path):
    return json.loads(Path(path).read_text())


def _file(root, rel):
    path = root / rel
    _require(not Path(rel).is_absolute() and path.resolve().is_relative_to(root) and
             path.is_file() and not path.is_symlink(), "RECOVERY_FILE_UNAVAILABLE", str(rel))
    return path


def _same_bytes(left, right):
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            chunk = a.read(1 << 20)
            if chunk != b.read(1 << 20):
                return False
            if not chunk:
                return True


def _custody(root, mirrors, rels):
    evidence = {}
    for rel in sorted(rels):
        source = _file(root, rel)
        for index, mirror in enumerate(mirrors):
            try:
                other = _file(mirror, rel)
            except RecoveryError:
                raise RecoveryError("RECOVERY_CUSTODY_MISSING", f"mirror{index + 1}:{rel}") from None
            _require(not source.samefile(other) and _same_bytes(source, other),
                     "RECOVERY_CUSTODY_MISMATCH", f"mirror{index + 1}:{rel}")
        evidence[rel] = DT._sha(source)
    return evidence


def _database(root, rel, tables):
    path = _file(root, rel)
    _require(not any(Path(str(path) + suffix).exists() for suffix in ("-wal", "-journal")),
             "RECOVERY_BACKUP_NOT_IMMUTABLE", rel)
    con = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return {table: [dict(row) for row in con.execute("SELECT * FROM " + table)] for table in tables}
    finally:
        con.close()


def _indexed(rows, key):
    result = {r[key]: r for r in rows}
    _require(len(result) == len(rows), "RECOVERY_DUPLICATE_RECORD", key)
    return result


def _publication(root, items):
    published = _json(root / "REPORTS.json")
    receipt = _json(root / "PUBLICATION.json")
    _require(isinstance(published, list) and bool(published), "RECOVERY_PUBLICATION_EMPTY")
    index = {}
    for report in published:
        sealed = O.seal_report(report)
        digest = sealed["report_sha256"]
        _require(digest not in index, "RECOVERY_PUBLICATION_DUPLICATE")
        item = items.get(digest) or {}
        _require(item.get("state") == "DELIVERED" and item.get("body") == canonical(sealed) and
                 item.get("readback_sha256") == O._rows_sha(O.expected_rows(sealed)),
                 "RECOVERY_PUBLICATION_UNVERIFIED", digest)
        index[digest] = sealed
    _require(set(index) == set(items) and receipt.get("state") == "DELIVERED" and
             receipt.get("expected") == receipt.get("delivered") == len(index) and
             receipt.get("pending", 0) == receipt.get("mismatch", 0) == 0,
             "RECOVERY_PUBLICATION_INCOMPLETE")
    return index


def _stored(root, unit, queue, reports, items, rels, code, source_binding=None):
    uid, ident = unit["unit_id"], unit["identity_sha256"]
    prefix = "terminals/" + ident
    spath = _file(root, prefix + "/TERMINAL.json")
    cpath = _file(root, prefix + "/CHILD_TERMINAL.json")
    stored, child = _json(spath), _json(cpath)
    identity = stored.get("identity") or {}
    _require(stored.get("schema") == "i6dmo.stored_terminal.v1" and stored.get("unit_id") == uid and
             stored.get("identity_sha256") == ident == C.unit_identity(identity) and identity == unit["identity"]
             and identity.get("code_commit") == code and identity.get("unit") == uid and
             identity.get("kind") == unit["kind"], "RECOVERY_STORED_IDENTITY", uid)
    _require(DT._sha(cpath) == stored.get("child_terminal_sha256") and child == stored.get("terminal") and
             child.get("unit_id") == uid and child.get("identity_sha256") == ident,
             "RECOVERY_CHILD_TERMINAL", uid)
    terminal = queue["terminals"].get(uid) or {}
    attempt = queue["attempts"].get(terminal.get("attempt_id")) or {}
    qunit = queue["units"].get(uid) or {}
    _require(qunit.get("identity_sha256") == ident and qunit.get("kind") == unit["kind"] and
             json.loads(qunit.get("spec", "null")).get("identity") == identity,
             "RECOVERY_QUEUE_UNIT_IDENTITY", uid)
    failed_spec = None
    if child.get("status") == "FAILED":
        candidates = [a for a in queue["attempts"].values() if a.get("unit_id") == uid]
        _require(bool(candidates) and source_binding is not None, "RECOVERY_FAILED_ATTEMPT", uid)
        attempt = max(candidates, key=lambda a: (a.get("idx", 0), a["attempt_id"]))
        info = stored.get("attempt") or {}
        _require(qunit.get("state") == "FAILED" and attempt.get("outcome") == "FAILED" and
                 all(info.get(k) == attempt.get(v) for k, v in
                     (("index", "idx"), ("kind", "kind"), ("token", "token"), ("role", "role"))),
                 "RECOVERY_FAILED_ATTEMPT", uid)
        spec_path = _file(root, attempt["attempt_dir"] + "/UNIT_SPEC.json")
        spec = _json(spec_path)
        _require(spec.get("unit_id") == uid and spec.get("kind") == unit["kind"] and
                 spec.get("identity_sha256") == ident and spec.get("identity") == identity and
                 spec.get("campaign_id") == source_binding["campaign_id"] and
                 spec.get("effective_config_sha256") == source_binding["effective_config_sha256"] and
                 spec.get("test_open") is False and
                 (spec.get("effective_config") or {}).get("code_commit") == code,
                 "RECOVERY_FAILED_UNIT_SPEC", uid)
        # Child arm views inherit the root seal; they are not independently sealed.
        _require(spec["effective_config"] == source_binding["effective_view"],
                 "RECOVERY_FAILED_UNIT_SPEC_VIEW", uid)
        failed_spec = dict(path=str(spec_path), sha256=DT._sha(spec_path))
    if child.get("status") != "FAILED" or terminal:
        _require(terminal.get("source") == "LOCAL" and terminal.get("identity_sha256") == ident and
             terminal.get("terminal_sha256") == stored["child_terminal_sha256"] and
             json.loads(terminal.get("terminal", "null")) == child and attempt.get("unit_id") == uid and
             attempt.get("outcome") == child.get("status"),
             "RECOVERY_QUEUE_SOURCE", uid)
    rels.update((prefix + "/TERMINAL.json", prefix + "/CHILD_TERMINAL.json"))
    paths = {}
    for name, art in (child.get("artifacts") or {}).items():
        rel = prefix + "/out/" + art["path"]
        path = _file(root, rel)
        _require(path.resolve().is_relative_to((root / prefix / "out").resolve()) and
                 DT._sha(path) == art.get("sha256"), "RECOVERY_ARTIFACT_DIGEST", uid + ":" + name)
        paths[name] = path
        rels.add(str(path.relative_to(root)))
    _require(not DT._has_fake_marker(child), "RECOVERY_FAKE_MARKER", uid)
    for report in (child.get("reports") or []) if child.get("status") == "COMPLETE" else []:
        sealed = O.seal_report(report)
        digest = sealed["report_sha256"]
        _require(reports.get(digest) == sealed and items[digest].get("source_unit") == uid and
                 report.get("config_sha256") == ident, "RECOVERY_UNIT_PUBLICATION", uid)
    if child.get("status") == "COMPLETE":
        _require(bool(child.get("reports")), "RECOVERY_UNIT_PUBLICATION", uid)
    ref = dict(path=str(spath), sha256=DT._sha(spath), unit_id=uid, identity_sha256=ident,
               identity=copy.deepcopy(identity), child_terminal_sha256=stored["child_terminal_sha256"])
    if failed_spec is not None:
        ref.update(unit_spec=failed_spec, authentication="FAILED_PLAN_QUEUE_ATTEMPT_UNIT_SPEC")
    return child, paths, ref


def _tensor_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(canonical([list(value.shape), str(value.dtype)]).encode("ascii"))
    digest.update(value.tobytes())
    return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())


def _donor(unit, child, paths, model, objective, provenance, population, scaler):
    i, feature = unit["feature_index"], unit["feature"]
    donor = _json(paths["donor_terminal"])
    corpus = _json(paths["corpus"])
    _require(child.get("status") == "COMPLETE" and not child.get("tf_imported_before_flags") and
             not DT._has_fake_marker(donor), "RECOVERY_DONOR_NOT_COMPLETE", feature)
    wanted = dict(schema="i6dmo.r2_pretraining_terminal.v1", status="COMPLETE",
        campaign_id=provenance["campaign_id"], unit_id=unit["unit_id"], feature=feature, feature_index=i,
        applied_seed=6102026 + i, seed_rule=SEED_RULE, code_pin=provenance["code_pin"],
        ruling_seal_sha256=provenance["ruling_seal_sha256"], objective=objective)
    _require(all(donor.get(k) == v for k, v in wanted.items()), "RECOVERY_DONOR_PROVENANCE", feature)
    _require(corpus == donor.get("learned_corpus") and corpus.get("feature_index") == i and
             corpus.get("sha256") == DT._canonical_sha({k: v for k, v in corpus.items() if k != "sha256"}) and
             corpus.get("sha256") == child.get("learned_corpus_sha256") and corpus.get("purge_rule") == PURGE_RULE,
             "RECOVERY_DONOR_CORPUS", feature)
    _require((donor.get("fit") or {}).get("stop_reason") not in ("max_updates", "max_seconds") and
             child.get("applied_seed") == 6102026 + i and child.get("feature") == feature and
             child.get("feature_index") == i, "RECOVERY_DONOR_FIT", feature)
    _require(paths["donor.npz"] == paths["donor_terminal"].parent / "donor.npz" and
             DT._sha(paths["donor.npz"]) == (donor.get("artifacts") or {}).get("donor.npz"),
             "RECOVERY_DONOR_NPZ_DIGEST", feature)
    branch = model["branch"]
    slots, filters = model["slots"], branch["filters"]
    expected = dict(schema="i6dmo.branch_donor.v1", arm="mg2m", rule=model["market_time"]["rule"], slots=slots,
        spacing_open_market_hours=model["market_time"]["spacing_open_market_hours"], feature_name=feature,
        feature_index=i, feature_names_sha256=DT._canonical_sha(model["features"]),
        branch_model_name=f"branch_{i:02d}", conv_layer_name=f"branch_{i:02d}_conv",
        branch_params={k: branch[k] for k in ("filters", "kernel_size", "strides", "activation")},
        input_shape=[slots, 1], output_shape=[slots, filters], scaler_sha256=scaler,
        population_sha256=population, objective=objective)
    manifest = donor.get("manifest") or {}
    _require(all(manifest.get(k) == v for k, v in expected.items()), "RECOVERY_DONOR_IDENTITY", feature)
    with np.load(paths["donor.npz"], allow_pickle=False) as z:
        _require(set(z.files) == {"kernel", "bias"}, "RECOVERY_DONOR_NPZ_SCHEMA", feature)
        weights = [z["kernel"], z["bias"]]
        _require([w.shape for w in weights] == [(branch["kernel_size"], 1, filters), (filters,)] and
                 all(w.dtype == np.float32 and np.isfinite(w).all() for w in weights),
                 "RECOVERY_DONOR_WEIGHTS", feature)
        _require([_tensor_digest(w) for w in weights] == manifest.get("weights"), "RECOVERY_DONOR_WEIGHTS", feature)
    return donor, dict(feature_index=i, branch_model_name=manifest["branch_model_name"],
        conv_layer_name=manifest["conv_layer_name"], input_shape=manifest["input_shape"],
        output_shape=manifest["output_shape"], weights=[{k: w[k] for k in ("shape", "dtype")} for w in manifest["weights"]])


def build_manifest(source_root, *, mirror_roots, source_pins, r0_target):
    """Authenticate custody and return private original refs plus the exact R0 predicate.

    source_pins: effective_sha256, plan_sha256, code_commit (external frozen pins).
    r0_target: contract, code_root, expected_code_files, expected_scales_sha256,
    current_monitor_identity. Target code bytes are checked, not merely asserted.
    This API returns a document; it writes no manifest or source artifact.
    """
    try:
        return _build(source_root, mirror_roots, source_pins, r0_target)
    except RecoveryError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError, sqlite3.Error) as exc:
        raise RecoveryError("RECOVERY_INVALID", type(exc).__name__) from None


def _build(source_root, mirror_roots, source_pins, r0_target):
    root = Path(source_root).resolve()
    mirrors = [Path(p).resolve() for p in mirror_roots]
    _require(len(mirrors) == 2 and len({root, *mirrors}) == 3 and all(p.is_dir() for p in mirrors) and
             not mirrors[0].samefile(mirrors[1]) and all(not root.samefile(p) for p in mirrors),
             "RECOVERY_MIRROR_ROOTS")
    _require(set(source_pins) == {"effective_sha256", "plan_sha256", "code_commit"}, "RECOVERY_SOURCE_PINS")
    for rel, pin in (("EFFECTIVE_CONFIG.json", "effective_sha256"), ("PLAN.json", "plan_sha256")):
        _require(DT._sha(_file(root, rel)) == source_pins[pin], "RECOVERY_SOURCE_PIN", rel)
    initial = _custody(root, mirrors, BASE_FILES)
    effective, plan = C.load_effective(root), _json(root / "PLAN.json")
    _require(effective.get("mode") == plan.get("mode") == "R2_CONTRAST" and
             effective.get("code_commit") == source_pins["code_commit"] and effective.get("fit_device") == "gpu"
             and not effective.get("engineering_cpu_fixture") and not plan.get("test_units_enumerated") and
             not (effective["r2"].get("engineering_objective_overrides")) and
             plan.get("effective_config_file_sha256") == source_pins["effective_sha256"] and
             plan.get("effective_config_sha256") == C.sha256_json(effective), "RECOVERY_SOURCE_IDENTITY")
    qrows = _database(root, "backup/queue.sqlite3", ("terminals", "attempts", "units"))
    queue = {name: _indexed(rows, "attempt_id" if name == "attempts" else "unit_id") for name, rows in qrows.items()}
    items = _indexed(_database(root, "backup/outbox.sqlite3", ("items",))["items"], "report_sha256")
    published = _publication(root, items)
    units = _indexed(plan["units"], "unit_id")
    features = C.r2_features(effective)
    model = effective["arms"]["mg2m"]["adapter_effective"]["model"]
    pretraining = sorted((u for u in units.values() if u["kind"] == "pretrain"), key=lambda u: u["feature_index"])
    _require(len(features) == len(set(features)) == len(pretraining) == 19 and model["features"] == features and
             [u["feature"] for u in pretraining] == features and
             [u["feature_index"] for u in pretraining] == list(range(19)), "RECOVERY_FEATURE_ORDER")
    provenance = C.r2_provenance(effective, plan)
    population = effective["data_pins"]["common_population"]["population_sha256"]
    objective = effective["r2"]["objective"]
    _require(isinstance(objective, dict) and objective.get("identity_sha256") and objective.get("fit_settings"),
             "RECOVERY_OBJECTIVE_UNRULED")
    rels, refs, donors, branches, corpora, unit_specs = set(BASE_FILES), {}, {}, {}, {}, {}
    scaler = None
    for unit in pretraining:
        feature = unit["feature"]
        child, paths, ref = _stored(root, unit, queue, published, items, rels, source_pins["code_commit"])
        # The immutable source's first donor fixes its original scaler identity;
        # every other branch must agree, and train import still checks the target.
        first = _json(paths["donor_terminal"])
        scaler = scaler or first["manifest"]["scaler_sha256"]
        _require(bool(scaler), "RECOVERY_DONOR_SCALER")
        donor, branch = _donor(unit, child, paths, model, objective, provenance, population, scaler)
        terminal_ref = dict(path=str(paths["donor_terminal"]), sha256=DT._sha(paths["donor_terminal"]))
        ref.update(donor_terminal=terminal_ref,
                   donor_npz=dict(path=str(paths["donor.npz"]), sha256=DT._sha(paths["donor.npz"])))
        refs[feature], donors[feature], branches[feature] = ref, dict(terminal=terminal_ref), branch
        unit_specs[feature] = copy.deepcopy(unit)
        corpora[feature] = donor["learned_corpus"]["sha256"]
    provenance["corpus"] = corpora
    context = dict(arm="mg2m", rule=model["market_time"]["rule"], slots=model["slots"],
        spacing_open_market_hours=model["market_time"]["spacing_open_market_hours"], features=features,
        feature_names_sha256=DT._canonical_sha(features), branch_params={k:model["branch"][k] for k in
        ("filters", "kernel_size", "strides", "activation")}, scaler_sha256=scaler,
        population_sha256=population, objective=objective, branches=branches, provenance=provenance)
    failed_child, failed_paths, failed_ref = _stored(root, units["r2.fit"], queue, published, items, rels,
        source_pins["code_commit"], source_binding=dict(campaign_id=plan["campaign_id"],
            effective_config_sha256=plan["effective_config_sha256"],
            effective_view=C.arm_view(effective, units["r2.fit"].get("arm") or "mg2m")))
    _require(failed_child.get("status") == "FAILED" and not failed_child.get("training") and
             (failed_child.get("costs") or {}).get("fit_seconds") is None, "RECOVERY_R2_ALREADY_TRAINED")
    if "RECEIPT" in failed_paths:
        _require(not _json(failed_paths["RECEIPT"]).get("training"), "RECOVERY_R2_ALREADY_TRAINED")
    r0_child, r0_paths, r0_ref = _stored(root, units["r0.fit"], queue, published, items, rels, source_pins["code_commit"])
    receipt = _json(r0_paths["RECEIPT"])
    train_dir = r0_paths["RECEIPT"].parent
    for rel, sha in (receipt.get("artifacts") or {}).items():
        path = _file(root, str((train_dir / rel).relative_to(root)))
        _require(path.resolve().is_relative_to(train_dir.resolve()) and DT._sha(path) == sha,
                 "RECOVERY_R0_ARTIFACT_DIGEST", rel)
        rels.add(str(path.relative_to(root)))
    evidence = _custody(root, mirrors, rels)
    _require(all(evidence[k] == v for k, v in initial.items()), "RECOVERY_SOURCE_CHANGED")
    target = dict(population_sha256=population, arms=DT._matched_arms(effective["arms"]))
    runtime = {a: MR.runtime_pin(e["design"]) for a, e in effective["arms"].items()}
    problems = DT._verify_fit(root, dict(units["r0.fit"], arm="mg2m"), target, runtime,
                              (queue["terminals"], queue["attempts"]), items)
    _require(not problems, "RECOVERY_R0_SCIENTIFIC_REFUSED", ";".join(problems))
    code_root = Path(r0_target["code_root"]).resolve()
    expected_files = r0_target["expected_code_files"]
    _require(bool(expected_files), "RECOVERY_TARGET_CODE_MISSING")
    for rel, sha in expected_files.items():
        _require(DT._sha(_file(code_root, rel)) == sha, "RECOVERY_TARGET_CODE_BYTES", rel)
    training = _json(r0_paths["TRAINING.json"])
    decision = R2.r0_reuse_decision(receipt, training.get("monitor_evaluation"), r0_target["contract"],
        expected_code_files=expected_files, expected_scales_sha256=r0_target["expected_scales_sha256"],
        current_monitor_identity=r0_target["current_monitor_identity"], r1_verdict="ACCEPT")
    # This ACCEPT is the verified per-fit SOURCE/PUBLICATION/CUSTODY result above,
    # not a claimed ACCEPT from the whole-root verifier or a fabricated CLOSE.
    _require(_custody(root, mirrors, rels) == evidence, "RECOVERY_SOURCE_CHANGED")
    _require(DT._sha(failed_ref["unit_spec"]["path"]) == failed_ref["unit_spec"]["sha256"],
             "RECOVERY_FAILED_UNIT_SPEC_CHANGED")
    doc = dict(schema=SCHEMA, status="ADOPTABLE" if decision["decision"] == "REUSE_R0" else
        "DONORS_ADOPTABLE_R0_INCOMPATIBLE", source=dict(root=str(root), campaign_id=plan["campaign_id"],
        code_commit=effective["code_commit"], pins=dict(source_pins), close_required=False),
        custody=dict(mirror_roots=[str(p) for p in mirrors], files=evidence, publication_reports=len(published),
            verification="DIRECT_BYTE_COMPARISON_BOTH_MIRROR_VIEWS; IMMUTABLE_QUEUE_AND_OUTBOX_READBACK",
            parent_obligations="Bind remote staging proof and physical-host separation; local paths do not attest these",
            failed_unit_spec_scope="Source-only file, bound to custody-verified PLAN identity and failed queue attempt"),
        pretraining_refs=refs, unit_specs=unit_specs, donors=donors, donor_context=context,
        donor_provenance=copy.deepcopy(provenance), r0_reuse=decision,
        r0_fit=r0_ref if decision["decision"] == "REUSE_R0" else None, original_r0_fit=r0_ref,
        failed_r2=dict(reference=failed_ref, terminal=failed_child), test_read=False, training_repeated=False,
        qualification="Original O1 INNER-only monitor retained; no mean FIT/INNER relabelling; R0 report must be fresh")
    return dict(doc, manifest_sha256=DT._canonical_sha(doc))


def verify_manifest(manifest, *, source_root, mirror_roots, source_pins, r0_target):
    """Reauthenticate original bytes/custody and require exact manifest parity."""
    _require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA and
             manifest.get("manifest_sha256") == DT._canonical_sha({k:v for k,v in manifest.items()
                                                                  if k != "manifest_sha256"}), "RECOVERY_MANIFEST_SEAL")
    actual = build_manifest(source_root, mirror_roots=mirror_roots, source_pins=source_pins, r0_target=r0_target)
    _require(manifest == actual, "RECOVERY_MANIFEST_DIFFERS_FROM_CUSTODY")
    return actual


def load_manifest(path, *, expected_sha256, r0_target):
    """Read an externally file-pinned private manifest and reauthenticate custody.

    The target is explicit: a manifest's own assertions cannot approve new code
    files or turn an incompatible R0 into an adoptable one.
    """
    _require(DT._sha(Path(path)) == expected_sha256, "RECOVERY_MANIFEST_FILE_DIGEST")
    manifest = _json(path)
    return verify_manifest(manifest, source_root=manifest["source"]["root"],
        mirror_roots=manifest["custody"]["mirror_roots"], source_pins=manifest["source"]["pins"],
        r0_target=r0_target)
