"""R1: the ONE verifier that decides whether a prior campaign root may donate terminals.

Both interfaces call :func:`verify_donor` with the same target descriptor and therefore give the
same verdict: preparation (``matched_runner._check_donor``) and actual adoption
(``worker._adoptable`` through ``campaign.donor_problem``).

What a scientific matched target requires of a donor root (each failure is a typed ``DONOR_*``
problem; any problem refuses the WHOLE root, nothing is adopted and nothing is fitted here):

1. The donor's sealed effective config verifies, same mode, same fit procedure fingerprint (entry
   points, fit device, engineering flag), scientific (not engineering, GPU), the same sealed arm
   designs, runtime pins, frozen population and retained input.
2. Every COMPLETE fit terminal of the donor plan verifies as stored (identity components, child
   digest, artifact bytes) and carries a trainer receipt (``RECEIPT`` artifact) that is
   ``COMPLETED``, ``scientific: true`` on ``gpu`` without overrides and without any fake marker,
   whose trainer identity equals the terminal's and binds: the expected population digest, the arm,
   the scientific design and effective config, the ACTUAL runtime equal to the sealed pin (TF from
   the receipt, Keras from the saved archive), the declared capacity record (recomputed digest),
   and the weights/archive hashes (archive bytes, MODEL.json weights digest). A runtime that differs
   from the donor's own seal is an INCOMPATIBLE donor; its seal is never rewritten.
3. Trust anchors outside the root's own self-description:
   - SOURCE: the custody-verified queue backup records the terminal as produced LOCALLY by a
     COMPLETE attempt with the same child-terminal digest (a terminal copied in, or adopted from
     elsewhere, is not an original source);
   - PUBLICATION: every report the fit published is DELIVERED with exact readback in the
     custody-verified outbox backup, and the fit's cost report binds the same identity and weights;
   - CUSTODY: every file used above (the fit terminals, the queue and outbox backups) is
     byte-identical in the donor's two physical mirrors, the second on another host.

See ``docs/i6n/direct_multi_output_static/matched_short/R1_DONOR_TRUST.md`` for the trust boundary
and what it does NOT protect against.
"""
import hashlib
import json
import shlex
import sqlite3
import subprocess
import zipfile
from pathlib import Path

CAPACITY_DOC = Path(__file__).resolve().parents[2] / \
    "docs/i6n/direct_multi_output_static/matched_short/CAPACITY_MATCHED.json"
MODEL_FAMILY = {"mg1": "short_mg1", "mg2m": "short_mg2"}
FAKE_KEYS = ("fake", "fixture")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def target_digest(target):
    return _canonical_sha(target)


# ----------------------------------------------------------------------------- targets
def _matched_arms(effective_arms):
    from tools.i6dmo import matched_runner as MR
    return {arm: dict(seal_sha256=entry["design"].get("seal_sha256"), design_sha256=entry["design_sha256"],
                      effective_config_sha256=(entry.get("adapter_effective") or {}).get("effective_config_sha256"),
                      runtime=MR.runtime_pin(entry["design"]))
            for arm, entry in sorted(effective_arms.items())}


def target_from_effective(effective):
    """The adoption interface's target: the campaign's own sealed effective config."""
    from tools.i6dmo import campaign as C
    target = dict(mode=C.mode_of(effective), fingerprint=C.procedure_fingerprint(effective),
                  scientific=C.mode_of(effective) == C.MODE_MATCHED and not effective.get("engineering_cpu_fixture"))
    if C.mode_of(effective) == C.MODE_MATCHED:
        pins = effective.get("data_pins") or {}
        target.update(arms=_matched_arms(effective["arms"]),
                      population_sha256=(pins.get("common_population") or {}).get("population_sha256"),
                      input_sha256=(pins.get("matched_input") or {}).get("sha256"))
    return target


def target_from_designs(design_docs, pins, entrypoints):
    """The preparation interface's target, built exactly as ``init_matched_contrast`` would seal it."""
    from tools.i6dmo import campaign as C
    from tools.i6dmo import design as D
    from tools.i6n_swarm.common import sha256_json
    arms = {}
    for arm, doc in sorted(design_docs.items()):
        effective = D.effective_config(D.validate_design(doc))
        arms[arm] = dict(design=doc, design_sha256=sha256_json(doc), adapter_effective=effective)
    pseudo = dict(mode=C.MODE_MATCHED, entrypoints=entrypoints, fit_device="gpu", engineering_cpu_fixture=False,
                  arms=arms, data_pins=pins)
    return target_from_effective(pseudo)


# ---------------------------------------------------------------------------- custody
def _custody(root, rels, run):
    """Byte identity of ``rels`` between the donor root and BOTH of its physical mirror targets."""
    problems, evidence = [], {}
    try:
        spec = json.loads((Path(root) / "private" / "deployment.json").read_text())
        mirrors = spec["mirrors"]
    except (OSError, ValueError, KeyError):
        return ["DONOR_CUSTODY_UNAVAILABLE:no_private_deployment_record"], evidence
    expected = {rel: _sha(Path(root) / rel) for rel in rels}
    alias = (spec["roles"].get(mirrors.get("second_host_role") or "", {}) or {}).get("ssh_alias")
    for name, dest, remote in (("local", mirrors.get("local"), None),
                               ("second_host", mirrors.get("second_host_path"), alias)):
        if not dest:
            problems.append(f"DONOR_CUSTODY_UNAVAILABLE:{name}")
            continue
        if remote:
            script = "cd " + shlex.quote(str(dest)) + " && sha256sum -- " + " ".join(shlex.quote(r) for r in sorted(rels))
            try:
                out = run(["ssh", "-o", "BatchMode=yes", remote, "--", script], capture_output=True, text=True,
                          timeout=300)
            except Exception as exc:
                problems.append(f"DONOR_CUSTODY_UNAVAILABLE:{name}:{type(exc).__name__}")
                continue
            seen = {}
            for line in (out.stdout or "").splitlines():
                sha, _, rel = line.partition("  ")
                seen[rel] = sha
        else:
            base = Path(dest)
            if not base.is_dir():
                problems.append(f"DONOR_CUSTODY_UNAVAILABLE:{name}")
                continue
            seen = {rel: _sha(base / rel) for rel in rels if (base / rel).is_file()}
        bad = sorted(rel for rel, sha in expected.items() if seen.get(rel) != sha)
        evidence[name] = dict(files=len(expected), mismatched=len(bad))
        if bad:
            problems.append(f"DONOR_CUSTODY_MISMATCH:{name}:{bad[0]}")
    return problems, evidence


# ------------------------------------------------------------------------- per fit
def _queue_backup(root):
    path = Path(root) / "backup" / "queue.sqlite3"
    if not path.is_file():
        return None
    # immutable: read the custody-verified backup without locks or WAL/SHM sidecars (never writes)
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        terminals = {r["unit_id"]: dict(r) for r in con.execute("SELECT * FROM terminals")}
        attempts = {r["attempt_id"]: dict(r) for r in con.execute("SELECT * FROM attempts")}
    finally:
        con.close()
    return terminals, attempts


def _outbox_backup(root):
    path = Path(root) / "backup" / "outbox.sqlite3"
    if not path.is_file():
        return None
    # immutable: read the custody-verified backup without locks or WAL/SHM sidecars (never writes)
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return {r["report_sha256"]: dict(r) for r in con.execute("SELECT * FROM items")}
    finally:
        con.close()


def _stored_status(folder):
    try:
        return (json.loads((folder / "TERMINAL.json").read_text()).get("terminal") or {}).get("status")
    except (OSError, ValueError):
        return None


def _has_fake_marker(*docs):
    return any(isinstance(d, dict) and any(d.get(k) for k in FAKE_KEYS) for d in docs)


def _keras_version(archive):
    try:
        with zipfile.ZipFile(archive) as z:
            return json.loads(z.read("metadata.json")).get("keras_version")
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None


def _verify_fit(root, plan_unit, target, donor_runtime, queue, outbox):
    from tools.i6dmo import worker as W
    from tools.i6dmo.outbox import _rows_sha, expected_rows, seal_report
    from tools.i6n_swarm.common import canonical
    unit_id, arm = plan_unit["unit_id"], plan_unit.get("arm")
    folder = Path(root) / "terminals" / plan_unit["identity_sha256"]
    try:
        stored = W._verify_stored(folder, plan_unit)
    except (OSError, ValueError, KeyError) as exc:
        return [f"DONOR_STORED_TERMINAL_INVALID:{unit_id}:{str(exc)[:80]}"]
    child = stored["terminal"]
    problems = []
    art = (child.get("artifacts") or {}).get("RECEIPT")
    receipt_path = folder / "out" / art["path"] if isinstance(art, dict) else None
    if receipt_path is None or not receipt_path.is_file():
        return [f"DONOR_FIT_RECEIPT_MISSING:{unit_id}"]
    receipt = json.loads(receipt_path.read_text())
    if _has_fake_marker(child, receipt):
        problems.append(f"DONOR_FAKE_MARKER:{unit_id}")
    if child.get("status") != "COMPLETE" or receipt.get("status") != "COMPLETED" or \
            receipt.get("scientific") is not True or receipt.get("device") != "gpu" or receipt.get("overrides"):
        problems.append(f"DONOR_FIT_NOT_SCIENTIFIC:{unit_id} scientific={receipt.get('scientific')!r} "
                        f"device={receipt.get('device')!r} overrides={bool(receipt.get('overrides'))}")
    identity = receipt.get("identity") or {}
    if child.get("trainer_identity") != identity:
        problems.append(f"DONOR_RECEIPT_TERMINAL_MISMATCH:{unit_id}")
    if identity.get("common_population_sha256") != target.get("population_sha256") or \
            (receipt.get("matched") or {}).get("common_population_sha256") not in (None, target.get("population_sha256")):
        problems.append(f"DONOR_POPULATION_MISMATCH:{unit_id}")
    if identity.get("matched_arm") != arm or (receipt.get("matched") or {}).get("arm") != arm:
        problems.append(f"DONOR_ARM_MISMATCH:{unit_id}")
    want = (target.get("arms") or {}).get(arm) or {}
    if identity.get("design_sha256") != want.get("design_sha256") or \
            identity.get("effective_config_sha256") != want.get("effective_config_sha256"):
        problems.append(f"DONOR_DESIGN_MISMATCH:{unit_id}")
    pin = donor_runtime.get(arm) or {}
    train = folder / "out" / "train"
    actual = dict(tensorflow=(receipt.get("resources") or {}).get("tensorflow_version"),
                  keras=_keras_version(train / "model" / "forecast_model.keras"))
    if actual != pin or pin != want.get("runtime"):
        problems.append(f"DONOR_RUNTIME_MISMATCH:{unit_id} declared={pin} actual={actual} "
                        f"target={want.get('runtime')} (INCOMPATIBLE_DONOR; its seal is never rewritten)")
    capacity = (receipt.get("matched") or {}).get("capacity") or {}
    try:
        declared = json.loads(CAPACITY_DOC.read_text())["declared_capacity"][MODEL_FAMILY.get(arm, "")]
    except (OSError, ValueError, KeyError):
        declared = None
    body = {k: v for k, v in capacity.items() if k != "capacity_sha256"}
    if capacity.get("arm") != arm or declared is None or \
            any((capacity.get("counts") or {}).get(k) != v for k, v in declared.items()) or \
            capacity.get("capacity_sha256") != _canonical_sha(body) or \
            identity.get("capacity_sha256") != capacity.get("capacity_sha256"):
        problems.append(f"DONOR_CAPACITY_MISMATCH:{unit_id}")
    for rel, sha in (receipt.get("artifacts") or {}).items():
        if not (train / rel).is_file() or _sha(train / rel) != sha:
            problems.append(f"DONOR_ARTIFACT_BYTES:{unit_id}:{rel}")
    archive = (receipt.get("artifacts") or {}).get("model/forecast_model.keras")
    if not archive or identity.get("model_archive_sha256") != archive:
        problems.append(f"DONOR_WEIGHTS_MISMATCH:{unit_id}:archive")
    model_doc = train / "model" / "MODEL.json"
    if "model/MODEL.json" not in (receipt.get("artifacts") or {}) or not model_doc.is_file() or \
            json.loads(model_doc.read_text()).get("weights_sha256") != identity.get("weights_sha256"):
        problems.append(f"DONOR_WEIGHTS_MISMATCH:{unit_id}:MODEL.json")
    # SOURCE anchor: produced here, by a complete attempt, with exactly this child terminal
    if queue is None:
        problems.append(f"DONOR_SOURCE_TERMINAL_UNTRUSTED:{unit_id}:no_queue_backup")
    else:
        terminals, attempts = queue
        row = terminals.get(unit_id)
        attempt = attempts.get((row or {}).get("attempt_id"))
        why = ("no_terminal_row" if row is None else "not_local_source" if row.get("source") != "LOCAL"
               else "identity" if row.get("identity_sha256") != folder.name
               else "child_digest" if row.get("terminal_sha256") != stored.get("child_terminal_sha256")
               else "attempt" if not attempt or attempt.get("outcome") != "COMPLETE" else None)
        if why:
            problems.append(f"DONOR_SOURCE_TERMINAL_UNTRUSTED:{unit_id}:{why}")
    # PUBLICATION anchor: what this fit published was delivered with exact readback, and binds it
    reports = child.get("reports") or []
    if outbox is None or not reports:
        problems.append(f"DONOR_PUBLICATION_UNVERIFIED:{unit_id}:{'no_outbox_backup' if outbox is None else 'no_report'}")
    else:
        for report in reports:
            sealed = seal_report(report)
            item = outbox.get(sealed["report_sha256"])
            if item is None or item.get("state") != "DELIVERED" or item.get("body") != canonical(sealed) or \
                    item.get("readback_sha256") != _rows_sha(expected_rows(sealed)):
                problems.append(f"DONOR_PUBLICATION_UNVERIFIED:{unit_id}:{sealed['report_sha256'][:12]}")
            elif str(report.get("experiment_key", "")).endswith(":cost") and (
                    report.get("config_sha256") != plan_unit["identity_sha256"]
                    or (report.get("tags") or {}).get("weights_sha256") != identity.get("weights_sha256")):
                problems.append(f"DONOR_PUBLICATION_MISMATCH:{unit_id}")
    return problems


# --------------------------------------------------------------------------- verifier
def verify_donor(prior_root, target, *, run=subprocess.run):
    """``dict(verdict=ACCEPTED|REFUSED, problems=[...], evidence={...})`` for one donor root."""
    from tools.i6dmo import campaign as C
    prior_root = Path(prior_root)
    problems, evidence = [], dict(target_sha256=target_digest(target))

    def done():
        return dict(verdict="REFUSED" if problems else "ACCEPTED", problems=problems, evidence=evidence)
    try:
        donor = json.loads((prior_root / "EFFECTIVE_CONFIG.json").read_text())
    except (OSError, ValueError):
        problems.append("DONOR_EFFECTIVE_CONFIG_UNREADABLE")
        return done()
    try:
        C._check_seal(donor, "DONOR_EFFECTIVE_CONFIG")
    except C.Refused:
        problems.append("DONOR_EFFECTIVE_CONFIG_SEAL")
        return done()
    if C.mode_of(donor) != target["mode"]:
        problems.append(f"DONOR_MODE_DIFFERS:{C.mode_of(donor)}")
        return done()
    if C.procedure_fingerprint(donor) != target["fingerprint"]:
        problems.append("DONOR_FIT_PROCEDURE_DIFFERS (entry points, fit device or engineering mode)")
        return done()
    if not target.get("scientific"):
        return done()  # elapsed/market-time/engineering targets: mode and procedure only
    if donor.get("engineering_cpu_fixture") or donor.get("fit_device") != "gpu":
        problems.append("DONOR_NOT_SCIENTIFIC: engineering or CPU donor")
        return done()
    donor_arms = _matched_arms(donor.get("arms") or {})
    for arm, want in (target.get("arms") or {}).items():
        have = donor_arms.get(arm) or {}
        if have.get("seal_sha256") != want.get("seal_sha256") or have.get("design_sha256") != want.get("design_sha256"):
            problems.append(f"DONOR_DESIGN_DIFFERS:{arm}")
        if have.get("runtime") != want.get("runtime"):
            problems.append(f"DONOR_RUNTIME_PIN_DIFFERS:{arm}")
    pins = donor.get("data_pins") or {}
    if (pins.get("common_population") or {}).get("population_sha256") != target.get("population_sha256"):
        problems.append("DONOR_POPULATION_DIFFERS")
    if (pins.get("matched_input") or {}).get("sha256") != target.get("input_sha256"):
        problems.append("DONOR_INPUT_DIFFERS")
    if problems:
        return done()
    plan = json.loads((prior_root / "PLAN.json").read_text())
    queue, outbox = _queue_backup(prior_root), _outbox_backup(prior_root)
    runtime = {arm: v["runtime"] for arm, v in donor_arms.items()}
    custody_files = [rel for rel in ("backup/queue.sqlite3", "backup/outbox.sqlite3") if (prior_root / rel).is_file()]
    verified = []
    for unit in plan["units"]:
        if unit["kind"] != "fit":
            continue
        folder = prior_root / "terminals" / unit["identity_sha256"]
        recorded = queue is not None and unit["unit_id"] in queue[0]
        if not (folder / "TERMINAL.json").exists() and not recorded:
            continue  # this unit has nothing to donate
        if not recorded and _stored_status(folder) == "FAILED":
            continue  # a failed fit is never adopted (worker refuses FAILED terminals)
        unit_problems = _verify_fit(prior_root, unit, target, runtime, queue, outbox)
        problems.extend(unit_problems)
        if folder.is_dir():
            custody_files += [str(p.relative_to(prior_root)) for p in sorted(folder.rglob("*")) if p.is_file()]
        if not unit_problems:
            verified.append(unit["unit_id"])
    if not verified and not problems:
        problems.append("DONOR_NO_VERIFIED_FIT")
    custody_problems, evidence["custody"] = _custody(prior_root, custody_files, run)
    problems.extend(custody_problems)
    evidence.update(verified_fits=verified, custody_files=len(custody_files))
    return done()


def warehouse_readback(prior_root, unit_ids, client):
    """Read-only live confirmation: each fit report's rows in the warehouse equal the expected rows.

    ``client`` is the existing configured publisher's client (``HttpWarehouse``); its token comes
    from the process environment set by the configured EnvironmentFile and is never read here."""
    from tools.i6dmo.outbox import expected_rows, seal_report, _row_key
    plan = json.loads((Path(prior_root) / "PLAN.json").read_text())
    out = {}
    for unit in plan["units"]:
        if unit["unit_id"] not in unit_ids:
            continue
        stored = json.loads((Path(prior_root) / "terminals" / unit["identity_sha256"] / "TERMINAL.json").read_text())
        for report in stored["terminal"].get("reports") or []:
            sealed = seal_report(report)
            rows = sorted(_row_key(r) for r in client.query_metrics(sealed["report_sha256"]))
            out[f"{unit['unit_id']}:{report['experiment_key'].rsplit(':', 1)[-1]}"] = dict(
                report_sha256=sealed["report_sha256"], rows=len(rows), exact=rows == expected_rows(sealed))
    return out
