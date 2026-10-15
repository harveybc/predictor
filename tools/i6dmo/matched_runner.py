"""Matched short contrast (NEXT_CONTRASTS_PLAN_20261015 section 2/4): MG1 (25 slots) vs MG2 (13 slots).

Finite graph inside the existing campaign machinery (no independent scheduler)::

    mg1.fit  -> mg1.report  \\
                             comparison -> publish/readback -> mirrors -> CLOSE
    mg2m.fit -> mg2m.report /

The queue, governed worker, stored-terminal verifier, outbox with exact readback, physical
mirrors and the automatic CLOSE are the ones every I6-DMO campaign uses (``tools.i6dmo.campaign``
mode ``MATCHED_SHORT_CONTRAST``). This module adds only:

- ``run_unit`` for the ``comparison`` kind (``unit_runner --kind comparison``): it authenticates
  both arm report terminals and their digest-pinned METRICS.json, requires that each report declares
  the campaign's frozen common-population digest and that each head was scored on identical rows in
  both arms, and applies the frozen decision rule. It never fits and never reads TEST.
- ``prepare`` / ``validate`` for the ``python -m tools.i6dmo matched-short`` entrypoint.

Decision rule (plan section 2): per arm, the equal-head mean over horizons 2,3,4,5,6,12 of the
annual pooled relative MAE ``(MAE_h - naive_h) / naive_h`` (negative is better) on the frozen
common DEVELOPMENT rows. The lower score wins; within 1e-8 the cheaper MG2 is retained. Any
missing arm or head -> INCOMPLETE_COMPARISON with the missing evidence (never a selection from
incomplete heads). A scientific negative is a completed result.

Child interface for the arm units (consumed by the model lane's trainer/report adapters): the
worker hands each arm unit (``spec["arm"]`` = ``mg1``|``mg2m``) an ``effective_config`` VIEW whose
``design`` is that arm's sealed EXPERIMENT document, and ``spec["pins"]`` with
``common_population`` (``path`` = the role's experiment storage root + relpath, ``sha256`` of
POPULATION.json, ``population_sha256`` = the frozen digest) and ``matched_input`` (``path`` of the
retained input directory, ``sha256`` of INPUT.npz). Consumers load with
``common_population.load_common_population(path, expected_digest=population_sha256,
input_dir=matched_input.path)`` and ``arm_tensors(pop, design, split)``. Reports must declare
``common_population_sha256`` (terminal ``lineage`` or METRICS.json) and per-head ``row_ids_sha256``.
The fit child is ``train --design --population --population-digest --input-dir --out --device``;
the worker claims an arm fit/report only once ``tools/i6dmo/train.py`` declares
``MATCHED_POPULATION_CLI = True`` / ``tools/i6dmo/reporting.py`` declares
``MATCHED_POPULATION_REPORT = True`` (typed WAIT_ARTIFACT naming the model lane otherwise).
"""
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

#: Arm ids are the data lane's (``design.matched_arm``): ``mg1`` control, ``mg2m`` matched reference.
ARMS = ("mg1", "mg2m")
ARM_SLOTS = {"mg1": 25, "mg2m": 13}
ARM_SPACING = {"mg1": 1, "mg2m": 2}
ARM_RESOURCE_CLASS = {"mg1": "short_mg1_matched", "mg2m": "short_mg2_matched"}
MATCHED_DIR = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static/matched_short"
HORIZONS = (2, 3, 4, 5, 6, 12)
TOLERANCE = 1e-8
TIE_KEEPS = "mg2m"
RULE = "EQUAL_HEAD_MEAN_ANNUAL_POOLED_RELATIVE_MAE_COMMON_DEVELOPMENT"
SEED = 6102026
DISPOSITIONS = ("MG1_SELECTED", "MG2_SELECTED", "MG2_RETAINED_WITHIN_TOLERANCE", "INCOMPLETE_COMPARISON")
GIB = 1024 ** 3
MIN_COORDINATOR_AVAILABLE = 4 * GIB
MIN_DISK = 20 * GIB
TEST_START = "2025-01-01"


class PrepareRefused(RuntimeError):
    """Typed refusal of the preparation step; the message starts with its cause token."""


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=1, allow_nan=False) + "\n")
    os.replace(tmp, path)


# ------------------------------------------------------------------------- decision rule
def decide(arm_scores, *, tolerance=TOLERANCE):
    """Pure decision. ``arm_scores``: {arm: {h: relative_error or None}} (missing arm = absent)."""
    missing = []
    for arm in ARMS:
        heads = arm_scores.get(arm)
        if heads is None:
            missing.append(f"ARM_MISSING:{arm}")
            continue
        for h in HORIZONS:
            value = heads.get(h)
            if value is None or not isinstance(value, (int, float)) or isinstance(value, bool) \
                    or not math.isfinite(value):
                missing.append(f"HEAD_MISSING:{arm}:h{h}")
    if missing:
        return dict(disposition="INCOMPLETE_COMPARISON", selected=None, scores=None, missing=missing,
                    rule=RULE, tolerance=tolerance)
    scores = {arm: sum(arm_scores[arm][h] for h in HORIZONS) / len(HORIZONS) for arm in ARMS}
    gap = scores["mg1"] - scores["mg2m"]
    if abs(gap) <= tolerance:
        disposition, selected = "MG2_RETAINED_WITHIN_TOLERANCE", TIE_KEEPS
    elif gap < 0:
        disposition, selected = "MG1_SELECTED", "mg1"
    else:
        disposition, selected = "MG2_SELECTED", "mg2m"
    return dict(disposition=disposition, selected=selected, scores=scores, score_gap_mg1_minus_mg2m=gap,
                missing=[], rule=RULE, tolerance=tolerance)


# --------------------------------------------------------------------- comparison hook
class _Refusal(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def _authenticated_report(unit_id, up, expected_population):
    """(per-head rows, record) from an authenticated report terminal, or raise a typed refusal."""
    path = Path(up["path"])
    if not path.is_file() or _sha(path) != up.get("sha256"):
        raise _Refusal(f"UPSTREAM_DIGEST_MISMATCH:{unit_id}")
    stored = json.loads(path.read_text())
    folder = path.parent
    if stored.get("schema") != "i6dmo.stored_terminal.v1" or stored.get("unit_id") != unit_id \
            or stored.get("identity_sha256") != folder.name:
        raise _Refusal(f"UPSTREAM_IDENTITY_MISMATCH:{unit_id}")
    child_path = folder / "CHILD_TERMINAL.json"
    if not child_path.is_file() or _sha(child_path) != stored.get("child_terminal_sha256") \
            or json.loads(child_path.read_text()) != stored.get("terminal"):
        raise _Refusal(f"UPSTREAM_CHILD_TERMINAL_MISMATCH:{unit_id}")
    child = stored["terminal"]
    if child.get("status") != "COMPLETE":
        raise _Refusal(f"UPSTREAM_NOT_COMPLETE:{unit_id}")
    art = (child.get("artifacts") or {}).get("metrics")
    if not isinstance(art, dict):
        raise _Refusal(f"UPSTREAM_ARTIFACT_MISSING:{unit_id}:metrics")
    metrics_path = folder / "out" / art["path"]
    if not metrics_path.is_file() or _sha(metrics_path) != art.get("sha256"):
        raise _Refusal(f"UPSTREAM_ARTIFACT_DIGEST:{unit_id}:metrics")
    metrics = json.loads(metrics_path.read_text())
    declared = (child.get("lineage") or {}).get("common_population_sha256") or metrics.get("common_population_sha256")
    if expected_population and declared != expected_population:
        raise _Refusal(f"COMMON_POPULATION_NOT_DECLARED_OR_DIFFERS:{unit_id}")
    if any(str(s).lower() == "test" for s in (metrics.get("splits_read") or [])):
        raise _Refusal(f"TEST_FEEDBACK:{unit_id}")
    rows = {}
    for h, row in (metrics.get("per_horizon") or {}).items():
        rows[int(h)] = row
    record = dict(identity_sha256=folder.name, stored_terminal_sha256=up["sha256"],
                  metrics_sha256=art["sha256"], common_population_sha256=declared)
    return rows, record


def _relative(row):
    mae, naive, n = row.get("mae"), row.get("naive_mae"), row.get("n")
    if not n or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                        for v in (mae, naive)) or naive <= 0:
        return None
    rel = (float(mae) - float(naive)) / float(naive)
    declared = row.get("relative_error")
    if declared is not None and abs(float(declared) - rel) > 1e-12 * max(1.0, abs(rel)):
        raise _Refusal("RELATIVE_ERROR_DECLARATION_DISAGREES")
    return rel


def run_unit(spec, out_dir, progress_path):
    """``unit_runner`` hook for the ``comparison`` unit; returns an ``i6dmo.terminal.v1`` dict."""
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def terminal(status, disposition, artifacts=None, gates=None, reports=None):
        return dict(schema="i6dmo.terminal.v1", unit_id=spec.get("unit_id"),
                    identity_sha256=spec.get("identity_sha256"), status=status, disposition=disposition,
                    artifacts=artifacts or {}, eligibility={}, gates=gates or {}, reports=reports or [],
                    costs=dict(wall_seconds=time.time() - started, fit_seconds=None,
                               score_seconds=time.time() - started, cgroup_peak_bytes=None, gpu_uuid=None))

    if spec.get("kind") != "comparison":
        return terminal("FAILED", "KIND_INVALID")
    if spec.get("test_open"):
        return terminal("FAILED", "TEST_OPEN_REFUSED")
    effective = spec.get("effective_config") or {}
    expected_population = ((effective.get("data_pins") or {}).get("common_population") or {}).get("population_sha256")
    arm_scores, records, rows_by_arm = {}, {}, {}
    try:
        for arm in ARMS:
            unit_id = f"{arm}.report"
            up = (spec.get("upstream") or {}).get(unit_id)
            if not up:
                continue  # technically failed/blocked arm: recorded as missing, never inferred
            rows, record = _authenticated_report(unit_id, up, expected_population)
            records[unit_id] = record
            rows_by_arm[arm] = rows
            arm_scores[arm] = {h: (_relative(rows[h]) if h in rows else None) for h in HORIZONS}
        common_rows = {}
        if all(a in rows_by_arm for a in ARMS):
            for h in HORIZONS:
                ids = [rows_by_arm[a].get(h, {}).get("row_ids_sha256") for a in ARMS]
                if all(ids) and ids[0] != ids[1]:
                    raise _Refusal(f"COMMON_ROWS_DIFFER:h{h}")
                ns = [rows_by_arm[a].get(h, {}).get("n") for a in ARMS]
                if all(n is not None for n in ns) and ns[0] != ns[1]:
                    raise _Refusal(f"COMMON_COUNTS_DIFFER:h{h}")
                common_rows[h] = dict(row_ids_sha256=ids[0], n=ns[0])
    except _Refusal as exc:
        return terminal("FAILED", exc.code)
    decision = decide(arm_scores)
    doc = dict(schema="i6dmo.matched_comparison.v1", decision=decision,
               per_head={arm: {str(h): v for h, v in heads.items()} for arm, heads in arm_scores.items()},
               per_head_rows={arm: {str(h): dict(n=r.get("n"), mae=r.get("mae"), naive_mae=r.get("naive_mae"))
                                    for h, r in rows.items()} for arm, rows in rows_by_arm.items()},
               common_rows={str(h): v for h, v in common_rows.items()}, inputs=records,
               common_population_sha256=expected_population, split="DEVELOPMENT", test_read=False,
               note="one seed establishes the recipe decision, not statistical superiority or a holdout")
    _atomic(out_dir / "COMPARISON.json", doc)
    artifacts = {"comparison": dict(path="COMPARISON.json", sha256=_sha(out_dir / "COMPARISON.json"))}
    metrics = []
    for arm, heads in arm_scores.items():
        for h, value in heads.items():
            if value is not None:
                metrics.append(dict(metric=f"{arm.upper()}_Relative_Error", value=value, split="validation",
                                    horizon=h, unit="ratio"))
    for arm, score in (decision["scores"] or {}).items():
        metrics.append(dict(metric=f"{arm.upper()}_Equal_Head_Mean_Relative_Error", value=score,
                            split="validation", horizon=None, unit="ratio"))
    metrics.append(dict(metric="Comparison_Complete", value=0 if decision["missing"] else 1, split="validation",
                        horizon=None, unit="bool"))
    report = dict(experiment_key=f"i6dmo:{spec['campaign_id']}:{spec['unit_id']}:decision",
                  experiment_set_key=f"i6dmo:{spec['campaign_id']}", actor="predictor", lake="LOCAL_RETAINED",
                  lineage="UNVERIFIED", config_sha256=spec["identity_sha256"], code_commit=effective.get("code_commit"),
                  project="predictor", phase="I6DMO",
                  tags={"unit": spec["unit_id"], "family": "comparison", "split": "development",
                        "test_read": "false", "disposition": decision["disposition"],
                        "selected": str(decision["selected"]), "rule": RULE,
                        "relative_error": "(MAE_h-naive_h)/naive_h; negative is better",
                        "evidence_sha256": artifacts["comparison"]["sha256"],
                        "missing": ",".join(decision["missing"])[:200]},
                  metrics=metrics)
    return terminal("COMPLETE", decision["disposition"], artifacts,
                    dict(comparison_status=decision["disposition"], selected=decision["selected"]), [report])


# --------------------------------------------------------------------------- preparation
def _inside_git(path):
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"], capture_output=True,
                         text=True, check=False)
    return out.returncode == 0 and out.stdout.strip() == "true"


def _under(base, path, what):
    path, base = Path(path).resolve(), Path(base).resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        raise PrepareRefused(f"MISSING_ARTIFACT:{what} is not under its declared root (replicate it there)")


def _check_test_free(design, what):
    splits = design.get("splits") or {}
    if (splits.get("test") or {}).get("open") is not False:
        raise PrepareRefused(f"TEST_REQUEST:{what} design does not keep TEST closed")
    for name, window in splits.items():
        if name == "test" or not isinstance(window, dict):
            continue
        end = str(window.get("end", ""))
        if end and end[:10] > TEST_START:
            raise PrepareRefused(f"TEST_REQUEST:{what} split {name} ends after {TEST_START}")


def storage_root_for(role_spec, env=None):
    """The role's experiment storage root: the private spec's ``experiment_storage_root`` or
    ``$EXPERIMENT_STORAGE_ROOT``; it must be a mounted, readable directory."""
    env = os.environ if env is None else env
    base = role_spec.get("experiment_storage_root") or env.get("EXPERIMENT_STORAGE_ROOT")
    if not base:
        raise PrepareRefused("UNAVAILABLE_MOUNT:experiment_storage_root is not configured "
                             "(private spec role.experiment_storage_root or $EXPERIMENT_STORAGE_ROOT)")
    base = Path(base)
    if not base.is_dir() or not os.access(base, os.R_OK):
        raise PrepareRefused("UNAVAILABLE_MOUNT:experiment_storage_root is not a readable directory")
    return base


def population_dir_from_seal(seal, storage_root):
    location = str(seal.get("private_location") or "")
    prefix = "$EXPERIMENT_STORAGE_ROOT/"
    if not location.startswith(prefix):
        raise PrepareRefused("UNFROZEN_COMPARISON: the committed seal names no private storage location")
    return Path(storage_root) / location[len(prefix):]


def verify_population(population_dir, expected_digest, input_dir=None):
    """The data lane's verified load; every typed refusal is an unfrozen comparison."""
    try:
        from tools.i6dmo import common_population as CP
    except ImportError:
        raise PrepareRefused("UNFROZEN_COMPARISON: tools.i6dmo.common_population is not available (data lane)")
    try:
        pop = CP.load_common_population(population_dir, expected_digest=expected_digest, input_dir=input_dir)
    except Exception as exc:
        raise PrepareRefused(f"UNFROZEN_COMPARISON:{getattr(exc, 'code', type(exc).__name__)}") from None
    if pop.digest != expected_digest:
        raise PrepareRefused("UNFROZEN_COMPARISON:POPULATION_DIGEST_MISMATCH")
    return pop


def runtime_pin(design):
    return dict((design.get("sealed_identities") or {}).get("runtime") or {})


def prepare(*, deployment, input_dir, out, role="coordinator", designs=None, population_seal=None,
            population_dir=None, prior_roots=(), test=False, probe=None, env=None):
    """Resolve the matched contrast's private pins BY PATH and write host-specific private JSON.

    The population directory and its expected digest come from the private spec (storage root) plus
    the COMMITTED ``POPULATION_SEAL.json``. The warehouse EnvironmentFile is checked by path only
    (never opened). Refusals are typed: TEST_REQUEST, PRIVATE_OUTPUT_INSIDE_GIT, MISSING_ARTIFACT,
    UNAVAILABLE_MOUNT, UNFROZEN_COMPARISON (digest/identity mismatch included), INCOMPATIBLE_DESIGN,
    INCOMPATIBLE_DONOR, INADEQUATE_MEMORY, INADEQUATE_DISK.
    """
    if test:
        raise PrepareRefused("TEST_REQUEST: the matched contrast has no TEST stage")
    out = Path(out).resolve()
    anchor = out.parent
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if _inside_git(anchor):
        raise PrepareRefused("PRIVATE_OUTPUT_INSIDE_GIT: the prepared JSON is host-specific and private")
    deployment = Path(deployment)
    if not deployment.is_file():
        raise PrepareRefused("MISSING_ARTIFACT:deployment private spec")
    spec = json.loads(deployment.read_text())
    if role not in (spec.get("roles") or {}):
        raise PrepareRefused(f"MISSING_ARTIFACT:role {role} not in the deployment spec")
    role_spec = spec["roles"][role]
    data_root = Path(role_spec.get("data_root") or "")
    run_root = Path(role_spec.get("run_root") or "")
    for what, path, mode in (("data_root", data_root, os.R_OK), ("run_root", run_root, os.W_OK)):
        if not str(path) or str(path) == "." or not path.is_dir() or not os.access(path, mode):
            raise PrepareRefused(f"UNAVAILABLE_MOUNT:{what}")
    storage = storage_root_for(role_spec, env)
    env_file = (spec.get("warehouse") or {}).get("environment_file")
    env_state = "ABSENT_PUBLICATION_WAITS"
    if env_file:
        # path only: existence and type; the content is never opened, read or echoed
        if not Path(env_file).is_file():
            raise PrepareRefused("MISSING_ARTIFACT:warehouse environment_file (configured path does not exist)")
        env_state = "PRESENT_BY_PATH"
    seal_path = Path(population_seal or MATCHED_DIR / "POPULATION_SEAL.json")
    if not seal_path.is_file():
        raise PrepareRefused("UNFROZEN_COMPARISON: committed POPULATION_SEAL.json missing")
    seal = json.loads(seal_path.read_text())
    if seal.get("schema") != "i6dmo.common_population.seal.v1" or seal.get("test_materialized") is not False:
        raise PrepareRefused("UNFROZEN_COMPARISON: not a TEST-free common population seal")
    digest = seal.get("population_digest")
    pop_dir = Path(population_dir) if population_dir else population_dir_from_seal(seal, storage)
    if not pop_dir.is_dir():
        raise PrepareRefused("UNAVAILABLE_MOUNT:population directory absent under the experiment storage root")
    if not (pop_dir / "POPULATION.json").is_file() or _sha(pop_dir / "POPULATION.json") != seal.get(
            "population_json_sha256"):
        raise PrepareRefused("UNFROZEN_COMPARISON:POPULATION_FILE_DIGEST_MISMATCH")
    input_dir = Path(input_dir)
    if not (input_dir / "INPUT.npz").is_file():
        raise PrepareRefused("MISSING_ARTIFACT:input INPUT.npz")
    pop = verify_population(pop_dir, digest, input_dir=input_dir)
    from tools.i6dmo import design as D
    sealed = {}
    for arm in ARMS:
        path = Path((designs or {}).get(arm) or MATCHED_DIR / f"EXPERIMENT.{arm}.json")
        if not path.is_file():
            raise PrepareRefused(f"MISSING_ARTIFACT:{arm} design")
        doc = json.loads(path.read_text())
        _check_test_free(doc, arm)
        try:
            D.validate_design(doc)
            if D.matched_arm(doc) != arm:
                raise PrepareRefused(f"INCOMPATIBLE_DESIGN:{arm} is not the {arm} matched identity")
        except D.DesignError as exc:
            raise PrepareRefused(f"INCOMPATIBLE_DESIGN:{arm}:{exc.code}") from None
        identity = (pop.identities.get("arms") or {}).get(arm) or {}
        if identity.get("experiment_seal_sha256") != doc.get("seal_sha256"):
            raise PrepareRefused(f"UNFROZEN_COMPARISON:{arm} design is not the one the population was frozen for")
        if not runtime_pin(doc):
            raise PrepareRefused(f"INCOMPATIBLE_DESIGN:{arm} declares no sealed runtime pin")
        sealed[arm] = dict(path=str(path.resolve()), sha256=_sha(path), seal_sha256=doc.get("seal_sha256"),
                           runtime=runtime_pin(doc), family=D.MATCHED_SCOPE[arm]["family"])
    pins = dict(
        common_population=dict(base="experiment_storage", relpath=_under(storage, pop_dir, "population"),
                               sha256=_sha(pop_dir / "POPULATION.json"), population_sha256=digest),
        matched_input=dict(relpath=_under(data_root, input_dir, "input"), sha256=_sha(input_dir / "INPUT.npz")))
    for prior in prior_roots:
        _check_donor(prior, sealed, pins)
    facts = (probe or _local_probe)(run_root, data_root)
    if facts.get("mem_available_bytes", 0) < MIN_COORDINATOR_AVAILABLE:
        raise PrepareRefused(f"INADEQUATE_MEMORY: {facts.get('mem_available_bytes')} < {MIN_COORDINATOR_AVAILABLE}")
    if facts.get("disk_free_bytes", 0) < MIN_DISK:
        raise PrepareRefused(f"INADEQUATE_DISK: {facts.get('disk_free_bytes')} < {MIN_DISK}")
    doc = dict(schema="i6dmo.matched_short_prepared.v1", deployment=str(deployment.resolve()),
               deployment_sha256=_sha(deployment), role=role, designs=sealed, data_pins=pins,
               experiment_storage_root=str(storage), population_seal=dict(path=str(seal_path.resolve()),
                                                                           sha256=_sha(seal_path)),
               warehouse_environment_file=env_state, prior_roots=[str(Path(p).resolve()) for p in prior_roots],
               decision_rule=dict(rule=RULE, tolerance=TOLERANCE, tie_keeps=TIE_KEEPS, horizons=list(HORIZONS)),
               host_facts=dict(mem_available_bytes=facts.get("mem_available_bytes"),
                               disk_free_bytes=facts.get("disk_free_bytes")),
               test_open=False, prepared_at=time.time())
    _atomic(out, doc)
    os.chmod(out, 0o600)
    return doc


def _local_probe(run_root, data_root):
    from tools.i6dmo import preflight
    mem = preflight._meminfo()
    return dict(mem_available_bytes=mem.get("MemAvailable", 0), disk_free_bytes=shutil.disk_usage(run_root).free)


def _check_donor(prior, sealed, pins):
    """A prior root is adoptable only when it is a matched contrast with the same sealed arms and data."""
    prior = Path(prior)
    try:
        effective = json.loads((prior / "EFFECTIVE_CONFIG.json").read_text())
    except (OSError, ValueError):
        raise PrepareRefused(f"INCOMPATIBLE_DONOR:{prior.name} has no sealed effective config")
    if effective.get("mode") != "MATCHED_SHORT_CONTRAST":
        raise PrepareRefused(f"INCOMPATIBLE_DONOR:{prior.name} is not a matched short contrast")
    for arm in ARMS:
        if ((effective.get("arms") or {}).get(arm) or {}).get("design", {}).get("seal_sha256") != sealed[arm]["seal_sha256"]:
            raise PrepareRefused(f"INCOMPATIBLE_DONOR:{prior.name} {arm} design differs")
    for key, pin in pins.items():
        if (effective.get("data_pins") or {}).get(key, {}).get("sha256") != pin["sha256"]:
            raise PrepareRefused(f"INCOMPATIBLE_DONOR:{prior.name} {key} differs")


def validate(prepared_path):
    """Re-verify a prepared JSON: designs, input, population (digest) still match; launches nothing."""
    doc = json.loads(Path(prepared_path).read_text())
    if doc.get("schema") != "i6dmo.matched_short_prepared.v1" or doc.get("test_open") is not False:
        raise PrepareRefused("PREPARED_SCHEMA")
    spec_path = Path(doc["deployment"])
    if not spec_path.is_file() or _sha(spec_path) != doc["deployment_sha256"]:
        raise PrepareRefused("MISSING_ARTIFACT:deployment spec changed since prepare")
    spec = json.loads(spec_path.read_text())
    data_root = Path(spec["roles"][doc["role"]]["data_root"])
    problems = []
    for arm, design in doc["designs"].items():
        if not Path(design["path"]).is_file() or _sha(design["path"]) != design["sha256"]:
            problems.append(f"DESIGN_CHANGED:{arm}")
    seal = doc["population_seal"]
    if not Path(seal["path"]).is_file() or _sha(seal["path"]) != seal["sha256"]:
        problems.append("POPULATION_SEAL_CHANGED")
    inp = doc["data_pins"]["matched_input"]
    if not (data_root / inp["relpath"] / "INPUT.npz").is_file() or \
            _sha(data_root / inp["relpath"] / "INPUT.npz") != inp["sha256"]:
        problems.append("PIN_CHANGED:matched_input")
    pop = doc["data_pins"]["common_population"]
    pop_dir = Path(doc["experiment_storage_root"]) / pop["relpath"]
    if not (pop_dir / "POPULATION.json").is_file() or _sha(pop_dir / "POPULATION.json") != pop["sha256"]:
        problems.append("PIN_CHANGED:common_population")
    if problems:
        raise PrepareRefused("MISSING_ARTIFACT:" + ",".join(problems))
    verify_population(pop_dir, pop["population_sha256"], input_dir=data_root / inp["relpath"])
    return dict(state="VALID", designs={a: d["seal_sha256"] for a, d in doc["designs"].items()},
                runtime={a: d["runtime"] for a, d in doc["designs"].items()},
                pins=sorted(doc["data_pins"]), common_population_sha256=pop["population_sha256"])


def runtime_problem(pin, facts):
    """Typed refusal when the executing worker's TF/Keras differ from the sealed runtime pin."""
    if not pin:
        return None
    actual = (facts or {}).get("runtime")
    if not actual:
        return "RUNTIME_UNVERIFIED: the worker probe reported no TensorFlow/Keras versions"
    wrong = [f"{k}={actual.get(k)}!={v}" for k, v in sorted(pin.items()) if actual.get(k) != v]
    return ("RUNTIME_PIN_MISMATCH:" + ",".join(wrong) + " (install the pinned runtime; never reseal)") if wrong else None


def preflight(prepared_path, out_dir, *, probe=None):
    """Per-role, read-only host preflight for the matched contrast (no launch, no fit).

    For every role: the probe answers; disk; the pinned input and population are present with their
    digests (relative to that role's data_root / experiment_storage_root); and on every GPU role that
    can run a fit, the ACTUAL TensorFlow/Keras equal the sealed runtime pin (typed
    RUNTIME_PIN_MISMATCH otherwise) and the initial-cap admission passes on the preferred role.
    """
    from tools.i6dmo import preflight as P, worker as W
    validate(prepared_path)
    doc = json.loads(Path(prepared_path).read_text())
    spec = json.loads(Path(doc["deployment"]).read_text())
    pins = doc["data_pins"]
    pin_runtime = next(iter(doc["designs"].values()))["runtime"]
    if any(d["runtime"] != pin_runtime for d in doc["designs"].values()):
        raise PrepareRefused("INCOMPATIBLE_DESIGN: the arms pin different runtimes")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roles = {}
    gpu_orders = [v.get("order", 0) for v in spec["roles"].values() if any(g.get("heavy") for g in v.get("gpus", []))]
    for role, role_spec in sorted(spec["roles"].items()):
        checks = []

        def check(name, ok, blocks, detail=None):
            checks.append(dict(check=name, state="PASS" if ok else "FAIL", blocks=blocks, detail=detail))
        files = [f"{pins['matched_input']['relpath']}/INPUT.npz"]
        storage = role_spec.get("experiment_storage_root") or (doc["experiment_storage_root"] if role == doc["role"]
                                                                else None)
        if storage:
            files.append(str(Path(storage) / pins["common_population"]["relpath"] / "POPULATION.json"))
        try:
            facts = (probe or (lambda r, f: P.probe_role(spec, r, files=f, check_strategy=False)))(role, files)
        except Exception as exc:
            check("probe", False, "LAUNCH", str(exc)[:200])
            roles[role] = checks
            continue
        heavy = next((g for g in role_spec.get("gpus", []) if g.get("heavy")), None)
        blocks = "LAUNCH" if heavy else "INFO"
        check("probe", True, "LAUNCH", dict(mem_available_bytes=facts.get("mem_available_bytes"),
                                            disk_free_bytes=facts.get("disk_free_bytes")))
        check("disk", facts.get("disk_free_bytes", 0) >= P.MIN_DISK, "LAUNCH",
              dict(disk_free_bytes=facts.get("disk_free_bytes"), required_bytes=P.MIN_DISK))
        found = facts.get("files") or {}
        check("pin:matched_input", found.get(files[0]) == pins["matched_input"]["sha256"], "LAUNCH")
        if storage:
            check("pin:common_population", found.get(files[1]) == pins["common_population"]["sha256"], blocks)
        else:
            check("pin:common_population", False, blocks, "role declares no experiment_storage_root")
        problem = runtime_problem(pin_runtime, facts)
        check("runtime_pin", problem is None, blocks, dict(pinned=pin_runtime, actual=facts.get("runtime"),
                                                           cause=problem))
        if heavy:
            visible = heavy["uuid"] in (facts.get("gpu_free_bytes") or {})
            check("gpu_visible", visible, "LAUNCH")
            launcher = dict(spec["launcher"], **role_spec.get("launcher", {}))
            ok, cause, evidence = W.admit(facts, cap=int(launcher["initial_cap_bytes"]), gpu_uuid=heavy["uuid"],
                                          reserve=int(role_spec.get("reserve_bytes", W.HOST_RESERVE)))
            first = role_spec.get("order", 0) == min(gpu_orders)
            check("admission_initial_cap", ok, "LAUNCH" if first else "ROUTED_TO_NEXT_GPU_ROLE",
                  dict(evidence, cause=cause, note="no measured 25-slot peak exists: initial cap, never a "
                                                   "13-slot peak"))
        roles[role] = checks
    blockers = [f"{r}:{c['check']}" for r, cs in roles.items() for c in cs
                if c["state"] != "PASS" and c["blocks"] == "LAUNCH"]
    result = dict(schema="i6dmo.matched_precheck.v1", state="PASS" if not blockers else "FAIL",
                  launch_blockers=blockers, roles=roles, runtime_pin=pin_runtime,
                  note="PRECHECK is evidence of headroom at probe time, not a launch")
    _atomic(out_dir / "PRECHECK.matched.json", result)
    os.chmod(out_dir / "PRECHECK.matched.json", 0o600)
    return result
