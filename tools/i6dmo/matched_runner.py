"""Matched short contrast (NEXT_CONTRASTS_PLAN_20261015 section 2/4): MG1 (25 slots) vs MG2 (13 slots).

Finite graph inside the existing campaign machinery (no independent scheduler)::

    mg1.fit -> mg1.report \\
                           comparison -> publish/readback -> mirrors -> CLOSE
    mg2.fit -> mg2.report /

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
worker hands each arm unit an ``effective_config`` VIEW in which ``design`` is that arm's sealed
design and the arm's ``<arm>_bundle``/``<arm>_input`` pins appear as ``short_bundle``/``short_input``;
the campaign's ``common_population`` pin (relpath, sha256, population_sha256) is passed through
unchanged. Reports must declare ``common_population_sha256`` (terminal ``lineage`` or METRICS.json).
"""
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

ARMS = ("mg1", "mg2")
ARM_SLOTS = {"mg1": 25, "mg2": 13}
ARM_SPACING = {"mg1": 1, "mg2": 2}
ARM_RESOURCE_CLASS = {"mg1": "short_mg1_matched", "mg2": "short_mg2_matched"}
HORIZONS = (2, 3, 4, 5, 6, 12)
TOLERANCE = 1e-8
TIE_KEEPS = "mg2"
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
    gap = scores["mg1"] - scores["mg2"]
    if abs(gap) <= tolerance:
        disposition, selected = "MG2_RETAINED_WITHIN_TOLERANCE", TIE_KEEPS
    elif gap < 0:
        disposition, selected = "MG1_SELECTED", "mg1"
    else:
        disposition, selected = "MG2_SELECTED", "mg2"
    return dict(disposition=disposition, selected=selected, scores=scores, score_gap_mg1_minus_mg2=gap,
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


def _rel(data_root, path, what):
    path, data_root = Path(path).resolve(), Path(data_root).resolve()
    try:
        return str(path.relative_to(data_root))
    except ValueError:
        raise PrepareRefused(f"MISSING_ARTIFACT:{what} is not under the role data_root (replicate it there)")


def _design_market(doc):
    return doc.get("market_time") or {}


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


def load_common_population(path, expected_digest, loader=None):
    """The data lane's frozen population, verified; an absent API is an unfrozen comparison."""
    if loader is None:
        try:
            from tools.i6dmo.common_population import load_common_population as loader
        except ImportError:
            raise PrepareRefused("UNFROZEN_COMPARISON: tools.i6dmo.common_population is not available "
                                 "(data lane) - the common population cannot be verified")
    try:
        return loader(path, expected_digest=expected_digest)
    except Exception as exc:
        raise PrepareRefused(f"UNFROZEN_COMPARISON: common population refused: "
                             f"{getattr(exc, 'code', type(exc).__name__)}") from None


def prepare(*, deployment, designs, bundles, inputs, common_population, population_digest, out,
            prior_roots=(), test=False, probe=None, population_loader=None, role="coordinator"):
    """Resolve pins BY PATH for the matched contrast and write the host-specific private JSON.

    Never opens the warehouse EnvironmentFile (existence/regular-file check only) and never prints
    a secret. Refuses explicitly: TEST requests, unfrozen comparison, incompatible donor, missing
    artifact, unavailable mount, inadequate memory/disk.
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
    data_root = Path(spec["roles"][role].get("data_root") or "")
    run_root = Path(spec["roles"][role].get("run_root") or "")
    for what, path in (("data_root", data_root), ("run_root", run_root)):
        if not str(path) or not path.is_dir() or not os.access(path, os.W_OK if what == "run_root" else os.R_OK):
            raise PrepareRefused(f"UNAVAILABLE_MOUNT:{what}")
    env_file = (spec.get("warehouse") or {}).get("environment_file")
    env_state = "ABSENT_PUBLICATION_WAITS"
    if env_file:
        # path only: existence and type; the content is never opened, read or echoed
        if not Path(env_file).is_file():
            raise PrepareRefused("MISSING_ARTIFACT:warehouse environment_file (configured path does not exist)")
        env_state = "PRESENT_BY_PATH"
    sealed = {}
    for arm in ARMS:
        path = Path(designs[arm])
        if not path.is_file():
            raise PrepareRefused(f"MISSING_ARTIFACT:{arm} design")
        doc = json.loads(path.read_text())
        _check_test_free(doc, arm)
        market = _design_market(doc)
        if market.get("observations") != ARM_SLOTS[arm] or market.get("spacing_open_market_hours") != ARM_SPACING[arm]:
            raise PrepareRefused(f"INCOMPATIBLE_DESIGN:{arm} must have {ARM_SLOTS[arm]} slots at "
                                 f"{ARM_SPACING[arm]} open-market hour spacing")
        if list(doc.get("horizons_hours") or []) != list(HORIZONS):
            raise PrepareRefused(f"INCOMPATIBLE_DESIGN:{arm} horizons must be {list(HORIZONS)}")
        if doc.get("seed") != SEED:
            raise PrepareRefused(f"INCOMPATIBLE_DESIGN:{arm} seed must be {SEED}")
        sealed[arm] = dict(path=str(path.resolve()), sha256=_sha(path), seal_sha256=doc.get("seal_sha256"),
                           rule=market.get("rule"))
    pins = {}
    for arm in ARMS:
        bundle = Path(bundles[arm])
        manifest_path = bundle / "BUNDLE.json"
        if not manifest_path.is_file():
            raise PrepareRefused(f"MISSING_ARTIFACT:{arm} bundle BUNDLE.json")
        manifest = json.loads(manifest_path.read_text())
        rule = (manifest.get("input_sampling") or {}).get("rule")
        if rule != sealed[arm]["rule"]:
            raise PrepareRefused(f"INCOMPATIBLE_BUNDLE:{arm} bundle rule {rule} != design rule {sealed[arm]['rule']}")
        pins[f"{arm}_bundle"] = dict(relpath=_rel(data_root, bundle, f"{arm} bundle"), sha256=_sha(manifest_path),
                                     bundle_sha256=manifest.get("bundle_sha256"), input_sampling=rule)
        source = Path(inputs[arm])
        if not (source / "INPUT.npz").is_file():
            raise PrepareRefused(f"MISSING_ARTIFACT:{arm} input INPUT.npz")
        pins[f"{arm}_input"] = dict(relpath=_rel(data_root, source, f"{arm} input"), sha256=_sha(source / "INPUT.npz"))
    population = Path(common_population)
    if not population.exists():
        raise PrepareRefused("UNFROZEN_COMPARISON: common population artifact missing")
    if not population_digest or len(str(population_digest)) != 64:
        raise PrepareRefused("UNFROZEN_COMPARISON: the expected population digest is required")
    load_common_population(population, population_digest, loader=population_loader)
    pins["common_population"] = dict(relpath=_rel(data_root, population, "common population"),
                                     sha256=artifact_sha256(population), population_sha256=population_digest)
    for prior in prior_roots:
        _check_donor(prior, sealed, pins)
    facts = (probe or _local_probe)(run_root, data_root)
    if facts.get("mem_available_bytes", 0) < MIN_COORDINATOR_AVAILABLE:
        raise PrepareRefused(f"INADEQUATE_MEMORY: {facts.get('mem_available_bytes')} < {MIN_COORDINATOR_AVAILABLE}")
    if facts.get("disk_free_bytes", 0) < MIN_DISK:
        raise PrepareRefused(f"INADEQUATE_DISK: {facts.get('disk_free_bytes')} < {MIN_DISK}")
    doc = dict(schema="i6dmo.matched_short_prepared.v1", deployment=str(deployment.resolve()),
               deployment_sha256=_sha(deployment), role=role, designs=sealed, data_pins=pins,
               warehouse_environment_file=env_state, prior_roots=[str(Path(p).resolve()) for p in prior_roots],
               decision_rule=dict(rule=RULE, tolerance=TOLERANCE, tie_keeps=TIE_KEEPS, horizons=list(HORIZONS)),
               host_facts=dict(mem_available_bytes=facts.get("mem_available_bytes"),
                               disk_free_bytes=facts.get("disk_free_bytes")),
               test_open=False, prepared_at=time.time())
    _atomic(out, doc)
    os.chmod(out, 0o600)
    return doc


def artifact_sha256(path):
    """File digest, or for a directory the digest of its sorted (relative path, file digest) listing."""
    path = Path(path)
    if path.is_file():
        return _sha(path)
    listing = [[str(p.relative_to(path)), _sha(p)] for p in sorted(path.rglob("*")) if p.is_file()]
    import hashlib
    return hashlib.sha256(json.dumps(listing, sort_keys=True).encode()).hexdigest()


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
    for key in ("mg1_bundle", "mg2_bundle", "mg1_input", "mg2_input", "common_population"):
        if (effective.get("data_pins") or {}).get(key, {}).get("sha256") != pins[key]["sha256"]:
            raise PrepareRefused(f"INCOMPATIBLE_DONOR:{prior.name} {key} differs")


def validate(prepared_path, *, population_loader=None):
    """Re-verify a prepared JSON: designs, bundles, inputs and the common population still match."""
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
    for key, pin in doc["data_pins"].items():
        base = data_root / pin["relpath"]
        if key.endswith("_bundle") or key.endswith("_input"):
            target = base / ("BUNDLE.json" if key.endswith("_bundle") else "INPUT.npz")
            ok = target.is_file() and _sha(target) == pin["sha256"]
        else:
            ok = base.exists() and artifact_sha256(base) == pin["sha256"]
        if not ok:
            problems.append(f"PIN_CHANGED:{key}")
    pop = doc["data_pins"]["common_population"]
    load_common_population(data_root / pop["relpath"], pop["population_sha256"], loader=population_loader)
    if problems:
        raise PrepareRefused("MISSING_ARTIFACT:" + ",".join(problems))
    return dict(state="VALID", designs={a: d["seal_sha256"] for a, d in doc["designs"].items()},
                pins=sorted(doc["data_pins"]), common_population_sha256=pop["population_sha256"])
