"""I6-DMO campaign: sealed effective config, enumerated work, finite tick, gates, readiness and close.

Finite sequence (PLAN section 1):
SHORT_DS_STATIC_VALIDATION -> LONG_DS_STATIC_VALIDATION -> PAIR_VALIDATION -> FINAL_PROCEDURE_FROZEN
-> TEST_READINESS -> FINAL_PAIR_TEST -> CLOSED.

- ``init`` seals the design (through the model lane's adapter) and enumerates every validation
  unit: exactly one fit per output family, one report per family that never fits, one pair
  validation. TEST is never materialized here: no unit carries a TEST window or ``test_open``.
- ``tick`` is finite: it activates dependencies, writes automatic gates, evaluates readiness and,
  only when every recorded gate passes and is bound to this campaign, enumerates the final refits
  and the single TEST pair unit. Otherwise readiness is a precise BLOCKED/TERMINAL disposition with
  the missing artifact and its owner.
- Gate receipts derived from TEST reads are refused, and selection inputs ignore TEST terminals.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from tools.i6n_swarm.common import atomic_json, canonical, file_sha256, sha256_json

from tools.i6dmo import queue as Q

CADENCES = ("NONE", "QUARTERLY", "MONTHLY", "WEEKLY")
STAGES = ("SHORT_DS_STATIC_VALIDATION", "LONG_DS_STATIC_VALIDATION", "PAIR_VALIDATION", "FINAL_PROCEDURE_FROZEN",
          "TEST_READINESS", "FINAL_PAIR_TEST", "CLOSED")
# Market-time pair (order 50d20d01): one finite development graph, no TEST stage at all.
MARKET_TIME_STAGES = ("SHORT_MARKET_TIME_VALIDATION", "LONG_MARKET_TIME_VALIDATION", "PAIR_VALIDATION",
                      "VALIDATION_STRATEGY", "CLOSED")
MODE_ELAPSED = "ELAPSED_STATIC"
MODE_MARKET_TIME = "MARKET_TIME_PAIR"
MODES = (MODE_ELAPSED, MODE_MARKET_TIME)
UNIT_KINDS = ("fit", "report", "pair", "strategy", "final_refit", "final_test")
IDENTITY_KEYS = ("unit", "family", "kind", "cadence", "effective_config_sha256", "data_sha256", "code_commit",
                 "seed", "upstream")
DEFAULT_ENTRYPOINTS = {kind: ["{python}", "-m", "tools.i6dmo.unit_runner", "--kind", kind]
                       for kind in ("fit", "report", "pair", "strategy", "final_refit", "final_test")}
FINAL_REFIT_TRAIN = ["2021-01-01T00:00:00Z", "2025-01-01T00:00:00Z"]
# Gate -> (owner, artifact, produced automatically by tick?)
GATES = {
    "SHORT_VALIDATION": ("ops lane tick", "gates/SHORT_VALIDATION.json", True),
    "LONG_VALIDATION": ("ops lane tick", "gates/LONG_VALIDATION.json", True),
    "PAIR_VALIDATION": ("strategy lane pair unit", "gates/PAIR_VALIDATION.json", True),
    "PROCEDURE_FREEZE": ("ops lane tick after pair PASS", "gates/PROCEDURE_FREEZE.json", True),
    "VALIDATION_DELIVERY": ("ops lane publish (outbox readback + mirrors)", "gates/VALIDATION_DELIVERY.json", True),
    "DATA_PROVENANCE": ("data lane: seven producers and availability clocks recovered without TEST",
                        "gates/DATA_PROVENANCE.json", False),
    "SOFTWARE_TRACEABILITY": ("Satoshi integration: TRACEABILITY.json with DMO01-DMO20 PASS",
                              "gates/SOFTWARE_TRACEABILITY.json", False),
    "EXECUTION_POLICY": ("strategy lane: receipt sealed automatically at init/deploy from the versioned "
                         "effective_config.strategy_execution.instrument; tampering is a final refusal",
                         "gates/EXECUTION_POLICY.json", True),
}


class Refused(RuntimeError):
    """A typed refusal; the message starts with its cause token."""


# ------------------------------------------------------------------ identity
def unit_identity(components):
    if set(components) != set(IDENTITY_KEYS):
        raise ValueError(f"IDENTITY_KEYS: need exactly {IDENTITY_KEYS}")
    if components["cadence"] not in CADENCES:
        raise ValueError(f"UNKNOWN_CADENCE: {components['cadence']}")
    if components["kind"] not in UNIT_KINDS:
        raise ValueError(f"UNKNOWN_KIND: {components['kind']}")
    return sha256_json(dict(schema="i6dmo.unit_identity.v1", **components))


def validate_entrypoints(entries):
    missing = [k for k in UNIT_KINDS if k not in entries]
    if missing:
        raise ValueError(f"ENTRYPOINT_MISSING: {missing}")
    for kind, argv in entries.items():
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
            raise ValueError(f"ENTRYPOINT_INVALID: {kind}")
    if entries["report"] in (entries["fit"], entries["final_refit"]):
        raise ValueError("REPORT_ENTRY_CALLS_FIT: weekly/annual reports must never call the fit entry")
    return entries


# ------------------------------------------------------------------- paths
def _paths(root):
    root = Path(root)
    return dict(effective=root / "EFFECTIVE_CONFIG.json", plan=root / "PLAN.json", queue=root / "state" / "queue.sqlite3",
                outbox=root / "state" / "outbox.sqlite3", gates=root / "gates", terminals=root / "terminals",
                deployment=root / "private" / "deployment.json", status=root / "STATUS.json",
                close=root / "CLOSE.json", long=root / "LONG_EFFECTIVE_CONFIG.json")


def open_queue(root, clock=time.time):
    return Q.Queue(_paths(root)["queue"], clock=clock)


def _seal(doc):
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    return dict(body, seal_sha256=sha256_json(body))


def _check_seal(doc, what):
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    if doc.get("seal_sha256") != sha256_json(body):
        raise Refused(f"{what}_SEAL: sealed document was modified")
    return doc


def load_effective(root):
    path = _paths(root)["effective"]
    if not path.exists():
        raise Refused("EFFECTIVE_CONFIG_SEAL: missing EFFECTIVE_CONFIG.json")
    doc = _check_seal(json.loads(path.read_text()), "EFFECTIVE_CONFIG")
    if doc["design"]["splits"]["test"].get("open") is not False or doc.get("test_open") is not False:
        raise Refused("TEST_OPEN_IN_DESIGN")
    return doc


def load_plan(root):
    return json.loads(_paths(root)["plan"].read_text())


def load_deployment(root):
    return json.loads(_paths(root)["deployment"].read_text())


def _write_plan(root, plan):
    atomic_json(_paths(root)["plan"], plan)


class ModelLaneAdapter:
    """Bridge to the model lane's typed design module (``tools.i6dmo.design``)."""

    def __init__(self, module):
        self.module = module

    def validate_design(self, design):
        try:
            effective = self.module.effective_config(self.module.validate_design(design))
        except Exception as exc:  # DesignError carries a typed code
            return dict(ok=False, errors=[getattr(exc, "code", type(exc).__name__)], detail=str(exc)[:300])
        return dict(ok=True, errors=[], design_sha256=effective["design_sha256"],
                    effective_config_sha256=effective.get("effective_config_sha256"))

    def effective_config(self, design):
        return self.module.effective_config(self.module.validate_design(design))


def _adapter(adapter):
    if adapter is not None:
        return adapter
    try:
        from tools.i6dmo import design as design_module  # model lane
    except ImportError as exc:
        raise Refused("MODEL_ADAPTER_UNAVAILABLE: tools.i6dmo.design (model lane) is not importable") from exc
    return ModelLaneAdapter(design_module)


def _git_head():
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2], check=False)
    if out.returncode != 0:
        raise Refused("CODE_COMMIT_UNRESOLVED")
    return out.stdout.strip()


ELAPSED_DESIGN_SCHEMA = "i6dmo.design.v1"
MARKET_TIME_DESIGN_SCHEMA = "i6dmo.market_time_design.v1"
# Elapsed-grid fields that must never appear beside a market_time block (plan section 2).
ELAPSED_TIME_FIELDS = ("input", "history_span_hours", "spacing_hours")
MARKET_TIME_REQUIRED = ("rule", "observations", "spacing_open_market_hours", "span_open_market_hours",
                        "include_origin", "elapsed_span_fixed", "require_exact_market_slots")


def _market_time_checks(design):
    """Validate the sealed market-time schema: required block, no contradictory elapsed fields."""
    errors = []
    block = design.get("market_time")
    if not isinstance(block, dict):
        return ["MARKET_TIME_BLOCK_MISSING"]
    missing = [k for k in MARKET_TIME_REQUIRED if k not in block]
    if missing:
        errors.append(f"MARKET_TIME_FIELD_MISSING:{missing[0]}")
    if any(f in design for f in ELAPSED_TIME_FIELDS):
        errors.append("MARKET_TIME_CONTRADICTION")
    if block.get("elapsed_span_fixed") is not False:
        errors.append("MARKET_TIME_ELAPSED_SPAN_MUST_VARY")
    if block.get("require_exact_market_slots") is not True:
        errors.append("MARKET_TIME_REQUIRES_EXACT_SLOTS")
    if block.get("include_origin") is not True:
        errors.append("MARKET_TIME_MUST_INCLUDE_ORIGIN")
    for key in ("observations", "spacing_open_market_hours", "span_open_market_hours"):
        value = block.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"MARKET_TIME_FIELD_INVALID:{key}")
    return errors


def _design_checks(design, *, mode=None):
    """Common design gates. ``mode`` selects which schema is admissible for this campaign."""
    errors = []
    schema = design.get("schema")
    expected = MARKET_TIME_DESIGN_SCHEMA if mode == MODE_MARKET_TIME else ELAPSED_DESIGN_SCHEMA
    if schema != expected:
        errors.append("SCHEMA")
    elif schema == MARKET_TIME_DESIGN_SCHEMA:
        errors.extend(_market_time_checks(design))
    splits = design.get("splits", {})
    if splits.get("test", {}).get("open") is not False:
        errors.append("TEST_OPEN_IN_DESIGN")
    if design.get("retraining_frequency") not in CADENCES:
        errors.append("UNKNOWN_CADENCE")
    if design.get("repetitions") != 1:
        errors.append("REPETITIONS_NOT_ONE")
    return errors


def _seal_family(design, adapter, family, *, mode=None):
    verdict = adapter.validate_design(design)
    if not verdict.get("ok"):
        raise Refused(f"DESIGN_INVALID: {family}: {verdict.get('errors')}")
    errors = _design_checks(design, mode=mode)
    if errors:
        raise Refused(f"{errors[0]}: {family}")
    return dict(design=design, design_sha256=sha256_json(design), validator=verdict,
                adapter_effective=adapter.effective_config(design))


# --------------------------------------------------------------------- init
def init_campaign(design_path, spec_path, root, *, adapter=None, code_commit=None, data_pins=None, entrypoints=None,
                  prior_roots=(), long_design=None, strategy_execution=None, mode=MODE_ELAPSED,
                  completed_short_fit_source=None):
    design = json.loads(Path(design_path).read_text())
    if mode not in MODES:
        raise Refused(f"UNKNOWN_CAMPAIGN_MODE: {mode}")
    if design.get("splits", {}).get("test", {}).get("open") is not False:
        raise Refused("TEST_OPEN_IN_DESIGN")
    if mode == MODE_MARKET_TIME and design.get("schema") != MARKET_TIME_DESIGN_SCHEMA:
        raise Refused(f"ELAPSED_DESIGN_IN_MARKET_TIME_CAMPAIGN: {design.get('schema')}")
    if mode == MODE_MARKET_TIME and long_design is None:
        raise Refused("MARKET_TIME_PAIR_REQUIRES_BOTH_DESIGNS: seal MG2 and MG12 before init")
    adapter = _adapter(adapter)
    short = _seal_family(design, adapter, "short", mode=mode)
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise Refused("ROOT_NOT_NEW: init never reuses or resurrects an existing campaign root")
    spec = json.loads(Path(spec_path).read_text())
    _validate_spec(spec)
    if data_pins is None:
        try:
            from tools.i6dmo import data as data_lane
            data_pins = data_lane.resolve_pins(design, spec)
        except (ImportError, AttributeError) as exc:
            raise Refused("DATA_PINS_REQUIRED: data lane resolve_pins unavailable") from exc
    if not data_pins or not all(isinstance(p, dict) and len(str(p.get("sha256", ""))) == 64
                                for p in data_pins.values()):
        raise Refused("DATA_PINS_REQUIRED: every pin needs a sha256")
    _check_bundle_pins(data_pins, mode=mode)
    entries = validate_entrypoints(dict(entrypoints or DEFAULT_ENTRYPOINTS))
    long_doc = None
    if long_design is not None:
        long_doc = _seal_family(json.loads(Path(long_design).read_text()), adapter, "long", mode=mode)
        if long_doc["design"]["retraining_frequency"] != design["retraining_frequency"]:
            raise Refused("LONG_CADENCE_DIFFERS")
    effective = _seal(dict(
        schema="i6dmo.effective_config.v1", experiment_id=design["experiment_id"], design=design,
        design_sha256=short["design_sha256"], validator=short["validator"],
        adapter_effective=short["adapter_effective"], cadence=design["retraining_frequency"], seed=design["seed"],
        code_commit=code_commit or _git_head(), data_pins=data_pins, data_sha256=sha256_json(data_pins),
        entrypoints=entries, prior_roots=[str(p) for p in prior_roots], test_open=False,
        strategy_execution=strategy_execution, mode=mode, deployment_sha256=sha256_json(spec), long=long_doc))
    if completed_short_fit_source is not None:
        from tools.i6dmo import recovery
        reused = recovery.prepare(completed_short_fit_source, effective)
        effective = _seal(dict(effective, reused_short_fit=reused))
    eff_sha = sha256_json(effective)
    campaign_id = "i6dmo-" + sha256_json(dict(experiment=design["experiment_id"], cadence=effective["cadence"],
                                              effective=eff_sha))[:12]
    paths = _paths(root)
    paths["deployment"].parent.mkdir(parents=True, exist_ok=True)
    os.chmod(paths["deployment"].parent, 0o700)
    atomic_json(paths["deployment"], spec, mode=0o600)
    atomic_json(paths["effective"], effective)
    for folder in ("gates", "terminals", "attempts"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    units = _validation_units(effective, eff_sha)
    market_time = mode == MODE_MARKET_TIME
    plan = dict(schema="i6dmo.plan.v1", campaign_id=campaign_id, effective_config_sha256=eff_sha,
                effective_config_file_sha256=file_sha256(paths["effective"]), mode=mode,
                stages=list(MARKET_TIME_STAGES if market_time else STAGES), units=units,
                test_units_enumerated=False,
                expected=dict(validation_units=len(units),
                              final_units=("NOT_ENUMERABLE_IN_MARKET_TIME_PAIR" if market_time
                                           else "NOT_ENUMERATED_UNTIL_READINESS")))
    _write_plan(root, plan)
    q = open_queue(root)
    try:
        for unit in units:
            q.ensure_unit(unit["unit_id"], unit, state=unit.pop("_state"))
        if effective.get("reused_short_fit"):
            recovery.install(root, effective["reused_short_fit"], q)
    finally:
        q.close()
    for unit in plan["units"]:
        unit.pop("_state", None)
    _write_plan(root, plan)
    _storage_gate(spec, "init")
    policy_wait = ensure_execution_policy(root)
    write_status(root)
    return dict(campaign_id=campaign_id, effective_config_sha256=eff_sha, root=str(root),
                execution_policy="SEALED" if policy_wait is None else policy_wait,
                units=[u["unit_id"] for u in plan["units"]])


SUPERSEDED_BUNDLE_PREFIXES = ("5221615d",)  # pre-ruling exact-slot build
# Retained ASOF diagnostic arm (audited AUD-DMO-01): real evidence under its own identity,
# never admissible as the exact-DS2 treatment of this campaign.
ASOF_DIAGNOSTIC_PREFIXES = ("3e42c3f5", "e88f72aa")
BUNDLE_SAMPLING_RULE = "DS2_EXACT"
BUNDLE_SAMPLING_BY_KEY = {"short_bundle": "DS2_EXACT", "long_bundle": "DS12_EXACT"}
MARKET_TIME_SAMPLING_BY_KEY = {"short_bundle": "MG2_MARKET_EXACT", "long_bundle": "MG12_MARKET_EXACT"}


def _check_bundle_pins(pins, *, mode=None):
    """*_bundle pins: {relpath (bundle dir), sha256 (BUNDLE.json file), bundle_sha256, input_sampling}.

    The expected sampling rule follows the campaign mode, so an elapsed-grid bundle can never be
    pinned into a market-time campaign (nor the reverse).
    """
    expected = MARKET_TIME_SAMPLING_BY_KEY if mode == MODE_MARKET_TIME else BUNDLE_SAMPLING_BY_KEY
    for key, pin in pins.items():
        if not key.endswith("_bundle"):
            continue
        identity = str(pin.get("bundle_sha256", ""))
        if len(identity) != 64:
            raise Refused(f"BUNDLE_PIN_INCOMPLETE: {key} needs bundle_sha256")
        if identity.startswith(SUPERSEDED_BUNDLE_PREFIXES):
            raise Refused(f"SUPERSEDED_BUNDLE: {key} {identity[:8]} must not be used")
        if identity.startswith(ASOF_DIAGNOSTIC_PREFIXES):
            raise Refused(f"ASOF_DIAGNOSTIC_ARM: {key} {identity[:8]} is the retained ASOF diagnostic, "
                          "not the exact-DS2 identity")
        rule = expected.get(key, BUNDLE_SAMPLING_RULE)
        if pin.get("input_sampling") != rule:
            raise Refused(f"BUNDLE_SAMPLING: {key} must be {rule}")


def pin_file(key, pin):
    """The file whose digest a pin carries: BUNDLE.json for *_bundle, INPUT.npz for *_input."""
    suffix = "BUNDLE.json" if key.endswith("_bundle") else "INPUT.npz" if key.endswith("_input") else None
    return pin["relpath"] if suffix is None else f"{pin['relpath'].rstrip('/')}/{suffix}"


def _storage_gate(spec, stage):
    """Run the versioned storage preflight before a stage that will produce archived artifacts."""
    config = (spec or {}).get("storage") or {}
    estimate = int(config.get("estimate_bytes", 0) or 0)
    if not estimate:
        return None
    from tools.i6dmo import storage as S
    try:
        base = S.storage_root()
    except S.StorageRefusal as exc:
        raise Refused(f"STORAGE_GATE_{exc.code}: stage={stage}") from None
    result = S.run_preflight(Path(base) / S.APP_SUBDIR, estimate,
                             int(config.get("reserve_bytes", S.DEFAULT_RESERVE_BYTES)))
    if result["returncode"] != 0:
        raise Refused(f"STORAGE_GATE_REFUSED: stage={stage} rc={result['returncode']} "
                      f"{result['stderr'][:120]}")
    return result


def _validate_spec(spec):
    if spec.get("schema") != "i6dmo.private_deployment.v1":
        raise Refused("DEPLOYMENT_SPEC_SCHEMA")
    roles = spec.get("roles") or {}
    heavy = [r for r, v in roles.items() if any(g.get("heavy") for g in v.get("gpus", []))]
    if not heavy:
        raise Refused("DEPLOYMENT_SPEC_NO_GPU_ROLE")
    if not any(v.get("cpu_units") for v in roles.values()):
        raise Refused("DEPLOYMENT_SPEC_NO_CPU_ROLE")
    for role, value in roles.items():
        if sum(1 for g in value.get("gpus", []) if g.get("heavy")) > 1:
            raise Refused(f"DEPLOYMENT_SPEC_TWO_HEAVY_SLOTS: {role} (one heavy GPU process per host)")
    for key in ("warehouse", "mirrors", "launcher"):
        if key not in spec:
            raise Refused(f"DEPLOYMENT_SPEC_MISSING: {key}")


def _windows(design):
    s = design["splits"]
    return dict(train=[s["train"]["start"], s["train"]["end"]], validation=[s["validation"]["start"],
                                                                            s["validation"]["end"]],
                inner_tail_weeks=s["inner_tail_weeks"])


def _base(unit_id, family, kind, stage, device, depends_on, windows, *, calls_fit, test_open=False):
    return dict(unit_id=unit_id, family=family, kind=kind, stage=stage, device=device, depends_on=depends_on,
                windows=windows, calls_fit=calls_fit, test_open=test_open, cadence=None, identity=None,
                identity_sha256=None)


def _identify(unit, effective, family_sha, upstream):
    components = dict(unit=unit["unit_id"], family=unit["family"], kind=unit["kind"], cadence=effective["cadence"],
                      effective_config_sha256=family_sha, data_sha256=effective["data_sha256"],
                      code_commit=effective["code_commit"], seed=effective["seed"], upstream=sorted(upstream))
    unit.update(cadence=effective["cadence"], identity=components, identity_sha256=unit_identity(components))
    return unit


def _family_sha(effective, family):
    if family == "short":
        return sha256_json(dict(design=effective["design"], adapter=effective["adapter_effective"]))
    if family == "long":
        doc = effective.get("long")
        return None if doc is None else sha256_json(dict(design=doc["design"], adapter=doc["adapter_effective"]))
    short, long_ = _family_sha(effective, "short"), _family_sha(effective, "long")
    return None if long_ is None else sha256_json(dict(short=short, long=long_))


def mode_of(effective):
    return effective.get("mode") or MODE_ELAPSED


def _market_time_units(effective):
    """short.fit -> short.report -> long.fit -> long.report -> pair.eligibility -> strategy.validation.

    ``long.fit`` depends on the short REPORT TERMINAL, not on short eligibility: a scientifically
    NOT_ELIGIBLE short family still answers the separate long question.
    """
    short_w, long_w = _windows(effective["design"]), _windows(effective["long"]["design"])
    st = MARKET_TIME_STAGES
    short_sha, long_sha = _family_sha(effective, "short"), _family_sha(effective, "long")
    units, upstream = [], {}
    spec = [("short.fit", "short", "fit", st[0], "gpu", [], short_w, True, short_sha),
            ("short.report", "short", "report", st[0], "cpu", ["short.fit"], short_w, False, short_sha),
            ("long.fit", "long", "fit", st[1], "gpu", ["short.report"], long_w, True, long_sha),
            ("long.report", "long", "report", st[1], "cpu", ["long.fit"], long_w, False, long_sha),
            ("pair.eligibility", "pair", "pair", st[2], "cpu", ["short.report", "long.report"],
             dict(validation=short_w["validation"]), False, _family_sha(effective, "pair")),
            ("strategy.validation", "pair", "strategy", st[3], "cpu", ["pair.eligibility"],
             dict(validation=short_w["validation"]), False, _family_sha(effective, "pair"))]
    for unit_id, family, kind, stage, device, deps, windows, calls_fit, fam_sha in spec:
        unit = _base(unit_id, family, kind, stage, device, deps, windows, calls_fit=calls_fit)
        _identify(unit, effective, fam_sha, [upstream[d] for d in deps])
        if unit_id == "short.fit" and effective.get("reused_short_fit"):
            unit = json.loads(json.dumps(effective["reused_short_fit"]["unit"]))
            unit["execution_policy"] = "ADOPT_ONLY"
        upstream[unit_id] = unit["identity_sha256"]
        unit["_state"] = "QUEUED" if not deps else "WAIT_DEPENDENCY"
        units.append(unit)
    return units


def _validation_units(effective, eff_sha):
    if mode_of(effective) == MODE_MARKET_TIME:
        return _market_time_units(effective)
    design = effective["design"]
    short_w = _windows(design)
    long_w = _windows(effective["long"]["design"]) if effective.get("long") else None
    short_fit = _identify(_base("short.fit", "short", "fit", STAGES[0], "gpu", [], short_w, calls_fit=True),
                          effective, _family_sha(effective, "short"), [])
    short_report = _identify(_base("short.report", "short", "report", STAGES[0], "cpu", ["short.fit"], short_w,
                                   calls_fit=False), effective, _family_sha(effective, "short"),
                             [short_fit["identity_sha256"]])
    long_fit = _base("long.fit", "long", "fit", STAGES[1], "gpu", ["short.report"], long_w, calls_fit=True)
    long_report = _base("long.report", "long", "report", STAGES[1], "cpu", ["long.fit"], long_w, calls_fit=False)
    pair = _base("pair.validation", "pair", "pair", STAGES[2], "cpu", ["short.report", "long.report"],
                 dict(validation=short_w["validation"]), calls_fit=False)
    short_fit["_state"], short_report["_state"] = "QUEUED", "WAIT_DEPENDENCY"
    if effective.get("long"):
        _identify(long_fit, effective, _family_sha(effective, "long"), [short_report["identity_sha256"]])
        _identify(long_report, effective, _family_sha(effective, "long"), [long_fit["identity_sha256"]])
        _identify(pair, effective, _family_sha(effective, "pair"),
                  [short_report["identity_sha256"], long_report["identity_sha256"]])
        long_fit["_state"] = long_report["_state"] = pair["_state"] = "WAIT_DEPENDENCY"
    else:
        long_fit["_state"] = long_report["_state"] = pair["_state"] = "WAIT_ARTIFACT"
    return [short_fit, short_report, long_fit, long_report, pair]


def seal_long(root, long_design_path, *, adapter=None):
    """Seal a long design prepared after init (identity fixed before any long fit)."""
    root = Path(root)
    effective = load_effective(root)
    if effective.get("long") is not None:
        raise Refused("LONG_ALREADY_SEALED")
    doc = _seal_family(json.loads(Path(long_design_path).read_text()), _adapter(adapter), "long",
                       mode=mode_of(load_effective(root)))
    sealed = _seal(dict(schema="i6dmo.long_effective_config.v1", campaign_effective_sha256=sha256_json(effective),
                        long=doc))
    atomic_json(_paths(root)["long"], sealed)
    return _apply_long(root)


def _effective_with_long(root):
    effective = load_effective(root)
    if effective.get("long") is None and _paths(root)["long"].exists():
        long_doc = _check_seal(json.loads(_paths(root)["long"].read_text()), "LONG_EFFECTIVE_CONFIG")
        if long_doc["campaign_effective_sha256"] != sha256_json(effective):
            raise Refused("LONG_EFFECTIVE_CONFIG_SEAL: bound to another campaign")
        effective = dict(effective, long=long_doc["long"])
    return effective


def _apply_long(root):
    effective = _effective_with_long(root)
    if effective.get("long") is None:
        return False
    plan = load_plan(root)
    fresh = {u["unit_id"]: u for u in _validation_units(effective, plan["effective_config_sha256"])}
    q = open_queue(root)
    try:
        changed = False
        for i, unit in enumerate(plan["units"]):
            if unit["identity_sha256"] is None and unit["unit_id"] in fresh:
                new = fresh[unit["unit_id"]]
                new.pop("_state", None)
                plan["units"][i] = new
                q.ensure_unit(new["unit_id"], new, state="WAIT_DEPENDENCY")
                changed = True
    finally:
        q.close()
    if changed:
        _write_plan(root, plan)
    return changed


# --------------------------------------------------------------------- gates
def record_gate(root, name, *, status, evidence, splits_read, effective_config_sha256=None, owner=None):
    if name not in GATES:
        raise Refused(f"UNKNOWN_GATE: {name}")
    if name == "EXECUTION_POLICY":
        raise Refused("EXECUTION_POLICY_IS_SEALED_NOT_RECORDED: it derives from the instrument configuration")
    if any(str(s).lower() == "test" for s in splits_read):
        raise Refused("TEST_FEEDBACK: a gate before TEST may not be built from TEST reads")
    plan = load_plan(root)
    receipt = dict(schema="i6dmo.gate.v1", gate=name, status=status, campaign_id=plan["campaign_id"],
                   effective_config_sha256=effective_config_sha256 or plan["effective_config_sha256"],
                   splits_read=sorted(splits_read), evidence=evidence, owner=owner or GATES[name][0],
                   recorded_at=time.time())
    receipt["receipt_sha256"] = sha256_json(receipt)
    atomic_json(_paths(root)["gates"] / f"{name}.json", receipt)
    return receipt


def gate_problem(root, name):
    return _gate_problem(root, name, load_plan(root))


def ensure_execution_policy(root):
    """Seal gates/EXECUTION_POLICY.json from the sealed instrument configuration (idempotent).

    Returns the typed cause while it cannot be sealed; a tampered existing receipt is left in
    place so verification keeps failing it (never overwritten into validity).
    """
    from tools.i6dmo import strategy_adapter as SA
    effective = load_effective(root)
    instrument = (effective.get("strategy_execution") or {}).get("instrument")
    if instrument is None:
        return "INSTRUMENT_CONFIG_MISSING: effective_config.strategy_execution.instrument owner=Satoshi (init)"
    try:
        existing = SA.load_execution_policy(root)
    except SA.StrategyAdapterError as exc:
        return f"EXECUTION_POLICY_TAMPERED: {exc}"
    derived = SA.derive_execution_policy(instrument)
    if existing is None:
        SA.seal_execution_policy(root, instrument)
        return None
    if existing["receipt_sha256"] != derived["receipt_sha256"]:
        return "EXECUTION_POLICY_MISMATCH: sealed receipt differs from the sealed instrument configuration"
    return None


def _execution_policy_problem(root):
    from tools.i6dmo import strategy_adapter as SA
    effective = load_effective(root)
    instrument = (effective.get("strategy_execution") or {}).get("instrument")
    if instrument is None:
        return "INSTRUMENT_CONFIG_MISSING"
    try:
        sealed = SA.load_execution_policy(root)
    except SA.StrategyAdapterError:
        return "EXECUTION_POLICY_TAMPERED"
    if sealed is None:
        return "ABSENT"
    if sealed["receipt_sha256"] != SA.derive_execution_policy(instrument)["receipt_sha256"]:
        return "MISMATCH"
    return None


def _gate_problem(root, name, plan):
    if name == "EXECUTION_POLICY":
        return _execution_policy_problem(root)
    path = _paths(root)["gates"] / f"{name}.json"
    if not path.exists():
        return "ABSENT"
    doc = json.loads(path.read_text())
    body = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    if doc.get("receipt_sha256") != sha256_json(body):
        return "RECEIPT_DIGEST_MISMATCH"
    if doc.get("campaign_id") != plan["campaign_id"] or \
            doc.get("effective_config_sha256") != plan["effective_config_sha256"]:
        return "BINDING_MISMATCH"
    if any(str(s).lower() == "test" for s in doc.get("splits_read", [])):
        return "TEST_FEEDBACK"
    if doc.get("status") != "PASS":
        return f"STATUS_{doc.get('status')}"
    return None


def selection_inputs(root):
    """Everything allowed to select models/eligibility: validation report terminals only, never TEST."""
    q = open_queue(root)
    try:
        units = [u for u in q.units() if u["kind"] == "report" and not u["spec"].get("test_open")
                 and u["state"] == "COMPLETE"]
        eligibility = {}
        for unit in units:
            term = q.terminal(unit["unit_id"])
            eligibility[unit["spec"]["family"]] = term["terminal"].get("eligibility", {})
        return dict(units=sorted(u["unit_id"] for u in units), eligibility=eligibility)
    finally:
        q.close()


def _auto_gates(root, plan, q):
    """Write the gates whose evidence is local and complete; never overwrite a PASS with a different body."""
    states = {u["unit_id"]: u["state"] for u in q.units()}
    plan_ids = {u["unit_id"]: u for u in plan["units"]}

    def done(*ids):
        return all(states.get(i) == "COMPLETE" for i in ids)

    gates_dir = _paths(root)["gates"]
    written = []

    def put(name, status, evidence):
        if (gates_dir / f"{name}.json").exists():
            return
        record_gate(root, name, status=status, evidence=evidence, splits_read=["train", "validation"])
        written.append(name)

    for family in ("short", "long"):
        ids = (f"{family}.fit", f"{family}.report")
        if done(*ids):
            put(f"{family.upper()}_VALIDATION", "PASS",
                dict(units={i: q.terminal(i)["terminal_sha256"] for i in ids},
                     identities={i: plan_ids[i]["identity_sha256"] for i in ids},
                     eligibility=q.terminal(ids[1])["terminal"].get("eligibility", {})))
    if done("pair.validation"):
        term = q.terminal("pair.validation")
        doc = term["terminal"]
        passed = doc.get("status") == "COMPLETE" and doc.get("gates", {}).get("pair_status") == "PASS"
        put("PAIR_VALIDATION", "PASS" if passed else doc.get("status", "FAILED"),
            dict(terminal_sha256=term["terminal_sha256"], gates=doc.get("gates", {})))
        if passed:
            effective = _effective_with_long(root)
            put("PROCEDURE_FREEZE", "PASS", dict(
                short_identity=plan_ids["short.fit"]["identity_sha256"],
                long_identity=plan_ids["long.fit"]["identity_sha256"],
                pair_identity=plan_ids["pair.validation"]["identity_sha256"],
                short_family_sha256=_family_sha(effective, "short"), long_family_sha256=_family_sha(effective, "long"),
                strategy_version=doc.get("gates", {}).get("strategy_version"),
                costs_sha256=doc.get("gates", {}).get("costs_sha256"), seed=effective["seed"],
                cadence=effective["cadence"], final_fit_policy=dict(train=FINAL_REFIT_TRAIN, inner_tail_weeks=8,
                                                                   weights_frozen_through_test=True)))
    return written


def pair_eligible(terminal):
    """True only when the pair unit declares PASS; anything else disqualifies the strategy."""
    doc = (terminal or {}).get("terminal") or {}
    if doc.get("status") != "COMPLETE":
        return False
    gates = doc.get("gates") or {}
    return gates.get("pair_status") == "PASS"


def _gate_strategy_on_pair(root, q):
    """Queue the validation strategy only on a qualified pair; otherwise close it PAIR_NOT_ELIGIBLE."""
    row = q.unit("strategy.validation")
    if row is None or row["state"] in ("COMPLETE", "FAILED", "RUNNING"):
        return
    pair = q.terminal("pair.eligibility")
    if pair is None:
        return
    if pair_eligible(pair):
        if row["state"] in ("WAIT_DEPENDENCY", "BLOCKED"):
            q.set_state("strategy.validation", "QUEUED")
        return
    doc = (pair["terminal"].get("gates") or {})
    cause = doc.get("pair_status") or pair["terminal"].get("status") or "PAIR_NOT_ELIGIBLE"
    if row["state"] != "BLOCKED" or row["disposition"] != "PAIR_NOT_ELIGIBLE":
        q.set_state("strategy.validation", "BLOCKED", f"PAIR_NOT_ELIGIBLE:{cause}",
                    disposition="PAIR_NOT_ELIGIBLE")
        q.event("PAIR_NOT_ELIGIBLE", "strategy.validation", dict(pair_cause=cause))


def enumerate_test_units(root):
    """Public entry: refused outright for a market-time pair campaign (no TEST stage exists)."""
    if mode_of(load_effective(root)) == MODE_MARKET_TIME:
        raise Refused("TEST_NOT_ENUMERABLE_IN_MARKET_TIME_PAIR: this development campaign has no TEST stage")
    plan = load_plan(root)
    ready = readiness(root)
    if ready["state"] != "READY":
        raise Refused(f"TEST_READINESS_NOT_READY: {ready['state']}")
    return _enumerate_finals(root, plan, ready)


def readiness(root):
    plan = load_plan(root)
    q = open_queue(root)
    try:
        units = {u["unit_id"]: u for u in q.units()}
        validation = [u for u in plan["units"] if u["kind"] in ("fit", "report", "pair")]
        for unit in validation:
            row = units[unit["unit_id"]]
            if row["state"] == "WAIT_DESIGN_RULING":
                return dict(state="BLOCKED", disposition="DESIGN_RULING_REQUIRED",
                            missing=[dict(gate="DESIGN_RULING", artifact="state/DESIGN_BLOCK.json",
                                          owner="plan owner (Master work plan)", reason=row["cause"])])
            if row["state"] in ("FAILED", "BLOCKED"):
                return dict(state="TERMINAL", disposition=f"UNIT_{row['state']}:{unit['unit_id']}",
                            cause=row["cause"], missing=[])
        pair = q.terminal("pair.validation")
        if pair is not None and pair["terminal"].get("status") != "COMPLETE":
            return dict(state="TERMINAL", disposition=pair["terminal"].get("status"), missing=[],
                        cause="pair validation terminal is not eligible for TEST")
        if pair is not None and pair["terminal"].get("gates", {}).get("pair_status") != "PASS":
            return dict(state="TERMINAL", disposition=pair["terminal"].get("gates", {}).get("pair_status")
                        or "PAIR_NOT_ELIGIBLE", missing=[], cause="pair validation did not pass")
    finally:
        q.close()
    if mode_of(load_effective(root)) == MODE_MARKET_TIME:
        q = open_queue(root)
        try:
            strategy = q.unit("strategy.validation")
            pair = q.terminal("pair.eligibility")
        finally:
            q.close()
        if strategy is not None and strategy["disposition"] == "PAIR_NOT_ELIGIBLE":
            return dict(state="TERMINAL", disposition="PAIR_NOT_ELIGIBLE", missing=[],
                        cause=strategy["cause"])
        if pair is not None and strategy is not None and strategy["state"] == "COMPLETE":
            return dict(state="TERMINAL", disposition="MARKET_TIME_PAIR_VALIDATION_COMPLETE", missing=[],
                        cause="development campaign; TEST is not part of this graph")
        return dict(state="IN_PROGRESS", disposition=None, missing=[],
                    cause="market-time pair validation is still advancing")
    missing = []
    for name, (owner, artifact, _auto) in GATES.items():
        problem = _gate_problem(root, name, plan)
        if problem:
            missing.append(dict(gate=name, artifact=artifact, owner=owner, reason=problem))
    return dict(state="READY" if not missing else "BLOCKED", disposition=None, missing=missing)


def _enumerate_finals(root, plan, ready):
    if plan["test_units_enumerated"]:
        return False
    effective = _effective_with_long(root)
    receipts = {name: json.loads((_paths(root)["gates"] / f"{name}.json").read_text())["receipt_sha256"]
                for name in GATES}
    readiness_sha = sha256_json(receipts)
    atomic_json(root / "gates" / "TEST_READINESS.json", dict(schema="i6dmo.test_readiness.v1", state="READY",
                                                             receipts=receipts, readiness_sha256=readiness_sha))
    test = effective["design"]["splits"]["test"]
    finals = []
    for family in ("short", "long"):
        unit = _base(f"final.{family}_refit", family, "final_refit", STAGES[5], "gpu", [],
                     dict(train=FINAL_REFIT_TRAIN, inner_tail_weeks=effective["design"]["splits"]["inner_tail_weeks"]),
                     calls_fit=True)
        finals.append(_identify(unit, effective, _family_sha(effective, family), [readiness_sha]))
    pair = _base("final.pair_test", "pair", "final_test", STAGES[5], "cpu",
                 ["final.short_refit", "final.long_refit"], dict(test=[test["start"], test["end"]]), calls_fit=False,
                 test_open=True)
    finals.append(_identify(pair, effective, _family_sha(effective, "pair"),
                            [readiness_sha] + [u["identity_sha256"] for u in finals]))
    q = open_queue(root)
    try:
        for unit in finals:
            q.ensure_unit(unit["unit_id"], unit, state="QUEUED" if not unit["depends_on"] else "WAIT_DEPENDENCY")
            q.event("TEST_OPENED_BY_READINESS", unit["unit_id"], dict(readiness_sha256=readiness_sha))
    finally:
        q.close()
    plan["units"].extend(finals)
    plan["test_units_enumerated"] = True
    plan["expected"]["final_units"] = len(finals)
    _write_plan(root, plan)
    return True


# ---------------------------------------------------------------------- tick
def _default_long_sealer(short_design_path, short_terminal_path, out_path):
    from tools.i6dmo import design as model_design  # model lane
    try:
        return model_design.seal_long_design(short_design_path, short_terminal_path, out_path)
    except model_design.DesignError as exc:
        return dict(status="FAILED", code=exc.code, detail=str(exc)[:300])


LONG_SEALER = _default_long_sealer
SHORT_CLOSURE_DISPOSITIONS = ("COMPLETE", "NOT_ELIGIBLE")


def _auto_seal_long(root, adapter=None):
    """After short closure (short.report COMPLETE or NOT_ELIGIBLE) derive, seal and apply the long design."""
    root = Path(root)
    effective = _effective_with_long(root)
    if effective.get("long") is not None:
        return None
    q = open_queue(root)
    try:
        row = q.unit("short.report")
        term = q.terminal("short.report")
        if row is None or row["state"] != "COMPLETE" or term is None or \
                term["terminal"].get("status") not in SHORT_CLOSURE_DISPOSITIONS:
            return None
        child = root / "terminals" / term["identity_sha256"] / "CHILD_TERMINAL.json"
        designs = root / "designs"
        designs.mkdir(exist_ok=True)
        short_path = designs / "EXPERIMENT.short.json"
        if not short_path.exists():
            short_path.write_text(json.dumps(effective["design"], sort_keys=True, indent=2) + "\n")
        result = LONG_SEALER(short_path, child, designs / "EXPERIMENT.long.json")
        if result.get("status") != "SEALED":
            cause = f"LONG_DESIGN_{result.get('code')}: owner=model lane (seal-long-design)"
            for unit_id in ("long.fit", "long.report", "pair.validation"):
                unit = q.unit(unit_id)
                if unit and unit["identity_sha256"] is None and unit["cause"] != cause:
                    q.set_state(unit_id, "WAIT_ARTIFACT", cause)
            return result
        q.event("LONG_DESIGN_DERIVED", "long.fit", dict(design_sha256=result.get("design_sha256")))
    finally:
        q.close()
    seal_long(root, designs / "EXPERIMENT.long.json", adapter=adapter)
    return result


LONG_EMPTY_SUPPORT_CAUSE = (
    "WAIT_DESIGN_RULING: LONG_EXACT_EMPTY_SUPPORT - the exact long bundle has 0 rows in every split; "
    "every 144h/12h ladder crosses a >=47h weekend closure, so the sealed long design cannot fit under "
    "exact sampling; plan-owner ruling required (ASOF arm under its own identity, or a market-time "
    "window redesign)")
LONG_CHAIN = ("long.fit", "long.report", "pair.validation")


def record_design_block(root, unit_ids, cause):
    """A sealed-design impossibility reported to the plan owner: the units hold, nothing else stops."""
    if not str(cause).startswith("WAIT_DESIGN_RULING:"):
        raise Refused("DESIGN_BLOCK_CAUSE_UNTYPED: the cause must start with WAIT_DESIGN_RULING:")
    doc = dict(schema="i6dmo.design_block.v1", units=sorted(unit_ids), cause=str(cause), recorded_at=time.time())
    atomic_json(Path(root) / "state" / "DESIGN_BLOCK.json", doc)
    _apply_design_block(root)
    return doc


def _design_block(root):
    path = Path(root) / "state" / "DESIGN_BLOCK.json"
    return json.loads(path.read_text()) if path.exists() else None


def _apply_design_block(root):
    block = _design_block(root)
    if block is None:
        return []
    held = []
    q = open_queue(root)
    try:
        for unit_id in block["units"]:
            row = q.unit(unit_id)
            if row is None or row["state"] in ("COMPLETE", "FAILED"):
                continue
            if row["state"] != "WAIT_DESIGN_RULING" or row["cause"] != block["cause"]:
                q.set_state(unit_id, "WAIT_DESIGN_RULING", block["cause"])
            held.append(unit_id)
    finally:
        q.close()
    return held


def tick(root, now=None, adapter=None):
    """One finite, atomic advance. Never waits, never trains, never reads a token."""
    root = Path(root)
    load_effective(root)
    effective = load_effective(root)
    market_time = mode_of(effective) == MODE_MARKET_TIME
    design_block = _design_block(root)
    held = set((design_block or {}).get("units", ()))
    if not held and not market_time:
        _auto_seal_long(root, adapter=adapter)
    _apply_long(root)
    plan = load_plan(root)
    q = open_queue(root)
    activated, blocked = [], []
    try:
        rows = {u["unit_id"]: u for u in q.units()}
        for unit in plan["units"]:
            row = rows[unit["unit_id"]]
            if unit["unit_id"] in held:
                continue
            if row["state"] != "WAIT_DEPENDENCY":
                continue
            deps = [rows[d] for d in unit["depends_on"]]
            bad = [d for d in deps if d["state"] in ("FAILED", "BLOCKED")]
            if bad:
                q.set_state(unit["unit_id"], "BLOCKED", f"UPSTREAM_{bad[0]['unit_id']}_{bad[0]['state']}")
                blocked.append(unit["unit_id"])
            elif all(d["state"] == "COMPLETE" for d in deps):
                q.set_state(unit["unit_id"], "QUEUED")
                activated.append(unit["unit_id"])
        for unit in plan["units"]:
            if rows[unit["unit_id"]]["state"] == "WAIT_ARTIFACT" and rows["short.report"]["state"] == "COMPLETE" \
                    and not rows[unit["unit_id"]]["cause"] and unit["identity_sha256"] is None:
                q.set_state(unit["unit_id"], "WAIT_ARTIFACT",
                            "LONG_DESIGN_NOT_SEALED: run `seal-long` with the sealed long design (model lane)")
        gates = _auto_gates(root, plan, q)
        if market_time:
            _gate_strategy_on_pair(root, q)
    finally:
        q.close()
    _apply_design_block(root)
    _delivery_gate(root, plan)
    ready = readiness(root)
    opened = (not market_time) and ready["state"] == "READY" and _enumerate_finals(root, load_plan(root), ready)
    status = write_status(root, now=now)
    return dict(activated=activated, blocked=blocked, gates_written=gates, readiness=ready, finals_enumerated=opened,
                campaign_state=status["campaign"]["state"])


def _delivery_gate(root, plan):
    path = _paths(root)["gates"] / "VALIDATION_DELIVERY.json"
    if path.exists() or not _paths(root)["outbox"].exists():
        return
    q = open_queue(root)
    try:
        validation = [u["unit_id"] for u in plan["units"] if u["kind"] in ("fit", "report", "pair")]
        if not all(q.unit(u)["state"] == "COMPLETE" for u in validation):
            return
    finally:
        q.close()
    from tools.i6dmo.outbox import Outbox
    box = Outbox.open_readonly(_paths(root)["outbox"])
    try:
        summary = box.summary()
    finally:
        box.close()
    mirrors = Path(root) / "MIRRORS.json"
    mirror_ok = mirrors.exists() and json.loads(mirrors.read_text()).get("state") == "VERIFIED"
    if summary["expected"] and summary["delivered"] == summary["expected"] and mirror_ok:
        record_gate(root, "VALIDATION_DELIVERY", status="PASS",
                    evidence=dict(delivered=summary["delivered"], mirrors_sha256=file_sha256(mirrors)),
                    splits_read=["train", "validation"])


# ------------------------------------------------------------------- publish
def publish(root, *, warehouse=None, mirror=True, run=None):
    """Bounded delivery pass (outbox + exact readback) and physical mirrors. Never trains."""
    from tools.i6dmo import mirror as M
    from tools.i6dmo.outbox import HttpWarehouse, Outbox
    root = Path(root)
    spec = load_deployment(root)
    if warehouse is None:
        wh = spec["warehouse"]
        warehouse = HttpWarehouse(wh["url"], wh["token_env"])
    box = Outbox(_paths(root)["outbox"])
    try:
        summary = box.pump(warehouse)
        reports = [json.loads(i["body"]) for i in box.items()]
    finally:
        box.close()
    atomic_json(root / "REPORTS.json", reports)
    state = "DELIVERED" if summary["expected"] and summary["delivered"] == summary["expected"] else (
        "MISMATCH" if summary["mismatch"] else ("NO_ITEMS" if not summary["expected"] else "PENDING"))
    publication = dict(schema="i6dmo.publication.v1", state=state, test_read=False,
                       **{k: v for k, v in summary.items() if k != "delivery_seconds"})
    atomic_json(root / "PUBLICATION.json", publication)
    mirrors = M.sync_all(root, spec, run=run) if mirror else None
    write_status(root)
    return dict(publication=publication, mirrors=mirrors)


# --------------------------------------------------------------------- close
def close_campaign(root):
    root = Path(root)
    plan = load_plan(root)
    missing = []
    ready = readiness(root)
    q = open_queue(root)
    try:
        rows = {u["unit_id"]: u for u in q.units()}
        validation = [u["unit_id"] for u in plan["units"] if u["kind"] in ("fit", "report", "pair")]
        for unit_id in validation:
            if rows[unit_id]["state"] not in ("COMPLETE", "FAILED", "BLOCKED"):
                missing.append(dict(item=unit_id, reason=f"NOT_TERMINAL_{rows[unit_id]['state']}"))
        if ready["state"] == "READY":
            if not plan["test_units_enumerated"]:
                missing.append(dict(item="final units", reason="NOT_ENUMERATED"))
            for unit in plan["units"]:
                if unit["kind"] in ("final_refit", "final_test") and rows[unit["unit_id"]]["state"] != "COMPLETE":
                    missing.append(dict(item=unit["unit_id"], reason=f"NOT_COMPLETE_{rows[unit['unit_id']]['state']}"))
        elif ready["state"] == "BLOCKED":
            missing.extend(dict(item=m["gate"], reason=m["reason"], owner=m["owner"]) for m in ready["missing"])
        terminals = {t["unit_id"]: t for t in q.terminals()}
    finally:
        q.close()
    for unit_id, term in terminals.items():
        stored = Path(root) / "terminals" / (term["identity_sha256"] or "none") / "TERMINAL.json"
        if term["identity_sha256"] and not stored.exists():
            missing.append(dict(item=unit_id, reason="TERMINAL_ARTIFACT_MISSING"))
    pub = root / "PUBLICATION.json"
    if not pub.exists() or json.loads(pub.read_text()).get("state") != "DELIVERED":
        missing.append(dict(item="PUBLICATION.json", reason="DELIVERY_NOT_VERIFIED"))
    mirrors = root / "MIRRORS.json"
    if not mirrors.exists() or json.loads(mirrors.read_text()).get("state") != "VERIFIED":
        missing.append(dict(item="MIRRORS.json", reason="MIRRORS_NOT_VERIFIED"))
    if missing:
        return dict(state="REFUSED", missing=missing, readiness=ready)
    doc = dict(schema="i6dmo.close.v1", state="CLOSED", campaign_id=plan["campaign_id"], readiness=ready,
               disposition=ready.get("disposition") or "FINAL_PAIR_TEST_COMPLETE",
               terminals={k: v["terminal_sha256"] for k, v in sorted(terminals.items())},
               publication_sha256=file_sha256(pub), mirrors_sha256=file_sha256(mirrors), closed_at=time.time())
    atomic_json(_paths(root)["close"], doc)
    write_status(root)
    return doc


def events(root, limit=500):
    q = open_queue(root)
    try:
        return q.events(limit)
    finally:
        q.close()


def write_status(root, now=None):
    from tools.i6dmo import status as S
    doc = S.compute(root, level="all", now=now)
    atomic_json(_paths(root)["status"], doc)
    return doc


def canonical_text(value):
    return canonical(value)
