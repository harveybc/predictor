"""Finite downsampled R0/R2 sequence: authenticated short recovery, then long.

An outer timer advances only after the preceding campaign's verified CLOSE.
Workers, scientific gates, OLAP publication and mirrors remain the existing
I6-DMO mechanisms. This module never trains or reads a credential itself.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import time

from tools.i6dmo import campaign as C, matched_runner as MR, raw_long_campaign as L
from tools.i6n_swarm.common import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs/i6n/direct_multi_output_static"
RELEASE = DOCS / "r2_downsampled/RELEASE_ACCEPTANCE.json"
FIELDS = {"schema", "work_root", "source_deployment", "short_source_campaign", "short_source_pins",
          "long_source_work_root", "input_dir"}
SCIENTIFIC_CLOSES = {"R0_RETAINED", "R2_SELECTED"}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS \
            or config.get("schema") != "i6dmo.r2_downsampled_sequence.v1":
        raise ValueError("SEQUENCE_CONFIG: exact explicit fields required")
    for key in FIELDS - {"schema", "short_source_pins"}:
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError(f"SEQUENCE_CONFIG: absolute {key} required")
    root = Path(config["work_root"]).resolve()
    if any(Path(config[k]).resolve().is_relative_to(root) for k in
           FIELDS - {"schema", "short_source_pins", "work_root"}):
        raise ValueError("SEQUENCE_CONFIG: source inside mutable sequence")
    pins = config["short_source_pins"]
    if not isinstance(pins, dict) or set(pins) != {"effective_sha256", "plan_sha256", "code_commit"}:
        raise ValueError("SEQUENCE_CONFIG: original source pins required")
    return dict(config)


def next_action(short, long):
    """A scientific negative advances; technical failure never becomes a result."""
    if any(item and item.get("state") == "ERROR" for item in (short, long)):
        return "ERROR"
    for item in (short, long):
        if item and (item.get("failed", 0) or item.get("state") in ("FAILED", "TECHNICAL_TERMINAL")
                     or item.get("state") == "CLOSED" and item.get("disposition") not in SCIENTIFIC_CLOSES):
            return "TECHNICAL_TERMINAL"
    if short is None:
        return "ERROR" if long is not None else "DEPLOY_SHORT"
    if short["state"] != "CLOSED":
        return "WAIT_SHORT"
    if long is None:
        return "DEPLOY_LONG"
    return "CLOSED" if long["state"] == "CLOSED" else "WAIT_LONG"


def campaign_status(root):
    from tools.i6dmo import status as S
    doc = S.compute(root, level="campaign")
    if doc.get("state") == "CLOSED":
        close = C.closed_doc(root)
        problems, _, _ = C.closure_problems(root)
        if close is None or problems:
            raise ValueError(f"SEQUENCE_CLOSE_UNVERIFIED:{problems}")
    return dict(doc, failed=doc.get("counts", {}).get("FAILED", 0) + doc.get("counts", {}).get("BLOCKED", 0),
                disposition=(doc.get("close") or {}).get("disposition"))


def status(root, *, use_journal=True):
    root = Path(root)
    families = {}
    for family in ("short", "long"):
        campaign = root / family / "campaign"
        try:
            if campaign.exists() and not (campaign / "PLAN.json").is_file():
                raise ValueError("SEQUENCE_INCOMPLETE_CAMPAIGN")
            if (root / family / "campaign.building").exists():
                raise ValueError("SEQUENCE_INITIALIZATION_INTERRUPTED")
            families[family] = campaign_status(campaign) if campaign.exists() else None
        except Exception as exc:
            families[family] = dict(state="ERROR", error_type=type(exc).__name__, error=str(exc),
                                    eta=dict(kind="UNMEASURED"))
    journal = root / "AUTOMATION_STATUS.json"
    try:
        retained = json.loads(journal.read_text()) if journal.exists() else None
    except (OSError, ValueError) as exc:
        retained = dict(action="ERROR", error=str(exc), error_type=type(exc).__name__)
    action = next_action(families["short"], families["long"])
    if use_journal and retained and retained.get("action") == "ERROR":
        action = "ERROR"
    return dict(schema="i6dmo.r2_sequence_status.v1", families=families,
                action=action, test_open=False, launcher_journal=retained)


def release():
    doc = json.loads(RELEASE.read_text())
    if doc.get("schema") != "i6dmo.r2_downsampled_release.v1" or doc.get("verdict") != "ACCEPT_FOR_EXECUTION":
        raise ValueError("SEQUENCE_RELEASE_NOT_ACCEPTED")
    required = {"campaign", "worker", "unit_runner", "donor_transport", "r2_recovery", "r2_runner",
                "r2_design", "model", "reporting", "raw_long_campaign", "r2_downsampled_campaign"}
    files = doc.get("code_files") or {}
    if not {f"tools/i6dmo/{name}.py" for name in required}.issubset(files):
        raise ValueError("SEQUENCE_RELEASE_SCOPE")
    for rel, digest in files.items():
        path = (REPO / rel).resolve()
        if not path.is_relative_to(REPO) or file_sha256(path) != digest:
            raise ValueError(f"SEQUENCE_REVIEW_BYTES_CHANGED:{rel}")
    return doc


def _remote_files(role, root):
    script = ("import pathlib,hashlib,json; "
              f"r=pathlib.Path({str(root)!r}); "
              "print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() "
              "for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()}))")
    return json.loads(L._remote(role, ["python3", "-c", script], 180))


def capture_remote_mirror(config):
    """Retain a byte-verified local view of the existing second-host custody."""
    root = Path(config["work_root"]) / "recovery"
    anchor = json.loads((DOCS / "r2_downsampled/SOURCE_PINS.json").read_text())
    effective = C.load_effective(config["short_source_campaign"])
    plan = C.load_plan(config["short_source_campaign"])
    if config["short_source_pins"] != anchor["pins"] or plan["campaign_id"] != anchor["source_campaign"] \
            or effective["seal_sha256"] != anchor["effective_seal_sha256"] \
            or C.sha256_json(effective) != anchor["published_effective_config_sha256"] \
            or plan["effective_config_sha256"] != anchor["published_effective_config_sha256"]:
        raise ValueError("RECOVERY_PUBLISHED_SOURCE_ANCHOR_MISMATCH")
    source = json.loads((Path(config["short_source_campaign"]) / "private/deployment.json").read_text())
    role_name = source["mirrors"]["second_host_role"]
    role = source["roles"][role_name]
    remote = source["mirrors"]["second_host_path"]
    view = root / "remote_mirror"
    proof = root / "REMOTE_CUSTODY.json"
    if proof.exists():
        retained = json.loads(proof.read_text())
        if retained.get("source_pins") != config["short_source_pins"]:
            raise ValueError("REMOTE_CUSTODY_SOURCE_CHANGED")
        for rel, digest in retained["files"].items():
            if file_sha256(view / rel) != digest:
                raise ValueError("REMOTE_CUSTODY_LOCAL_VIEW_CHANGED")
        if _remote_files(role, remote) != retained["files"]:
            raise ValueError("REMOTE_CUSTODY_ORIGINAL_CHANGED")
    else:
        before = _remote_files(role, remote)
        if not before:
            raise ValueError("REMOTE_CUSTODY_EMPTY")
        view.mkdir(parents=True, exist_ok=True)
        L._run(["rsync", "-a", "--checksum", f"{role['ssh_alias']}:{remote}/", str(view) + "/"], 600)
        after = _remote_files(role, remote)
        if before != after or any(file_sha256(view / rel) != sha for rel, sha in before.items()):
            raise ValueError("REMOTE_CUSTODY_TRANSPORT_MISMATCH")
        atomic_json(proof, dict(schema="i6dmo.remote_custody_view.v1", role=role_name,
            source_pins=config["short_source_pins"], files=before,
            verification="REMOTE_SHA_BEFORE_AND_AFTER_COPY_EQUALS_LOCAL_BYTES"), mode=0o600)
    return [source["mirrors"]["local"], str(view)]


def recovery_target(contract, population_dir, input_dir):
    """Derive R0 reuse requirements from current FIT data/code, not donor assertions."""
    from tools.i6dmo import train as T, design as D
    design = D.load_design(DOCS / "matched_short/EXPERIMENT.mg2m.json")
    effective = D.effective_config(design)
    pop = T.load_matched_population(population_dir, contract["shared"]["population_sha256"],
                                   effective["horizons_hours"], effective["feature_manifest"]["features"], input_dir)
    fit = T.common_split_arrays(T.arm_split(pop, design, "FIT"), pop.splits["FIT"], effective, "fit")
    scales, _ = T.matched_scales(pop, effective, fit)
    return dict(contract=contract, code_root=str(REPO),
        expected_code_files={p: file_sha256(REPO / p) for p in T.CODE_FILES},
        expected_scales_sha256=scales["scales_sha256"], current_monitor_identity=dict(
            procedure=T.MONITOR_EVAL, batch_size=effective["training"]["batch_size"],
            partition=T.MONITOR_PARTITION, monitor_instance=T.MONITOR_INSTANCE,
            input_materialization=T.MONITOR_INPUT, monitor_forward=T.monitor_forward_identity()))


def _source(config, family):
    if family == "short":
        root = Path(config["short_source_campaign"])
        source_dir = DOCS / "matched_short"
        pop_seal = source_dir / "POPULATION_SEAL.json"
        effective = C.load_effective(root)
        return source_dir, pop_seal, effective["data_pins"]["common_population"]
    work = Path(config["long_source_work_root"])
    effective = C.load_effective(work / "campaign")
    if C.closed_doc(work / "campaign") is None:
        raise ValueError("RAW_LONG_SOURCE_NOT_CLOSED")
    return DOCS / "r2_downsampled/long_source", work / "POPULATION_SEAL.json", \
        effective["data_pins"]["common_population"]


def _prepare_family(config, family, code, reviewed_files):
    """Prepare only the fixed existing population; never regenerate or refit data."""
    from tools.i6dmo import r2_design as RD, r2_recovery as RR
    work = Path(config["work_root"]) / family
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    spec = copy.deepcopy(json.loads(Path(config["source_deployment"]).read_text()))
    source_dir, seal_path, pop_pin = _source(config, family)
    storage = Path(spec["roles"]["coordinator"]["experiment_storage_root"])
    population_dir = storage / pop_pin["relpath"]
    arm = "mg2m" if family == "short" else "mg12m"
    source_design = source_dir / f"EXPERIMENT.{arm}.json"
    contract_path = DOCS / "matched_short/R2_CONTRAST_DESIGN.ruled.json" if family == "short" \
        else DOCS / "r2_downsampled/R2_CONTRAST_DESIGN.long.ruled.json"
    contract = json.loads(contract_path.read_text())
    RD.validate_r2_design(contract, for_execution=True)
    spec["launcher"]["initial_cap_bytes"] = 5 * 1024 ** 3
    spec["timer_cadence"] = dict(worker_seconds=5, cpu_worker_seconds=10, tick_seconds=10,
                                  publish_seconds=60, boot_seconds=30, accuracy_seconds=1)
    for setting in spec["roles"].values():
        setting.setdefault("environment", {}).update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
            TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1")
    prepared = dict(population_dir=str(population_dir), population_relpath=pop_pin["relpath"],
                    launch_config=dict(input_dir=config["input_dir"]))
    spec = L._provision(spec, prepared, code, reviewed_files=reviewed_files)
    mirror_name = Path(config["work_root"]).name + "-" + family
    spec["mirrors"]["local"] = str(storage / "i6dmo-mirrors" / mirror_name)
    role = spec["roles"][spec["mirrors"]["second_host_role"]]
    spec["mirrors"]["second_host_path"] = str(Path(role["experiment_storage_root"]) / "i6dmo-mirrors" / mirror_name)
    L._remote(role, ["mkdir", "-p", spec["mirrors"]["second_host_path"]], 30)
    deployment = work / "deployment.json"
    atomic_json(deployment, spec, mode=0o600)
    designs = {a: str(DOCS / ("matched_short" if family == "short" else "matched_long") /
                       f"EXPERIMENT.{a}.json") for a in (MR.ARMS if family == "short" else MR.LONG_ARMS)}
    MR.prepare(deployment=deployment, input_dir=config["input_dir"], out=work / "prepared.json",
        designs=designs, population_seal=seal_path, population_dir=population_dir, long=family == "long")
    check = MR.preflight(work / "prepared.json", work / "preflight")
    if check["state"] != "PASS":
        return dict(state="WAIT_RESOURCE", launch_blockers=check["launch_blockers"])
    recovery = None
    if family == "short":
        mirrors = capture_remote_mirror(config)
        target = recovery_target(contract, population_dir, config["input_dir"])
        manifest = RR.build_manifest(config["short_source_campaign"], mirror_roots=mirrors,
                                     source_pins=config["short_source_pins"], r0_target=target)
        path = work / "RECOVERY.json"
        atomic_json(path, manifest, mode=0o600)
        proof = Path(config["work_root"]) / "recovery/REMOTE_CUSTODY.json"
        recovery = dict(path=str(path), sha256=file_sha256(path), r0_target=target,
                        physical_custody=dict(path=str(proof), sha256=file_sha256(proof)))
    return dict(state="PREPARED", contract=str(contract_path), design=str(source_design),
                deployment=str(deployment), prepared=str(work / "prepared.json"), recovery=recovery)


def _setup_identity(config, reviewed):
    if L._run(["git", "-C", str(REPO), "status", "--porcelain"]).strip():
        raise ValueError("SEQUENCE_SOURCE_NOT_FROZEN")
    source = Path(config["short_source_campaign"])
    pins = config["short_source_pins"]
    if file_sha256(source / "PLAN.json") != pins["plan_sha256"] or \
            file_sha256(source / "EFFECTIVE_CONFIG.json") != pins["effective_sha256"] or \
            C.load_effective(source)["code_commit"] != pins["code_commit"]:
        raise ValueError("SEQUENCE_SOURCE_PINS_CHANGED")
    long = Path(config["long_source_work_root"])
    if C.closed_doc(long / "campaign") is None:
        raise ValueError("RAW_LONG_SOURCE_NOT_CLOSED")
    paths = [Path(config["source_deployment"]), source / "PLAN.json", source / "EFFECTIVE_CONFIG.json",
        long / "campaign/PLAN.json", long / "campaign/EFFECTIVE_CONFIG.json", long / "campaign/CLOSE.json",
        long / "POPULATION_SEAL.json", DOCS / "matched_short/POPULATION_SEAL.json",
        DOCS / "matched_short/EXPERIMENT.mg2m.json",
        DOCS / "matched_short/R2_CONTRAST_DESIGN.ruled.json",
        DOCS / "r2_downsampled/long_source/EXPERIMENT.mg12m.json",
        DOCS / "r2_downsampled/R2_CONTRAST_DESIGN.long.ruled.json",
        DOCS / "r2_downsampled/SOURCE_PINS.json"]
    return dict(schema="i6dmo.r2_sequence_setup.v2", config=config, code_commit=C._git_head(),
        release_sha256=file_sha256(RELEASE), code_files=reviewed["code_files"],
        deployment_source_sha256=file_sha256(config["source_deployment"]),
        source_files={str(p): file_sha256(p) for p in paths})


def _binding(prepared, family, code, files):
    paths = {k: prepared[k] for k in ("contract", "design", "deployment", "prepared")}
    if prepared["recovery"]:
        paths["recovery"] = prepared["recovery"]["path"]
        paths["physical_custody"] = prepared["recovery"]["physical_custody"]["path"]
    return dict(schema="i6dmo.r2_sequence_initialization.v1", family=family, code_commit=code,
        code_files=files, prepared=prepared, files={k: dict(path=str(p), sha256=file_sha256(p))
                                                  for k, p in paths.items()})


def _verify_campaign(root, binding, *, initialized=True):
    for pin in binding["files"].values():
        if file_sha256(pin["path"]) != pin["sha256"]:
            raise ValueError(f"SEQUENCE_INITIALIZATION_INPUT_CHANGED:{pin['path']}")
    prepared = binding["prepared"]
    contract = json.loads(Path(prepared["contract"]).read_text())
    design = json.loads(Path(prepared["design"]).read_text())
    spec = json.loads(Path(prepared["deployment"]).read_text())
    pins = json.loads(Path(prepared["prepared"]).read_text())["data_pins"]
    effective, plan = C.load_effective(root), C.load_plan(root)
    expected = dict(code_commit=binding["code_commit"], mode=C.MODE_R2, design=design, data_pins=pins,
        data_sha256=C.sha256_json(pins), deployment_sha256=C.sha256_json(spec),
        fit_device="gpu", engineering_cpu_fixture=False, test_open=False, prior_roots=[],
        entrypoints=C.R2_ENTRYPOINTS)
    if any(effective.get(k) != v for k, v in expected.items()) or C.load_deployment(root) != spec:
        raise ValueError("SEQUENCE_EFFECTIVE_BINDING_CHANGED")
    r2 = effective["r2"]
    reuse = "FRESH_R0_REQUIRED"
    expected_r2 = dict(contract_file_sha256=file_sha256(prepared["contract"]),
        contract_seal_sha256=contract["seal_sha256"], objective=contract["pretraining"]["objective"],
        strict_reports=True)
    recovery = prepared["recovery"]
    if recovery:
        manifest = json.loads(Path(recovery["path"]).read_text())
        if file_sha256(recovery["path"]) != recovery["sha256"]:
            raise ValueError("SEQUENCE_RECOVERY_CHANGED")
        physical = recovery["physical_custody"]
        if file_sha256(physical["path"]) != physical["sha256"]:
            raise ValueError("SEQUENCE_PHYSICAL_CUSTODY_CHANGED")
        reuse = manifest["r0_reuse"]["decision"]
        expected_r2.update(recovered_unit_specs=manifest["unit_specs"],
            recovered_donor_provenance=manifest["donor_provenance"], donor_recovery=dict(
                manifest_sha256=manifest["manifest_sha256"], file_sha256=recovery["sha256"],
                source_campaign=manifest["source"]["campaign_id"],
                physical_custody_sha256=physical["sha256"],
                original_donors={f: dict(identity_sha256=ref["identity_sha256"],
                    stored_terminal_sha256=ref["sha256"],
                    donor_terminal_sha256=ref["donor_terminal"]["sha256"],
                    donor_weights_sha256=ref["donor_npz"]["sha256"])
                                 for f, ref in manifest["pretraining_refs"].items()}))
    elif any(r2.get(k) for k in ("donor_recovery", "recovered_unit_specs", "recovered_donor_provenance")):
        raise ValueError("SEQUENCE_UNEXPECTED_RECOVERY")
    expected_r2["r0_reuse"] = reuse
    if any(r2.get(k) != v for k, v in expected_r2.items()):
        raise ValueError("SEQUENCE_RECOVERY_OR_CONTRACT_BINDING_CHANGED")
    if plan.get("effective_config_sha256") != C.sha256_json(effective) or \
            plan.get("effective_config_file_sha256") != file_sha256(root / "EFFECTIVE_CONFIG.json") or \
            plan.get("mode") != C.MODE_R2 or plan.get("test_units_enumerated") is not False:
        raise ValueError("SEQUENCE_PLAN_BINDING_CHANGED")
    from tools.i6dmo.queue import Queue
    q = Queue.open_readonly(root / "state/queue.sqlite3")
    try:
        rows = q.units()
        if {r["unit_id"]: r["spec"] for r in rows} != {u["unit_id"]: u for u in plan["units"]}:
            raise ValueError("SEQUENCE_QUEUE_PLAN_MISMATCH")
        for row in rows:
            if row["spec"].get("test_open") is not False:
                raise ValueError("SEQUENCE_TEST_UNIT_REFUSED")
            if row["spec"].get("execution_policy") == "ADOPT_ONLY":
                terminal = q.terminal(row["unit_id"])
                if not terminal or terminal["source"] != "ADOPTED" or \
                        terminal["identity_sha256"] != row["identity_sha256"]:
                    raise ValueError("SEQUENCE_RECOVERY_ADOPTION_INCOMPLETE")
    finally:
        q.close()
    marker = dict(schema="i6dmo.r2_sequence_initialized.v1", binding_sha256=C.sha256_json(binding),
        effective_file_sha256=file_sha256(root / "EFFECTIVE_CONFIG.json"),
        plan_file_sha256=file_sha256(root / "PLAN.json"))
    if initialized and json.loads((root / "INITIALIZED.json").read_text()) != marker:
        raise ValueError("SEQUENCE_INITIALIZED_IDENTITY_CHANGED")
    return marker


def _quarantine_building(root):
    building = root.with_name("campaign.building")
    if building.exists():
        building.rename(root.with_name(f"campaign.incomplete-{time.time_ns()}"))


def _deploy_family(config, family, code, files):
    from tools.i6dmo import deploy, r2_recovery as RR
    root = Path(config["work_root"]) / family / "campaign"
    binding_path = root.parent / "INITIALIZATION.json"
    _quarantine_building(root)
    if root.exists():
        binding = json.loads(binding_path.read_text())
        if binding["code_commit"] != code or binding["code_files"] != files:
            raise ValueError("SEQUENCE_INITIALIZATION_CODE_CHANGED")
        _verify_campaign(root, binding)
    else:
        prepared = _prepare_family(config, family, code, files)
        if prepared["state"] != "PREPARED":
            return prepared
        binding = _binding(prepared, family, code, files)
        atomic_json(binding_path, binding, mode=0o600)
        building = root.with_name("campaign.building")
        pins = json.loads(Path(prepared["prepared"]).read_text())["data_pins"]
        caps = {k: dict(cap_bytes=5*1024**3, basis="PROVISIONAL conservative new-contrast cap; own-class peak governs subsequent jobs")
                for k in C.R2_DEFAULT_CLASS_CAPS}
        reuse = (json.loads(Path(prepared["recovery"]["path"]).read_text())["r0_reuse"]
                 if prepared["recovery"] else dict(decision="FRESH_R0_REQUIRED", failed=["new_long_monitor_training_code"]))
        C.init_r2_contrast(prepared["contract"], prepared["design"], prepared["deployment"], building,
            code_commit=code, data_pins=pins, class_caps=caps, r0_reuse=reuse,
            donor_recovery=prepared["recovery"])
        marker = _verify_campaign(building, binding, initialized=False)
        atomic_json(building / "INITIALIZED.json", marker, mode=0o600)
        building.rename(root)
        fd = os.open(root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    if not (root / "DEPLOY.json").exists() and C.closed_doc(root) is None and not C._finalized(root):
        prepared = binding["prepared"]
        MR.validate(prepared["prepared"])
        if prepared["recovery"]:
            recovery = prepared["recovery"]
            RR.load_manifest(recovery["path"], expected_sha256=recovery["sha256"], r0_target=recovery["r0_target"])
        for role in C.load_deployment(root)["roles"].values():
            L._verify_runtime(role, role["workdir"], code, reviewed_files=files)
        deploy.deploy(root)
    return campaign_status(root)


def _coordinator_storage(spec):
    storage = spec.get("roles", {}).get("coordinator", {}).get("experiment_storage_root")
    if not isinstance(storage, str) or not Path(storage).is_absolute():
        raise ValueError("SEQUENCE_STORAGE_ROOT_REQUIRED: absolute coordinator experiment_storage_root")
    return storage


def run(config):
    config = validate_config(config)
    spec = json.loads(Path(config["source_deployment"]).read_text())
    storage = _coordinator_storage(spec)
    previous = os.environ.get("EXPERIMENT_STORAGE_ROOT")
    os.environ["EXPERIMENT_STORAGE_ROOT"] = storage
    try:
        return _run(config)
    finally:
        if previous is None:
            os.environ.pop("EXPERIMENT_STORAGE_ROOT", None)
        else:
            os.environ["EXPERIMENT_STORAGE_ROOT"] = previous


def _run(config):
    root = Path(config["work_root"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "sequence.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            reviewed = release()
            identity = _setup_identity(config, reviewed)
            setup = root / "SETUP.json"
            if setup.exists():
                if json.loads(setup.read_text()) != identity:
                    raise ValueError("SEQUENCE_IDENTITY_CHANGED")
            else:
                atomic_json(setup, identity, mode=0o600)
            code = identity["code_commit"]
            for family in ("short", "long"):
                campaign = root / family / "campaign"
                _quarantine_building(campaign)
                if campaign.exists():
                    binding = json.loads((campaign.parent / "INITIALIZATION.json").read_text())
                    if binding["code_commit"] != code or binding["code_files"] != reviewed["code_files"]:
                        raise ValueError("SEQUENCE_INITIALIZATION_CODE_CHANGED")
                    _verify_campaign(campaign, binding)
            result = status(root, use_journal=False)
            action = result["action"]
            family = (action.removeprefix("DEPLOY_").lower() if action.startswith("DEPLOY_") else
                      "short" if action == "WAIT_SHORT" else "long" if action == "WAIT_LONG" else None)
            if family and not (root / family / "campaign/DEPLOY.json").exists():
                deployed = _deploy_family(config, family, code, reviewed["code_files"])
                result = dict(status(root, use_journal=False), deployment=deployed)
            atomic_json(root / "AUTOMATION_STATUS.json", dict(schema="i6dmo.r2_sequence_journal.v1",
                action=result["action"], as_of=time.time(), code_commit=code,
                deployment=result.get("deployment")), mode=0o600)
            return result
        except Exception as exc:
            atomic_json(root / "AUTOMATION_STATUS.json", dict(action="ERROR", as_of=time.time(),
                error_type=type(exc).__name__, error=str(exc)), mode=0o600)
            raise


def install(spec_path):
    from tools.i6dmo import deploy
    path = Path(spec_path).resolve()
    config = validate_config(json.loads(path.read_text()))
    release()
    spec = json.loads(Path(config["source_deployment"]).read_text())
    storage_environment = json.dumps("EXPERIMENT_STORAGE_ROOT=" + _coordinator_storage(spec)).replace("%", "%%")
    name = "i6dmo-r2-sequence-" + C.sha256_json(config)[:12]
    unit_dir = Path.home() / ".config/systemd/user"
    marker = "X-I6DMO-Sequence=" + C.sha256_json(config)
    argv = [spec["launcher"]["crispdm_run"], "-m", "4G", "-t", "1800s", "-n", name, "--",
            spec["coordinator"]["python"], "-m", "tools.i6dmo.r2_downsampled_campaign", "run", "--spec", str(path)]
    docs = dict(service=f"[Unit]\nDescription=Finite downsampled R0/R2 continuation\n{marker}\n\n[Service]\n"
        f"Type=oneshot\nWorkingDirectory={REPO}\nEnvironment=CUDA_VISIBLE_DEVICES=\n"
        f"Environment={storage_environment}\n"
        "Environment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=OPENBLAS_NUM_THREADS=2\nEnvironment=OMP_NUM_THREADS=2\n"
        f"ExecStart={shlex.join(argv)}\nMemoryMax=5G\nMemorySwapMax=0\nTimeoutStartSec=1900\nNice=10\n",
        timer=f"[Unit]\nDescription=Short to long R2 continuation\n{marker}\n\n[Timer]\nOnBootSec=30s\n"
        f"OnUnitInactiveSec=30s\nAccuracySec=1s\nPersistent=true\nUnit={name}.service\n\n[Install]\nWantedBy=timers.target\n")
    unit_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in docs.items():
        unit = unit_dir / f"{name}.{suffix}"
        if unit.exists() and marker not in unit.read_text():
            raise ValueError("FOREIGN_SEQUENCE_UNIT")
        unit.write_text(content)
    for args in (("daemon-reload",), ("enable", "--now", name + ".timer")):
        if deploy.default_systemctl(*args)["rc"]:
            raise ValueError("SEQUENCE_INSTALL_FAILED")
    atomic_json(Path(config["work_root"]) / "LAUNCHER.json", dict(name=name), mode=0o600)
    return dict(state="ARMED", unit=name + ".timer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "install", "status"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.action == "status":
        if not args.root:
            parser.error("status requires --root")
        result = status(args.root)
    else:
        if not args.spec:
            parser.error("run/install requires --spec")
        config = validate_config(json.loads(args.spec.read_text()))
        try:
            result = install(args.spec) if args.action == "install" else run(config)
        except Exception as exc:
            atomic_json(Path(config["work_root"]) / "AUTOMATION_STATUS.json", dict(
                action="ERROR", as_of=time.time(), error_type=type(exc).__name__, error=str(exc)), mode=0o600)
            print(json.dumps(dict(state="ERROR", error_type=type(exc).__name__, error=str(exc))))
            return 2
        if args.action == "run" and result.get("action") in ("CLOSED", "TECHNICAL_TERMINAL"):
            from tools.i6dmo import deploy
            expected = "i6dmo-r2-sequence-" + C.sha256_json(config)[:12]
            deploy.default_systemctl("disable", "--now", expected + ".timer")
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
