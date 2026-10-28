"""Prepare, deploy and query the finite RAW long control without agent polling.

Private launch JSON contains paths, never token values. Existing campaigns are
queried, never reinitialised. Scientific execution requires the reviewed source
manifest; models, queues, publication and closure use the existing I6-DMO APIs.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from tools.i6dmo import campaign as C, matched_runner as MR
from tools.i6n_swarm.common import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[2]
RELEASE = REPO / "docs/i6n/direct_multi_output_static/matched_long/RELEASE_ACCEPTANCE.json"
FIELDS = {"schema", "work_root", "source_deployment", "input_dir", "short_campaign", "strategy_inputs"}
RUNTIME = dict(tensorflow="2.21.0", keras="3.13.2")


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS or config.get("schema") != "i6dmo.raw_long_launch.v1":
        raise ValueError("LAUNCH_CONFIG: exact explicit path fields required")
    if any(not isinstance(config[k], str) or not Path(config[k]).is_absolute() for k in FIELDS - {"schema"}):
        raise ValueError("LAUNCH_CONFIG: paths must be absolute")
    root = Path(config["work_root"]).resolve()
    if any(Path(config[k]).resolve().is_relative_to(root) for k in FIELDS - {"schema", "work_root"}):
        raise ValueError("LAUNCH_CONFIG: source inputs must remain outside the writable work root")
    return dict(config)


def record(root, phase, state, **values):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    doc = dict(schema="i6dmo.raw_long_automation_status.v1", phase=phase, state=state,
               as_of=time.time(), eta="UNMEASURED", **values)
    atomic_json(root / "AUTOMATION_STATUS.json", doc, mode=0o600)
    return doc


def _existing_status(root):
    from tools.i6dmo import status as S
    doc = S.compute(root, level="campaign")
    state = "CLOSED" if C.closed_doc(root) is not None else "TECHNICAL_TERMINAL" if C._finalized(root) else "DISPATCHED"
    return dict(doc, state=state)


def status(root):
    root = Path(root)
    journal = root / "AUTOMATION_STATUS.json"
    doc = json.loads(journal.read_text()) if journal.exists() else dict(state="NOT_PREPARED", eta="UNMEASURED")
    if (root / "campaign" / "PLAN.json").exists():
        from tools.i6dmo import status as S
        execution = S.compute(root / "campaign", level="campaign")
        counts = execution.get("counts") or {}
        doc = dict(doc, launcher_journal=dict(doc), execution=execution,
                   state=execution["state"], eta=execution.get("eta", dict(kind="UNMEASURED")),
                   completed=counts.get("COMPLETE", 0),
                   total=(execution.get("expected") or {}).get("validation_units", sum(counts.values())),
                   phase="closed" if execution["state"] == "CLOSED" else "execution")
        if "generated_at" in execution:
            doc["as_of"] = execution["generated_at"]
    return doc


def _run(argv, timeout=180):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"COMMAND_FAILED:{Path(argv[0]).name}:rc={result.returncode}:"
                           f"{result.stderr[-800:]}")
    return result.stdout


def _remote(role, argv, timeout=180):
    alias = role.get("ssh_alias")
    if not alias:
        return _run(argv, timeout)
    return _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", alias, "--",
                 shlex.join([str(x) for x in argv])], timeout)


def verify_release(path=RELEASE):
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != "i6dmo.raw_long_release.v1" or doc.get("verdict") != "ACCEPT_FOR_EXECUTION":
        raise ValueError("RELEASE_NOT_ACCEPTED")
    files = doc.get("code_files") or {}
    required = {f"tools/i6dmo/{n}.py" for n in ("raw_long_campaign", "campaign", "design", "model",
        "common_population", "train", "worker", "unit_runner", "matched_runner", "reporting", "raw_pair",
        "strategy_inputs", "deploy")}
    if not required.issubset(files):
        raise ValueError("RELEASE_SCOPE_INCOMPLETE")
    for rel, digest in files.items():
        path = (REPO / rel).resolve()
        if not path.is_relative_to(REPO) or file_sha256(path) != digest:
            raise ValueError(f"REVIEW_SOURCE_CHANGED:{rel}")
    return doc


def _short_pin(root):
    from tools.i6dmo import donor_trust
    if C.closed_doc(root) is None:
        raise ValueError("SHORT_CAMPAIGN_NOT_CLOSED")
    effective = C.load_effective(root)
    trust = donor_trust.verify_donor(root, donor_trust.target_from_effective(effective))
    if trust["verdict"] != "ACCEPTED":
        raise ValueError("SHORT_CUSTODY_REJECTED:" + ";".join(trust["problems"])[:300])
    unit = next(u for u in C.load_plan(root)["units"] if u["unit_id"] == "mg2m.report")
    path = Path(root) / "terminals" / unit["identity_sha256"] / "TERMINAL.json"
    return dict(path=str(path), sha256=file_sha256(path), unit_id="mg2m.report")


def prepare(config):
    """CPU-only preparation: freeze data and private pins before any GPU claim."""
    from tools.i6dmo import common_population as CP, design as D, raw_pair
    config = validate_config(config)
    if _run(["git", "-C", str(REPO), "status", "--porcelain"]).strip():
        raise ValueError("SOURCE_NOT_FROZEN: real population generation requires committed source")
    work = Path(config["work_root"])
    if MR._inside_git(work if work.exists() else work.parent):
        raise ValueError("PRIVATE_OUTPUT_INSIDE_GIT")
    record(work, "prepare", "RUNNING", completed=0, total=6)
    prepared = work / "PREPARED.json"
    if prepared.exists():
        doc = json.loads(prepared.read_text())
        if doc.get("launch_config") != config:
            raise ValueError("PREPARATION_CONFIG_CHANGED")
        if not doc.get("generation_code"):
            raise ValueError("PREPARATION_UNFROZEN_GENERATOR")
        CP.load_common_population(doc["population_dir"], expected_digest=doc["population_digest"],
                                  input_dir=config["input_dir"])
        raw_pair._report("mg2m.report", doc["raw_pair"]["short_report"], "mg2m")
        for key in ("strategy_bars", "strategy_calendar"):
            ref = doc["raw_pair"][key]
            if file_sha256(ref["path"]) != ref["sha256"]:
                raise ValueError(f"STRATEGY_SOURCE_DIGEST:{key}")
        return doc
    spec = json.loads(Path(config["source_deployment"]).read_text())
    base = Path(spec["roles"]["coordinator"]["experiment_storage_root"])
    if not base.is_dir():
        raise ValueError("STORAGE_NOT_MOUNTED")
    design_dir = work / "designs"
    D.seal_matched_long(REPO / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json", design_dir,
                        runtime=RUNTIME)
    arms = MR.LONG_ARMS
    designs = {a: json.loads((design_dir / f"EXPERIMENT.{a}.json").read_text()) for a in arms}
    data_designs = {a: json.loads((design_dir / f"DESIGN.{a}.json").read_text()) for a in arms}
    record(work, "population", "RUNNING", completed=1, total=6)
    population_dir = base / "i6dmo" / work.name / "population"
    checkpoint = work / "POPULATION_BUILD.json"
    if population_dir.exists() and any(population_dir.iterdir()):
        if not checkpoint.is_file():
            raise ValueError("POPULATION_BUILD_CHECKPOINT_MISSING")
        build = json.loads(checkpoint.read_text())
        if build.get("input_sha256") != file_sha256(Path(config["input_dir"]) / "INPUT.npz") \
                or build.get("design_seals") != {a: d["seal_sha256"] for a, d in designs.items()}:
            raise ValueError("POPULATION_BUILD_CHECKPOINT_MISMATCH")
        pop = CP.load_common_population(population_dir, expected_digest=build["population_digest"],
                                        input_dir=config["input_dir"])
        seal = CP.seal_summary(json.loads((population_dir / "POPULATION.json").read_text()),
                               file_sha256(population_dir / "POPULATION.json"))
        if any(pop.identities["arms"][a]["experiment_seal_sha256"] != designs[a]["seal_sha256"] for a in arms):
            raise ValueError("EXISTING_POPULATION_DESIGN_DIFFERS")
    else:
        pop = CP.build_common_population(config["input_dir"], designs, data_designs,
                                         code_commit=C._git_head())
        atomic_json(checkpoint, dict(population_digest=pop.digest,
            input_sha256=file_sha256(Path(config["input_dir"]) / "INPUT.npz"),
            design_seals={a: d["seal_sha256"] for a, d in designs.items()},
            generation_code=C._git_head()), mode=0o600)
        staging = population_dir.with_name(f"population.building-{time.time_ns()}")
        seal = CP.write_common_population(pop, staging)
        staging.rename(population_dir)
    seal["private_location"] = "$EXPERIMENT_STORAGE_ROOT/" + str(population_dir.relative_to(base))
    atomic_json(work / "POPULATION_SEAL.json", seal, mode=0o600)
    short_pin = _short_pin(Path(config["short_campaign"]))
    raw_pair._report(short_pin["unit_id"], short_pin, "mg2m")  # real retained contract, no predictions read
    strategy = Path(config["strategy_inputs"])
    proof = json.loads((strategy / "RECONCILIATION.json").read_text())
    reconciliation = proof.get("reconciliation") or {}
    retained = proof.get("retained") or {}
    input_root = Path(config["input_dir"])
    original = json.loads((input_root / "PREPARATION.json").read_text())
    archive_digest = next(x["sha256"] for x in original["archives"] if x["year"] == "2024")
    if proof.get("schema") != "i6dmo.strategy_inputs.v1" or proof.get("test_read") is not False \
            or proof.get("status") != "PREPARED" or reconciliation.get("retained_2024_rows", 0) <= 0 \
            or reconciliation.get("matched_retained_rows") != reconciliation.get("retained_2024_rows") \
            or reconciliation.get("missing_retained_timestamps") != [] \
            or reconciliation.get("mismatched_timestamps") != [] \
            or proof.get("archive_sha256") != archive_digest \
            or retained.get("input_sha256") != file_sha256(input_root / "INPUT.npz") \
            or retained.get("calendar_sha256") != file_sha256(input_root / "CALENDAR.json") \
            or proof.get("code_sha256") != file_sha256(REPO / "tools/i6dmo/strategy_inputs.py"):
        raise ValueError("STRATEGY_SOURCE_NOT_VERIFIED")
    for name in ("STRATEGY_BARS.npz", "CALENDAR.json"):
        if not (strategy / name).is_file() or file_sha256(strategy / name) != proof["outputs"][name]:
            raise ValueError(f"STRATEGY_SOURCE_DIGEST:{name}")
    from tools.i6dmo import strategy_adapter as SA
    instrument = SA.instrument_from_config(SA.load_instrument_config(REPO / "tools/i6dmo/instrument_eurusd.v1.json"))
    pair = dict(short_report=short_pin, policy=dict(min_valid_long_heads=2, min_valid_short_heads=1),
        strategy_bars=dict(path=str(strategy / "STRATEGY_BARS.npz"), sha256=proof["outputs"]["STRATEGY_BARS.npz"]),
        strategy_calendar=dict(path=str(strategy / "CALENDAR.json"), sha256=proof["outputs"]["CALENDAR.json"]),
        strategy_execution=dict(instrument=instrument))
    doc = dict(schema="i6dmo.raw_long_prepared.v1", launch_config=config, designs=str(design_dir),
               population_dir=str(population_dir), population_digest=pop.digest,
               population_relpath=str(population_dir.relative_to(base)), raw_pair=pair,
               source_deployment_sha256=file_sha256(config["source_deployment"]),
               generation_code=pop.identities["code"]["commit"])
    atomic_json(prepared, doc, mode=0o600)
    record(work, "prepared", "PREPARED", completed=2, total=6, population_digest=pop.digest,
           counts={s: {str(h): int(v) for h, v in x.counts.items()} for s, x in pop.splits.items()})
    return doc


def _provision(spec, prepared, code, *, reviewed_files=None):
    runtime = f"/var/tmp/i6dmo-runtime-{code[:8]}"
    source = Path(prepared["launch_config"]["input_dir"])
    coordinator = spec["roles"]["coordinator"]
    input_rel = str(source.relative_to(Path(coordinator["data_root"])))
    for role, setting in spec["roles"].items():
        old = setting["workdir"]
        exists = _remote(setting, ["python3", "-c", f"import pathlib;print(int(pathlib.Path({runtime!r}).is_dir()))"], 30).strip() == "1"
        if not exists:
            if setting.get("ssh_alias"):
                _remote(setting, ["git", "-C", old, "fetch", "origin", code], 180)
            repo = old if setting.get("ssh_alias") else str(REPO)
            _remote(setting, ["git", "-C", repo, "worktree", "add", "--detach", runtime, code], 180)
        if _remote(setting, ["git", "-C", runtime, "rev-parse", "HEAD"], 30).strip() != code:
            raise ValueError(f"RUNTIME_PIN_MISMATCH:{role}")
        if reviewed_files is None:
            _verify_runtime(setting, runtime, code)
        else:
            _verify_runtime(setting, runtime, code, reviewed_files=reviewed_files)
        if setting.get("ssh_alias"):
            dest = Path(setting["experiment_storage_root"]) / prepared["population_relpath"]
            _remote(setting, ["mkdir", "-p", str(dest)], 30)
            _run(["rsync", "-a", "--checksum", prepared["population_dir"] + "/",
                  f"{setting['ssh_alias']}:{dest}/"], 1800)
            inp = Path(setting["data_root"]) / input_rel
            _remote(setting, ["mkdir", "-p", str(inp)], 30)
            _run(["rsync", "-a", "--checksum", str(source) + "/", f"{setting['ssh_alias']}:{inp}/"], 600)
        setting["workdir"] = runtime
        setting.setdefault("environment", {}).update(PYTHONPATH=runtime, PYTHONNOUSERSITE="1",
                                                       PYTHONSAFEPATH="1", PYTHONDONTWRITEBYTECODE="1",
                                                       PYTHONHOME="")
    spec["coordinator"]["workdir"] = runtime
    return spec


def _verify_runtime(role, runtime, code, *, reviewed_files=None):
    """Check executing bytes and import resolution, not merely a same-HEAD label."""
    files = verify_release()["code_files"] if reviewed_files is None else reviewed_files
    versions = RUNTIME if any(g.get("heavy") for g in role.get("gpus", [])) else {}
    script = "\n".join((
        "import hashlib,importlib.util,importlib.metadata,json,os,pathlib,subprocess,sys",
        f"root=pathlib.Path({runtime!r}).resolve(); expected={files!r}; code={code!r}; versions={versions!r}",
        "def git(*args): return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()",
        "assert git('rev-parse','HEAD')==code, 'RUNTIME_HEAD_MISMATCH'",
        "assert not git('status','--porcelain','--untracked-files=all'), 'RUNTIME_DIRTY'",
        "ignored=git('ls-files','--others','--ignored','--exclude-standard').splitlines()",
        "assert not any(x.endswith(('.py','.pyc','.so')) for x in ignored), 'RUNTIME_IMPORT_OVERRIDE'",
        "assert all(importlib.metadata.version(k)==v for k,v in versions.items()), 'RUNTIME_PACKAGE_VERSION'",
        "os.chdir(root); sys.path.insert(0,str(root))",
        "for rel,digest in expected.items():",
        " p=(root/rel).resolve(); assert p.is_relative_to(root), 'RUNTIME_PATH_ESCAPE'",
        " assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, 'RUNTIME_BYTES:'+rel",
        " if rel.startswith('tools/i6dmo/') and rel.endswith('.py'):",
        "  name=rel[:-3].replace('/','.')",
        "  if name.endswith('.__init__'): name=name[:-9]",
        "  spec=importlib.util.find_spec(name)",
        "  assert spec and spec.origin and pathlib.Path(spec.origin).resolve()==p, 'RUNTIME_IMPORT:'+name",
        "print(json.dumps({'verified_files':len(expected),'code_commit':code}))"))
    result = _remote(role, [role["python"], "-I", "-B", "-c", script], 120)
    observed = json.loads(result)
    if observed.get("verified_files") != len(files) or observed.get("code_commit") != code:
        raise ValueError("RUNTIME_BYTE_ATTESTATION_MISMATCH")
    return observed


def _resume(config, campaign):
    """Resume only this prepared deployment; never resurrect a technical terminal."""
    verify_release()
    work = Path(config["work_root"])
    prepared = json.loads((work / "PREPARED.json").read_text())
    effective = C.load_effective(campaign)
    spec = C.load_deployment(campaign)
    if prepared.get("launch_config") != config or effective.get("mode") != C.MODE_MATCHED_LONG \
            or effective.get("raw_pair") != prepared.get("raw_pair") \
            or effective["data_pins"]["common_population"]["population_sha256"] != prepared["population_digest"] \
            or effective.get("deployment_sha256") != C.sha256_json(spec) \
            or file_sha256(config["source_deployment"]) != prepared["source_deployment_sha256"]:
        raise ValueError("RESUME_IDENTITY_MISMATCH")
    if C.closed_doc(campaign) is not None or C._finalized(campaign):
        return _existing_status(campaign)
    if not (campaign / "DEPLOY.json").exists():
        for role in spec["roles"].values():
            _verify_runtime(role, role["workdir"], effective["code_commit"])
        from tools.i6dmo import deploy
        deploy.deploy(campaign)
    return _existing_status(campaign)


def run(config):
    config = validate_config(config)
    work = Path(config["work_root"])
    campaign = work / "campaign"
    if (campaign / "PLAN.json").exists():
        return _resume(config, campaign)
    verify_release()
    code = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", code) or _run(["git", "-C", str(REPO), "status", "--porcelain"]).strip():
        raise ValueError("SOURCE_NOT_FROZEN: commit reviewed code before execution")
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (work / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (campaign / "PLAN.json").exists():
            return _resume(config, campaign)
        prepared = prepare(config)
        spec = copy.deepcopy(json.loads(Path(config["source_deployment"]).read_text()))
        spec["launcher"]["initial_cap_bytes"] = 6 * 1024 ** 3
        spec["timer_cadence"] = dict(worker_seconds=5, cpu_worker_seconds=10, tick_seconds=10,
                                      publish_seconds=60, boot_seconds=30, accuracy_seconds=1)
        spec["mirrors"]["local"] = str(Path(spec["roles"]["coordinator"]["experiment_storage_root"]) /
                                        "i6dmo-mirrors" / work.name)
        spec["mirrors"]["second_host_path"] = str(Path(spec["roles"]["secondary"]["experiment_storage_root"]) /
                                                   "i6dmo-mirrors" / work.name)
        record(work, "replication", "RUNNING", completed=2, total=6)
        spec = _provision(spec, prepared, code)
        private = work / "deployment.json"
        atomic_json(private, spec, mode=0o600)
        paths = {a: str(Path(prepared["designs"]) / f"EXPERIMENT.{a}.json") for a in MR.LONG_ARMS}
        MR.prepare(deployment=private, input_dir=config["input_dir"], out=work / "matched.prepared.json",
                   designs=paths, population_seal=work / "POPULATION_SEAL.json",
                   population_dir=prepared["population_dir"], long=True)
        preflight = MR.preflight(work / "matched.prepared.json", work / "preflight")
        if preflight["state"] != "PASS":
            return record(work, "preflight", "BLOCKED_RESOURCE", completed=3, total=6,
                          launch_blockers=preflight["launch_blockers"])
        pins = json.loads((work / "matched.prepared.json").read_text())["data_pins"]
        C.init_matched_contrast(paths, private, campaign, code_commit=code, data_pins=pins,
                                raw_pair=prepared["raw_pair"])
        from tools.i6dmo import deploy
        deployed = deploy.deploy(campaign)
        record(work, "dispatch", "DISPATCHED", completed=4, total=6,
               campaign_id=deployed["campaign_id"], code_commit=code)
        return _existing_status(campaign)


def install(spec_path, *, systemctl=None, unit_dir=None):
    """Install a bounded retry timer for preparation/admission, not a model-training loop."""
    from tools.i6dmo import deploy
    spec_path = Path(spec_path).resolve()
    config = validate_config(json.loads(spec_path.read_text()))
    verify_release()
    ctl = systemctl or deploy.default_systemctl
    settings = json.loads(Path(config["source_deployment"]).read_text())
    python = settings["coordinator"]["python"]
    launcher = settings["launcher"]["crispdm_run"]
    name = "i6dmo-raw-long-launch-" + C.sha256_json(config)[:12]
    path = Path(unit_dir or Path.home() / ".config/systemd/user")
    path.mkdir(parents=True, exist_ok=True)
    marker = f"X-I6DMO-Automation={C.sha256_json(config)}"
    argv = [launcher, "-m", "4G", "-t", "1800s", "-n", name, "--", python,
            "-m", "tools.i6dmo.raw_long_campaign", "run", "--spec", str(spec_path)]
    service = (f"[Unit]\nDescription=RAW long finite preparation and admission\n{marker}\n\n"
        f"[Service]\nType=oneshot\nWorkingDirectory={REPO}\nEnvironment=CUDA_VISIBLE_DEVICES=\n"
        f"Environment=PYTHONDONTWRITEBYTECODE=1\nExecStart={shlex.join(argv)}\n"
        "MemoryMax=5G\nMemorySwapMax=0\nTimeoutStartSec=1900\nNice=10\n")
    timer = (f"[Unit]\nDescription=RAW long autonomous continuation\n{marker}\n\n"
        f"[Timer]\nOnBootSec=30s\nOnUnitInactiveSec=60s\nAccuracySec=1s\nPersistent=true\nUnit={name}.service\n\n"
        "[Install]\nWantedBy=timers.target\n")
    for suffix, content in (("service", service), ("timer", timer)):
        unit = path / f"{name}.{suffix}"
        if unit.exists() and marker not in unit.read_text():
            raise ValueError("FOREIGN_LAUNCH_UNIT")
        if not unit.exists() or unit.read_text() != content:
            unit.write_text(content)
    atomic_json(Path(config["work_root"]) / "LAUNCHER.json", dict(name=name, marker=marker), mode=0o600)
    for args in (("daemon-reload",), ("enable", "--now", f"{name}.timer")):
        if ctl(*args)["rc"]:
            raise RuntimeError("LAUNCH_TIMER_INSTALL_FAILED")
    return dict(state="ARMED", unit=f"{name}.timer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status", "install"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.action == "status":
        if not args.root:
            parser.error("status requires --root")
        result = status(args.root)
    else:
        if not args.spec:
            parser.error("prepare/run require --spec")
        config = json.loads(args.spec.read_text())
        try:
            result = prepare(config) if args.action == "prepare" else install(args.spec) if args.action == "install" else run(config)
            if args.action == "run" and result.get("state") in ("CLOSED", "TECHNICAL_TERMINAL"):
                receipt = Path(config["work_root"]) / "LAUNCHER.json"
                if receipt.exists():
                    from tools.i6dmo import deploy
                    name = json.loads(receipt.read_text())["name"]
                    if name != "i6dmo-raw-long-launch-" + C.sha256_json(config)[:12]:
                        raise ValueError("FOREIGN_LAUNCH_UNIT")
                    deploy.default_systemctl("disable", "--now", f"{name}.timer")
        except Exception as exc:
            record(config["work_root"], "automation", "FAILED", error_type=type(exc).__name__, error=str(exc))
            print(json.dumps(dict(state="FAILED", error_type=type(exc).__name__, error=str(exc))))
            return 2
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
