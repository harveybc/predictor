"""Finite short-input/long-target bridge using the existing governed campaign.

Preparation, admission and deployment resume from a bounded systemd timer;
workers own fit -> report -> comparison -> publication -> mirrors -> CLOSE.
No agent polling, new scheduler, DOIN or TEST path is introduced.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import shlex
import time

from tools.i6dmo import campaign as C, matched_runner as MR, raw_long_campaign as L
from tools.i6n_swarm.common import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs/i6n/direct_multi_output_static/context_bridge"
RELEASE = DOCS / "RELEASE_ACCEPTANCE.json"
FIELDS = {"schema", "work_root", "source_deployment", "input_dir"}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS or config.get("schema") != "i6dmo.context_bridge_launch.v1":
        raise ValueError("BRIDGE_CONFIG_FIELDS")
    for key in FIELDS - {"schema"}:
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute():
            raise ValueError("BRIDGE_CONFIG_ABSOLUTE_PATHS")
    root = Path(config["work_root"]).resolve()
    if any(Path(config[k]).resolve().is_relative_to(root) for k in ("input_dir", "source_deployment")):
        raise ValueError("BRIDGE_CONFIG_SOURCE_INSIDE_OUTPUT")
    return dict(config)


def release():
    doc = json.loads(RELEASE.read_text())
    if doc.get("schema") != "i6dmo.context_bridge_release.v1" or doc.get("verdict") != "ACCEPT_FOR_EXECUTION":
        raise ValueError("BRIDGE_NOT_REVIEWED")
    required = {f"tools/i6dmo/{n}.py" for n in ("context_bridge_campaign", "campaign", "design", "model",
        "common_population", "train", "worker", "unit_runner", "matched_runner", "reporting", "deploy")}
    if not required.issubset(doc.get("code_files", {})):
        raise ValueError("BRIDGE_REVIEW_SCOPE")
    for rel, digest in doc["code_files"].items():
        path = (REPO / rel).resolve()
        if not path.is_relative_to(REPO) or file_sha256(path) != digest:
            raise ValueError("BRIDGE_REVIEW_BYTES_CHANGED:" + rel)
    return doc


def status(root):
    root = Path(root)
    if (root / "campaign/PLAN.json").exists():
        from tools.i6dmo import status as S
        result = S.compute(root / "campaign", level="campaign")
        if result["state"] == "CLOSED" and C.closure_problems(root / "campaign")[0]:
            raise ValueError("BRIDGE_CLOSE_NOT_VERIFIED")
        return result
    p = root / "AUTOMATION_STATUS.json"
    return json.loads(p.read_text()) if p.exists() else dict(state="NOT_PREPARED", eta="UNMEASURED")


def prepare(config):
    """Freeze both designs and the common population before any scientific fit."""
    from tools.i6dmo import common_population as CP, design as D
    config = validate_config(config)
    work = Path(config["work_root"])
    if MR._inside_git(work if work.exists() else work.parent):
        raise ValueError("PRIVATE_OUTPUT_INSIDE_GIT")
    source = json.loads(Path(config["source_deployment"]).read_text())
    storage = Path(source["roles"]["coordinator"]["experiment_storage_root"])
    design_dir = work / "designs"
    D.seal_context_bridge(REPO / "docs/i6n/direct_multi_output_static/EXPERIMENT.short.json", design_dir,
                          runtime=L.RUNTIME)
    arms = MR.CONTEXT_BRIDGE_ARMS
    experiments = {a: json.loads((design_dir / f"EXPERIMENT.{a}.json").read_text()) for a in arms}
    data_designs = {a: json.loads((design_dir / f"DESIGN.{a}.json").read_text()) for a in arms}
    population_dir = storage / "i6dmo" / work.name / "population"
    checkpoint = work / "POPULATION_BUILD.json"
    binding = dict(input_sha256=file_sha256(Path(config["input_dir"]) / "INPUT.npz"),
                   seals={a: d["seal_sha256"] for a, d in experiments.items()},
                   source_deployment_sha256=file_sha256(config["source_deployment"]))
    prepared_path = work / "PREPARED.json"
    if prepared_path.exists():
        old = json.loads(prepared_path.read_text())
        if old["launch_config"] != config or old["binding"] != binding:
            raise ValueError("BRIDGE_PREPARATION_CHANGED")
    L.record(work, "population", "RUNNING", completed=1, total=4)
    if (population_dir / "POPULATION.json").exists():
        saved = json.loads(checkpoint.read_text())
        if saved["binding"] != binding:
            raise ValueError("BRIDGE_POPULATION_BINDING_CHANGED")
        pop = CP.load_common_population(population_dir, expected_digest=saved["population_digest"],
                                        input_dir=config["input_dir"])
    else:
        if population_dir.exists() and any(population_dir.iterdir()):
            raise ValueError("BRIDGE_PARTIAL_POPULATION")
        pop = CP.build_common_population(config["input_dir"], experiments, data_designs, code_commit=C._git_head())
        atomic_json(checkpoint, dict(binding=binding, population_digest=pop.digest), mode=0o600)
        staging = population_dir.with_name("population.building-" + str(time.time_ns()))
        CP.write_common_population(pop, staging)
        if population_dir.exists():
            population_dir.rmdir()  # only the verified empty preparation directory
        staging.rename(population_dir)
    seal = CP.seal_summary(json.loads((population_dir / "POPULATION.json").read_text()),
                           file_sha256(population_dir / "POPULATION.json"))
    seal["private_location"] = "$EXPERIMENT_STORAGE_ROOT/" + str(population_dir.relative_to(storage))
    atomic_json(work / "POPULATION_SEAL.json", seal, mode=0o600)
    prepared = dict(schema="i6dmo.context_bridge_prepared.v1", launch_config=config, binding=binding,
                    population_dir=str(population_dir), population_digest=pop.digest,
                    population_relpath=str(population_dir.relative_to(storage)), designs=str(design_dir))
    atomic_json(prepared_path, prepared, mode=0o600)
    return prepared


def run(config):
    config = validate_config(config)
    files = release()["code_files"]
    work = Path(config["work_root"])
    work.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (work / "launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = work / "campaign"
        if (campaign / "PLAN.json").exists():
            saved = json.loads((work / "PREPARED.json").read_text())
            effective = C.load_effective(campaign)
            if saved["launch_config"] != config or set(effective["arms"]) != set(MR.CONTEXT_BRIDGE_ARMS) \
                    or file_sha256(config["source_deployment"]) != saved["binding"]["source_deployment_sha256"] \
                    or effective["data_pins"]["common_population"]["population_sha256"] != saved["population_digest"]:
                raise ValueError("BRIDGE_RESUME_IDENTITY")
            state = status(work)
            if not (campaign / "DEPLOY.json").exists() and state["state"] not in ("CLOSED", "FAILED", "BLOCKED"):
                from tools.i6dmo import deploy
                for role in C.load_deployment(campaign)["roles"].values():
                    L._verify_runtime(role, role["workdir"], effective["code_commit"], reviewed_files=files)
                deploy.deploy(campaign)
            return status(work)
        if L._run(["git", "-C", str(REPO), "status", "--porcelain"]).strip():
            raise ValueError("SOURCE_NOT_FROZEN")
        code = C._git_head()
        prepared = prepare(config)
        spec = copy.deepcopy(json.loads(Path(config["source_deployment"]).read_text()))
        spec["launcher"]["initial_cap_bytes"] = 4 * 1024 ** 3
        spec["timer_cadence"] = dict(worker_seconds=5, cpu_worker_seconds=10, tick_seconds=10,
                                      publish_seconds=30, boot_seconds=30, accuracy_seconds=1)
        # Same-class sequential fits on the preferred GPU; independent CPU reports may use other roles.
        for name, role in spec["roles"].items():
            if name != "preferred":
                role["gpus"] = []
        spec["mirrors"]["local"] = str(Path(spec["roles"]["coordinator"]["experiment_storage_root"]) /
                                        "i6dmo-mirrors" / work.name)
        spec["mirrors"]["second_host_path"] = str(Path(spec["roles"]["secondary"]["experiment_storage_root"]) /
                                                   "i6dmo-mirrors" / work.name)
        L.record(work, "replication", "RUNNING", completed=2, total=4)
        spec = L._provision(spec, prepared, code, reviewed_files=files)
        private = work / "deployment.json"
        atomic_json(private, spec, mode=0o600)
        paths = {a: str(Path(prepared["designs"]) / f"EXPERIMENT.{a}.json") for a in MR.CONTEXT_BRIDGE_ARMS}
        MR.prepare(deployment=private, input_dir=config["input_dir"], out=work / "matched.prepared.json",
                   designs=paths, population_seal=work / "POPULATION_SEAL.json",
                   population_dir=prepared["population_dir"], long=True)
        precheck = MR.preflight(work / "matched.prepared.json", work / "preflight")
        if precheck["state"] != "PASS":
            return L.record(work, "preflight", "BLOCKED_RESOURCE", launch_blockers=precheck["launch_blockers"])
        pins = json.loads((work / "matched.prepared.json").read_text())["data_pins"]
        C.init_matched_contrast(paths, private, campaign, code_commit=code, data_pins=pins)
        from tools.i6dmo import deploy
        deploy.deploy(campaign)
        L.record(work, "dispatch", "DISPATCHED", completed=3, total=4)
        return status(work)


def install(spec_path):
    """Restart-safe bounded preparation timer; the campaign owns scientific progress."""
    from tools.i6dmo import deploy
    spec_path = Path(spec_path).resolve()
    config = validate_config(json.loads(spec_path.read_text()))
    release()
    name = "i6dmo-context-bridge-" + C.sha256_json(config)[:12]
    deployment = json.loads(Path(config["source_deployment"]).read_text())
    argv = [deployment["launcher"]["crispdm_run"], "-m", "4G", "-t", "1800s", "-n", name, "--",
            deployment["coordinator"]["python"], "-B", "-m", "tools.i6dmo.context_bridge_campaign", "run",
            "--spec", str(spec_path)]
    marker = "X-I6DMO-Automation=" + C.sha256_json(config)
    directory = Path.home() / ".config/systemd/user"
    directory.mkdir(parents=True, exist_ok=True)
    documents = {
        "service": f"[Unit]\nDescription=Finite R0 context bridge\n{marker}\n\n[Service]\nType=oneshot\n"
        f"WorkingDirectory={REPO}\nEnvironment=CUDA_VISIBLE_DEVICES=\nEnvironment=OPENBLAS_NUM_THREADS=1\n"
        f"Environment=PYTHONDONTWRITEBYTECODE=1\nExecStart={shlex.join(argv)}\nMemoryMax=5G\nMemorySwapMax=0\n"
        "TimeoutStartSec=1900\nNice=10\n",
        "timer": f"[Unit]\nDescription=Context bridge autonomous admission\n{marker}\n\n[Timer]\nOnBootSec=30s\n"
        f"OnUnitInactiveSec=60s\nAccuracySec=1s\nPersistent=true\nUnit={name}.service\n\n[Install]\nWantedBy=timers.target\n"}
    for suffix, content in documents.items():
        path = directory / f"{name}.{suffix}"
        if path.exists() and marker not in path.read_text():
            raise ValueError("FOREIGN_LAUNCH_UNIT")
        path.write_text(content)
    atomic_json(Path(config["work_root"]) / "LAUNCHER.json", dict(name=name, marker=marker), mode=0o600)
    for args in (("daemon-reload",), ("enable", "--now", name + ".timer")):
        if deploy.default_systemctl(*args)["rc"]:
            raise RuntimeError("BRIDGE_TIMER_INSTALL")
    return dict(state="ARMED", timer=name + ".timer")


def stop_launcher(config):
    """Stop only the timer bound to this exact private launch config."""
    from tools.i6dmo import deploy
    path = Path(config["work_root"]) / "LAUNCHER.json"
    if not path.exists():
        return
    name = "i6dmo-context-bridge-" + C.sha256_json(config)[:12]
    if json.loads(path.read_text())["name"] != name:
        raise ValueError("FOREIGN_LAUNCH_UNIT")
    if deploy.default_systemctl("disable", "--now", name + ".timer")["rc"]:
        raise RuntimeError("BRIDGE_TIMER_STOP_FAILED")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "install", "status", "prepare"))
    p.add_argument("--spec", type=Path)
    p.add_argument("--root", type=Path)
    a = p.parse_args()
    if a.action == "status":
        if not a.root:
            p.error("status requires --root")
        result = status(a.root)
    else:
        if not a.spec:
            p.error("--spec required")
        config = validate_config(json.loads(a.spec.read_text()))
        try:
            result = install(a.spec) if a.action == "install" else prepare(config) if a.action == "prepare" else run(config)
        except Exception as exc:
            L.record(config["work_root"], "automation", "FAILED", error_type=type(exc).__name__, error=str(exc))
            if a.action == "run":
                stop_launcher(config)
            raise
        if a.action == "run" and result.get("state") in ("CLOSED", "FAILED", "BLOCKED"):
            stop_launcher(config)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
