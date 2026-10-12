"""Deploy two distinct TRAIN-only cost pilots from pinned C/E artifacts.

No polling, no fallback, no automatic retry. Runtime configuration stays private.
The preferred device measures the larger S1 trajectory workload; the secondary
measures the minimal h2 seed. Neither measurement certifies an entire class.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from tools import i6n_search
from tools.i6n_swarm import common


def command(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def deploy(args):
    repo = Path.cwd()
    commit = command(["git", "rev-parse", "HEAD"])
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = Path(args.weekly_plan).resolve()
    views = json.loads((source / "VIEWS.private.json").read_text())
    plan = json.loads((repo / "docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json").read_text())
    inventory = json.loads(Path(args.inventory).read_text())
    code = Path("/var/tmp") / ("i6n-pilot-code-" + commit[:12])
    archive = root / "code.tar"
    command(["git", "archive", "--format=tar", "-o", str(archive), commit,
             "tools", "app", "predictor_plugins", "optimizer_plugins", "pipeline_plugins",
             "preprocessor_plugins", "target_plugins"])
    reports = []
    for role in args.role:
        worker = next(w for w in inventory if w["role"] == role)
        remote = worker["host"]
        ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", remote]
        if args.size_class != "S1" and role != "preferred":
            raise ValueError("larger-class pilots are scheduled on the preferred GPU only")
        horizon = {"S2": 36, "S3": 72}.get(args.size_class, 12 if role == "preferred" else 2)
        entry = views["folds"][str(horizon)][0]
        cell = common.read_pinned(entry["cell_config"], "C cell")
        genome = copy.deepcopy(plan["initial_genotypes"][str(horizon)]["seed_evaluator_genome"])
        if role == "preferred":
            width = {"S1":32, "S2":64, "S3":128}[args.size_class]
            genome.update(multiplier=4 if args.size_class == "S1" else 6,
                          delta_x={"S1":2, "S2":4, "S3":1}[args.size_class],
                          delta_y=horizon//4, output_mode="trajectory")
            genome["core"].update(entrance="conv_transformer", width=width, transformer_blocks=2,
                                 reducer_filters=[width, width, width], reducer_strides=[1, 1, 1],
                                 adapter="lstm", adapter_width=min(width,64), n_horizons=4)
        directory = root / role
        directory.mkdir(exist_ok=True, mode=0o700)
        cell_pin = entry["cell_config"]
        if args.size_class == "S3":
            # Extend only the declared resource contract for the prescribed n=433
            # pilot. Data, dates, features, stopping and history are unchanged.
            cell = copy.deepcopy(cell)
            cell["resources"]["max_samples"] = 433
            pilot_cell = directory / "CELL.private.json"
            common.atomic_json(pilot_cell, cell, mode=0o600)
            cell_pin = dict(path=str(pilot_cell), sha256=common.file_sha256(pilot_cell))
        cfg = dict(output=str(directory / "run"), code_commit=commit,
                   cell=cell_pin, fold=entry["fold_id"],
                   experiment="i6n-swarm-weekly-cost-20261011", python=worker["fit_python"],
                   effective_config=i6n_search.effective_config(genome), workdir=str(code),
                   environment=worker["environment"],
                   device=dict(cls="rtx5090" if role == "preferred" else "rtx4090",
                               cuda_visible_devices=worker["environment"]["CUDA_VISIBLE_DEVICES"]),
                   admission_ledger=str(Path.home()/".local/state/crispdm/admission/ledger.jsonl"),
                   launcher=dict(crispdm_run=worker["admission"], memory_bytes=7936*1024**2,
                                 wall_seconds=3600, label_prefix="i6n-cost-"+role+"-"+commit[:8]))
        cfg_path = directory / "CONFIG.private.json"
        if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
            raise ValueError("existing pilot config differs; not overwriting")
        common.atomic_json(cfg_path, cfg, mode=0o600)
        files = [Path(cell_pin["path"]),
                 Path(cell["input"]["path"]), Path(cell["source_clock"]["path"])]
        command(ssh+["mkdir -p "+shlex.quote(str(code))+" "+shlex.quote(str(directory))])
        command(["rsync", "-a", str(archive), remote+":"+str(root/"code.tar")])
        command(ssh+["tar -xf "+shlex.quote(str(root/"code.tar"))+" -C "+shlex.quote(str(code))])
        for file in files+[cfg_path]:
            command(ssh+["mkdir -p "+shlex.quote(str(file.parent))])
            command(["rsync", "-a", str(file), remote+":"+str(file)])
            expected = common.file_sha256(file)
            actual = command(ssh+["sha256sum "+shlex.quote(str(file))]).split()[0]
            if actual != expected:
                raise ValueError("remote input digest mismatch")
        unit = "i6n-cost-"+role+"-"+commit[:8]+".service"
        text = ("[Unit]\nDescription=I6N single TRAIN cost pilot\nAfter=network-online.target\n"
                "[Service]\nType=oneshot\nWorkingDirectory="+str(code)+"\n"
                "Environment=CUDA_VISIBLE_DEVICES=\nEnvironment=OPENBLAS_NUM_THREADS=1\n"
                "Environment=OMP_NUM_THREADS=1\nMemoryMax=384M\nTimeoutStartSec=4200\nRestart=no\n"
                "ExecStart="+shlex.join([worker["light_python"], "-m", "tools.i6n_swarm.cost_pilot",
                                         "--config", str(cfg_path)])+"\n")
        unit_path = directory / unit
        unit_path.write_text(text)
        command(ssh+["mkdir -p ~/.config/systemd/user"])
        command(["rsync", "-a", str(unit_path), remote+":.config/systemd/user/"+unit])
        command(ssh+["systemctl --user daemon-reload"])
        if args.start:
            command(ssh+["systemctl --user start --no-block "+shlex.quote(unit)])
        reports.append(dict(role=role, unit=unit, horizon=horizon, fold=entry["fold_id"],
                            genotype_sha256=i6n_search.effective_identity(genome),
                            code_commit=commit, config_sha256=common.file_sha256(cfg_path),
                            start_requested=args.start))
    common.atomic_json(root/"DEPLOYMENT.json", dict(pilots=reports), mode=0o600)
    print(json.dumps(dict(pilots=reports), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True)
    p.add_argument("--weekly-plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", action="append", choices=("preferred", "secondary"), required=True)
    p.add_argument("--size-class", choices=("S1", "S2", "S3"), default="S1")
    p.add_argument("--start", action="store_true")
    deploy(p.parse_args())


if __name__ == "__main__":
    main()
