"""Bounded sequential endpoint engineering driver; not a DOIN swarm or campaign."""

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from optimizer_plugins import i6n_optimizer as adapter
from tools import i6n_historical_evaluator as bridge, i6n_search as search

cell, digest = bridge.cell, bridge.c1.canonical_digest
SCHEMA = "i6n.search_controller.v1"
GIB = 1024**3


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError("exact private schema keys required: " + keys)


def pinned(binding):
    exact(binding, "path sha256")
    if not Path(binding["path"]).is_absolute() or cell.file_hash(binding["path"]) != binding["sha256"]:
        raise ValueError("absolute path and matching digest required")
    return Path(binding["path"]).read_text()


def device_environment(cfg):
    dev = cfg["device"]
    exact(dev, "cuda_visible_devices required_gpu_name")
    if dev["required_gpu_name"] is None:
        if dev["cuda_visible_devices"] != "":
            raise ValueError("CPU verification must hide GPUs")
    elif (dev["required_gpu_name"] not in ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 4090 Laptop GPU")
          or not dev["cuda_visible_devices"].startswith("GPU-") or "," in dev["cuda_visible_devices"]):
        raise ValueError("explicit single admitted GPU UUID required; no fallback")
    return dict(os.environ, CUDA_VISIBLE_DEVICES=dev["cuda_visible_devices"],
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
                TF_NUM_INTEROP_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def aborted_exclusion(envelope, genome, base):
    """Exclude a dispatched predecessor; never import its fitness or authorize retry."""
    exact(envelope, "payload sha256")
    old = envelope["payload"]
    if digest(old) != envelope["sha256"]:
        raise ValueError("aborted controller state digest mismatch")
    if (old["state"] != "INTERRUPTED"
            or old["scope"] != f"ENDPOINT_ONLY_ENGINEERING_H{base['horizon_hours']}"):
        raise ValueError("aborted exclusion requires an interrupted same-horizon controller")
    previous = old["identity"]["cell"]["config"]
    fields = ("schema", "calendar", "features", "asset", "unit", "role", "horizon_hours",
              "training", "evaluation_week_id", "evaluation_split", "coverage_mode")
    if (any(previous[k] != base[k] for k in fields)
            or any(previous[k]["sha256"] != base[k]["sha256"] for k in ("input", "source_clock"))):
        raise ValueError("aborted exclusion historical context mismatch")
    bridge.candidate_config(base, search.effective_config(genome))
    key = search.effective_identity(genome)
    record = old["records"].get(key)
    if not isinstance(record, dict) or record.get("state") not in ("RUNNING", "FAILED"):
        raise ValueError("aborted exclusion requires a dispatched RUNNING or FAILED record")
    wire = record["wire"]
    exact(wire, "schema genome identity context generation")
    if (wire["schema"] != adapter.SCHEMA or wire["identity"] != key
            or type(wire["generation"]) is not int or wire["generation"] < 0
            or search.normalize_genome(wire["genome"]) != wire["genome"]
            or search.effective_identity(wire["genome"]) != key):
        raise ValueError("aborted record canonical genome identity mismatch")
    return key


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    exact(cfg, "schema scope binding plan external_pilots output_root seed budget admission resources device")
    if cfg["schema"] != SCHEMA or not Path(cfg["output_root"]).is_absolute():
        raise ValueError("controller schema or output path invalid")
    exact(cfg["plan"], "path sha256 section")
    pinned({k: cfg["plan"][k] for k in ("path", "sha256")})
    if not isinstance(cfg["plan"]["section"], str) or not cfg["plan"]["section"].strip():
        raise ValueError("plan section required")
    opt = json.loads(pinned(cfg["binding"]))
    allowed = {"i6n_horizon", "i6n_support", "i6n_evaluation_context", "i6n_evaluator",
               "i6n_evaluator_command", "i6n_evaluator_timeout"}
    if set(opt) - allowed or cfg["scope"] != f"ENDPOINT_ONLY_ENGINEERING_H{opt['i6n_horizon']}":
        raise ValueError("binding or horizon-specific scope mismatch")
    g = search.default_genome(opt["i6n_horizon"])
    base, _ = bridge.load_request(dict(config=search.effective_config(g), generation=0,
        context=opt["i6n_evaluation_context"], support=opt["i6n_support"]))
    # load_request maps model genes; retain the actual base for pilot comparison.
    base = cell.read_config(opt["i6n_evaluation_context"]["base_config"])
    b, a, r = cfg["budget"], cfg["admission"], cfg["resources"]
    exact(b, "max_candidates population_size generations patience max_proposals")
    exact(a, "min_spacing_hours max_samples max_core_width")
    exact(r, "child_memory_bytes child_wall_seconds total_wall_seconds min_mem_available_bytes poll_seconds crispdm_run admission_root")
    for name, value in {**b, **a, **{k: v for k, v in r.items() if k not in ("crispdm_run", "admission_root")}, "seed": cfg["seed"]}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"positive integer required: {name}")
    if (b["max_candidates"] > 24 or b["population_size"] > 8 or b["generations"] > 3
            or b["max_proposals"] > 32*b["max_candidates"] or b["patience"] > b["generations"]
            or a["min_spacing_hours"] < 2 or a["max_samples"] > 25 or a["max_core_width"] > 32):
        raise ValueError("outside initial endpoint allocation")
    env = device_environment(cfg)
    gpu = cfg["device"]["required_gpu_name"] is not None
    cap, minimum = ((3*GIB, 6*GIB) if cfg["device"]["required_gpu_name"] == "NVIDIA GeForce RTX 4090 Laptop GPU" else (5*GIB, 8*GIB))
    if (base["device"] != ("gpu" if gpu else "cpu") or cfg["seed"] != base["training"]["seed"]
            or (gpu and (r["child_memory_bytes"] != cap or r["min_mem_available_bytes"] < minimum))
            or (gpu and cfg["seed"] != 6102026) or base["resources"]["max_parameters"] > 1_000_000
            or (not gpu and r["child_memory_bytes"] > 2*GIB) or r["child_wall_seconds"] > 3600
            or r["total_wall_seconds"] > 90000):
        raise ValueError("device, seed or resource envelope mismatch")
    for key in ("crispdm_run", "admission_root"):
        if not Path(r[key]).is_absolute():
            raise ValueError("absolute launcher/admission paths required")
    excluded = set()
    if not isinstance(cfg["external_pilots"], list):
        raise ValueError("explicit external pilot exclusions required")
    for item in cfg["external_pilots"]:
        if "reason" in item:
            exact(item, "receipt genome reason")
            if item["reason"] != "ABORTED_AFTER_DISPATCH":
                raise ValueError("unsupported external exclusion reason")
            excluded.add(aborted_exclusion(json.loads(pinned(item["receipt"])), item["genome"], base))
            continue
        exact(item, "receipt genome")
        pilot = json.loads(pinned(item["receipt"]))
        pc, effective = pilot["identity"]["config"], search.effective_config(item["genome"])
        mapped = bridge.candidate_config(base, effective)
        if (pilot["state"] != "COMPLETED" or effective["input_transform"] != "identity"
                or pc["input"]["sha256"] != base["input"]["sha256"]
                or any(pc[k] != mapped[k] for k in ("model", "samples", "spacing_hours", "horizon_hours", "training", "evaluation_week_id"))):
            raise ValueError("external pilot exclusion does not match its declared genome/data")
        excluded.add(search.effective_identity(item["genome"]))
    opt.pop("i6n_evaluator", None)
    opt.update(i6n_evaluator_command=[sys.executable, "-m", "tools.i6n_historical_evaluator", "evaluate"],
        i6n_evaluator_timeout=base["resources"]["max_seconds"]+120, i6n_budget=b["max_candidates"],
        n_generations=b["generations"], optimization_patience=b["patience"], i6n_max_proposals=b["max_proposals"])
    governing = cell.identity(base)
    governing["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
    identity = dict(config_sha256=cell.file_hash(path), cell=governing, context=opt["i6n_evaluation_context"],
        controller_sha256=cell.file_hash(__file__), adapter_sha256=cell.file_hash(adapter.__file__),
        search_sha256=cell.file_hash(search.__file__), bridge_sha256=cell.file_hash(bridge.__file__),
        launcher_sha256=cell.file_hash(r["crispdm_run"]))
    return cfg, opt, excluded, identity


class EndpointPlugin(adapter.Plugin):
    """Same DEAP tournament/mutation; restrict proposals BEFORE reservation."""
    def __init__(self, admission, excluded, filtered=None):
        super().__init__()
        self.admission, self.excluded, self.filtered = admission, excluded, filtered if filtered is not None else {}

    def reason(self, genome):
        effective = search.effective_config(genome)
        if effective["output_mode"] != "endpoint":
            return "TRAJECTORY_UNSUPPORTED"
        if search.effective_identity(genome) in self.excluded:
            return "EXTERNAL_PILOT_EXCLUDED"
        a, core = self.admission, effective["core"]
        widths = [core["width"], *core["reducer_filters"], core["adapter_width"] or 0]
        if effective["delta_x"] < a["min_spacing_hours"] or effective["samples"] > a["max_samples"] or max(widths) > a["max_core_width"]:
            return "OUTSIDE_COST_CLASS"

    def _propose(self, tracker, contract, settings, rng, parents):
        seen, population = set(tracker["seen"]), []
        while len(population) < settings["population_size"] and len(seen) < settings["budget"] and tracker["attempts"] < settings["max_proposals"]:
            tracker["attempts"] += 1
            genome = (search.mutate(adapter.tools.selBest(rng.choices(parents, k=3), 1)[0].genome, rng)
                      if parents and rng.random() < .8 else search.sample_genome(contract["horizon"], rng))
            wire = self._wire(genome, tracker["context"], tracker["generation"])
            reason = self.reason(genome)
            if reason:
                self.filtered[wire["identity"]] = dict(reason=reason, genome=wire["genome"])
            elif wire["identity"] not in seen:
                seen.add(wire["identity"])
                population.append(wire)
        tracker.update(seen=sorted(seen), pending=[w["identity"] for w in population])
        return population


def read_state(root):
    envelope = json.loads((root / "STATE.json").read_text())
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("controller state digest mismatch")
    return envelope["payload"]


def summary(state):
    records = list(state["records"].values())
    finished = [r for r in records if r["state"] in ("COMPLETED", "FAILED")]
    durations = [r["elapsed_seconds"] for r in finished]
    remaining = state["budget"] - len(finished)
    return dict(schema=SCHEMA, scope=state["scope"], state=state["state"], stop_reason=state.get("stop_reason"),
        reserved=len(state.get("shared", {}).get("innovation_tracker", {}).get("seen", [])),
        started=len(records), completed=sum(r["state"] == "COMPLETED" for r in records),
        failed=sum(r["state"] == "FAILED" for r in records), filtered=len(state["filtered"]),
        attempts=state.get("shared", {}).get("innovation_tracker", {}).get("attempts", 0), best=state["best"],
        active=next((r for r in records if r["state"] == "RUNNING"), None),
        eta_seconds=(sum(durations)/len(durations)*remaining if durations and state["state"] == "RUNNING" else None),
        deadline_epoch=state["deadline_epoch"])


def save(root, state):
    cell.atomic_json(root / "STATE.json", dict(payload=state, sha256=digest(state)))
    cell.atomic_json(root / "STATUS.json", summary(state))


def finish(root, state, status, reason):
    state.update(state=status, stop_reason=reason)
    save(root, state)
    cell.atomic_json(root / "TERMINAL.json", dict(status=summary(state), state_sha256=cell.file_hash(root / "STATE.json"),
        identity=state["identity"], scope=state["scope"]))
    return summary(state)


def resource_reason(record):
    resource = record.get("resource")
    if not resource or type(resource.get("observed_peak_bytes")) is not int or resource["observed_peak_bytes"] <= 0:
        return "resource_evidence_missing"
    if resource.get("reclaim_cause") != "LEASE_RELEASED" or record.get("returncode") not in (0, 1):
        return "resource_or_timeout_stop"


def host_gate(cfg):
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    if available < cfg["resources"]["min_mem_available_bytes"]:
        raise ValueError("MemAvailable below fresh host gate; no cap lowering")
    root = os.environ.get("CRISPDM_ADMISSION_DIR", str(Path(os.environ.get("XDG_STATE_HOME", str(Path.home()/".local/state"))) / "crispdm/admission"))
    if Path(root).resolve() != Path(cfg["resources"]["admission_root"]).resolve():
        raise ValueError("admission store differs from launcher's shared host store")
    if cfg["device"]["required_gpu_name"]:
        name = subprocess.check_output(["nvidia-smi", "-i", cfg["device"]["cuda_visible_devices"],
            "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=10).strip()
        if name != cfg["device"]["required_gpu_name"]:
            raise ValueError("preferred GPU unavailable; no fallback")


def command(cfg, path, candidate):
    r = cfg["resources"]
    label = "i6n-" + digest(dict(root=cfg["output_root"], candidate=candidate))[:24]
    child = [sys.executable, "-m", "tools.i6n_search_controller", "worker", "--config", str(path), "--candidate", candidate]
    return [r["crispdm_run"], "-m", str(r["child_memory_bytes"]), "-t", f"{r['child_wall_seconds']}s",
            "-n", label, "-L", label, "--", *child], child, label


def launch(cfg, opt, wire, directory, heartbeat):
    argv, child, label = command(cfg, cfg["_path"], wire["identity"])
    started = time.monotonic()
    with (directory / "worker.log").open("xb") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=device_environment(cfg), cwd=cell.BASE)
        try:
            while process.poll() is None:
                locator = directory / "LOCATOR.json"
                if locator.exists():
                    status = Path(json.loads(locator.read_text())["root"]) / "STATUS.json"
                    if status.exists():
                        heartbeat(json.loads(status.read_text()))
                if time.monotonic()-started > cfg["resources"]["child_wall_seconds"]+60:
                    raise TimeoutError("child supervisor overrun; no successor authorized")
                time.sleep(cfg["resources"]["poll_seconds"])
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=40)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    argv_hash = hashlib.sha256(b"".join(s.encode()+b"\0" for s in child)).hexdigest()
    matches = []
    for path in (Path(cfg["resources"]["admission_root"]) / "retained").glob("*.json"):
        body = json.loads(path.read_text())
        lease = body.get("lease", {})
        if lease.get("label") == label and lease.get("argv_sha256") == argv_hash:
            if lease.get("cap_bytes") != cfg["resources"]["child_memory_bytes"]:
                raise ValueError("retained admission cap differs")
            matches.append(body)
    resource = matches[0] if len(matches) == 1 else None
    cell.atomic_json(directory / "RESOURCE.json", dict(returncode=process.returncode, resource=resource))
    return dict(returncode=process.returncode, resource=resource)


def verify_bridge_result(result, opt, wire):
    request = dict(config=search.effective_config(wire["genome"]), generation=wire["generation"],
                   context=opt["i6n_evaluation_context"], support=opt["i6n_support"])
    cfg, _ = bridge.load_request(request)
    expected = bridge.identity(cfg, request)
    expected["cell"]["cuda_visible_devices"] = result["cuda_visible_devices"]
    root = Path(request["context"]["output_root"]) / "candidates" / digest(expected)
    status = json.loads((root / "STATUS.json").read_text())
    raw = root / "RESULT.json"
    if status["state"] != "COMPLETED" or status["identity"] != expected or cell.file_hash(raw) != status["result_sha256"]:
        raise ValueError("bridge completion identity/digest mismatch")
    saved = json.loads(raw.read_text())
    if saved["receipt"] != result["result"]["metrics"] or set(saved["artifacts"]) != bridge.ARTIFACTS:
        raise ValueError("worker receipt differs from historical bridge")
    for name, sha in saved["artifacts"].items():
        if cell.file_hash(root / name) != sha:
            raise ValueError("bridge artifact digest mismatch")


def worker(path, candidate):
    cfg, opt, _, identity = load_config(path)
    root = Path(cfg["output_root"])
    state = read_state(root)
    record = state["records"][candidate]
    directory = root / "candidates" / candidate
    if state["identity"] != identity or state["state"] != "RUNNING" or record["state"] != "RUNNING":
        raise ValueError("worker lacks matching durable reservation")
    with (directory / "STARTED").open("x"):
        pass
    try:
        wire = record["wire"]
        request = dict(config=search.effective_config(wire["genome"]), generation=wire["generation"],
            context=opt["i6n_evaluation_context"], support=opt["i6n_support"])
        mapped, _ = bridge.load_request(request)
        location = Path(request["context"]["output_root"]) / "candidates" / digest(bridge.identity(mapped, request))
        cell.atomic_json(directory / "LOCATOR.json", dict(root=str(location)))
        result = adapter.Plugin().evaluate_single_genome(wire, wire["generation"], opt)
        cell.atomic_json(directory / "WORKER.json", dict(state="COMPLETED", result=result,
            cuda_visible_devices=cfg["device"]["cuda_visible_devices"]))
        return 0
    except Exception as exc:
        cell.atomic_json(directory / "WORKER.json", dict(state="FAILED", error=f"{type(exc).__name__}: {exc}"))
        return 1


def run(path):
    path = Path(path).resolve()
    cfg, opt, excluded, identity = load_config(path)
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "RUN.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "STATE.json").exists():
            state = read_state(root)
            if state["identity"] != identity:
                raise ValueError("controller identity changed; no resume")
            for key, record in state["records"].items():
                if record["state"] in ("FAILED", "COMPLETED") and resource_reason(record):
                    return finish(root, state, "STOPPED", resource_reason(record))
                if record["state"] == "COMPLETED":
                    output = root / "candidates" / key / "WORKER.json"
                    if cell.file_hash(output) != record["worker_sha256"]:
                        raise ValueError("worker result digest mismatch")
                    verify_bridge_result(json.loads(output.read_text()), opt, record["wire"])
            if state["state"] != "RUNNING":
                return finish(root, state, state["state"], state["stop_reason"])
            if any(r["state"] == "RUNNING" for r in state["records"].values()):
                return finish(root, state, "INTERRUPTED", "unreconciled_reservation_no_retry")
        else:
            if any(p.name != "RUN.lock" for p in root.iterdir()):
                raise ValueError("orphan controller artifacts prohibit starting")
            state = dict(identity=identity, state="RUNNING", scope=cfg["scope"], records={}, filtered={}, best=None,
                budget=cfg["budget"]["max_candidates"], deadline_epoch=time.time()+cfg["resources"]["total_wall_seconds"])
            save(root, state)
        plugin = EndpointPlugin(cfg["admission"], excluded, state["filtered"])
        try:
            if "shared" not in state:
                try:
                    state["shared"] = plugin.create_shared_population(cfg["budget"]["population_size"], opt, cfg["seed"])
                except RuntimeError:
                    return finish(root, state, "COMPLETED", "attempt_budget_no_admissible_candidates")
                save(root, state)
            while True:
                shared = state["shared"]
                for wire in shared["population"]:
                    key = wire["identity"]
                    if key in state["records"]:
                        continue
                    if time.time()+cfg["resources"]["child_wall_seconds"]+60 > state["deadline_epoch"]:
                        return finish(root, state, "COMPLETED", "wall_budget")
                    if load_config(path)[3] != identity:
                        raise ValueError("governing identity changed during stage")
                    try:
                        host_gate(cfg)
                    except Exception as exc:
                        return finish(root, state, "REFUSED", str(exc))
                    directory = root / "candidates" / key
                    directory.mkdir(parents=True, exist_ok=False)
                    record = dict(wire=wire, state="RUNNING", progress=None)
                    state["records"][key] = record
                    save(root, state)
                    started = time.monotonic()
                    def heartbeat(progress):
                        record["progress"] = {k: progress.get(k) for k in ("state", "progress", "elapsed_seconds", "eta_seconds")}
                        save(root, state)
                    outcome = launch(dict(cfg, _path=str(path)), opt, wire, directory, heartbeat)
                    record.update(outcome, elapsed_seconds=time.monotonic()-started, state="FAILED", fitness=None)
                    if resource_reason(record):
                        return finish(root, state, "STOPPED", resource_reason(record))
                    output = directory / "WORKER.json"
                    if outcome["returncode"] == 0 and output.exists():
                        completed = json.loads(output.read_text())
                        if completed.get("cuda_visible_devices") != cfg["device"]["cuda_visible_devices"]:
                            raise ValueError("worker GPU visibility mismatch")
                        verify_bridge_result(completed, opt, wire)
                        result = completed["result"]
                        score, _ = search._score(result["metrics"], opt["i6n_support"])
                        if result["hyper_dict"] != search.effective_config(wire["genome"]) or result["fitness"] != score:
                            raise ValueError("worker effective genome or fitness mismatch")
                        record.update(state="COMPLETED", fitness=score, worker_sha256=cell.file_hash(output))
                    elif output.exists():
                        record["error"] = json.loads(output.read_text()).get("error")
                    save(root, state)
                    if not output.exists():
                        return finish(root, state, "STOPPED", "worker_receipt_missing")
                evaluated = [dict(w, fitness=state["records"][w["identity"]]["fitness"]) for w in shared["population"]]
                eligible = [w for w in evaluated if w["fitness"] is not None]
                best = min(eligible, key=lambda w: w["fitness"]) if eligible else None
                improved = best is not None and (state["best"] is None or best["fitness"] < state["best"]["fitness"])
                if improved:
                    state["best"] = dict(identity=best["identity"], fitness=best["fitness"], genome=best["genome"])
                stale = 0 if improved else shared["no_improve_count"]+1
                state["shared"] = plugin.reproduce_shared(evaluated, shared["generation"], cfg["seed"]+shared["generation"]+1,
                    opt, shared["innovation_tracker"], shared["stage_schedule"], shared["param_defaults"], 0, stale)
                save(root, state)
                if state["shared"]["converged"]:
                    return finish(root, state, "COMPLETED", state["shared"]["stop_reason"])
        except BaseException as exc:
            finish(root, state, "INTERRUPTED", f"{type(exc).__name__}: {exc}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "status", "worker"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    if args.command == "worker":
        return worker(args.config, args.candidate)
    if args.command == "run":
        result = run(args.config)
    else:
        cfg, _, excluded, identity = load_config(args.config)
        result = (summary(read_state(Path(cfg["output_root"]))) if args.command == "status" else
                  dict(state="CHECKED_NOT_LAUNCHED", scope=cfg["scope"], excluded=sorted(excluded), identity=digest(identity)))
    print(json.dumps(result, allow_nan=False))
    return 0 if result["state"] in ("COMPLETED", "CHECKED_NOT_LAUNCHED", "RUNNING") else 1


if __name__ == "__main__":
    sys.exit(main())
