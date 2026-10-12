"""I6-N swarm production deployment: one idempotent top-level command (orders 2.B).

    python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json>            # apply (idempotent)
    python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json> --dry-run  # validate + plan only
    python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json> --status   # read-only
    python -m tools.i6n_swarm_deploy --spec <DEPLOY.private.json> --join R   # gate (timer-run)

The private deploy spec (never committed) names the integration READINESS pin,
the sibling pins, the private worker inventory, the data root and the pilot
receipts. ``apply`` performs, each step only once and verified:

1. export the pinned code trees (predictor + siblings) into one code directory
   with a sha256 manifest and a per-host python wrapper (PYTHONPATH = pinned
   trees, I6N_DATA_ROOT = that host's data root);
2. derive the effective PRIVATE campaign config (B schema) from the inventory,
   the pilot receipts and lane C's plan; validate every substituted argument;
   class caps keep 1.25 x (measured peak + declared-model coverage adjustment);
3. ``plan`` -> ``preflight`` -> ``install``/``start`` coordinator (B's CLI);
4. per worker (secondary first): copy code, data root, campaign root and the
   ledger token to the SAME absolute paths, verify every byte remotely with
   ``sha256sum -c``, then ``install`` its units and either ``start`` them
   (immediate) or leave them to the coordinator join gate (after_pilot).

The join gate (a oneshot unit + timer on the coordinator) starts a worker only
after its running pilot unit is no longer active and the pilot's terminal state
is verified: COMPLETED with a remote RESULT sha256 equal to its status record,
or FAILED with the cause recorded. A failed pilot never pauses the campaign;
the worker joins for its admitted smaller classes. Nothing here stops, edits or
duplicates the pilot, trains, lowers a cap or retries an OOM. Host names,
addresses and tokens live only in private files and are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import secrets
import shlex
import socket
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

SPEC_SCHEMA = "i6n.swarm_deploy.private.v1"
STATE_SCHEMA = "i6n.swarm_deploy.state.v1"
CAMPAIGN_SCHEMA = "i6n.swarm_campaign.private.v1"
SEED = 6102026
SAFETY = 1.25
ROW_FEATURE_COLUMNS = 41  # 20 features x (value, mask) + market flag: cell.prepare row model
DEVICE_CLASS = {"preferred": "rtx5090", "secondary": "rtx4090"}
WORKER_ORDER = ("secondary", "preferred")
SPLITS_FOLD = ("TRAIN", "INNER")
SPLITS_WEEK = ("TRAIN", "INNER", "VALIDATION")


class DeployError(RuntimeError):
    """A typed refusal; nothing after the failing step is attempted."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


# --------------------------------------------------------------------------- small helpers
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_private_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def _pin(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _redact(text: str, secrets_: Iterable[str]) -> str:
    for s in secrets_:
        if s:
            text = text.replace(s, "<private>")
    return text


# --------------------------------------------------------------------------- class admission
def row_bytes(samples: int) -> int:
    """Declared per-row materialization of ``tools.i6n_historical_cell.prepare``."""
    return samples * ROW_FEATURE_COLUMNS * 4 + 8 + 1 + 8


def max_samples_at(h: int, class_max: int, multipliers=(1, 2, 4, 6)) -> int:
    """Largest sample count a class can reach at horizon h (dx = 1 always divides)."""
    history = max(max(24, m * h) for m in multipliers)
    return min(class_max, 1 + history)


def unit_rows(unit: Mapping[str, Any], h: int, role: str = "MANAGEMENT") -> int:
    splits = SPLITS_WEEK if unit["kind"] == "WEEKLY_REFIT" else SPLITS_FOLD
    support = unit["support"][role][str(h)]
    return sum(int(support[s]["n"] or 0) for s in splits if s in support)


def class_admission(c_plan: Mapping[str, Any], e_classes: List[Mapping[str, Any]],
                    pilots: Mapping[str, Mapping[str, Any]], cell_resources: Mapping[str, int], *,
                    child_wall_seconds: int) -> Dict[str, Any]:
    """Admission per class from a COMPLETED pilot at the class corner plus the declared model.

    ``pilots[class_id] = {receipt: {path, sha256}, horizon, fold, samples, width}``. A class is
    admitted only when its pilot ran at the class corner (samples and width = class maxima,
    measured coverage) and every reachable workload fits the cell guards. The coverage
    adjustment is the largest positive declared materialization above the pilot's own, so the
    cap keeps 1.25 x (measured peak + adjustment). Nothing is admitted without a receipt.
    """
    units = [u for u in c_plan["units"] if u["data"]["state"] in ("READY", "READY_WITH_DECLARED_CLOCK_ASSUMPTION")]
    by_id = {u["unit_id"]: u for u in units}
    horizons = [int(h) for h in c_plan["horizons_hours"]]
    out: Dict[str, Any] = {}
    for cls in e_classes:
        cid, cmax, wmax = cls["class_id"], int(cls["max_samples"]), int(cls["max_width"])
        pilot = pilots.get(cid)
        if pilot is None:
            out[cid] = {"admitted": False, "cause": "NO_COMPLETED_PILOT_RECEIPT"}
            continue
        receipt = json.loads(Path(pilot["receipt"]["path"]).read_text())
        if sha256_file(Path(pilot["receipt"]["path"])) != pilot["receipt"]["sha256"]:
            raise DeployError("PILOT_RECEIPT_DIGEST", cid)
        peak = receipt.get("observed_peak_bytes")
        if receipt.get("state") != "COMPLETED" or type(peak) is not int or peak <= 0:
            out[cid] = {"admitted": False, "cause": "PILOT_NOT_COMPLETED"}
            continue
        if int(pilot["samples"]) != cmax or int(pilot["width"]) != wmax:
            out[cid] = {"admitted": False, "cause": "PILOT_NOT_AT_CLASS_CORNER"}
            continue
        pilot_unit = by_id.get(pilot["fold"])
        if pilot_unit is None:
            raise DeployError("PILOT_FOLD_UNKNOWN", f"{cid}: {pilot['fold']}")
        pilot_bytes = unit_rows(pilot_unit, int(pilot["horizon"])) * row_bytes(cmax)
        worst, worst_at, guard = pilot_bytes, None, []
        for h in horizons:
            samples = max_samples_at(h, cmax)
            rb = row_bytes(samples)
            for u in units:
                rows = unit_rows(u, h)
                declared = rows * rb
                if declared > worst:
                    worst, worst_at = declared, {"unit_id": u["unit_id"], "horizon": h, "samples": samples,
                                                 "rows": rows}
                if declared > cell_resources["max_materialized_bytes"]:
                    guard.append(f"materialized {u['unit_id']} h{h}")
            batch = 256 * samples * (2 * ROW_FEATURE_COLUMNS * 4 + 2 * 8 + 2)
            if batch > cell_resources["max_batch_bytes"] or samples > cell_resources["max_samples"]:
                guard.append(f"batch/samples h{h}")
        if guard:
            out[cid] = {"admitted": False, "cause": "CELL_GUARD_EXCEEDED: " + "; ".join(sorted(set(guard))[:5])}
            continue
        adjustment = max(0, worst - pilot_bytes)
        derivation = {"model": "rows x (samples*41*4+17) over C's sealed TRAIN/INNER(/VALIDATION) support",
                      "pilot": {"fold": pilot["fold"], "horizon": int(pilot["horizon"]), "samples": cmax,
                                "declared_bytes": pilot_bytes},
                      "worst": worst_at, "worst_declared_bytes": worst, "plan_sha256": c_plan["plan_sha256"]}
        out[cid] = {"admitted": True, "receipt": dict(pilot["receipt"]), "measured_peak_bytes": peak,
                    "coverage_adjustment_bytes": adjustment, "coverage_derivation": derivation,
                    "coverage_derivation_sha256": sha256_json(derivation),
                    "child_memory_bytes": math.ceil(SAFETY * (peak + adjustment)),
                    "child_wall_seconds": int(child_wall_seconds), "small_pilot": False}
    return out


# --------------------------------------------------------------------------- remote hosts
class Remote:
    """ssh/rsync to one inventory host; BatchMode, bounded timeouts, host never printed."""

    def __init__(self, host: str, timeout: int = 120):
        self.host, self.timeout = host, timeout

    def run(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.host, command],
                                capture_output=True, text=True, timeout=self.timeout)
        if check and result.returncode != 0:
            raise DeployError("REMOTE_COMMAND_FAILED", _redact(result.stderr[-400:], [self.host]))
        return result

    def put(self, local: Path, remote: str) -> None:
        self.run("mkdir -p " + shlex.quote(str(Path(remote).parent)))
        result = subprocess.run(["rsync", "-a", "--checksum", str(local), f"{self.host}:{remote}"],
                                capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            raise DeployError("RSYNC_FAILED", _redact(result.stderr[-400:], [self.host]))


    def put_many(self, paths: List[str]) -> None:
        """One rsync for many absolute paths (same path on both ends; --files-from, --relative)."""
        listing = Path(os.environ.get("TMPDIR", "/tmp")) / f"i6n-deploy-files-{os.getpid()}.txt"
        listing.write_text("".join(p.lstrip("/") + "\n" for p in sorted(paths)))
        try:
            result = subprocess.run(["rsync", "-a", "--omit-dir-times", "--checksum", "--files-from", str(listing), "/",
                                     f"{self.host}:/"], capture_output=True, text=True, timeout=3600)
        finally:
            listing.unlink(missing_ok=True)
        if result.returncode != 0:
            raise DeployError("RSYNC_FAILED", _redact(result.stderr[-400:], [self.host]))


class LocalHost(Remote):
    """The coordinator itself (no ssh)."""

    def __init__(self):
        super().__init__("localhost")

    def run(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=self.timeout)
        if check and result.returncode != 0:
            raise DeployError("LOCAL_COMMAND_FAILED", result.stderr[-400:])
        return result

    def put(self, local: Path, remote: str) -> None:
        if str(local) != remote:
            Path(remote).parent.mkdir(parents=True, exist_ok=True)
            Path(remote).write_bytes(Path(local).read_bytes())


# --------------------------------------------------------------------------- manifests
def manifest_lines(files: Mapping[str, Path]) -> str:
    """sha256sum -c format over absolute destination paths."""
    return "".join(f"{sha256_file(src)}  {dst}\n" for dst, src in sorted(files.items()))


def remote_mismatches(remote: Remote, manifest_text: str, manifest_path: str) -> List[str]:
    """Paths whose remote bytes differ from the manifest (missing counts as differing)."""
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"i6n-deploy-{os.getpid()}-{secrets.token_hex(4)}.sha256"
    tmp.write_text(manifest_text)
    try:
        remote.put(tmp, manifest_path)
    finally:
        tmp.unlink(missing_ok=True)
    result = remote.run("sha256sum -c --quiet " + shlex.quote(manifest_path) + " 2>/dev/null", check=False)
    if result.returncode == 0:
        return []
    bad = []
    for line in result.stdout.splitlines():
        if line.rstrip().endswith("FAILED") or "FAILED open or read" in line:
            path = line.rsplit(":", 1)[0]
            if len(path) > 1 and path[0] == path[-1] == "'":  # newer coreutils quote names with spaces
                path = path[1:-1].replace("'\\''", "'")
            bad.append(path)
    return bad or ["<unparsed sha256sum failure>"]


# --------------------------------------------------------------------------- spec
REQUIRED_SPEC = ("schema", "campaign_id", "experiment", "pin", "repos", "code_dir", "campaign_root",
                 "data_root", "inventory", "c_plan", "c_views", "multi_output", "search_plan",
                 "pilots", "ports", "workers", "deploy_dir")


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != SPEC_SCHEMA:
        raise DeployError("SPEC_SCHEMA")
    missing = [k for k in REQUIRED_SPEC if k not in spec]
    if missing:
        raise DeployError("SPEC_INCOMPLETE", ",".join(missing))
    if os.stat(path).st_mode & 0o077:
        raise DeployError("SPEC_NOT_PRIVATE", "deploy spec must be 0600")
    for role in spec["workers"]:
        if role not in DEVICE_CLASS:
            raise DeployError("UNKNOWN_WORKER_ROLE", role)
        if spec["workers"][role].get("start") not in ("immediate", "after_pilot"):
            raise DeployError("WORKER_START_POLICY", role)
    return spec


def inventory(spec: Mapping[str, Any]) -> Dict[str, dict]:
    return {w["role"]: w for w in json.loads(Path(spec["inventory"]).read_text())}


def resolve_address(host: str) -> str:
    return socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]


def local_source_address(peer: str) -> str:
    out = subprocess.run(["ip", "-4", "route", "get", peer], capture_output=True, text=True, check=True).stdout
    tokens = out.split()
    return tokens[tokens.index("src") + 1]


# --------------------------------------------------------------------------- code bundle
def export_code(spec: Mapping[str, Any], run: Callable = subprocess.run) -> Dict[str, Path]:
    """git archive each pinned repo into code_dir/<name>; return {dest: src} for the manifest."""
    code = Path(spec["code_dir"])
    files: Dict[str, Path] = {}
    for name, repo in sorted(spec["repos"].items()):
        target = code / name
        marker = code / f".{name}.pin"
        if not (marker.exists() and marker.read_text().strip() == repo["pin"]):
            if target.exists() and any(target.iterdir()):
                raise DeployError("CODE_DIR_HOLDS_OTHER_PIN", name)
            target.mkdir(parents=True, exist_ok=True)
            data = run(["git", "-C", repo["path"], "archive", "--format=tar", repo["pin"], *repo.get("paths", [])],
                       capture_output=True, check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                tar.extractall(target, filter="data")
            marker.write_text(repo["pin"] + "\n")
        for path in sorted(p for p in target.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            files[str(path)] = path
    return files


def wrapper_text(spec: Mapping[str, Any], python: str, data_root: str) -> str:
    code = Path(spec["code_dir"])
    order = sorted(spec["repos"].items(), key=lambda kv: (kv[0] != "predictor", kv[0]))  # predictor first
    paths = [str(code / name / sub) for name, repo in order for sub in repo.get("pythonpath", ["."])]
    return ("#!/bin/sh\n# I6-N swarm pinned python wrapper (generated; do not edit)\n"
            f"export PYTHONPATH={shlex.quote(':'.join(paths))}\n"
            f"export I6N_DATA_ROOT={shlex.quote(data_root)}\nexport PYTHONDONTWRITEBYTECODE=1\n"
            f"exec {shlex.quote(python)} \"$@\"\n")


# --------------------------------------------------------------------------- effective config
def effective_campaign_config(spec: Mapping[str, Any], inv: Mapping[str, dict], admission: Mapping[str, Any],
                              addresses: Mapping[str, str], token_file: str) -> dict:
    root = Path(spec["campaign_root"])
    code = Path(spec["code_dir"])
    wrapper = str(code / "bin" / "i6n-python")
    workdir = str(code / "predictor")
    coordinator_inv = inv.get("control", {})
    hosts = [dict(role="coordinator", ledger_host=True, devices=[], python=wrapper, workdir=workdir,
                  data_dir=str(root / "hostdata" / "coordinator"), doin_port=None,
                  shared_data_root=spec["data_root"],
                  environment=dict(CUDA_VISIBLE_DEVICES=""))]
    for role in WORKER_ORDER:
        if role not in spec["workers"]:
            continue
        w = inv[role]
        env = dict(w.get("environment", {}))
        env["I6N_DATA_ROOT"] = spec["data_root"]
        hosts.append(dict(role=role, ledger_host=False, python=wrapper, workdir=workdir,
                          data_dir=str(root / "hostdata" / role), doin_port=int(spec["ports"]["doin"][role]),
                          shared_data_root=spec["data_root"],
                          environment=env,
                          devices=[dict(cls=DEVICE_CLASS[role], cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES", ""))]))
    admitted = {}
    for c, a in admission.items():
        if not a["admitted"]:
            continue
        entry = {k: a[k] for k in ("receipt", "child_wall_seconds", "small_pilot", "coverage_adjustment_bytes")}
        entry["coverage_derivation"] = a.get("coverage_derivation_pin") or a["coverage_derivation_sha256"]
        admitted[c] = entry
    if not admitted:
        raise DeployError("NO_ADMITTED_CLASS")
    launcher_paths = {inv[r]["admission"] for r in spec["workers"]}
    if len(launcher_paths) != 1:
        raise DeployError("LAUNCHER_PATH_DIFFERS_BETWEEN_HOSTS")
    cfg = dict(
        schema=CAMPAIGN_SCHEMA, campaign_id=spec["campaign_id"], mode="production", experiment=spec["experiment"],
        seed=SEED, prior_candidates=spec.get("prior_candidates"),
        evaluator_capabilities=list(spec.get("evaluator_capabilities",
                                             ["VALIDATION_OUTER_WEEK", "DIRECT_MULTI_OUTPUT_ARM"])),
        shared_data_root=spec["data_root"], search_plan=dict(spec["search_plan"]),
        weekly_views=dict(spec["c_views"]), weekly_plan=dict(spec["c_plan"]), multi_output=dict(spec["multi_output"]),
        hosts=hosts,
        ledger=dict(bind=addresses["coordinator"], port=int(spec["ports"]["ledger"]), token_file=token_file,
                    url=f"http://{addresses['coordinator']}:{int(spec['ports']['ledger'])}"),
        launcher=dict(crispdm_run=launcher_paths.pop(), label_prefix=f"i6n-swarm-{spec['campaign_id']}"),
        doin=dict(node_command=[wrapper, "-m", "doin_node.cli"],
                  shared_min_peers=int(spec.get("shared_min_peers", 0)), claim_timeout=600,
                  peers={r: f"{addresses[r]}:{int(spec['ports']['doin'][r])}" for r in spec["workers"]}),
        lease=dict(ttl_seconds=120, renew_seconds=30, max_wait_seconds=60, poll_seconds=5),
        class_admission=admitted, freeze_caps=dict(child_memory_bytes=4 * 1024 ** 3, child_wall_seconds=3600),
        tick_interval_seconds=300, local_roles=["coordinator"])
    if coordinator_inv:
        cfg["coordinator_inventory_role"] = "control"
    return cfg


def validate_campaign_config(cfg: Mapping[str, Any], *, local_exists: Callable[[str], bool] = os.path.exists) -> None:
    """Every substituted argument resolved: no placeholder, pinned inputs match, ports distinct."""
    text = json.dumps(cfg)
    if "<" in text and ">" in text:
        raise DeployError("PLACEHOLDER_LEFT", "a '<...>' placeholder remains in the effective config")
    for key in ("search_plan", "weekly_views", "weekly_plan", "multi_output"):
        pin = cfg[key]
        if not local_exists(pin["path"]) or sha256_file(Path(pin["path"])) != pin["sha256"]:
            raise DeployError("INPUT_PIN_MISMATCH", key)
    for c, a in cfg["class_admission"].items():
        if sha256_file(Path(a["receipt"]["path"])) != a["receipt"]["sha256"]:
            raise DeployError("PILOT_RECEIPT_DIGEST", c)
    ports = [h["doin_port"] for h in cfg["hosts"] if h["doin_port"]] + [cfg["ledger"]["port"]]
    if len(set(ports)) != len(ports) or any(not 1024 < p < 65536 for p in ports):
        raise DeployError("PORTS_INVALID")
    if cfg["seed"] != SEED or cfg["mode"] != "production":
        raise DeployError("SEED_OR_MODE")


# --------------------------------------------------------------------------- state
def _state_path(spec) -> Path:
    return Path(spec["deploy_dir"]) / "DEPLOY_STATE.json"


def read_state(spec) -> dict:
    path = _state_path(spec)
    return json.loads(path.read_text()) if path.exists() else {"schema": STATE_SCHEMA, "steps": {}}


def mark(spec, state: dict, step: str, **detail) -> None:
    state["steps"][step] = dict(detail, at=time.time())
    write_private_json(_state_path(spec), state)


# --------------------------------------------------------------------------- deploy
def data_files(spec: Mapping[str, Any]) -> Dict[str, Path]:
    root = Path(spec["data_root"])
    return {str(p): p for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith(".tmp")}


def campaign_files(spec: Mapping[str, Any]) -> Dict[str, Path]:
    root = Path(spec["campaign_root"])
    files = {str(root / "PLAN.json"): root / "PLAN.json"}
    files.update({str(p): p for p in sorted((root / "private").rglob("*")) if p.is_file()})
    return files


class Deployer:
    def __init__(self, spec: Mapping[str, Any], *, remotes: Optional[Mapping[str, Remote]] = None,
                 local: Optional[Remote] = None, run: Callable = subprocess.run,
                 addresses: Optional[Mapping[str, str]] = None):
        self.spec, self.run_local = spec, run
        self.inv = inventory(spec)
        self.remotes = dict(remotes) if remotes is not None else {
            r: Remote(self.inv[r]["host"]) for r in spec["workers"]}
        self.local = local or LocalHost()
        self._addresses = dict(addresses) if addresses else None
        self.state = read_state(spec)
        self.secrets = [self.inv[r]["host"] for r in spec["workers"] if r in self.inv]

    # -- addresses (private, never printed)
    def addresses(self) -> Dict[str, str]:
        if self._addresses is None:
            peers = {r: resolve_address(self.inv[r]["host"]) for r in self.spec["workers"]}
            first = next(iter(peers.values()))
            self._addresses = dict(peers, coordinator=local_source_address(first))
        self.secrets += list(self._addresses.values())
        return self._addresses

    def campaign_cli(self, *args: str) -> dict:
        code = Path(self.spec["code_dir"])
        cmd = [str(code / "bin" / "i6n-python"), "-m", "tools.i6n_swarm_campaign", *args]
        result = self.run_local(cmd, capture_output=True, text=True, cwd=str(code / "predictor"),
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        if result.returncode not in (0, 3):
            raise DeployError("CAMPAIGN_CLI_FAILED", _redact(f"{args[0]}: {result.stderr[-600:]}", self.secrets))
        try:
            return json.loads(result.stdout)
        except ValueError as exc:
            raise DeployError("CAMPAIGN_CLI_OUTPUT", args[0]) from exc

    # -- steps
    def prepare(self) -> dict:
        spec = self.spec
        code_files = export_code(spec, self.run_local)
        code = Path(spec["code_dir"])
        (code / "bin").mkdir(parents=True, exist_ok=True)
        coordinator_python = spec.get("coordinator_python") or self.inv.get("control", {}).get("fit_python")
        if not coordinator_python or not os.path.exists(coordinator_python):
            raise DeployError("COORDINATOR_PYTHON_MISSING")
        wrapper = code / "bin" / "i6n-python"
        wrapper.write_text(wrapper_text(spec, coordinator_python, spec["data_root"]))
        wrapper.chmod(0o755)
        self.code_manifest = manifest_lines(code_files)
        (Path(spec["deploy_dir"])).mkdir(parents=True, exist_ok=True)
        c_plan = json.loads(Path(spec["c_plan"]["path"]).read_text())
        e_plan = json.loads(Path(spec["search_plan"]["path"]).read_text())
        template = json.loads(Path(spec["cell_template"]["path"]).read_text())
        admission = class_admission(c_plan, e_plan["classes"], spec["pilots"], template["resources"],
                                    child_wall_seconds=template["resources"]["max_seconds"] + 120)
        for c, a in admission.items():
            if a.get("admitted"):
                path = Path(spec["deploy_dir"]) / f"coverage_{c}.json"
                write_private_json(path, a["coverage_derivation"])
                a["coverage_derivation_pin"] = {"path": str(path), "sha256": sha256_file(path)}
        token = Path(spec["deploy_dir"]) / "ledger.token"
        if not token.exists():
            fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_hex(32))
        cfg = effective_campaign_config(spec, self.inv, admission, self.addresses(), str(token))
        validate_campaign_config(cfg)
        cfg_path = Path(spec["deploy_dir"]) / "campaign.private.json"
        if cfg_path.exists() and json.loads(cfg_path.read_text()) != cfg:
            if (Path(spec["campaign_root"]) / "PLAN.json").exists():
                raise DeployError("CONFIG_CHANGED_AFTER_PLAN", "write a new campaign root instead")
        write_private_json(cfg_path, cfg)
        public_admission = {c: {k: v for k, v in a.items() if k not in ("receipt",)} for c, a in admission.items()}
        write_private_json(Path(spec["deploy_dir"]) / "ADMISSION.json", public_admission)
        mark(spec, self.state, "prepare", config_sha256=sha256_file(cfg_path), code_files=len(code_files),
             admitted=sorted(c for c, a in admission.items() if a["admitted"]))
        return cfg

    def coordinator(self) -> None:
        spec, root = self.spec, Path(self.spec["campaign_root"])
        cfg_path = Path(spec["deploy_dir"]) / "campaign.private.json"
        if not (root / "PLAN.json").exists():
            self.campaign_cli("plan", "--config", str(cfg_path), "--output", str(root))
        pre = self.campaign_cli("preflight", "--root", str(root), "--json")
        if pre.get("state") != "READY":
            failed = [c["name"] + ": " + str(c.get("cause")) for c in pre.get("checks", []) if not c.get("ok")]
            raise DeployError("PREFLIGHT_BLOCKED", _redact("; ".join(failed), self.secrets))
        self.campaign_cli("install", "--root", str(root), "--role", "coordinator")
        self.campaign_cli("start", "--root", str(root), "--role", "coordinator")
        mark(spec, self.state, "coordinator", preflight="READY")

    def worker(self, role: str, activate: bool = True) -> None:
        spec = self.spec
        remote = self.remotes[role]
        code = Path(spec["code_dir"])
        python = self.inv[role]["fit_python"]
        remote.run("test -x " + shlex.quote(python))
        remote.run("test -x " + shlex.quote(self.inv[role]["admission"]))
        files = {line.split("  ", 1)[1]: Path(line.split("  ", 1)[1]) for line in self.code_manifest.splitlines()}
        files.update(data_files(spec))
        files.update(campaign_files(spec))
        token = Path(spec["deploy_dir"]) / "ledger.token"
        files[str(token)] = token
        manifest = manifest_lines(files)
        manifest_path = str(Path(spec["deploy_dir"]) / "verify" / f"{role}.sha256")
        differing = remote_mismatches(remote, manifest, manifest_path)
        unknown = [p for p in differing if p not in files]
        if unknown:
            raise DeployError("REMOTE_VERIFY_UNPARSED", "cannot map a sha256sum failure to a file")
        if differing:
            if hasattr(remote, "put_many") and all(str(files[p]) == p for p in differing):
                remote.put_many(differing)
            else:
                for path in differing:
                    remote.put(files[path], path)
        remote.run("chmod 700 " + shlex.quote(str(Path(spec["campaign_root"]) / "private")) + " && chmod 600 "
                   + shlex.quote(str(token)))
        wrapper = Path(spec["deploy_dir"]) / f"i6n-python.{role}"
        wrapper.write_text(wrapper_text(spec, python, spec["data_root"]))
        remote.put(wrapper, str(code / "bin" / "i6n-python"))
        remote.run("chmod 755 " + shlex.quote(str(code / "bin" / "i6n-python")))
        still = remote_mismatches(remote, manifest, manifest_path)
        if still:
            raise DeployError("REMOTE_BYTES_DIFFER", f"{len(still)} file(s) differ after copy on {role}")
        self.ensure_git_pin(role)
        if not activate:
            mark(spec, self.state, f"replicated:{role}", files=len(files), verified=True)
            return
        root = shlex.quote(spec["campaign_root"])
        cli = f"cd {shlex.quote(str(code / 'predictor'))} && {shlex.quote(str(code / 'bin' / 'i6n-python'))} -m tools.i6n_swarm_campaign"
        remote.run(f"{cli} install --root {root} --role {role}")
        started = spec["workers"][role]["start"] == "immediate"
        if started:
            remote.run(f"{cli} start --root {root} --role {role}")
        mark(spec, self.state, f"worker:{role}", files=len(files), verified=True,
             started=started, policy=spec["workers"][role]["start"])

    def ensure_git_pin(self, role: str) -> None:
        """Make the predictor deploy commit present in the worker's own repo (refs only; no checkout change)."""
        repo_spec = self.spec["repos"]["predictor"]
        ref = repo_spec.get("bundle_ref")
        if not ref:
            return
        remote = self.remotes[role]
        pin = repo_spec["pin"]
        candidates = [self.inv[role].get("worktree"), "~/Documents/GitHub/predictor"]
        probe = " || ".join(f"( test -e {c}/.git && echo {c} )" for c in candidates if c)
        repo = remote.run(probe, check=False).stdout.strip().splitlines()
        if not repo:
            raise DeployError("WORKER_REPO_MISSING", role)
        repo = repo[0]
        if remote.run(f"git -C {repo} cat-file -e {shlex.quote(pin)}^{{commit}}", check=False).returncode == 0:
            return
        bundle = Path(self.spec["deploy_dir"]) / f"predictor-{pin[:12]}.bundle"
        if not bundle.exists():
            self.run_local(["git", "-C", repo_spec["path"], "bundle", "create", str(bundle), ref],
                           capture_output=True, check=True)
        remote.put(bundle, str(bundle))
        remote.run(f"git -C {repo} fetch -q {shlex.quote(str(bundle))} "
                   f"{shlex.quote(ref)}:refs/remotes/i6n-deploy/{shlex.quote(ref.split('/')[-1])}")
        if remote.run(f"git -C {repo} cat-file -e {shlex.quote(pin)}^{{commit}}", check=False).returncode != 0:
            raise DeployError("WORKER_PIN_ABSENT_AFTER_FETCH", role)

    def install_join_gate(self, role: str) -> None:
        spec = self.spec
        name = f"i6n-swarm-{spec['campaign_id']}-join-{role}"
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        code = Path(spec["code_dir"])
        service = ("[Unit]\nDescription=I6-N swarm join gate for " + role + "\n"
                   f"X-I6N-Swarm-Campaign={spec['campaign_id']}\n\n[Service]\nType=oneshot\n"
                   f"WorkingDirectory={code / 'predictor'}\nEnvironment=CUDA_VISIBLE_DEVICES=\nMemoryMax=256M\n"
                   "ExecStart=" + " ".join(shlex.quote(a) for a in [
                       str(code / "bin" / "i6n-python"), "-m", "tools.i6n_swarm_deploy", "--spec",
                       spec["spec_path"], "--join", role]) + "\n")
        timer = ("[Unit]\nDescription=I6-N swarm join gate timer for " + role + "\n"
                 f"X-I6N-Swarm-Campaign={spec['campaign_id']}\n\n[Timer]\nOnBootSec=4min\n"
                 f"OnUnitInactiveSec=300s\nPersistent=true\nUnit={name}.service\n\n[Install]\nWantedBy=timers.target\n")
        for suffix, text in ((".service", service), (".timer", timer)):
            path = unit_dir / (name + suffix)
            if path.exists() and f"X-I6N-Swarm-Campaign={spec['campaign_id']}" not in path.read_text():
                raise DeployError("FOREIGN_UNIT", path.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.local.run("systemctl --user daemon-reload && systemctl --user enable --now "
                       + shlex.quote(name + ".timer"))
        mark(spec, self.state, f"join_gate:{role}", unit=name + ".timer")

    def replicate(self) -> dict:
        """Copy and byte-verify code, data root and campaign root on every worker; no install/start."""
        self.prepare()
        if not (Path(self.spec["campaign_root"]) / "PLAN.json").exists():
            cfg_path = Path(self.spec["deploy_dir"]) / "campaign.private.json"
            self.campaign_cli("plan", "--config", str(cfg_path), "--output", self.spec["campaign_root"])
        for role in WORKER_ORDER:
            if role in self.spec["workers"]:
                self.worker(role, activate=False)
        return status(self.spec)

    def apply(self) -> dict:
        self.prepare()
        self.coordinator()
        for role in WORKER_ORDER:
            if role in self.spec["workers"]:
                self.worker(role)
                if self.spec["workers"][role]["start"] == "after_pilot":
                    self.install_join_gate(role)
        return status(self.spec)

    # -- join gate (timer-run, idempotent)
    def join(self, role: str) -> dict:
        spec = self.spec
        if self.state["steps"].get(f"joined:{role}"):
            return {"role": role, "state": "JOINED"}
        gate = spec["workers"][role].get("pilot")
        remote = self.remotes[role]
        verdict = {"role": role, "pilot": None}
        if gate:
            active = remote.run("systemctl --user is-active " + shlex.quote(gate["unit"]), check=False).stdout.strip()
            if active in ("active", "activating", "reloading"):
                verdict.update(state="WAIT_DEPENDENCY", cause=f"PILOT_LEASE_HELD: {gate['unit']} {active}")
                mark(spec, self.state, f"join_wait:{role}", cause=verdict["cause"])
                return verdict
            out = remote.run(f"cd {shlex.quote(gate['code_dir'])} && {shlex.quote(gate['python'])} -m "
                             f"tools.i6n_swarm.cost_pilot --config {shlex.quote(gate['config'])} --status").stdout
            pilot = json.loads(out)
            if pilot.get("state") == "COMPLETED":
                result = pilot.get("result") or {}
                actual = remote.run("sha256sum " + shlex.quote(result.get("path", "/nonexistent"))).stdout.split()[0]
                if actual != result.get("sha256"):
                    verdict.update(state="FAILED", cause="PILOT_RESULT_DIGEST_MISMATCH")
                    mark(spec, self.state, f"join_refused:{role}", cause=verdict["cause"])
                    return verdict
                verdict["pilot"] = {"state": "COMPLETED", "observed_peak_bytes": pilot.get("observed_peak_bytes"),
                                    "result_sha256": actual}
            elif pilot.get("state") == "FAILED":
                verdict["pilot"] = {"state": "FAILED", "cause": str(pilot.get("reason"))[:300]}
            else:
                verdict.update(state="WAIT_DEPENDENCY", cause=f"PILOT_NOT_TERMINAL: {pilot.get('state')}")
                mark(spec, self.state, f"join_wait:{role}", cause=verdict["cause"])
                return verdict
        code = Path(spec["code_dir"])
        cli = (f"cd {shlex.quote(str(code / 'predictor'))} && {shlex.quote(str(code / 'bin' / 'i6n-python'))} "
               f"-m tools.i6n_swarm_campaign start --root {shlex.quote(spec['campaign_root'])} --role {role}")
        remote.run(cli)
        verdict.update(state="JOINED")
        mark(spec, self.state, f"joined:{role}", pilot=verdict["pilot"])
        return verdict


# --------------------------------------------------------------------------- status
def status(spec: Mapping[str, Any], run: Callable = subprocess.run) -> dict:
    """Read-only: deploy steps, admission, campaign status (B's status command)."""
    state = read_state(spec)
    out = {"schema": "i6n.swarm_deploy.status.v1", "campaign_root": spec["campaign_root"],
           "steps": {k: {kk: vv for kk, vv in v.items()} for k, v in state["steps"].items()},
           "status_command": f"python -m tools.i6n_swarm_campaign status --root {spec['campaign_root']} --json"}
    admission = Path(spec["deploy_dir"]) / "ADMISSION.json"
    if admission.exists():
        out["admission"] = {c: {k: a.get(k) for k in ("admitted", "cause", "measured_peak_bytes",
                                                     "coverage_adjustment_bytes", "child_memory_bytes")}
                            for c, a in json.loads(admission.read_text()).items()}
    if (Path(spec["campaign_root"]) / "PLAN.json").exists():
        code = Path(spec["code_dir"])
        result = run([str(code / "bin" / "i6n-python"), "-m", "tools.i6n_swarm_campaign", "status", "--root",
                      spec["campaign_root"], "--json"], capture_output=True, text=True, cwd=str(code / "predictor"),
                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        out["campaign"] = json.loads(result.stdout) if result.returncode == 0 else {"error": result.stderr[-300:]}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--join", choices=sorted(DEVICE_CLASS))
    group.add_argument("--replicate", action="store_true", help="copy + verify on workers only")
    group.add_argument("--write-status", action="store_true", help="write <campaign root>/STATUS.json (read-only otherwise)")
    args = parser.parse_args(argv)
    spec = dict(load_spec(Path(args.spec)), spec_path=str(Path(args.spec).resolve()))
    if args.status or args.write_status:
        result = status(spec)
        if args.write_status:
            result["generated_at"] = time.time()
            result["generated_by"] = "python -m tools.i6n_swarm_deploy --spec <private spec> --write-status"
            result["live_status"] = "status/STATUS.json (rewritten by the coordinator tick timer every 5 min)"
            (Path(spec["campaign_root"]) / "STATUS.json").write_text(
                json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
        print(json.dumps(result, indent=1, sort_keys=True, default=str))
        return 0
    deployer = Deployer(spec)
    try:
        if args.dry_run:
            cfg = deployer.prepare()
            print(json.dumps({"state": "PREPARED", "admitted": sorted(cfg["class_admission"]),
                              "hosts": [h["role"] for h in cfg["hosts"]]}, indent=1))
        elif args.replicate:
            result = deployer.replicate()
            print(json.dumps({"state": "REPLICATED", "steps": sorted(result["steps"])}, indent=1))
        elif args.join:
            print(json.dumps(deployer.join(args.join), indent=1, sort_keys=True))
        else:
            result = deployer.apply()
            print(json.dumps({"state": "APPLIED", "steps": sorted(result["steps"])}, indent=1))
    except DeployError as exc:
        print(json.dumps({"state": "REFUSED", "code": exc.code,
                          "detail": _redact(str(exc), deployer.secrets)}, indent=1))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
