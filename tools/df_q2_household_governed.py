#!/usr/bin/env python3
"""Q2 household lane: prove the governed chain end to end, and measure a scope peak from inside a child.

Two bounded units, each with its own campaign and its own delivery, through the client path the Q2
block itself uses (`tools/df_e1_governed.py` -> `tools/governed_run.py`).  Nothing here fits a model,
selects an arm, re-binds a historical cell or opens a reserve.

  probe   the mechanical unit: campaign -> governed delivery of the household panel -> a CHILD that
          re-verifies the consumed bytes by sha256 inside itself, reads the panel mechanically, and
          reports its own cgroup peak -> terminal -> accounting -> read BACK out of the warehouse.
  pilot   the bounded memory pilot: the same governed shape, and the child materialises the W1440
          data-stage footprint (window gathering into the block's own dtype, bounded by
          --max-windows) so that the SCOPE peak is measured from inside the child rather than
          inferred from a release that may never happen.

Why the peak is read from inside the child, and what it is NOT:

  * `resource.getrusage(RUSAGE_SELF).ru_maxrss` -- which `tools/df_e1_block.py:run_cell` records as
    `peak_rss_bytes` -- is ONE PROCESS'S RESIDENT SET.  It is not a cgroup peak and it may not size
    a cap.
  * `df_e1_block.py:run_units` launches each cell with a bare `subprocess.run([sys.executable, ...])`.
    No Q2 cell child ever took a launcher scope, so NO PER-CELL CGROUP PEAK EXISTS for any Q2 cell.
  * the launcher's own tree-peak sampler writes its number back only on a SUCCESSFUL release, so the
    measurement an incident most needs is the one an incident most easily loses.  Reading
    `memory.peak` of this process's own cgroup, from inside the child, does not depend on that.
  * a peak for a child shorter than the sampler's interval is a FLOOR, not a measurement, and a null
    peak means NOT MEASURED, never small.  Both are recorded as such.

usage (both stages run under the launcher, which supplies the scope this reads):
  crispdm-run -m 3G -t 900 -n q2h-probe -- python3 tools/df_q2_household_governed.py probe \\
      --root ROOT --api-key-file KEY --gov-url URL --warehouse-url URL --warehouse-token-file T
  crispdm-run -m 6G -t 1800 -n q2h-pilot -- python3 tools/df_q2_household_governed.py pilot \\
      --root ROOT --api-key-file KEY --gov-url URL --max-windows 20000
  python3 tools/df_q2_household_governed.py child --root ROOT --unit UNIT   (internal)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCHEMA = "df_q2_household_governed.v1"
LAKE = "public_panels"
RESOURCE = "uci_235_individual_household_power/panel.parquet"
PANEL_SHA256 = "b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db"
PANEL_BYTES = 10890295
W_LONG, H0 = 1440, 60


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------- the measurement, from inside

def own_cgroup() -> Path | None:
    try:
        line = Path("/proc/self/cgroup").read_text(encoding="utf-8").strip().splitlines()[0]
    except OSError:
        return None
    if "::" not in line:
        return None
    rel = line.split("::", 1)[1].lstrip("/")
    path = Path("/sys/fs/cgroup") / rel
    return path if path.is_dir() else None


def _int_file(path: Path):
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value in ("", "max"):
        return value or None
    try:
        return int(value)
    except ValueError:
        return None


def scope_memory(label: str) -> dict:
    """The whole-cgroup facts this process can read about ITSELF, with their provenance named."""
    cg = own_cgroup()
    out = {"at": now_iso(), "label": label,
           "cgroup": str(cg) if cg else None,
           "cgroup_is_a_launcher_scope": bool(cg and cg.name.endswith(".scope")),
           "peak_bytes": None, "current_bytes": None, "max_bytes": None, "high_bytes": None,
           "peak_provenance": "NOT_MEASURED_NO_CGROUP_READABLE",
           "rss_self_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
           "rss_children_bytes": int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024,
           "rss_is_not_a_cgroup_peak": "ru_maxrss is one process's resident set; it may not size a cap"}
    if cg is None:
        return out
    out["peak_bytes"] = _int_file(cg / "memory.peak")
    out["current_bytes"] = _int_file(cg / "memory.current")
    out["max_bytes"] = _int_file(cg / "memory.max")
    out["high_bytes"] = _int_file(cg / "memory.high")
    if out["peak_bytes"] is None:
        out["peak_provenance"] = "NOT_MEASURED_NO_MEMORY_PEAK_FILE"
    elif not out["cgroup_is_a_launcher_scope"]:
        out["peak_provenance"] = ("CGROUP_PEAK_OF_A_NON_SCOPE_CGROUP: this is the peak of whatever "
                                  "cgroup this process was placed in, not of a job's own scope")
    else:
        out["peak_provenance"] = ("WHOLE_CGROUP_PEAK_OF_THIS_JOB_S_OWN_SCOPE, read from inside the "
                                  "child and therefore independent of a successful release")
    return out


# ----------------------------------------------------------------------------------- the design

def design_of(unit: str, extra: dict) -> dict:
    body = {"schema": SCHEMA, "unit": unit, "lake": LAKE, "resource": RESOURCE,
            "expected_panel_sha256": PANEL_SHA256, "expected_panel_bytes": PANEL_BYTES,
            "classification": "NON_GOVERNING", "phase": "DEVELOPMENT",
            "estimand": "NONE: this unit measures a chain and a memory footprint; it scores no model",
            "not_an_arm": ("no fit, no arm, no seed, no checkpoint selection, no re-binding of any "
                           "historical cell and no reserve access"),
            **extra}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["design_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return body


# ---------------------------------------------------------------------------------- the child

def child(root: Path, unit: str) -> dict:
    root = Path(root)
    design = json.loads((root / f"DESIGN.{unit}.json").read_text(encoding="utf-8"))
    entry = scope_memory("child_entry")
    G = _module("df_e1_governed")
    # THE READER: a unit may only read a panel delivered to THAT unit; this refuses otherwise
    delivered = G.require_delivery(root, design, unit)["delivery"]
    path = Path(delivered["path"])
    consumed_sha = sha_file(path)                       # re-verified by sha256 INSIDE the child
    if consumed_sha != delivered["sha256"]:
        raise SystemExit("REFUSED: the bytes this child consumed are not the bytes delivered to it")
    if consumed_sha != design["expected_panel_sha256"]:
        raise SystemExit("REFUSED: the delivered panel is not the characterised household panel")
    after_read = scope_memory("after_bytes_verified")

    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    numeric = [f.name for f in table.schema if str(f.type).startswith(("double", "float", "int"))]
    rows, cols = table.num_rows, table.num_columns
    after_parquet = scope_memory("after_parquet_read")

    work: dict = {"rows": rows, "columns": cols, "numeric_columns": len(numeric)}
    if design.get("stage") == "pilot":
        # the W1440 DATA STAGE ONLY: the channel matrix in the block's dtype, and window gathering
        # into a bounded number of windows.  No model, no optimizer, no loss, no fit.
        chans = numeric[:int(design["channels"])]
        X = np.column_stack([np.asarray(table[c].to_numpy(zero_copy_only=False),
                                        dtype=np.float32) for c in chans])
        after_matrix = scope_memory("after_channel_matrix")
        n_max = int(design["max_windows"])
        usable = max(0, X.shape[0] - W_LONG - H0)
        n = min(n_max, usable)
        idx = np.arange(n, dtype=np.int64)
        offs = np.arange(W_LONG, dtype=np.int64)
        block = (idx[:, None] + offs[None, :])
        windows = X[block]                              # (n, 1440, channels) float32
        checksum = float(np.nansum(windows[:, -1, :], dtype=np.float64))
        after_windows = scope_memory("after_window_gather")
        work.update({"channels_used": chans, "matrix_shape": list(X.shape),
                     "windows_shape": list(windows.shape),
                     "window_bytes": int(windows.nbytes), "matrix_bytes": int(X.nbytes),
                     "usable_origins_at_W1440_h60": int(usable),
                     "windows_materialised": int(n),
                     "bounded_by": "max_windows" if n == n_max else "usable_origins",
                     "last_step_checksum": checksum,
                     "bytes_per_window": int(windows.nbytes // max(1, n)),
                     "full_population_window_bytes_if_all_origins_materialised":
                         int(usable) * W_LONG * len(chans) * 4,
                     "note": ("this is the DATA STAGE footprint of one W1440 arm, not a cell: no "
                              "model, gradients, optimizer slots or Keras graph are built here, so "
                              "it is a LOWER BOUND on a W1440 cell and never a cap for one")})
        samples = [entry, after_read, after_parquet, after_matrix, after_windows]
        del windows, block, X
    else:
        samples = [entry, after_read, after_parquet]
    exit_sample = scope_memory("child_exit")
    samples.append(exit_sample)
    peaks = [s["peak_bytes"] for s in samples if s["peak_bytes"] is not None]
    record = {"schema": f"{SCHEMA}.child", "unit": unit, "design_sha256": design["design_sha256"],
              "host_kind": os.environ.get("CRISPDM_HOST_KIND", "UNDECLARED"),
              "delivery": {"delivery_id": delivered.get("delivery_id"),
                           "cached": delivered.get("cached"),
                           "verification_state": delivered.get("verification_state"),
                           "availability_contract_sha256": delivered.get("availability_contract_sha256"),
                           "declared_sha256": delivered["sha256"], "bytes": delivered.get("bytes")},
              "consumed": {"path_is_content_addressed_cache": "cache" in str(path),
                           "sha256_reverified_in_child": consumed_sha,
                           "bytes_on_disk": path.stat().st_size,
                           "matches_delivery": consumed_sha == delivered["sha256"],
                           "matches_characterised_panel": consumed_sha == PANEL_SHA256},
              "work": work,
              "memory": {"samples": samples,
                         "scope_peak_bytes": max(peaks) if peaks else None,
                         "scope_peak_provenance": (exit_sample["peak_provenance"] if peaks
                                                   else "NOT_MEASURED_NULL_IS_NEVER_SMALL"),
                         "rss_self_peak_bytes": max(s["rss_self_bytes"] for s in samples),
                         "declared_cap_bytes_as_the_kernel_holds_it": exit_sample["max_bytes"]},
              "interpreter": {"python": sys.version.split()[0], "executable": sys.executable},
              "at": now_iso()}
    write(root / "CHILD" / f"{unit}.json", record)
    print(json.dumps({"unit": unit, "scope_peak_bytes": record["memory"]["scope_peak_bytes"],
                      "rss_self_peak_bytes": record["memory"]["rss_self_peak_bytes"],
                      "consumed_sha256_ok": record["consumed"]["matches_delivery"]}), flush=True)
    return record


# --------------------------------------------------------------------------------- the warehouse

REGISTRY = (Path.home() / ".local/state/crispdm-data-foundation"
            / "musashi-store-adoption-20260914T181826Z/5055.runtime.json")


def _warehouse_token(a):
    """The warehouse credential, read in process from a path an existing client already reads.

    It is never printed, copied into an artifact, written to a second file or logged.
    """
    path = getattr(a, "warehouse_token_file", None)
    if path and Path(path).is_file():
        text = Path(path).read_text(encoding="utf-8")
        for line in text.splitlines():                # a deployed systemd EnvironmentFile
            if line.startswith("DATA_GOV_LAKE_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        return text.strip().strip('"').strip("'")     # or a bare one-line token file
    lake_id = getattr(a, "warehouse_token_from_registry", None)
    if lake_id and REGISTRY.is_file():
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        entry = next((lake for lake in registry.get("lakes", [])
                      if lake.get("lake_id") == lake_id), None)
        if entry and entry.get("lake_service_token"):
            return entry["lake_service_token"]
    return None


def warehouse_readback(a, campaign_sha256: str, unit: str) -> dict:
    """Read the accepted unit BACK out of the live warehouse, through the existing close client."""
    url = a.warehouse_url
    out = {"url_scheme_only": url.split("://", 1)[0], "campaign_sha256": campaign_sha256,
           "unit": unit, "ok": False,
           "reader": "tools/df_mod_e0_close.py:warehouse_terminals (the existing close client)"}
    token = _warehouse_token(a)
    if not token:
        out["state"] = "NOT_ATTEMPTED_NO_WAREHOUSE_CREDENTIAL_REACHABLE"
        return out
    try:
        C = _module("df_mod_e0_close")
        found = C.warehouse_terminals(url, token, campaign_sha256)
    except Exception as exc:                                     # the read is evidence, not control
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return out
    row = (found.get("current") or {}).get(unit)
    out["rows_all_generations"] = found.get("rows_all_generations")
    if row:
        out["terminal"] = {k: row.get(k) for k in
                           ("unit_id", "status", "generation", "terminal_sha256", "config_sha256",
                            "started_at", "finished_at")}
        out["metric_rows"] = len(row.get("metrics") or [])
        out["metrics"] = [{k: m.get(k) for k in ("metric", "split", "horizon", "unit", "value")}
                          for m in (row.get("metrics") or [])]
        out["artifact_rows"] = len(row.get("artifacts") or [])
        out["artifacts"] = row.get("artifacts")
        out["ok"] = row.get("status") == "COMPLETED"
    else:
        out["state"] = "NO_TERMINAL_ROW_FOR_THIS_UNIT_IN_THE_WAREHOUSE"
    return out


# --------------------------------------------------------------------------------------- stages

def run_unit(a, unit: str, design: dict) -> dict:
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    write(root / f"DESIGN.{unit}.json", design)
    G = _module("df_e1_governed")
    G._load("governed_run")
    G._load("df_e1_receipts")
    U = _module("df_utility_run")
    started = U._z(U.now_iso())
    parent_before = scope_memory("parent_before_child")

    acquired = G.acquire(run_id=a.run_id, root=root, lake=LAKE, resource=RESOURCE, unit_id=unit,
                         role="panel", gov_url=a.gov_url, api_key_file=a.api_key_file,
                         design_sha256=design["design_sha256"], cache_dir=root / "cache",
                         expect_sha256=PANEL_SHA256)
    delivery = acquired["units"][unit]

    wall = time.monotonic()
    cpu0 = time.process_time()
    with open(root / f"{unit}.log", "w") as log:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "child",
                               "--root", str(root), "--unit", unit],
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"},
                              stdout=log, stderr=subprocess.STDOUT, timeout=a.child_timeout)
    ok = proc.returncode == 0 and (root / "CHILD" / f"{unit}.json").is_file()
    rec = json.loads((root / "CHILD" / f"{unit}.json").read_text()) if ok else None
    parent_after = scope_memory("parent_after_child")

    metrics = []
    if rec:
        metrics.append(U._metric("q2h.consumed_bytes", float(rec["consumed"]["bytes_on_disk"]), "bytes"))
        metrics.append(U._metric("q2h.panel_rows", float(rec["work"]["rows"]), "rows"))
        peak = rec["memory"]["scope_peak_bytes"]
        if peak is not None:
            metrics.append(U._metric("q2h.scope_peak_bytes", float(peak), "bytes"))
        if rec["work"].get("window_bytes"):
            metrics.append(U._metric("q2h.window_tensor_bytes", float(rec["work"]["window_bytes"]), "bytes"))
    terminal = U._terminal(
        status="COMPLETED" if ok else "FAILED",
        reason=None if ok else f"child exited {proc.returncode}; the retained log says why",
        cost={"wall_seconds": time.monotonic() - wall, "cpu_seconds": time.process_time() - cpu0},
        metrics=metrics, started=started, finished=U._z(U.now_iso()),
        tags={"purpose": design["schema"], "classification": "NON_GOVERNING", "phase": "DEVELOPMENT",
              "unit": unit, "role": design.get("stage", "PROBE").upper(),
              "design_sha256": design["design_sha256"],
              "estimand": "NONE_THIS_UNIT_SCORES_NO_MODEL"})
    terminal["artifacts"] = ([{"role": "record", "sha256": sha_file(root / "CHILD" / f"{unit}.json"),
                               "bytes": (root / "CHILD" / f"{unit}.json").stat().st_size}] if ok else [])
    write(root / "TERMINALS" / f"{unit}.json", terminal)
    reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url,
                                 api_key_file=a.api_key_file,
                                 outbox_dir=str(root / "outbox"), started_at=started)
    accepted = not (reported["flushed"]["pending"] or reported["flushed"]["failures"])
    readback = warehouse_readback(a, delivery["campaign_sha256"], unit)
    out = {"schema": f"{SCHEMA}.unit", "unit": unit, "design_sha256": design["design_sha256"],
           "child_exit": proc.returncode, "child_ok": ok,
           "delivery": {k: delivery.get(k) for k in
                        ("delivery_id", "sha256", "bytes", "cached", "verification_state",
                         "availability_use", "availability_label", "availability_contract_sha256",
                         "campaign_sha256", "campaign_key", "bytes_on_disk_sha256")},
           "terminal": {"status": terminal["status"], "accepted": accepted,
                        "flushed": reported["flushed"],
                        "receipt_terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256"),
                        "reconciliation": reported["reconciliation"]},
           "warehouse": readback,
           "child_record": rec,
           "parent_scope_memory": {"before": parent_before, "after": parent_after},
           "at": now_iso()}
    write(Path(a.root) / f"UNIT.{unit}.json", out)
    print(json.dumps({k: out[k] for k in ("unit", "child_ok", "terminal", "warehouse")},
                     indent=1, default=str))
    if not (ok and accepted):
        raise SystemExit(f"REFUSED: unit {unit} did not complete and be accepted")
    return out


def cmd_probe(a) -> int:
    design = design_of(a.unit or "probe",
                       {"stage": "probe",
                        "work": "verify the consumed bytes by sha256 inside the child and read the "
                                "panel mechanically; no tensors are materialised"})
    run_unit(a, a.unit or "probe", design)
    return 0


def cmd_pilot(a) -> int:
    design = design_of(a.unit or "pilot",
                       {"stage": "pilot", "max_windows": int(a.max_windows),
                        "channels": int(a.channels), "window": W_LONG, "horizon": H0,
                        "work": "materialise the W1440 DATA-STAGE footprint only, bounded by "
                                "max_windows, and read this job's own cgroup peak from inside the "
                                "child; no model is built and nothing is fitted",
                        "declares": "the result is a FLOOR for a W1440 cell's data stage and is not "
                                    "a cap for any cell; a successor cap is a separate declaration"})
    run_unit(a, a.unit or "pilot", design)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["probe", "pilot", "child", "readback"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--unit")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--gov-url", default="http://127.0.0.1:5055")
    ap.add_argument("--api-key-file", type=Path)
    ap.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    ap.add_argument("--warehouse-token-file", type=Path,
                    help="a bare token file, or the deployed systemd EnvironmentFile that carries "
                         "DATA_GOV_LAKE_TOKEN; read in process, never printed, copied or logged")
    ap.add_argument("--warehouse-token-from-registry", default=None, metavar="LAKE_ID",
                    help="read the warehouse credential in process from the governance registry "
                         "entry of this lake id; it is never printed, copied or logged")
    ap.add_argument("--max-windows", type=int, default=20000)
    ap.add_argument("--channels", type=int, default=7)
    ap.add_argument("--child-timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    if a.command == "child":
        child(a.root, a.unit)
        return 0
    if a.command == "readback":
        held = json.loads((Path(a.root) / f"UNIT.{a.unit}.json").read_text(encoding="utf-8"))
        out = warehouse_readback(a, held["delivery"]["campaign_sha256"], a.unit)
        write(Path(a.root) / f"WAREHOUSE.{a.unit}.json", out)
        held["warehouse"] = out
        write(Path(a.root) / f"UNIT.{a.unit}.json", held)
        print(json.dumps(out, indent=1, default=str))
        return 0 if out["ok"] else 1
    if not a.api_key_file:
        raise SystemExit("REFUSED: --api-key-file is required; the key is read from its file by the "
                         "existing client and is never printed, copied or logged")
    a.run_id = a.run_id or f"q2h-{int(time.time())}"
    return cmd_probe(a) if a.command == "probe" else cmd_pilot(a)


if __name__ == "__main__":
    sys.exit(main())
