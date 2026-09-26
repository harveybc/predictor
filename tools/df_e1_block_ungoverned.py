#!/usr/bin/env python3
"""UNGOVERNED development driver for a SEALED `tools/df_e1_block.py` design.

Why it exists, stated before anything it produces is read: this host holds no
data-gov service key, so `df_e1_block.py`'s governed path
(`acquire` -> `child` -> `report_terminal`) cannot run here at all — it raises
`TypeError` on a `None` `--api-key-file` before any data is opened. Per
`docs/GOVERNED_RUN.md`, "Small mechanics tests may use other runners. Their
results are non-governing and cannot promote a model, transformation, feature or
experiment." This driver is such a runner, and everything it writes is
NON_GOVERNING.

What it does NOT change: the design, the recipe, the rows, the scaler, the
cadence, the ceiling, the seeds, the enumeration, the model builders, the
training loop, the scoring, the reload parity, the fresh-process replay or the
closure. Every one of those comes from `df_e1_block` itself, imported and
called; nothing is reimplemented here.

What it does NOT have, and says so in `UNGOVERNED_RUN.json` and in every
closure this produces:

  * no data-gov acquisition, so no `DELIVERIES.json` and no delivery id;
  * no accepted terminal and no receipt, so no `TERMINALS/` and no
    `TERMINAL_RECEIPTS.json`;
  * no warehouse read, so every row's custody is `UNCHECKED` and the
    preparation's custody is `PREPARATION_LOCAL_ONLY`.

Its data custody is the one thing it can honestly assert: the panel bytes it
reads are addressed by content, and it refuses unless their sha256 equals the
design's `source_run.panel_sha256` — the digest a previous GOVERNED acquisition
verified. That is bytes-identity, not custody of the transfer, and the record
says which.

It writes no terminal, no receipt and no delivery record, so nothing it produces
can be mistaken for a governed artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _block():
    spec = importlib.util.spec_from_file_location("df_e1_block", HERE / "df_e1_block.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["df_e1_block"] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UngovernedRefusal(SystemExit):
    pass


def bind_panel(design: dict, panel: Path) -> dict:
    """The only custody claim this driver makes: the bytes read hash to the digest the design names."""
    got = sha_file(panel)
    want = design["source_run"]["panel_sha256"]
    if got != want:
        raise UngovernedRefusal(f"REFUSED: the panel bytes hash to {got[:16]}, the design names {want[:16]}")
    return {"path": str(panel), "sha256": got, "bytes": panel.stat().st_size,
            "custody": "BYTES_IDENTITY_ONLY",
            "why": "the file's sha256 equals the design's source_run.panel_sha256, which a previous GOVERNED acquisition "
                   "recorded as VERIFIED_TRANSFER; this driver re-verified the BYTES, not the transfer, and holds no "
                   "delivery id, no availability contract and no acceptance"}


def write_run_record(root: Path, design: dict, delivery: dict, extra: dict) -> dict:
    doc = {"schema": "df_e1_block_ungoverned_run.v1", "at": _now(),
           "design_sha256": design["design_sha256"], "block": design["block"],
           "classification": "NON_GOVERNING",
           "governance": {"data_gov_acquisition": "ABSENT", "accepted_terminal": "ABSENT", "receipt": "ABSENT",
                          "warehouse_read": "ABSENT",
                          "why": "no data-gov service key is held on this host; the governed runner refuses before opening "
                                 "data, so this driver ran instead and its results promote nothing"},
           "consequences": ["every closure row's custody is UNCHECKED: no accepted payload anchors the score",
                            "the preparation's custody is PREPARATION_LOCAL_ONLY",
                            "the closure's `verified` flag is False for that reason and for no other unless it names one",
                            "these numbers are DEVELOPMENT measurements and cannot select, rank or promote anything"],
           "delivery": delivery,
           "interpreter": {"python": platform.python_version(), "executable": sys.executable},
           "sealed_code": design["source_code"]}
    doc.update(extra)
    p = root / "UNGOVERNED_RUN.json"
    held = json.loads(p.read_text()) if p.is_file() else {"steps": []}
    held.setdefault("steps", []).append(doc)
    p.write_text(json.dumps(held, indent=1, default=str))
    return doc


def cmd_prepare(a) -> int:
    import pandas as pd
    B = _block()
    root = Path(a.root)
    design = json.loads((root / "DESIGN.json").read_text())
    B.validate(design)
    delivery = bind_panel(design, Path(a.panel))
    rec = B.prepare(design, root, frame=pd.read_parquet(delivery["path"]))
    write_run_record(root, design, delivery, {"step": "prepare", "data_sha256": rec["data_sha256"],
                                              "common_evaluation": rec["common_evaluation"],
                                              "binding_to_source": rec["binding_to_source"]})
    print(json.dumps({k: rec[k] for k in ("common_evaluation", "binding_to_source", "feasibility")}, indent=1))
    return 0


def cmd_cell(a) -> int:
    """One cell, in this process; the caller runs this as a fresh subprocess so peak RSS and CPU are the cell's own."""
    B = _block()
    root = Path(a.root)
    design = json.loads((root / "DESIGN.json").read_text())
    B.validate(design)
    cpu = int(design["limits"]["child_cpu_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
    data = B.load_data(root, design)
    cell = next(c for c in design["pilots"] + design["cells"] if c["cell_id"] == a.unit)
    rec = B.run_cell(design, data, cell, root / "attempts" / a.unit, pilot=cell.get("role") == "COST_PILOT")
    print(json.dumps({"unit": a.unit, "mae_z": rec["scores"]["mae_z"], "mae_kw": rec["scores"]["mae_kw"],
                      "stop": rec["training"]["stop_reason"], "censoring": rec["training"]["censoring"]["verdict"],
                      "updates": rec["training"]["updates"], "cpu_seconds": rec["cost"]["cpu_seconds"],
                      "peak_rss_bytes": rec["cost"]["peak_rss_bytes"]}))
    return 0


def cmd_run(a) -> int:
    """The requested units, each in a fresh subprocess, one at a time. A unit whose attempt folder exists is NEVER
    repeated: a recorded attempt is reused as it landed, never re-trained to get a different number."""
    B = _block()
    root = Path(a.root)
    design = json.loads((root / "DESIGN.json").read_text())
    B.validate(design)
    delivery = json.loads((root / "UNGOVERNED_RUN.json").read_text())["steps"][0]["delivery"]
    known = {c["cell_id"]: c for c in design["pilots"] + design["cells"]}
    unknown = [u for u in a.units if u not in known]
    if unknown:
        raise UngovernedRefusal(f"REFUSED: not units of this design: {unknown}")
    out = []
    for unit in a.units:
        if (root / "attempts" / unit / "cell.json").is_file():
            rec = json.loads((root / "attempts" / unit / "cell.json").read_text())
            out.append({"unit": unit, "reused": True, "mae_z": rec["scores"]["mae_z"]})
            print(json.dumps(out[-1]), flush=True)
            continue
        with open(root / f"{unit}.log", "a") as log:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "cell", "--root", str(root), "--unit", unit],
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TF_CPP_MIN_LOG_LEVEL": "3"},
                                  stdout=subprocess.PIPE, stderr=log, text=True,
                                  timeout=int(design["limits"]["child_wall_seconds"]))
        if proc.returncode != 0 or not (root / "attempts" / unit / "cell.json").is_file():
            out.append({"unit": unit, "ok": False, "exit": proc.returncode})
            print(json.dumps(out[-1]), flush=True)
            write_run_record(root, design, delivery, {"step": "run", "units": out,
                                                      "stopped": f"{unit} failed; the remaining units were not started"})
            raise UngovernedRefusal(f"REFUSED: {unit} failed (exit {proc.returncode}); the remaining units are not started")
        line = json.loads(proc.stdout.strip().splitlines()[-1])
        out.append({**line, "ok": True})
        print(json.dumps(out[-1]), flush=True)
    write_run_record(root, design, delivery, {"step": "run", "units": out})
    return 0


def cmd_pilot_report(a) -> int:
    """The cost projection of THIS design from ITS OWN pilots, by df_e1_block's own arithmetic. The decision field is
    advisory here: this driver never gates a fit on it, it publishes it."""
    B = _block()
    root = Path(a.root)
    design = json.loads((root / "DESIGN.json").read_text())
    results = [{"unit": c["cell_id"], "ok": True, "record": json.loads((root / "attempts" / c["cell_id"] / "cell.json").read_text())}
               for c in design["pilots"]]
    proj = B.projection(design, results)
    spent = B.spent_cpu(root)
    fits = proj["with_headroom_25_percent"] + spent + design["limits"]["closure_reserve_seconds"] <= design["limits"]["campaign_cpu_seconds"]
    doc = {"schema": "df_e1_block_pilot_report.v1", "design_sha256": design["design_sha256"], "spent_cpu_seconds": spent,
           "projection": proj, "fits_the_ceiling": fits,
           "decision": "EXECUTE" if fits else "BUDGET_LIMITED_BEFORE_ANY_OUTCOME",
           "governance": "NON_GOVERNING: produced by tools/df_e1_block_ungoverned.py, no accepted terminal"}
    (root / "REPORT.pilot.json").write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(doc, indent=1, default=str))
    return 0 if fits else 2


def cmd_baselines(a) -> int:
    """The design's three declared references on the common evaluation rows, by df_e1_block.baselines. A failed closure
    suppresses these on purpose; they are published here separately so a reader is never left without the naive."""
    B = _block()
    root = Path(a.root)
    design = json.loads((root / "DESIGN.json").read_text())
    data = B.load_data(root, design)
    out = {"schema": "df_e1_block_ungoverned_baselines.v1", "design_sha256": design["design_sha256"],
           "common_evaluation_rows": int(data["common_eval"].size), "baselines": B.baselines(data, design),
           "classification": "NON_GOVERNING"}
    (root / "BASELINES.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
    return 0


def cmd_close(a) -> int:
    """df_e1_block.close, unchanged, with no warehouse. It will record `verified: false` and name the reason; the report
    is published exactly as it lands."""
    B = _block()
    root = Path(a.root)
    ns = argparse.Namespace(root=root, warehouse_url=None, warehouse_token_file=None, replay_evidence=None,
                            skip_replay=False)
    try:
        report = B.close(ns)
        verdict = "verified"
    except SystemExit as exc:
        report = json.loads((root / "REPORT.json").read_text())
        verdict = f"refused: {exc}"
    design = json.loads((root / "DESIGN.json").read_text())
    delivery = json.loads((root / "UNGOVERNED_RUN.json").read_text())["steps"][0]["delivery"]
    write_run_record(root, design, delivery, {"step": "close", "closure_verified": report.get("verified"),
                                              "closure_problems": report.get("problems"), "verdict": verdict})
    print(json.dumps({"verified": report.get("verified"), "problems": report.get("problems"),
                      "rows": len(report.get("rows") or [])}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "cell", "run", "pilot-report", "baselines", "close"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--panel", type=Path, help="content-addressed panel parquet; its sha256 must equal the design's panel_sha256")
    ap.add_argument("--unit")
    ap.add_argument("--units", nargs="*", default=[])
    a = ap.parse_args(argv)
    return {"prepare": cmd_prepare, "cell": cmd_cell, "run": cmd_run, "pilot-report": cmd_pilot_report,
            "baselines": cmd_baselines, "close": cmd_close}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
