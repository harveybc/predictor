#!/usr/bin/env python3
"""C178: the four D2 grains loaded in bounded parts; historical reanalysis and fresh confirmation stay separate by run.

  c178_d2_load.py --mode throwaway|real --receipt <write-once json> --table-dir DIR [--table-dir DIR ...] [--chunk N]

Each table directory holds table-named JSONL files of df_fact_d2_unit_denoising, df_fact_d2_unit_snr,
df_fact_d2_decision and df_fact_d2_historical_reanalysis, plus TABLES_MANIFEST.json. Rows are streamed in chunks,
never a whole table in memory, and inserted with load_data_foundation.load: every row validated, row_sha256
primary key, ON CONFLICT DO NOTHING, one load-receipt row per chunk.

throwaway: a new database, the schema, every chunk loaded twice (the second pass must insert nothing), the
database dropped. real: the row count of every table already in the cube is taken before and after; every table
except the D2 grains, df_fact_load_receipt and df_dim_run must be unchanged. Both modes check, per table, that
offered = inserted + already present + refused, that no row is refused, and that the input holds no duplicate
row identity (so nothing is deduplicated silently). The running outbox loader is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]     # docs/audits/evidence/repro_runs/c166_c184_tools/<this file>
TOOLS = REPO / "tools"
D2_TABLES = ("df_fact_d2_unit_denoising", "df_fact_d2_unit_snr", "df_fact_d2_decision", "df_fact_d2_historical_reanalysis")
WRITTEN = set(D2_TABLES) | {"df_fact_load_receipt", "df_dim_run"}


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load("load_data_foundation")
DL = _load("df_load_d0_d2")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def chunks(path: Path, size: int):
    buf = []
    with open(path) as f:
        for line in f:
            buf.append(json.loads(line))
            if len(buf) >= size:
                yield buf
                buf = []
    if buf:
        yield buf


def run_rows(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        m = json.loads((d / "TABLES_MANIFEST.json").read_text())
        run_id = m.get("comparison_run_id") or m["run_id"]
        rows.append({"run_id": run_id, "module": m.get("label") or m.get("module") or f"D2 tables: {d.parent.name}",
                     "code_sha256": m.get("code_sha256") or "0" * 64,
                     "inputs_sha256": hashlib.sha256("".join(sorted(m["sha256"].values())).encode()).hexdigest(),
                     "status": "COMPLETED", "cpu_seconds": None,
                     "details": {"table_dir": d.parent.name, "counts": m["counts"]}})
    return rows


def load_pass(engine, dirs: list[Path], size: int, seen: dict | None) -> dict:
    out: dict = {}
    for d in dirs:
        for t in D2_TABLES:
            p = d / f"{t}.jsonl"
            if not p.is_file():
                continue
            acc = out.setdefault(t, {"rows_offered": 0, "rows_inserted": 0, "rows_already_present": 0,
                                     "rows_refused": 0, "chunks": 0, "refusals_sample": []})
            for i, rows in enumerate(chunks(p, size)):
                if seen is not None:
                    ids = seen.setdefault(t, set())
                    for r in rows:
                        ids.add(L.row_sha256(t, r))
                rec = L.load(engine, t, rows, f"{rows[0]['run_id']}:{d.parent.name}:chunk{i:04d}")
                for k in ("rows_offered", "rows_inserted", "rows_already_present", "rows_refused"):
                    acc[k] += rec[k]
                acc["chunks"] += 1
                if rec["rows_refused"] and len(acc["refusals_sample"]) < 5:
                    acc["refusals_sample"] += rec["refusals"][:5]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("throwaway", "real"), required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--table-dir", type=Path, action="append", required=True)
    ap.add_argument("--chunk", type=int, default=100_000)
    a = ap.parse_args()
    if a.receipt.exists():
        raise SystemExit("REFUSED: the receipt exists; each load is write-once")
    dirs = [Path(d) for d in a.table_dir]
    receipt = {"schema": "crispdm.data_foundation.d2_load_receipt.v1", "mode": a.mode, "chunk_rows": a.chunk,
               "table_dirs": {d.parent.name + "/" + d.name: {"manifest_sha256": sha_file(d / "TABLES_MANIFEST.json"),
                                                            "files_sha256": json.loads((d / "TABLES_MANIFEST.json").read_text())["sha256"]}
                              for d in dirs},
               "loader_code_sha256": sha_file(TOOLS / "load_data_foundation.py"), "script_sha256": sha_file(Path(__file__))}
    seen: dict = {}
    runs = run_rows(dirs)
    if a.mode == "throwaway":
        from sqlalchemy import text
        name = "c178_d2_throwaway_" + uuid.uuid4().hex[:10]
        admin = DL._engine("postgres")
        with admin.connect() as c:
            c.execute(text(f'CREATE DATABASE "{name}"'))
        eng = DL._engine(name)
        try:
            L.ensure_schema(eng)
            receipt["runs"] = L.load(eng, "df_dim_run", runs, "c178_d2_runs")
            receipt["first_load"] = load_pass(eng, dirs, a.chunk, seen)
            receipt["second_load"] = load_pass(eng, dirs, a.chunk, None)
            receipt["idempotent"] = all(v["rows_inserted"] == 0 for v in receipt["second_load"].values())
        finally:
            eng.dispose()
            with admin.connect() as c:
                c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            admin.dispose()
        receipt["throwaway_database_dropped"] = True
        load = receipt["first_load"]
    else:
        eng = DL._engine()
        from sqlalchemy import inspect
        existing = sorted(inspect(eng).get_table_names(schema="public"))
        before = DL._counts(eng, existing)
        L.ensure_schema(eng)
        receipt["runs"] = L.load(eng, "df_dim_run", runs, "c178_d2_runs")
        receipt["load"] = load_pass(eng, dirs, a.chunk, seen)
        after = DL._counts(eng, existing)
        receipt["tables_before"] = len(existing)
        receipt["changed_tables_outside_d2"] = {t: (before[t], after.get(t)) for t in existing
                                                if t not in WRITTEN and before[t] != after.get(t)}
        receipt["historical_unchanged"] = not receipt["changed_tables_outside_d2"]
        receipt["d2_counts_after"] = DL._counts(eng, list(D2_TABLES))
        eng.dispose()
        load = receipt["load"]
    receipt["distinct_offered"] = {t: len(v) for t, v in seen.items()}
    receipt["duplicates_in_input"] = {t: load[t]["rows_offered"] - len(seen.get(t, ())) for t in load}
    receipt["accounted"] = all(v["rows_offered"] == v["rows_inserted"] + v["rows_already_present"] + v["rows_refused"]
                               for v in load.values())
    receipt["rows_refused"] = {t: v["rows_refused"] for t, v in load.items()}
    receipt["no_silent_dedup"] = receipt["accounted"] and all(n == 0 for n in receipt["duplicates_in_input"].values())
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str).replace(str(Path.home()), "~") + "\n")
    a.receipt.chmod(0o444)
    print(json.dumps({k: receipt.get(k) for k in ("mode", "idempotent", "throwaway_database_dropped", "historical_unchanged",
                                                  "accounted", "no_silent_dedup", "rows_refused", "distinct_offered")},
                     default=str))
    ok = receipt["no_silent_dedup"] and not any(receipt["rows_refused"].values()) and \
        (receipt.get("idempotent") if a.mode == "throwaway" else receipt["historical_unchanged"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
