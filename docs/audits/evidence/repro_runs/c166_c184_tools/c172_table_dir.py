#!/usr/bin/env python3
"""C172/C178: the reanalysis rows and the comparison rows as table-named JSONL files for the loader.

Reads the 18 v2 shard roots (re-hashing every COMPLETED output against its terminal) and the merged
C172_COMPARISON.json; validates every row with df_d2_adjudicate.validate_proposed_row; writes
<comparison_root>/tables/df_fact_d2_unit_denoising.jsonl, df_fact_d2_unit_snr.jsonl and
df_fact_d2_historical_reanalysis.jsonl (write-once), plus TABLES_MANIFEST.json with counts and digests.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
import df_d2_adjudicate as A  # noqa: E402
import df_d2_design as D  # noqa: E402
import df_isolated_runner as IR  # noqa: E402

S = Path.home() / ".local/state/crispdm-data-foundation"
SHARDS = S / "d2_reanalysis_c172_v2"
ROOT = S / "d2_reanalysis_c172_v2_comparison"
OUT = ROOT / "tables"
C137_RUN_ID = "c137_cc9cca240a2549547f6ad2e3"

if OUT.exists():
    raise SystemExit("REFUSED: tables dir exists (write-once)")
OUT.mkdir()
files = {t: open(OUT / f"{t}.jsonl", "w") for t in ("df_fact_d2_unit_denoising", "df_fact_d2_unit_snr")}
counts = {t: 0 for t in files}
run_ids = set()
shard_runs = []
for sr in sorted(SHARDS.glob("*/shard_*")):
    m = json.loads((sr / "RUN_MANIFEST.json").read_text())
    if m["mode"] != D.HISTORICAL_MODE:
        raise SystemExit(f"REFUSED: {sr} is not a historical root")
    shard_runs.append(m["run_id"])
    latest = {}
    for p in sorted((sr / "terminals").glob("*.attempt-*.json")):
        name, att = p.name.rsplit(".attempt-", 1)
        n = int(att.split(".")[0])
        if name not in latest or n > latest[name][0]:
            latest[name] = (n, json.loads(p.read_text()))
    for name, (_, term) in sorted(latest.items()):
        if IR.validate_terminal(term):
            raise SystemExit(f"REFUSED: {name}: terminal does not validate")
        if term["status"] != "COMPLETED":
            continue
        out = sr / term["output_file"]
        if IR.sha_file(out) != term["output_sha256"]:
            raise SystemExit(f"REFUSED: {name}: output does not re-hash")
        with open(out) as f:
            for line in f:
                obj = json.loads(line)
                t, r = obj["table"], obj["row"]
                problems = A.validate_proposed_row(t, r)
                if problems:
                    raise SystemExit(f"REFUSED: {name}: {t}: {problems[:2]}")
                run_ids.add(r["run_id"])
                files[t].write(json.dumps(r, sort_keys=True) + "\n")
                counts[t] += 1
for f in files.values():
    f.close()
rep = json.loads((ROOT / "C172_COMPARISON.json").read_text())
run_id = "d2v2_hist_" + hashlib.sha256(json.dumps(sorted(shard_runs)).encode()).hexdigest()[:24]
rows = A.historical_rows(rep, run_id, C137_RUN_ID)
with open(OUT / "df_fact_d2_historical_reanalysis.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, sort_keys=True) + "\n")
counts["df_fact_d2_historical_reanalysis"] = len(rows)
manifest = {"schema": "crispdm.data_foundation.c172_tables_manifest.v1", "comparison_run_id": run_id,
            "historical_run_id": C137_RUN_ID, "shard_run_ids": sorted(shard_runs), "unit_row_run_ids": sorted(run_ids),
            "label": "HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY", "grants_consumption": False,
            "counts": counts, "sha256": {t: IR.sha_file(OUT / f"{t}.jsonl") for t in counts},
            "comparison_sha256": IR.sha_file(ROOT / "C172_COMPARISON.json")}
IR.atomic_write_once(OUT / "TABLES_MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
for p in OUT.iterdir():
    p.chmod(0o444)
OUT.chmod(0o555)
print(json.dumps({"counts": counts, "run_id": run_id, "shard_runs": len(shard_runs)}))
