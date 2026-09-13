#!/usr/bin/env python3
"""C172: compare the current-code historical reanalysis (collected v2 roots) with C137, in parts to bound memory.

  split   --reanalysis <dir holding ROLE/shard_NN roots> --work <scratch dir>
          one streaming pass: re-hashes every COMPLETED terminal output, then writes per-family JSONL of the
          reanalysis denoising rows (SNR rows skipped) and of the C137 runs / compared metrics / decisions.
  compare --work <dir> --family F --design <design>      one family, own process; writes work/report_F.json
  merge   --work <dir> --out <write-once report>          merges family reports; counts kept apart
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
C137 = Path.home() / ".local/state/crispdm-data-foundation/lab_evaluation_c137_v1"
COMPARED = ("support", "snr_improvement_db", "rmse_ratio")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Sinks:
    def __init__(self, work: Path):
        self.work, self.files = work, {}

    def write(self, kind: str, family: str, obj: dict):
        k = (kind, family)
        if k not in self.files:
            self.files[k] = open(self.work / f"{kind}__{family}.jsonl", "w")
        self.files[k].write(json.dumps(obj, sort_keys=True) + "\n")

    def close(self):
        for f in self.files.values():
            f.close()


def split(reanalysis: Path, work: Path) -> int:
    work.mkdir(parents=True, exist_ok=False)
    sinks = Sinks(work)
    census = {"shards": 0, "terminals": {}, "markers": [], "rows": 0, "units": set(), "roles": {}}
    for shard in sorted(reanalysis.glob("*/shard_*")):
        census["shards"] += 1
        role = shard.parent.name
        census["markers"] += sorted(p.name for p in shard.glob("ROOT_INVALIDATED__*.json"))
        latest = {}
        for p in sorted((shard / "terminals").glob("*.attempt-*.json")):
            name, att = p.name.rsplit(".attempt-", 1)
            n = int(att.split(".")[0])
            if name not in latest or n > latest[name][0]:
                latest[name] = (n, json.loads(p.read_text()))
        for name, (_, term) in sorted(latest.items()):
            census["terminals"][term["status"]] = census["terminals"].get(term["status"], 0) + 1
            census["roles"][role] = census["roles"].get(role, 0) + 1
            if term["status"] != "COMPLETED":
                continue
            out = shard / term["output_file"]
            if sha_file(out) != term["output_sha256"]:
                raise SystemExit(f"REFUSED: {shard.name}/{name} output does not re-hash to its terminal")
            with open(out) as f:
                for line in f:
                    o = json.loads(line)
                    if o["table"] != "df_fact_d2_unit_denoising":
                        continue
                    r = o["row"]
                    census["rows"] += 1
                    census["units"].add(r["unit_id"])
                    sinks.write("re", r["regime"]["family"], r)
    run_family = {}
    with open(C137 / "df_fact_operator_run.jsonl") as f:
        for line in f:
            r = json.loads(line)
            fam = r["regime"]["family"]
            sinks.write("hrun", fam, r)
            import load_data_foundation as L  # noqa: E402
            run_family[L.row_sha256("df_fact_operator_run", r)] = fam
    with open(C137 / "df_fact_operator_signal_metric.jsonl") as f:
        for line in f:
            if not any(f'"metric": "{m}"' in line for m in COMPARED):
                continue
            m = json.loads(line)
            if m["metric"] in COMPARED and m["partition"] in ("calibration", "confirmation"):
                fam = run_family.get(m["operator_run_sha256"])
                if fam is not None:
                    sinks.write("hmet", fam, m)
    with open(C137 / "df_fact_lab_decision.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sinks.write("hdec", d["regime"]["family"], d)
    sinks.close()
    census["units"] = len(census["units"])
    census["families"] = sorted({fam for (_, fam) in sinks.files})
    (work / "CENSUS.json").write_text(json.dumps(census, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in census.items() if k != "markers"} | {"markers": len(census["markers"])}))
    return 0


def _read(p: Path):
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f]


def compare(work: Path, family: str, design_path: Path) -> int:
    import df_d2_adjudicate as A
    design = json.loads(design_path.read_text())
    rep = A.compare_with_historical(_read(work / f"hdec__{family}.jsonl"), _read(work / f"hrun__{family}.jsonl"),
                                    _read(work / f"hmet__{family}.jsonl"), _read(work / f"re__{family}.jsonl"),
                                    design)
    (work / f"report__{family}.json").write_text(json.dumps(rep, sort_keys=True, default=str) + "\n")
    print(json.dumps({"family": family, "flips": rep["flips"], "units_equal": rep["units_equal"],
                      "truth_content_equal": rep["truth_content_equal"]}))
    return 0


def merge(work: Path, out: Path) -> int:
    if out.exists():
        raise SystemExit("REFUSED: report is write-once")
    census = json.loads((work / "CENSUS.json").read_text())
    merged = {"stratum": None, "grants_consumption": False, "rows": [], "flips_by_primary_cause": {}, "flips": 0,
              "decision_counts": {"historical": {}, "reanalysis": {}},
              "status_counts": {"historical": {}, "reanalysis": {}}, "units": {"historical": 0, "reanalysis": 0},
              "units_equal_per_family": {}, "truth_content_equal_per_family": {}}
    for fam in census["families"]:
        rep = json.loads((work / f"report__{fam}.json").read_text())
        merged["stratum"] = rep["stratum"]
        merged["rows"] += rep["rows"]
        merged["flips"] += rep["flips"]
        for c, n in rep["flips_by_primary_cause"].items():
            merged["flips_by_primary_cause"][c] = merged["flips_by_primary_cause"].get(c, 0) + n
        for side in ("historical", "reanalysis"):
            for k, n in rep["decision_counts"][side].items():
                merged["decision_counts"][side][k] = merged["decision_counts"][side].get(k, 0) + n
            for k, n in rep["status_counts"][side].items():
                merged["status_counts"][side][k] = merged["status_counts"][side].get(k, 0) + n
            merged["units"][side] += rep["units"][side]
        merged["units_equal_per_family"][fam] = rep["units_equal"]
        merged["truth_content_equal_per_family"][fam] = rep["truth_content_equal"]
    merged["units_equal"] = all(merged["units_equal_per_family"].values())
    merged["truth_content_equal"] = all(merged["truth_content_equal_per_family"].values())
    merged["label"] = "HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY"
    merged["reanalysis_census"] = census
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=1, sort_keys=True, default=str).replace(str(Path.home()), "~") + "\n")
    out.chmod(0o444)
    print(json.dumps({k: merged[k] for k in ("flips", "flips_by_primary_cause", "decision_counts", "status_counts",
                                              "units", "units_equal", "truth_content_equal")}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("split", "compare", "merge"))
    ap.add_argument("--reanalysis", type=Path)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--family")
    ap.add_argument("--design", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.step == "split":
        raise SystemExit(split(a.reanalysis, a.work))
    if a.step == "compare":
        raise SystemExit(compare(a.work, a.family, a.design))
    raise SystemExit(merge(a.work, a.out))
