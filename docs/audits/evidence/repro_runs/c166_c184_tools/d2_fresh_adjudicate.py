#!/usr/bin/env python3
"""C174-C177: one confirmation root from the fresh shard roots, then adjudication in bounded parts.

  collect --shards <dir with ROLE/shard_NN> --out <write-once root> --design <design>
      hardlinks every terminal and its re-hashed output into one root with a RUN_MANIFEST that lists the shard
      manifests; invalidation markers are linked too, so df_d2_adjudicate.load_fresh_root refuses as designed.
  split   --root <root> --work <dir>            one streaming pass: re-hash, keep confirmation + COST rows, one
                                                JSONL per family and table
  decide  --root <root> --work <dir> --family F --design <design>
                                                check_rows + decide_snr + decide_denoising + decision_rows for
                                                one family (regimes never cross families), own process
  merge   --root <root> --work <dir>            DECISIONS.jsonl + ADJUDICATION_SUMMARY.json, write-once
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "tools"))
import df_d2_adjudicate as A  # noqa: E402
import df_d2_design as D  # noqa: E402
import df_isolated_runner as IR  # noqa: E402

DEN, SNR = "df_fact_d2_unit_denoising", "df_fact_d2_unit_snr"


def collect(shards: Path, out: Path, design: dict) -> int:
    if out.exists():
        raise SystemExit("REFUSED: root exists (write-once)")
    roots = sorted(p for p in shards.glob("*/shard_*") if (p / "RUN_MANIFEST.json").is_file())
    out.mkdir(parents=True)
    (out / "terminals").mkdir()
    manifests, n, markers, by_status = [], 0, [], {}
    for sr in roots:
        m = json.loads((sr / "RUN_MANIFEST.json").read_text())
        if m["mode"] != D.FRESH_MODE or m["design_sha256"] != design["design_sha256"]:
            raise SystemExit(f"REFUSED: {sr} is not a fresh root of this design")
        for mk in sr.glob("ROOT_INVALIDATED__*.json"):
            os.link(mk, out / mk.name)
            markers.append(mk.name)
        for t in sorted((sr / "terminals").glob("*.attempt-*.json")):
            term = json.loads(t.read_text())
            problems = IR.validate_terminal(term)
            if problems:
                raise SystemExit(f"REFUSED: {t.name}: {problems[:2]}")
            dst = out / "terminals" / t.name
            if dst.exists():
                raise SystemExit(f"REFUSED: duplicate terminal {t.name}")
            os.link(t, dst)
            by_status[term["status"]] = by_status.get(term["status"], 0) + 1
            if term["status"] == "COMPLETED":
                src = sr / term["output_file"]
                if IR.sha_file(src) != term["output_sha256"]:
                    raise SystemExit(f"REFUSED: {t.name}: output does not re-hash to its terminal")
                d = out / term["output_file"]
                d.parent.mkdir(parents=True, exist_ok=True)
                os.link(src, d)
            n += 1
        manifests.append({"shard": sr.name, "role": sr.parent.name, "run_id": m["run_id"],
                          "code_sha256_at_creation": m["code_sha256_at_creation"], "terminals": len(list((sr / "terminals").glob("*.json")))})
    run_id = "d2v2_fresh_" + hashlib.sha256(json.dumps(sorted(m["run_id"] for m in manifests)).encode()).hexdigest()[:24]
    manifest = {"schema": "crispdm.data_foundation.d2_run_manifest.v1", "mode": D.FRESH_MODE,
                "design_sha256": design["design_sha256"], "run_id": run_id, "host_role": "COORDINATOR",
                "code_sha256_at_creation": D.lab_code_sha256(), "collected_from": manifests, "terminals": n,
                "terminals_by_status": by_status, "invalidation_markers": markers,
                "layout": "hardlinks to the shard roots under d2_fresh_c174_v1/<ROLE>/shard_NN, re-hashed at collection"}
    IR.atomic_write_once(out / "RUN_MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"run_id": run_id, "shards": len(roots), "terminals": n, "by_status": by_status, "markers": markers}))
    return 0


def _latest(root: Path) -> dict:
    latest: dict = {}
    for p in sorted((root / "terminals").glob("*.attempt-*.json")):
        name, attempt = p.name.rsplit(".attempt-", 1)
        k = int(attempt.split(".")[0])
        if name not in latest or k > latest[name][0]:
            latest[name] = (k, json.loads(p.read_text()))
    return latest


def split(root: Path, work: Path) -> int:
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    if manifest["mode"] != D.FRESH_MODE:
        raise SystemExit("REFUSED: not a fresh root")
    markers = sorted(p.name for p in root.glob("ROOT_INVALIDATED__*.json"))
    if markers:
        raise SystemExit(f"REFUSED: root invalidated as a whole by {markers}")
    work.mkdir(parents=True, exist_ok=False)
    files: dict = {}
    counts: dict = {}
    excluded = kept = 0
    units_by_family: dict = {}
    for name, (_, term) in sorted(_latest(root).items()):
        problems = IR.validate_terminal(term)
        if problems:
            raise SystemExit(f"REFUSED: {name}: {problems[:2]}")
        counts[term["status"]] = counts.get(term["status"], 0) + 1
        if term["status"] != "COMPLETED":
            continue
        out = root / term["output_file"]
        if IR.sha_file(out) != term["output_sha256"]:
            raise SystemExit(f"REFUSED: {name}: output does not re-hash to its terminal")
        with open(out) as f:
            for line in f:
                obj = json.loads(line)
                r = obj["row"]
                keep = r["partition"] == "confirmation" or (obj["table"] == DEN and r["branch"] == "COST")
                if not keep:
                    excluded += 1
                    continue
                fam = r["regime"]["family"]
                units_by_family.setdefault(fam, set()).add(r["unit_id"])
                key = (obj["table"], fam)
                if key not in files:
                    files[key] = open(work / f"{obj['table']}__{fam}.jsonl", "w")
                files[key].write(json.dumps(r, sort_keys=True) + "\n")
                kept += 1
    for f in files.values():
        f.close()
    census = {"run_id": manifest["run_id"], "terminal_counts": counts, "rows_kept": kept,
              "non_governing_rows_excluded": excluded, "families": sorted({fam for (_, fam) in files}),
              "units_by_family": {k: len(v) for k, v in sorted(units_by_family.items())}}
    (work / "CENSUS.json").write_text(json.dumps(census, indent=1, sort_keys=True) + "\n")
    print(json.dumps(census))
    return 0


def _read(p: Path) -> list:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f]


def decide(root: Path, work: Path, family: str, design: dict) -> int:
    run_id = json.loads((root / "RUN_MANIFEST.json").read_text())["run_id"]
    den = _read(work / f"{DEN}__{family}.jsonl")
    snr = _read(work / f"{SNR}__{family}.jsonl")
    decisions = A.decide_snr(snr, design) + A.decide_denoising(den, design)
    rows = A.decision_rows(decisions, run_id)
    with open(work / f"decisions__{family}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=float) + "\n")
    tally: dict = {}
    for r in rows:
        tally[(r["subject_kind"], r["decision"])] = tally.get((r["subject_kind"], r["decision"]), 0) + 1
    print(json.dumps({"family": family, "denoising_rows": len(den), "snr_rows": len(snr), "decisions": len(rows),
                      "tally": {f"{k[0]}:{k[1]}": v for k, v in sorted(tally.items())}}))
    return 0


def merge(root: Path, work: Path) -> int:
    census = json.loads((work / "CENSUS.json").read_text())
    out = root / "DECISIONS.jsonl"
    summ = root / "ADJUDICATION_SUMMARY.json"
    if out.exists() or summ.exists():
        raise SystemExit("REFUSED: adjudication outputs exist (write-once)")
    tally: dict = {}
    per_subject: dict = {}
    n = 0
    with open(out, "w") as f:
        for fam in census["families"]:
            for r in _read(work / f"decisions__{fam}.jsonl"):
                f.write(json.dumps(r, sort_keys=True) + "\n")
                n += 1
                tally.setdefault(r["subject_kind"], {}).setdefault(r["decision"], 0)
                tally[r["subject_kind"]][r["decision"]] += 1
                s = per_subject.setdefault(r["subject_kind"], {}).setdefault(r["subject"] if r["subject_kind"] == "SNR_ESTIMATOR" else f"{r['subject']} {json.dumps(r['operator_params'], sort_keys=True)}", {})
                s[r["decision"]] = s.get(r["decision"], 0) + 1
    doc = {"schema": "crispdm.data_foundation.d2_adjudication_summary.v1", "run_id": census["run_id"],
           "stratum": D.FRESH_MODE, "grants_consumption": False, "externally_reviewed": False,
           "decision_rows": n, "decisions_by_kind": tally, "per_subject": per_subject, "census": census,
           "decisions_sha256": IR.sha_file(out)}
    IR.atomic_write_once(summ, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.chmod(out, 0o444)
    print(json.dumps({"decision_rows": n, "decisions_by_kind": tally}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("collect", "split", "decide", "merge"))
    ap.add_argument("--shards", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--family")
    ap.add_argument("--design", type=Path)
    a = ap.parse_args()
    design = json.loads(a.design.read_text()) if a.design else None
    if a.step == "collect":
        raise SystemExit(collect(a.shards, a.out, design))
    if a.step == "split":
        raise SystemExit(split(a.root, a.work))
    if a.step == "decide":
        raise SystemExit(decide(a.root, a.work, a.family, design))
    raise SystemExit(merge(a.root, a.work))
