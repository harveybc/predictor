#!/usr/bin/env python3
"""RP58 (owner's correction, 2026-09-20): two lineages for the published legacy runs, bound by identity.

The owner reports that the only improvement he considers valid came in PHASE 3, after a causal leak
in a wavelet decomposition of an input was corrected, and that before that fix there was an
abnormally low error (~0.001). That statement is recorded here as **OWNER_REPORTED**. It is not
bound to any run until the identities below bind it, and it does not by itself validate or
invalidate anything.

A run is bound by IDENTITY, never by a folder name, a modification time or the flag a config carries
today:

  * the commit that introduced the published bytes (git, by content);
  * the configuration that NAMES that file as its own output — adjacency proves nothing;
  * the input files that configuration declares, resolved AS OF THAT COMMIT, with their digests and
    their column lists at that moment — the files sitting in the working tree today may be different
    bytes entirely, and usually are;
  * the transformation flags that configuration declares.

A precomputed column carries its producer's causality regardless of what the training run does, so
`use_wavelets: false` today is not evidence of anything about the columns actually consumed.

DISPOSITIONS, and what each one does NOT mean:

  INVALID_CAUSAL_LEAK      a leak was DEMONSTRATED on the consumed representation. Nothing reaches
                           this state from a suspicious score alone.
  CAUSALITY_UNVERIFIED     the checks could not be run on the consumed representation — the producer
                           is absent, the inputs are absent, or a decomposition is declared and not
                           reconstructed. NOT a claim of leakage.
  NO_DECOMPOSITION_DECLARED_OR_CONSUMED
                           the configuration declares no decomposition and the columns consumed at
                           the producing commit are only a time index and the target's own history.
                           This is a FACT about its inputs, not a certificate of validity.

The skill against the naive is computed and reported as an `anomaly_indicator`, apart from the
disposition, because a very good score does not prove a leak and a near-naive score does not prove
its absence.

    python tools/df_causal_lineage.py --inventory INVENTORY.json --out LINEAGE.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DECOMPOSITION_FLAGS = ("use_wavelets", "use_stl", "use_multi_tapper", "use_predicted_decompositions",
                       "use_real_decompositions")
DECOMPOSITION_COLUMN = re.compile(r"wave|dwt|cwt|stl|trend|season|resid|tapper|approx|detail|cA\d|cD\d", re.I)
OWNER_STATEMENT = {
    "status": "OWNER_REPORTED",
    "says": "the only improvement the owner considers valid was in phase 3, after correcting a causal "
            "leak in the wavelet decomposition of an input; before that fix there was an abnormally "
            "low error near 0.001",
    "bound_to_an_identity": False,
    "why": "no run in this repository has been bound to that description yet: the producer of the "
           "decomposed inputs is not in this repository, and the inputs those runs declare are absent "
           "from every commit checked",
    "consequence": "it neither validates nor invalidates any run here, and it is not used as one",
}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout


def _blob(commit: str, path: str) -> bytes | None:
    out = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO, capture_output=True)
    return out.stdout if out.returncode == 0 else None


def _columns(blob: bytes) -> list:
    try:
        reader = csv.reader(io.StringIO(blob.decode("utf-8", errors="replace")))
        return next(reader, [])
    except Exception:
        return []


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inputs_at_commit(config: dict, commit: str) -> dict:
    """Every declared input, as it was AT the producing commit — not as it is now."""
    out = {}
    for key in ("x_train_file", "y_train_file", "x_validation_file", "y_validation_file",
                "x_test_file", "y_test_file"):
        declared = config.get(key)
        if not declared:
            continue
        rel = declared[2:] if declared.startswith("./") else declared
        blob = _blob(commit, rel)
        here = REPO/rel
        entry = {"declared": declared,
                 "present_at_the_producing_commit": blob is not None,
                 "present_in_the_working_tree": here.is_file()}
        if blob is not None:
            entry.update(sha256_at_that_commit=_sha(blob), rows_at_that_commit=blob.count(b"\n"),
                         columns_at_that_commit=_columns(blob))
            entry["decomposition_columns"] = [c for c in entry["columns_at_that_commit"]
                                              if DECOMPOSITION_COLUMN.search(c)]
        if here.is_file():
            data = here.read_bytes()
            entry["sha256_now"] = _sha(data)
            entry["bytes_changed_since"] = (blob is not None and entry["sha256_now"] !=
                                            entry["sha256_at_that_commit"])
        out[key] = entry
    return out


def disposition(config: dict, inputs: dict) -> dict:
    flags = {k: config.get(k) for k in DECOMPOSITION_FLAGS if k in config}
    declares = any(bool(v) for v in flags.values())
    absent = [k for k, v in inputs.items() if not v["present_at_the_producing_commit"]]
    columns = sorted({c for v in inputs.values() for c in v.get("decomposition_columns", [])})
    reasons = []
    if declares:
        reasons.append(f"the configuration declares {sorted(k for k, v in flags.items() if v)}")
    if columns:
        reasons.append(f"the consumed columns include {columns}")
    if absent:
        reasons.append(f"the inputs {absent} do not exist at the producing commit, so the consumed "
                       f"representation cannot be reconstructed here")
    if declares or columns or absent:
        state = "CAUSALITY_UNVERIFIED"
    else:
        state = "NO_DECOMPOSITION_DECLARED_OR_CONSUMED"
        reasons.append("no decomposition flag is set and no consumed column at the producing commit "
                       "matches a decomposition name")
    return {"disposition": state, "because": reasons, "flags": flags,
            "not_a_verdict_on_the_score": "the disposition is about what can be CHECKED, not about "
                                          "how good the numbers look",
            "usable_as_a_benchmark": False,
            "why_not_usable": "until the consumed representation is reconstructed and the causal "
                              "battery runs on it, no published legacy run is a reference performance"}


def anomaly(entry: dict) -> dict:
    """The skill against the naive, reported apart from the disposition."""
    got = (entry.get("recomputed") or {}).get("horizons") or {}
    skills = {h: v.get("mae_skill_percent_vs_that_naive") for h, v in got.items()
              if v.get("mae_skill_percent_vs_that_naive") is not None}
    if not skills:
        return {"measured": False, "why": "no reconstructable predictions beside this table"}
    best = max(skills.values())
    return {"measured": True, "mae_skill_percent_vs_naive_by_horizon": skills,
            "largest": best,
            "flagged_as_extraordinary": bool(best >= 50.0),
            "rule": "50% over the naive at these horizons is flagged for checking. A flag is not a "
                    "finding: a very good score does not prove a leak, and a near-naive score does "
                    "not prove its absence."}


def lineage(inventory: dict) -> dict:
    entries, counts = {}, {}
    for rel, entry in inventory["entries"].items():
        commit = (entry.get("commit") or {}).get("introduced_in")
        named = entry.get("configs_that_name_this_output") or []
        config = {}
        if named:
            blob = _blob(commit, named[0]) if commit else None
            if blob is not None:
                try:
                    config = json.loads(blob)
                except ValueError:
                    config = {}
            if not config and (REPO/named[0]).is_file():
                config = json.loads((REPO/named[0]).read_text())
        inputs = inputs_at_commit(config, commit) if (config and commit) else {}
        verdict = disposition(config, inputs) if config else {
            "disposition": "CAUSALITY_UNVERIFIED",
            "because": ["no configuration in this repository names this file as its own output"],
            "usable_as_a_benchmark": False}
        entries[rel] = {"producing_commit": commit,
                        "config_that_names_it": named[0] if named else None,
                        "config_read_from": ("the producing commit" if config and commit and
                                             _blob(commit, named[0] if named else "") else
                                             "the working tree (it did not exist at that commit)"),
                        "inputs_at_that_commit": inputs,
                        **verdict, "anomaly_indicator": anomaly(entry)}
        counts[verdict["disposition"]] = counts.get(verdict["disposition"], 0)+1
    flagged = sorted((rel for rel, e in entries.items()
                      if e["anomaly_indicator"].get("flagged_as_extraordinary")),
                     key=lambda r: -entries[r]["anomaly_indicator"]["largest"])
    return {"schema": "df_causal_lineage.v1",
            "at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "owner_statement": OWNER_STATEMENT,
            "lineages": {
                "BEFORE_THE_CAUSAL_FIX": {"runs": [], "state": "UNBOUND",
                                          "why": "the fix is in the producer of the decomposed inputs, "
                                                 "which is not this repository; no commit here dates it"},
                "AFTER_THE_CAUSAL_FIX": {"runs": [], "state": "UNBOUND", "why": "as above"}},
            "entries": entries, "counts": counts,
            "flagged_for_checking": flagged[:20],
            "rule": "identity binds a run: the commit that introduced its bytes, the config that names "
                    "it as its own output, and the inputs as they were at that commit. Never the "
                    "folder, the modification time or today's flag."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    doc = lineage(json.loads(a.inventory.read_text()))
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({"counts": doc["counts"], "flagged_for_checking": doc["flagged_for_checking"][:8],
                      "lineages": {k: v["state"] for k, v in doc["lineages"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
