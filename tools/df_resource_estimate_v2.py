#!/usr/bin/env python3
"""C169 (order 2026-09-13): resource estimates v2, one row per physical estimate.

The C162 child wrote its ADF/KPSS memory estimates before the block identity
was bound, so in block mode the three offsets of one run produced identical
rows and the v1 load kept one of each triple. This tool rebuilds those rows
WITHOUT re-running ADF or KPSS, from what the sealed roots already hold:

* for every unit-root estimate of the child (stage CHILD_RUNTIME), the block
  it licensed is read from the profile statistic row of the same dataset,
  variable, partition and test, in gate order: `block_offset`,
  `temporal_universe` (the block), `run_universe`, `run_length`,
  `block_length`, `maxlag` and `policy_sha256` of its estimator params. Only
  statistic rows whose gate was actually called are paired (COMPLETED, FAILED,
  NON_FINITE_RESULT, NOT_RUN_RESOURCE_BOUND); counts must match exactly or the
  cell is reported UNRESOLVED;
* the partition start comes from the `counts` estimate of that partition
  (train starts at 0, calibration at n_train, confirmation at n_train + n_cal),
  so every range is also given in dataset row coordinates;
* the estimate is recomputed with the planner formula (pure arithmetic,
  df_memory_plan.estimate on the row's own sizes and context) when the row's
  constants digest equals the current one, and compared with the v1 value;
* a metadata preflight estimate of an EXACT unit-root cell binds its whole
  partition as the upper-bound run (basis PARTITION_UPPER_BOUND); a v1
  BLOCK_APPROX preflight row stood for all three offsets and is kept as such
  (UNIT_ROOT_PREFLIGHT_V1_ALL_OFFSETS), never split into invented rows;
* every other estimate row is carried over with identity kind NOT_UNIT_ROOT.

Rows that already carry `params.block_identity` (the current planner) are
taken as emitted. Every v2 row validates for df_fact_resource_estimate_v2.

Output (write-once directory; refused if it exists, refused under ~/.local
unless --allow-state-parent is given): df_fact_resource_estimate_v2.jsonl,
V1_TO_V2_MAP.jsonl (one line per offered v1 row) and SUMMARY.json with the v1
rows offered, distinct v1 identities, the v1 rows that collapsed (recorded,
never deleted), the v2 rows and distinct v2 identities. Nothing is written
inside a root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY_SCHEMA = "crispdm.data_foundation.resource_estimate_v2_recompute.v1"
STAGE_FILES = (("PREFLIGHT_METADATA_UPPER_BOUND", "preflight_estimates.jsonl"),
               ("CHILD_RUNTIME", "resource_estimates.jsonl"))
UNIT_ROOT = {"unit_root_adf": "adf", "unit_root_kpss": "kpss"}
PARTITIONS = ("train", "calibration", "confirmation")
GATE_CALLED_REASONS = ("NON_FINITE_RESULT", "NOT_RUN_RESOURCE_BOUND")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_memory_plan")
L = _load("load_data_foundation")


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _jsonl(p: Path):
    with open(p) as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                yield i, json.loads(line)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ v2 rows
def v2_row(v1: dict, *, attempt: str, identity: dict | None, kind: str, derivation: str,
           estimated_peak_bytes: int | None = None) -> dict:
    """`v1` is a v1 table row (the estimate row plus its stage)."""
    params = dict(v1["params"])
    if identity is not None:
        params["block_identity"] = dict(identity)
    i = identity or {}
    row = {k: v1[k] for k in ("run_id", "bank", "dataset_id", "variable_id", "partition", "module", "metric",
                              "estimator", "formula", "budget_bytes", "decision", "stage", "code_sha256")}
    row.update(
        estimated_peak_bytes=int(v1["estimated_peak_bytes"] if estimated_peak_bytes is None else estimated_peak_bytes),
        params=params, attempt=attempt, identity_kind=kind,
        block_offset=i.get("block_offset", "NONE"), universe_basis=i.get("universe_basis", "NONE"),
        partition_start=i.get("partition_start"),
        range_start=i["range_absolute"][0] if i else None, range_end=i["range_absolute"][1] if i else None,
        run_start=i["run_absolute"][0] if i else None, run_end=i["run_absolute"][1] if i else None,
        n_used=i.get("n_used"), lag_used=i.get("lag_used"),
        unit_root_policy_sha256=i.get("unit_root_policy_sha256"),
        variant=str(params.get("variant", v1["estimator"])), derivation=derivation,
        v1_row_sha256=L.row_sha256("df_fact_resource_estimate", v1))
    return row


def _partition_starts(estimates: list[dict]) -> dict:
    """{partition: start} from the `counts` estimates (sizes.n = partition length)."""
    n = {}
    for r in estimates:
        if r["module"] == "df_profile_univariate" and r["metric"] == "counts" and r["partition"] in PARTITIONS:
            n.setdefault(r["partition"], int(r["params"]["sizes"]["n"]))
    if not all(p in n for p in PARTITIONS):
        return {}
    return {"train": 0, "calibration": n["train"], "confirmation": n["train"] + n["calibration"]}


def _statistic_blocks(profile: Path) -> dict:
    """{(variable_id, partition, test): [estimator params, ...]} in gate order, gate-called rows only."""
    out = defaultdict(list)
    if not profile.is_file():
        return out
    with open(profile) as f:
        for line in f:
            if '_statistic' not in line or ('"adf_statistic' not in line and '"kpss_statistic' not in line):
                continue
            item = json.loads(line)
            r = item["row"]
            m = r["metric"]
            if item["module"] != "df_profile_univariate" or m.endswith("_spread") or "variable_id" not in r:
                continue
            test = m.split("_", 1)[0]
            if m not in (f"{test}_statistic", *(f"{test}_statistic_block_{o}" for o in ("start", "middle", "end"))):
                continue
            called = r["status"] in ("COMPLETED", "FAILED") or r["reason"] in GATE_CALLED_REASONS
            if called:
                out[(r["variable_id"], r["partition"], test)].append(r["estimator"]["params"])
    return out


def attempt_v2_rows(root: Path, attempt_dir: Path, *, stats=None):
    """Yield (file_name, line_no, v1_row, v2_row_or_None, note) for one attempt directory."""
    attempt = str(attempt_dir.relative_to(Path(root) / "attempts"))
    files = {stage: [(i, dict(r, stage=stage)) for i, r in _jsonl(attempt_dir / name)]
             for stage, name in STAGE_FILES if (attempt_dir / name).is_file()}
    consts = P.constants_sha256()
    stats = stats if stats is not None else {}
    blocks = None
    for stage, name in STAGE_FILES:
        rows = files.get(stage, [])
        starts = _partition_starts([r for _, r in rows])
        cursor = Counter()
        for i, v1 in rows:
            group = v1["metric"]
            if not (v1["module"] == "df_profile_univariate" and group in UNIT_ROOT):
                yield name, i, v1, v2_row(v1, attempt=attempt, identity=None, kind="NOT_UNIT_ROOT",
                                          derivation="CARRIED_OVER"), None
                continue
            params = v1["params"]
            if isinstance(params.get("block_identity"), dict):
                yield name, i, v1, v2_row(v1, attempt=attempt, identity=params["block_identity"],
                                          kind="UNIT_ROOT_BLOCK", derivation="EMITTED_WITH_BLOCK_IDENTITY"), None
                continue
            sizes, variant = params["sizes"], params.get("variant", v1["estimator"])
            ps = starts.get(v1["partition"])
            if stage == "PREFLIGHT_METADATA_UPPER_BOUND":
                if variant != "EXACT":
                    yield name, i, v1, v2_row(v1, attempt=attempt, identity=None,
                                              kind="UNIT_ROOT_PREFLIGHT_V1_ALL_OFFSETS",
                                              derivation="CARRIED_OVER"), None
                    continue
                if ps is None:
                    yield name, i, v1, None, "UNRESOLVED_PARTITION_START"
                    continue
                n = int(sizes["n_run"])
                ident = P.unit_root_block_identity(offset="exact", block=(0, n), run=(0, n), partition_start=ps,
                                                   lag=sizes.get("lag") if group == "unit_root_adf" else None,
                                                   policy_sha256=P.UNIT_ROOT_POLICY_SHA256, variant="EXACT",
                                                   basis="PARTITION_UPPER_BOUND")
                yield name, i, v1, v2_row(v1, attempt=attempt, identity=ident, kind="UNIT_ROOT_BLOCK",
                                          derivation="RECOMPUTED_FROM_PROFILE_STATISTICS"), None
                continue
            # child: pair with the statistic row of the same variable, partition and test, in gate order
            if blocks is None:
                blocks = _statistic_blocks(attempt_dir / "profile.jsonl")
            cell = (v1["variable_id"], v1["partition"], UNIT_ROOT[group])
            k = cursor[cell]
            cursor[cell] += 1
            seq = blocks.get(cell, [])
            if k >= len(seq):
                yield name, i, v1, None, "UNRESOLVED_NO_STATISTIC_ROW"
                continue
            sp = seq[k]
            offset = sp.get("block_offset", "exact")
            block, run = sp.get("temporal_universe"), sp.get("run_universe")
            lag = sizes.get("lag") if group == "unit_root_adf" else None
            problems = []
            if ps is None:
                problems.append("UNRESOLVED_PARTITION_START")
            if not (isinstance(block, list) and isinstance(run, list)):
                problems.append("UNRESOLVED_NO_UNIVERSE")
            elif int(sizes["n_run"]) != block[1] - block[0]:
                problems.append("UNRESOLVED_LENGTH_DISAGREES")
            if group == "unit_root_adf" and sp.get("maxlag") != lag:
                problems.append("UNRESOLVED_LAG_DISAGREES")
            if (sp.get("policy") or "EXACT") != variant:
                problems.append("UNRESOLVED_VARIANT_DISAGREES")
            if problems:
                yield name, i, v1, None, ";".join(problems)
                continue
            ident = P.unit_root_block_identity(offset=offset, block=block, run=run, partition_start=ps, lag=lag,
                                               policy_sha256=sp["policy_sha256"], variant=variant)
            est_bytes = None
            if params.get("constants_sha256") == consts:
                est_bytes, _, _ = P.estimate(v1["module"], group, sizes, params["context"])
                fits = est_bytes <= v1["budget_bytes"]
                decision = ("RUN_EXACT" if variant == "EXACT" else "RUN_BOUNDED") if fits else "NOT_RUN_RESOURCE_BOUND"
                stats["recomputed"] = stats.get("recomputed", 0) + 1
                if est_bytes != v1["estimated_peak_bytes"]:
                    stats["estimate_mismatch"] = stats.get("estimate_mismatch", 0) + 1
                if decision != v1["decision"]:
                    stats["decision_mismatch"] = stats.get("decision_mismatch", 0) + 1
            else:
                stats["not_recomputable_constants_differ"] = stats.get("not_recomputable_constants_differ", 0) + 1
            yield name, i, v1, v2_row(v1, attempt=attempt, identity=ident, kind="UNIT_ROOT_BLOCK",
                                      derivation="RECOMPUTED_FROM_PROFILE_STATISTICS",
                                      estimated_peak_bytes=est_bytes), None
        for cell, used in cursor.items():
            if stage == "CHILD_RUNTIME" and blocks is not None and used != len(blocks.get(cell, [])):
                stats.setdefault("unpaired_statistic_cells", []).append([*cell, used, len(blocks.get(cell, []))])


def root_v2_rows(root: Path) -> list[dict]:
    """Every v2 estimate row of a sealed root (used by the D0-D2 loader); refuses an unresolved row."""
    out = []
    for adir in sorted(Path(root).glob("attempts/*/attempt-*")):
        for name, i, v1, v2, note in attempt_v2_rows(root, adir):
            if v2 is None:
                raise SystemExit(f"REFUSED: {adir.name} {name}:{i} {note}")
            out.append(v2)
    return out


# ---------------------------------------------------------------- recompute
def recompute(roots: list[Path], out: Path, *, allow_state_parent: bool = False) -> dict:
    out = Path(out)
    if out.exists():
        raise SystemExit(f"REFUSED: {out.name} exists; the v2 recompute output is write-once")
    state = (Path.home() / ".local").resolve()
    if not allow_state_parent and (out.resolve() == state or state in out.resolve().parents):
        raise SystemExit("REFUSED: the output would be under ~/.local; pass --allow-state-parent explicitly")
    for r in roots:
        if (Path(r).resolve() == out.resolve()) or Path(r).resolve() in out.resolve().parents:
            raise SystemExit("REFUSED: nothing is written inside a root")
        if not (Path(r) / "PROFILE_RUN_RECEIPT.json").is_file():
            raise SystemExit(f"REFUSED: {Path(r).name} is not a sealed profile root")
    out.mkdir(parents=True)
    v1_seen, v2_seen = set(), set()
    v1_offered = v2_rows = 0
    v1_collapsed = Counter()
    kinds, derivations, notes, refusals = Counter(), Counter(), Counter(), []
    per_root, stats = {}, {}
    with open(out / "df_fact_resource_estimate_v2.jsonl", "w") as fv2, open(out / "V1_TO_V2_MAP.jsonl", "w") as fmap:
        for root in roots:
            root = Path(root)
            rc = Counter()
            for adir in sorted(root.glob("attempts/*/attempt-*")):
                for name, i, v1, v2, note in attempt_v2_rows(root, adir, stats=stats):
                    v1_offered += 1
                    rc["v1_offered"] += 1
                    d1 = bytes.fromhex(L.row_sha256("df_fact_resource_estimate", v1))
                    if d1 in v1_seen:
                        v1_collapsed[(v1["stage"], v1["metric"])] += 1
                    v1_seen.add(d1)
                    entry = {"root": root.name, "attempt": str(adir.relative_to(root / "attempts")), "file": name,
                             "line": i, "stage": v1["stage"], "v1_row_sha256": d1.hex()}
                    if v2 is None:
                        notes[note] += 1
                        entry.update(v2_row_sha256=None, note=note)
                    else:
                        problems = L.validate_row("df_fact_resource_estimate_v2", v2)
                        if problems:
                            refusals.append({"root": root.name, "file": name, "line": i, "problems": problems[:3]})
                        d2 = L.row_sha256("df_fact_resource_estimate_v2", v2)
                        v2_seen.add(bytes.fromhex(d2))
                        v2_rows += 1
                        rc["v2_rows"] += 1
                        kinds[v2["identity_kind"]] += 1
                        derivations[v2["derivation"]] += 1
                        entry.update(v2_row_sha256=d2, derivation=v2["derivation"],
                                     identity_kind=v2["identity_kind"], block_offset=v2["block_offset"])
                        fv2.write(json.dumps(v2, sort_keys=True) + "\n")
                    fmap.write(json.dumps(entry, sort_keys=True) + "\n")
            per_root[root.name] = dict(rc, receipt_sha256=_sha_file(root / "PROFILE_RUN_RECEIPT.json"))
    unresolved = sum(notes.values())
    summary = {
        "schema": SUMMARY_SCHEMA, "code_sha256": code_sha256(), "planner_code_sha256": P.CODE_SHA256,
        "planner_constants_sha256": P.constants_sha256(), "roots": per_root,
        "v1": {"rows_offered": v1_offered, "distinct_identities": len(v1_seen),
               "collapsed_rows": v1_offered - len(v1_seen),
               "collapsed_by_stage_metric": {f"{s}:{m}": c for (s, m), c in sorted(v1_collapsed.items())},
               "note": "v1 collapses are recorded here and in V1_TO_V2_MAP.jsonl; the v1 table is not modified"},
        "v2": {"rows": v2_rows, "distinct_identities": len(v2_seen), "collapsed_rows": v2_rows - len(v2_seen),
               "offered_equals_distinct": v2_rows == len(v2_seen), "identity_kinds": dict(kinds),
               "derivations": dict(derivations), "validation_refusals": len(refusals),
               "refusal_examples": refusals[:5]},
        "unresolved_v1_rows": unresolved, "unresolved_by_reason": dict(notes),
        "every_v1_row_mapped": v1_offered == v2_rows + unresolved,
        "estimate_recompute": {k: v for k, v in stats.items() if k != "unpaired_statistic_cells"},
        "unpaired_statistic_cells": len(stats.get("unpaired_statistic_cells", [])),
        "unpaired_examples": stats.get("unpaired_statistic_cells", [])[:5],
        "adf_kpss_rerun": False,
    }
    text = json.dumps(summary, indent=1, sort_keys=True).replace(str(Path.home()), "~")
    (out / "SUMMARY.json").write_text(text + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, action="append", required=True, help="sealed profile root; repeatable")
    ap.add_argument("--out", type=Path, required=True, help="write-once output directory")
    ap.add_argument("--allow-state-parent", action="store_true",
                    help="allow an output directory under ~/.local (the lead's explicit choice)")
    a = ap.parse_args(argv)
    s = recompute(a.root, a.out, allow_state_parent=a.allow_state_parent)
    print(json.dumps({k: s[k] for k in ("v1", "v2", "unresolved_v1_rows", "every_v1_row_mapped",
                                         "estimate_recompute", "unpaired_statistic_cells")}, indent=1))
    ok = s["v2"]["offered_equals_distinct"] and s["unresolved_v1_rows"] == 0 and s["v2"]["validation_refusals"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
