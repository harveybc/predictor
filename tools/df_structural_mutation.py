#!/usr/bin/env python3
"""C168 (order 2026-09-13): structural, isolated mutations of every causal check
of the D2 boundary.

For each of the 17 checks:

1. the minimal source surface its probe needs (tools/df_contract, df_snapshot,
   df_operators, df_synthetic_contract, df_synthetic_bank, df_guard_probes; for
   ``exhaustive_cuts`` also tools/df_causal_battery and tests/df_causal_reference)
   is copied, byte for byte, into two fresh temporary directories: INTACT and
   MUTANT;
2. in the MUTANT copy exactly one check is removed by an AST transformation:
   the single ``if <cond>: raise ...`` statement inside the named function whose
   raise carries the locator text becomes ``pass``. The target must match
   exactly once or the harness fails loudly. For ``exhaustive_cuts`` the single
   ``return`` of ``cuts_for_prefix`` in the battery copy is replaced by the seven
   historical cuts ``(0, 5, 19, 23, 60, 150, 238)``;
3. ``tools/df_guard_probes.py --guard NAME`` runs in a NEW interpreter process
   on each copy, by default under ``crispdm-run -m 1G -t 5m -n mut-...``;
4. this process never imports any tools module, intact or mutant: it parses
   source text, writes files and reads one JSON line from each child;
5. the record keeps the sha256 of the original file, of the mutant file and of
   the probe, the files each child actually loaded (with their digests), and
   both outcomes.

DETECTED only when (a) the intact copy refuses with the expected text (for
``fit_mode_enforcement`` also the wavelet case: the EXPANDING Haar fit is
refused) and (b) the mutant bites: the expected refusal no longer fires (for
``fit_mode_enforcement`` also outputs at t <= 30 move when train rows 31..59
are rescaled x5+3; for ``exhaustive_cuts`` the one-sample leak at t=100 is
accepted).

CLI: ``python tools/df_structural_mutation.py --out DIR`` writes, write-once,
df_fact_causal_test.jsonl (test_class GUARD_MUTATION), structural_mutations.jsonl,
mutants/ and STRUCTURAL_MUTATION_SUMMARY.json.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROBE_FILE = "tools/df_guard_probes.py"
RESULT_MARKER = "STRUCTURAL_PROBE_RESULT "
HISTORICAL_SEVEN_CUTS = (0, 5, 19, 23, 60, 150, 238)
REMOVE_IF_RAISE = "REMOVE_IF_RAISE"
REDUCE_CUTS = "REDUCE_CUTS_TO_SEVEN"
SURFACE = ("tools/df_contract.py", "tools/df_snapshot.py", "tools/df_operators.py", "tools/df_synthetic_contract.py",
           "tools/df_synthetic_bank.py", PROBE_FILE)
BATTERY_SURFACE = SURFACE + ("tools/df_causal_battery.py", "tests/df_causal_reference.py")
MODULES_NEVER_IMPORTED_HERE = ("df_contract", "df_snapshot", "df_operators", "df_synthetic_contract",
                               "df_synthetic_bank", "df_guard_probes", "df_causal_battery", "df_causal_reference")

SNAPSHOT, OPERATORS, BATTERY = "tools/df_snapshot.py", "tools/df_operators.py", "tools/df_causal_battery.py"

# guard, target file, enclosing function, locator in the raise, transformation, expected refusal, operator, n
MUTATIONS = (
    ("contract_digest", SNAPSHOT, "_verify_common", "contract digest does not re-derive", REMOVE_IF_RAISE),
    ("matrix_digest", SNAPSHOT, "_verify_common", "matrix digest does not re-derive", REMOVE_IF_RAISE),
    ("source_rederive", SNAPSHOT, "_verify_common", "source bytes do not re-derive the snapshot matrix",
     REMOVE_IF_RAISE),
    ("snapshot_digest", SNAPSHOT, "_verify_common", "snapshot digest does not re-derive", REMOVE_IF_RAISE),
    ("monotonic_timestamps", SNAPSHOT, "_verify_common", "timestamps are not strictly increasing", REMOVE_IF_RAISE),
    ("monotonic_timestamps.step", OPERATORS, "_check_bound_stream", "is not the next row", REMOVE_IF_RAISE),
    ("range_in_partition", SNAPSHOT, "verify_fit_snapshot", "is not inside the", REMOVE_IF_RAISE),
    ("role_allowed", SNAPSHOT, "verify_fit_snapshot", "is not allowed by the design", REMOVE_IF_RAISE),
    ("later_partition_exclusion", SNAPSHOT, "verify_fit_snapshot", "later partitions are not excluded",
     REMOVE_IF_RAISE),
    ("availability", SNAPSHOT, "verify_transform_snapshot", "available after its decision instant", REMOVE_IF_RAISE),
    ("artifact_bound", OPERATORS, "_check_license", "not bound to a FitSnapshot", REMOVE_IF_RAISE),
    ("dataset_binding", OPERATORS, "_check_license", "another dataset or contract", REMOVE_IF_RAISE),
    ("column_identity", OPERATORS, "_check_license", "columns differ from the fitted columns", REMOVE_IF_RAISE),
    ("transform_partition_license", OPERATORS, "_check_license", "is not licensed", REMOVE_IF_RAISE),
    ("fit_mode_enforcement", OPERATORS, "_build_artifact", "is not implemented for kind", REMOVE_IF_RAISE),
    ("stream_binding", OPERATORS, "_check_bound_stream", "another series", REMOVE_IF_RAISE),
    ("exhaustive_cuts", BATTERY, "cuts_for_prefix", None, REDUCE_CUTS),
)
GUARDS_BY_NAME = {m[0]: m for m in MUTATIONS}
# the operator each probe exercises and the rows it consumes (df_guard_probes.PROBES)
PROBE_OPERATOR = {"fit_mode_enforcement": ("trailing_haar_threshold", {"levels": 2, "threshold_k": 3.0})}
EWMA_OPERATOR = ("ewma", {"alpha": 0.3})
PROBE_N = {"exhaustive_cuts": 240, "role_allowed": 300}


class MutationTargetError(RuntimeError):
    """The structural target of a mutation is not found exactly once."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def redact(x) -> str:
    return str(x).replace(str(Path.home()), "~")


# ------------------------------------------------------------ AST mutation
def _stmt_lists(node):
    for fld in ("body", "orelse", "finalbody"):
        lst = getattr(node, fld, None)
        if isinstance(lst, list) and lst and isinstance(lst[0], ast.stmt):
            yield lst
    for h in getattr(node, "handlers", []) or []:
        yield h.body
    for c in getattr(node, "cases", []) or []:
        yield c.body


def _function(tree: ast.AST, name: str) -> ast.AST:
    hits = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    if len(hits) != 1:
        raise MutationTargetError(f"function {name!r} found {len(hits)} times, expected exactly once")
    return hits[0]


def _if_raise_sites(fn: ast.AST, locator: str) -> list:
    """(statement list, index) of every `if <cond>: raise <msg containing locator>` inside fn, at any depth."""
    sites, stack = [], [fn]
    while stack:
        node = stack.pop()
        for lst in _stmt_lists(node):
            for i, st in enumerate(lst):
                if (isinstance(st, ast.If) and not st.orelse and len(st.body) == 1
                        and isinstance(st.body[0], ast.Raise) and locator in ast.unparse(st.body[0])):
                    sites.append((lst, i))
                stack.append(st)
    return sites


def mutate_source(text: str, function: str, locator, transformation: str) -> dict:
    """-> {"mutant": source text, "removed": source of the node removed, "lineno", "replacement"}."""
    tree = ast.parse(text)
    fn = _function(tree, function)
    if transformation == REMOVE_IF_RAISE:
        if not locator:
            raise MutationTargetError("a REMOVE_IF_RAISE mutation needs a locator")
        sites = _if_raise_sites(fn, locator)
        if len(sites) != 1:
            raise MutationTargetError(f"`if ...: raise` carrying {locator!r} in {function} found {len(sites)} times, "
                                      "expected exactly once")
        lst, i = sites[0]
        node = lst[i]
        removed = ast.get_source_segment(text, node)
        lst[i] = ast.copy_location(ast.Pass(), node)
        replacement = "pass"
    elif transformation == REDUCE_CUTS:
        rets = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        if len(rets) != 1:
            raise MutationTargetError(f"{function} has {len(rets)} return statements, expected exactly one")
        node = rets[0]
        removed = ast.get_source_segment(text, node)
        replacement = f"return [t for t in {HISTORICAL_SEVEN_CUTS!r} if t < n]"
        node.value = ast.parse(replacement[len("return "):], mode="eval").body
    else:
        raise MutationTargetError(f"unknown transformation {transformation!r}")
    ast.fix_missing_locations(tree)
    mutant = ast.unparse(tree) + "\n"
    compile(mutant, f"<mutant {function}>", "exec")
    if ast.dump(ast.parse(mutant)) != ast.dump(tree):
        raise MutationTargetError("the mutant source does not round-trip to the transformed tree")
    return {"mutant": mutant, "removed": removed, "lineno": node.lineno, "replacement": replacement}


# --------------------------------------------------------------- children
def default_child_prefix(name: str) -> list:
    exe = shutil.which("crispdm-run")
    if exe is None:
        raise RuntimeError("crispdm-run is not on PATH; pass --child-prefix explicitly (never run children uncapped "
                           "by accident)")
    return [exe, "-m", "1G", "-t", "5m", "-n", name, "--"]


def _prefix(child_prefix, name: str) -> list:
    if child_prefix is None:
        return default_child_prefix(name)
    return [part.replace("{name}", name) for part in shlex.split(child_prefix)]


def _copy_surface(root: Path, dest: Path, files) -> dict:
    digests = {}
    for rel in files:
        src, dst = root / rel, dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dst.write_bytes(data)
        digests[rel] = sha256_bytes(data)
    return digests


def run_child(guard: str, tree: Path, python: str, child_prefix, variant: str) -> dict:
    name = f"mut-{guard.replace('.', '-')}-{variant}"
    cmd = _prefix(child_prefix, name) + ["env", "-u", "PYTHONPATH", "CUDA_VISIBLE_DEVICES=", python, "-B",
                                         str(tree / PROBE_FILE), "--guard", guard]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(tree), capture_output=True, text=True, timeout=900)
    wall = round(time.time() - t0, 2)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith(RESULT_MARKER)]
    if p.returncode != 0 or len(lines) != 1:
        return {"outcome": "CHILD_FAILED", "message": f"exit {p.returncode}; stderr tail: {p.stderr[-800:]}",
                "expected_refusal": False, "loaded_files": {}, "wall_seconds": wall}
    res = json.loads(lines[0][len(RESULT_MARKER):])
    res["wall_seconds"] = wall
    return res


# ------------------------------------------------------------- one guard
def run_mutation(guard: str, *, python: str = sys.executable, child_prefix=None, root: Path = ROOT,
                 keep_mutant_to: Path | None = None) -> dict:
    _, target, function, locator, transformation = GUARDS_BY_NAME[guard]
    surface = BATTERY_SURFACE if target == BATTERY else SURFACE
    original = (root / target).read_text()
    mut = mutate_source(original, function, locator, transformation)
    rec = {"guard": guard, "target_file": target, "target_function": function, "locator": locator,
           "transformation": transformation, "removed_source": mut["removed"], "removed_lineno": mut["lineno"],
           "replacement": mut["replacement"], "original_sha256": sha256_file(root / target),
           "mutant_sha256": sha256_bytes(mut["mutant"].encode()), "probe_file": PROBE_FILE,
           "probe_sha256": sha256_file(root / PROBE_FILE), "surface": list(surface)}
    with tempfile.TemporaryDirectory(prefix=f"c168_intact_{guard}_") as a, \
            tempfile.TemporaryDirectory(prefix=f"c168_mutant_{guard}_") as b:
        intact_dir, mutant_dir = Path(a), Path(b)
        copied = _copy_surface(root, intact_dir, surface)
        _copy_surface(root, mutant_dir, surface)
        (mutant_dir / target).write_text(mut["mutant"])
        if sha256_file(mutant_dir / target) != rec["mutant_sha256"]:
            raise RuntimeError("mutant bytes on disk differ from the transformed source")
        rec["surface_sha256"] = copied
        intact = run_child(guard, intact_dir, python, child_prefix, "intact")
        mutant = run_child(guard, mutant_dir, python, child_prefix, "mutant")
    if keep_mutant_to is not None:
        keep_mutant_to.mkdir(parents=True, exist_ok=True)
        with open(keep_mutant_to / f"{guard}__{Path(target).name}", "x") as fh:
            fh.write(mut["mutant"])
    rec["intact"], rec["mutant"] = intact, mutant
    rec["intact_loaded_target_sha256"] = intact.get("loaded_files", {}).get(target)
    rec["mutant_loaded_target_sha256"] = mutant.get("loaded_files", {}).get(target)
    rec["children_loaded_the_right_bytes"] = (rec["intact_loaded_target_sha256"] == rec["original_sha256"]
                                              and rec["mutant_loaded_target_sha256"] == rec["mutant_sha256"])
    refused = bool(intact.get("expected_refusal"))
    bit = mutant["outcome"] in ("REFUSED", "ACCEPTED", "EXCEPTION") and not mutant.get("expected_refusal")
    if guard == "fit_mode_enforcement":
        refused = refused and intact.get("wavelet", {}).get("outcome") == "REFUSED"
        bit = bit and mutant.get("wavelet", {}).get("outputs_t_le_30_moved") is True
    if guard == "exhaustive_cuts":
        bit = bit and mutant["outcome"] == "ACCEPTED"
    rec["refused"] = refused
    rec["bit"] = bit
    rec["mutant_consumed_the_input"] = mutant["outcome"] == "ACCEPTED"
    rec["detected"] = bool(refused and bit and rec["children_loaded_the_right_bytes"])
    return rec


def code_sha256s(root: Path = ROOT) -> dict:
    files = sorted(set(BATTERY_SURFACE) | {"tools/df_structural_mutation.py"})
    return {rel: sha256_file(root / rel) for rel in files}


def code_sha256(root: Path = ROOT) -> str:
    return sha256_bytes(json.dumps(code_sha256s(root), sort_keys=True).encode())


def fact_row(rec: dict, run_id: str, code: str) -> dict:
    kind, params = PROBE_OPERATOR.get(rec["guard"], EWMA_OPERATOR)
    wav = ""
    if rec["guard"] == "fit_mode_enforcement":
        wav = (f"; wavelet intact={rec['intact'].get('wavelet')}; wavelet mutant={rec['mutant'].get('wavelet')}")
    reason = (f"intact: {rec['intact']['outcome']}: {rec['intact']['message']}; "
              f"mutant: {rec['mutant']['outcome']}: {rec['mutant']['message'] or 'no refusal'}{wav}; "
              f"removed {rec['target_file']}:{rec['removed_lineno']} in {rec['target_function']}; "
              f"original_sha256={rec['original_sha256']}; mutant_sha256={rec['mutant_sha256']}; "
              f"probe_sha256={rec['probe_sha256']}")
    return {"run_id": run_id, "operator_kind": kind, "operator_params": dict(params, guard=rec["guard"]),
            "level": None, "test_class": "GUARD_MUTATION", "case_id": rec["guard"],
            "n": PROBE_N.get(rec["guard"], 100), "cuts_tested": 240 if rec["guard"] == "exhaustive_cuts" else 1,
            "outcome": "DETECTED" if rec["detected"] else "NOT_DETECTED", "reason": redact(reason),
            "code_sha256": code}


def run_all(*, python: str = sys.executable, child_prefix=None, only=None, mutants_dir: Path | None = None,
            progress=None) -> tuple:
    code = code_sha256()
    run_id = "c168_" + code[:24]
    recs = []
    for guard, *_ in MUTATIONS:
        if only and guard not in only:
            continue
        rec = run_mutation(guard, python=python, child_prefix=child_prefix, keep_mutant_to=mutants_dir)
        recs.append(rec)
        if progress:
            progress(f"{guard}: intact {rec['intact']['outcome']} / mutant {rec['mutant']['outcome']} -> "
                     f"{'DETECTED' if rec['detected'] else 'NOT_DETECTED'}")
    leaked = sorted(m for m in MODULES_NEVER_IMPORTED_HERE if m in sys.modules)
    return run_id, code, recs, leaked


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable, help="interpreter of the child processes")
    ap.add_argument("--child-prefix", default=None,
                    help="command prefix of each child ({name} is replaced); default: crispdm-run -m 1G -t 5m -n NAME --")
    ap.add_argument("--only", action="append", choices=[m[0] for m in MUTATIONS], default=None,
                    help="run only these guards (the summary then says so and all_detected is false)")
    a = ap.parse_args(argv)
    out = a.out.expanduser().resolve()
    local = (Path.home() / ".local").resolve()
    if out == local or local in out.parents:
        print("REFUSED: the structural mutation harness never writes under ~/.local", file=sys.stderr)
        return 2
    if out.exists():
        print(f"REFUSED: {out.name} exists; outputs are write-once", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    t0 = time.time()
    run_id, code, recs, leaked = run_all(python=a.python, child_prefix=a.child_prefix, only=a.only,
                                         mutants_dir=out / "mutants",
                                         progress=lambda m: print(m, file=sys.stderr, flush=True))
    rows = [fact_row(r, run_id, code) for r in recs]
    complete = a.only is None and len(recs) == len(MUTATIONS)
    summary = {"schema": "crispdm.data_foundation.structural_mutation_summary.v1", "run_id": run_id,
               "code_sha256": code, "code_sha256s": code_sha256s(), "mutations": len(recs),
               "declared_mutations": len(MUTATIONS), "complete": complete,
               "detected": sum(r["detected"] for r in recs),
               "not_detected": [r["guard"] for r in recs if not r["detected"]],
               "all_detected": complete and all(r["detected"] for r in recs),
               "evidence_process_imported_tools_modules": leaked,
               "child_prefix": redact(a.child_prefix or "crispdm-run -m 1G -t 5m -n mut-<guard>-<variant> --"),
               "child_python": redact(a.python), "wall_seconds": round(time.time() - t0, 1),
               "child_peak_rss_bytes_max": max([int(r[v].get("peak_rss_bytes") or 0) for r in recs
                                                for v in ("intact", "mutant")] or [0]),
               "harness_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
               "table": [{"guard": r["guard"], "intact": r["intact"]["outcome"], "mutant": r["mutant"]["outcome"],
                          "original_sha256": r["original_sha256"], "mutant_sha256": r["mutant_sha256"],
                          "probe_sha256": r["probe_sha256"], "detected": r["detected"]} for r in recs]}
    if leaked:
        summary["all_detected"] = False
    text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False)
    if str(Path.home()) in text:
        print("REFUSED: absolute home path in the summary", file=sys.stderr)
        return 2
    with open(out / "df_fact_causal_test.jsonl", "x") as fh:
        fh.write("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    with open(out / "structural_mutations.jsonl", "x") as fh:
        fh.write("".join(redact(json.dumps(r, sort_keys=True, allow_nan=False)) + "\n" for r in recs))
    with open(out / "STRUCTURAL_MUTATION_SUMMARY.json", "x") as fh:
        fh.write(text + "\n")
    print(json.dumps({k: summary[k] for k in ("run_id", "mutations", "detected", "not_detected", "all_detected",
                                              "wall_seconds", "child_peak_rss_bytes_max")}, indent=1))
    return 0 if summary["all_detected"] else 1


if __name__ == "__main__":
    sys.exit(main())
