#!/usr/bin/env python3
"""RP22: equivalence of the CONSUMED training code between two commits, function by function (AST
normalised, docstrings and comments ignored). The training path of a MOD-E0 cell is enumerated
explicitly; a differing function is listed with its normalised diff so that its effect on learning can
be argued, not assumed. Documents, do not hide.

    python tools/df_code_equivalence.py --base COMMIT --head COMMIT --file tools/df_mod_e0.py --functions generate,... --out OUT.json
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAINING_PATH = ["group_params", "latent_groups", "causal_oracle", "deterministic_component", "generate", "acf", "welch_bands", "decompose_moving_average",
                 "trend_seasonal_strength", "trend_seasonal_strength_v1", "profiles", "average_linkage", "adjusted_rand", "random_assignments", "same_partition",
                 "boundaries", "make_windows", "prepare", "mase", "_tf", "_tcn_block", "branch_extractor", "branch_reach", "support_reach", "build_modular",
                 "extractor_layer_names", "freeze_extractor", "count_params", "fit", "_sx", "_sy", "_uy", "_bind_donor", "run_cell", "worker_main"]


def _source(commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout


def _functions(src: str) -> dict:
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = list(node.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            node2 = ast.FunctionDef(name=node.name, args=node.args, body=body or [ast.Pass()], decorator_list=node.decorator_list, returns=node.returns, type_params=[])
            text = ast.unparse(ast.fix_missing_locations(node2))
            out[node.name] = text
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            consts[node.targets[0].id] = ast.unparse(node.value)
    return {"functions": out, "constants": consts}


def compare(base: str, head: str, path: str, functions: list) -> dict:
    a, b = _functions(_source(base, path)), _functions(_source(head, path))
    report = {"schema": "code_equivalence.v1", "file": path, "base": base, "head": head, "functions": {}, "constants": {}, "equivalent_training_path": True}
    for name in functions:
        fa, fb = a["functions"].get(name), b["functions"].get(name)
        entry = {"in_base": fa is not None, "in_head": fb is not None,
                 "sha_base": hashlib.sha256((fa or "").encode()).hexdigest()[:16], "sha_head": hashlib.sha256((fb or "").encode()).hexdigest()[:16]}
        entry["equal"] = fa is not None and fa == fb
        if not entry["equal"]:
            entry["diff"] = "\n".join(difflib.unified_diff((fa or "").splitlines(), (fb or "").splitlines(), "base", "head", lineterm="", n=1))[:4000]
            report["equivalent_training_path"] = False
        report["functions"][name] = entry
    for k in sorted(set(a["constants"]) | set(b["constants"])):
        if a["constants"].get(k) != b["constants"].get(k):
            report["constants"][k] = {"base": a["constants"].get(k), "head": b["constants"].get(k)}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--file", default="tools/df_mod_e0.py")
    parser.add_argument("--functions", default=",".join(TRAINING_PATH))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rep = compare(args.base, args.head, args.file, args.functions.split(","))
    args.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"equivalent_training_path": rep["equivalent_training_path"],
                      "differing": [n for n, e in rep["functions"].items() if not e["equal"]], "constants_changed": list(rep["constants"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
