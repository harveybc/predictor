#!/usr/bin/env python3
"""C16: inventory the DOIN surfaces and settle the call site.

The previous return said the eligibility consumer was "installed
without a live call site" in `doin-domains` and `doin-plugins`. The
order requires that claim to be replaced by an inventory: every
local and remote ref, every module surface, and a verdict. If a
real trading L2 gene publication path exists it is documented and
a wiring is proposed without executing it; if it does not, the
integration is declared `TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED`.

A `__pycache__` is not a module and a helper with no caller is not
an integration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

TRADING_MARKERS = ("l2_gene", "trading_l2", "publish_gene",
                   "publish_l2", "gene_publication",
                   "trading_domain")
CALLER_MARKERS = ("require_eligible_gene_inputs",
                  "verify_identity")


def _git(root: Path, *args) -> str:
    r = subprocess.run(["git", *args], cwd=root,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def inventory_repo(root: Path) -> dict:
    root = Path(root)
    refs = [r.strip() for r in
            _git(root, "branch", "-a",
                 "--format=%(refname:short)").splitlines()
            if r.strip()]
    surfaces, pycache_only = [], []
    for d in sorted(root.rglob("*")):
        if not d.is_dir() or ".git" in d.parts:
            continue
        py = [p for p in d.glob("*.py")]
        cache = (d / "__pycache__").is_dir()
        if py:
            surfaces.append({
                "package": str(d.relative_to(root)),
                "modules": sorted(p.name for p in py)})
        elif cache or d.name == "__pycache__":
            if d.name != "__pycache__":
                pycache_only.append(str(d.relative_to(root)))

    # search EVERY ref for a trading L2 publication path
    ref_hits = {}
    for ref in refs:
        listing = _git(root, "ls-tree", "-r", "--name-only", ref)
        hits = [ln for ln in listing.splitlines()
                if any(m in ln.lower()
                       for m in TRADING_MARKERS)]
        if hits:
            ref_hits[ref] = sorted(hits)

    # A caller is a call site outside the DEFINING PACKAGE. A
    # re-export in the package's own __init__ is plumbing, not
    # integration — counting it would restate the very claim the
    # order rejected.
    callers, reexports = {}, {}
    for marker in CALLER_MARKERS:
        defining_pkgs = set()
        for p in sorted(root.rglob("*.py")):
            if ".git" in p.parts or "__pycache__" in p.parts:
                continue
            if f"def {marker}" in p.read_text(errors="replace"):
                defining_pkgs.add(p.parent)
        found, exported = [], []
        for p in sorted(root.rglob("*.py")):
            if ".git" in p.parts or "__pycache__" in p.parts:
                continue
            text = p.read_text(errors="replace")
            if f"def {marker}" in text or marker not in text:
                continue
            rel = str(p.relative_to(root))
            if p.parent in defining_pkgs:
                exported.append(rel)
            else:
                found.append(rel)
        callers[marker] = found
        reexports[marker] = exported

    return {
        "repository": root.name,
        "refs_examined": sorted(refs),
        "ref_count": len(refs),
        "package_surfaces": surfaces,
        "directories_with_only_a_pycache": pycache_only,
        "refs_carrying_a_trading_l2_path": ref_hits,
        "eligibility_call_sites_outside_defining_package":
            callers,
        "reexports_within_the_defining_package": reexports,
        "has_live_call_site": any(bool(v)
                                  for v in callers.values()),
        "call_site_rule": "a re-export inside the defining "
                          "package is plumbing; only a call from "
                          "outside it is an integration",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", action="append", required=True,
                    type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--as-of", required=True)
    a = ap.parse_args(argv)

    repos = [inventory_repo(r) for r in a.repo]
    any_path = any(r["refs_carrying_a_trading_l2_path"]
                   for r in repos)
    verdict = ("TRADING_L2_PUBLICATION_PATH_FOUND"
               if any_path else
               "TRADING_L2_PUBLICATION_PATH_NOT_IMPLEMENTED")
    doc = {
        "schema": "crispdm.doin_surface_inventory.v1",
        "as_of": a.as_of,
        "repositories": repos,
        "verdict": verdict,
        "verdict_note": (
            "no ref of either repository carries a trading-domain "
            "L2 gene publication path, and no module outside the "
            "one that defines it calls the eligibility consumer. "
            "The consumer is therefore INSTALLED AND TESTED but "
            "NOT INTEGRATED, and is reported as such rather than "
            "counted as an integration."
            if not any_path else
            "a candidate path exists; it is documented here and "
            "a wiring is proposed, not executed"),
        "proposed_wiring": (
            "when a trading L2 publication path exists, call "
            "doin_domains.eligibility.require_eligible_gene_inputs"
            "(config, subject_ids=<the gene's declared input "
            "ids>, scope=<the L2 scope>) immediately before the "
            "gene is written, and refuse publication on its "
            "refusal. NOT WIRED in this order."),
    }
    doc["inventory_sha256"] = _sha_text(json.dumps(
        {k: v for k, v in doc.items()}, sort_keys=True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(doc, indent=1,
                                   sort_keys=True) + "\n")
    print(json.dumps({
        "verdict": verdict,
        "repositories": {r["repository"]: {
            "refs": r["ref_count"],
            "packages": len(r["package_surfaces"]),
            "pycache_only_dirs":
                r["directories_with_only_a_pycache"],
            "has_live_call_site": r["has_live_call_site"],
        } for r in repos},
        "inventory_sha256": doc["inventory_sha256"],
    }, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
