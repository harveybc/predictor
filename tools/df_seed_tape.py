#!/usr/bin/env python3
"""C173 (order 2026-09-13): a fresh seed tape, sealed before any unit exists.

``scan_prior_seeds`` collects every seed the C128-C163 roots used, read-only:

* ``synthetic_bank_c128_v1``: every ``UNIT.json`` (``seed`` and each value of
  ``derived_seeds``) and ``BANK_MANIFEST.json`` (``units[].seed``,
  ``matrix.seeds``);
* ``lab_evaluation_c137_v1``, ``lab_delay_cost_c137_v2``,
  ``snr_calibration_c163_v1``: in every ``.json`` file, any integer under a key
  containing ``seed`` (nested dictionaries of integers included); in every
  ``.jsonl`` file, the same by line (``"...seed...": <int>`` and
  ``"derived_seeds": {...}``); in every string, unit identifiers ``seed<N>``;
* the declared code constants ``CODE_CONSTANT_SEEDS`` (bootstrap and battery
  seeds).

``build_tape`` derives, for every regime of a sealed D2 v2 design, exactly the
design's ``n_seeds`` seeds from a declared master entropy:

    seed(regime, i) = int(sha256(canonical{master_entropy, design_sha256, regime_key, i, counter})[:15 hex])

(values below 2**60; ``counter`` starts at 0). A seed or any of its generator
derived seeds (clean/noise/missing) that appears in the prior set, or repeats
inside the tape, REFUSES the tape: nothing is silently skipped. The tape binds
the design digest, the scan (roots, fields, count and sha256 of the sorted
prior set) and is sealed by ``tape_sha256``.

``materialize_from_tape`` generates the units with the reviewed generator
(``df_synthetic_bank.materialize_unit``) into a write-once root, after
verifying the tape seal: every unit keeps clean, noise, observed, missing mask,
events, its sealed contract (``CONTRACT.json``) and its TRAIN / CALIBRATION /
CONFIRMATION partitions, and is re-verified by regeneration. The same unit
feeds every paired arm.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAPE_SCHEMA = "crispdm.data_foundation.d2_seed_tape.v1"
ROOT_SCHEMA = "crispdm.data_foundation.d2_fresh_reserve_root.v1"
DEFAULT_PRIOR_ROOTS = ("synthetic_bank_c128_v1", "lab_evaluation_c137_v1", "lab_delay_cost_c137_v2",
                       "snr_calibration_c163_v1")
CODE_CONSTANT_SEEDS = {"df_snr.DEFAULT_BOOTSTRAP.seed": 20260912, "df_causal_battery.SEED": 20260913,
                       "df_synthetic_bank.SEEDS": [11, 12, 13]}
_KEY_INT = re.compile(r'"([^"]*seed[^"]*)"\s*:\s*(-?\d+)\b', re.IGNORECASE)
_DERIVED = re.compile(r'"derived_seeds"\s*:\s*\{([^{}]*)\}')
_INT = re.compile(r'(-?\d+)')
_UNIT_SEED = re.compile(r'seed(\d+)')


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("df_d2_design")


class TapeRefusal(ValueError):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


# -------------------------------------------------------------------- scan
def _walk_json(node, key_has_seed, found, field_counts, path):
    if isinstance(node, dict):
        for k, v in node.items():
            seedy = key_has_seed or "seed" in str(k).lower()
            _walk_json(v, seedy, found, field_counts, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for v in node:
            _walk_json(v, key_has_seed, found, field_counts, path + "[]")
    elif isinstance(node, bool):
        return
    elif isinstance(node, int) and key_has_seed:
        found.add(int(node))
        field_counts[path] = field_counts.get(path, 0) + 1
    elif isinstance(node, str):
        for m in _UNIT_SEED.finditer(node):
            found.add(int(m.group(1)))
            field_counts["<string>seed<N>"] = field_counts.get("<string>seed<N>", 0) + 1


def _scan_text_line(line: str, found: set, field_counts: dict):
    for m in _KEY_INT.finditer(line):
        found.add(int(m.group(2)))
        field_counts[m.group(1)] = field_counts.get(m.group(1), 0) + 1
    for m in _DERIVED.finditer(line):
        for v in _INT.findall(m.group(1)):
            found.add(int(v))
            field_counts["derived_seeds.*"] = field_counts.get("derived_seeds.*", 0) + 1
    for m in _UNIT_SEED.finditer(line):
        found.add(int(m.group(1)))
        field_counts["<string>seed<N>"] = field_counts.get("<string>seed<N>", 0) + 1


def scan_prior_seeds(state_root: Path, roots=DEFAULT_PRIOR_ROOTS, *, include_code_constants: bool = True) -> dict:
    """Every seed of the prior roots (read-only). -> {"seeds": sorted list, "report": ...}."""
    state_root = Path(state_root)
    found: set = set()
    report = {"roots": {}, "code_constants": {}}
    for name in roots:
        root = state_root / name
        entry = {"present": root.is_dir(), "files_scanned": 0, "fields": {}, "seeds_found": 0}
        report["roots"][name] = entry
        if not root.is_dir():
            continue
        local: set = set()
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix == ".json":
                try:
                    _walk_json(json.loads(p.read_text()), False, local, entry["fields"], "")
                except ValueError:
                    with open(p, errors="replace") as f:
                        for line in f:
                            _scan_text_line(line, local, entry["fields"])
            elif p.suffix == ".jsonl":
                with open(p, errors="replace") as f:
                    for line in f:
                        if "seed" in line:
                            _scan_text_line(line, local, entry["fields"])
            else:
                continue
            entry["files_scanned"] += 1
        entry["seeds_found"] = len(local)
        found |= local
    if include_code_constants:
        for k, v in CODE_CONSTANT_SEEDS.items():
            vals = v if isinstance(v, list) else [v]
            found.update(int(x) for x in vals)
            report["code_constants"][k] = vals
    seeds = sorted(found)
    report["prior_seed_count"] = len(seeds)
    report["prior_seeds_sha256"] = hashlib.sha256(json.dumps(seeds).encode()).hexdigest()
    return {"seeds": seeds, "report": report}


# -------------------------------------------------------------------- tape
def _candidate(master_entropy: str, design_sha: str, rk: str, i: int, counter: int) -> int:
    blob = D.canonical({"master_entropy": master_entropy, "design_sha256": design_sha, "regime_key": rk, "index": i,
                        "counter": counter})
    return int(hashlib.sha256(blob).hexdigest()[:15], 16)


def build_tape(design: dict, master_entropy: str, prior: dict) -> dict:
    """Seeds per regime from the sealed design, refusing any collision."""
    D.require_valid(design)
    BANK = _load("df_synthetic_bank")
    if not isinstance(master_entropy, str) or len(master_entropy) < 16:
        raise TapeRefusal("master entropy must be a declared string of at least 16 characters")
    prior_set = set(prior["seeds"])
    used: set = set()
    regimes = []
    cells = {D.regime_key(D.regime_of_cell(c)): c for c in design["bank"]["cells"]}
    for rk in sorted(design["seeds_per_regime"]):
        ent = design["seeds_per_regime"][rk]
        cell = cells[rk]
        seeds = []
        for i in range(ent["n_seeds"]):
            s = _candidate(master_entropy, design["design_sha256"], rk, i, 0)
            derived = BANK.derived_seeds(cell, s)
            hits = [x for x in [s, *derived.values()] if x in prior_set or x in used]
            if hits:
                raise TapeRefusal(f"seed collision in regime {rk} index {i}: {hits} already used by a prior root "
                                  "or by this tape")
            used.update([s, *derived.values()])
            seeds.append({"index": i, "seed": s, "derived_seeds": derived})
        regimes.append({"regime_key": rk, "cell": cell, "n_seeds": ent["n_seeds"], "power_status": ent["status"],
                        "seeds": seeds})
    doc = {"schema": TAPE_SCHEMA, "design_sha256": design["design_sha256"], "design_id": design["design_id"],
           "master_entropy": master_entropy,
           "derivation": "int(sha256(canonical{master_entropy, design_sha256, regime_key, index, counter=0})[:15 hex])",
           "generator": {"version": BANK.GENERATOR_VERSION, "code_sha256": BANK.code_sha256()},
           "prior_scan": prior["report"], "regimes": regimes,
           "sealed_before_generation": True, "tape_sha256": ""}
    doc["tape_sha256"] = D.sha_obj({k: v for k, v in doc.items() if k != "tape_sha256"})
    return doc


def verify_tape(tape: dict, design: dict | None = None) -> list[str]:
    p = []
    if not isinstance(tape, dict) or tape.get("schema") != TAPE_SCHEMA:
        return ["not a seed tape"]
    if D.sha_obj({k: v for k, v in tape.items() if k != "tape_sha256"}) != tape.get("tape_sha256"):
        p.append("tape digest does not re-derive")
    if tape.get("sealed_before_generation") is not True:
        p.append("tape was not sealed before generation")
    if design is not None:
        if tape["design_sha256"] != design["design_sha256"]:
            p.append("tape binds another design")
        want = {rk: e["n_seeds"] for rk, e in design["seeds_per_regime"].items()}
        got = {r["regime_key"]: len(r["seeds"]) for r in tape["regimes"]}
        if want != got:
            p.append("tape seeds per regime differ from the design")
    all_seeds = [s["seed"] for r in tape["regimes"] for s in r["seeds"]]
    if len(set(all_seeds)) != len(all_seeds):
        p.append("a seed repeats inside the tape")
    return p


def write_tape(tape: dict, path: Path) -> str:
    problems = verify_tape(tape)
    if problems:
        raise TapeRefusal("; ".join(problems))
    _load("df_isolated_runner").atomic_write_once(Path(path), json.dumps(tape, indent=1, sort_keys=True) + "\n")
    return tape["tape_sha256"]


# ------------------------------------------------------------ materialize
def materialize_from_tape(tape: dict, design: dict, out_root: Path) -> dict:
    """Generate every unit of the tape into a write-once root."""
    BANK = _load("df_synthetic_bank")
    SYNC = _load("df_synthetic_contract")
    IR = _load("df_isolated_runner")
    D.require_valid(design)
    problems = verify_tape(tape, design)
    if problems:
        raise TapeRefusal("; ".join(problems))
    if tape["generator"]["code_sha256"] != BANK.code_sha256():
        raise TapeRefusal("the generator code differs from the one the tape was sealed with")
    out_root = Path(out_root)
    if out_root.exists():
        raise TapeRefusal(f"{out_root.name} exists; the fresh reserve is write-once")
    out_root.mkdir(parents=True)
    IR.atomic_write_once(out_root / "SEED_TAPE.json", json.dumps(tape, indent=1, sort_keys=True) + "\n")
    units = []
    for reg in tape["regimes"]:
        cell = dict(reg["cell"], blocks=["D2_FRESH"])
        for s in reg["seeds"]:
            rec = BANK.materialize_unit(cell, s["seed"], out_root)
            ud = out_root / rec["unit_id"]
            if rec["derived_seeds"] != s["derived_seeds"]:
                raise TapeRefusal(f"{rec['unit_id']}: derived seeds differ from the tape")
            check = BANK.verify_unit(ud)
            if not check["ok"]:
                raise TapeRefusal(f"{rec['unit_id']}: regeneration mismatch {check['mismatches']}")
            contract = SYNC.unit_contract(ud)
            IR.atomic_write_once(ud / "CONTRACT.json", json.dumps(contract, indent=1, sort_keys=True) + "\n")
            units.append({"unit_id": rec["unit_id"], "seed": s["seed"], "regime_key": reg["regime_key"],
                          "unit_json_sha256": hashlib.sha256((ud / "UNIT.json").read_bytes()).hexdigest(),
                          "contract_sha256": contract["contract_sha256"], "content_sha256": contract["content_sha256"],
                          "partitions": rec["partitions"]})
    manifest = {"schema": ROOT_SCHEMA, "mode": D.FRESH_MODE, "design_sha256": design["design_sha256"],
                "tape_sha256": tape["tape_sha256"], "generator": tape["generator"], "units": units,
                "complete": len(units) == sum(len(r["seeds"]) for r in tape["regimes"]),
                "rule": "no plot, metric or sample of this root may change the design"}
    IR.atomic_write_once(out_root / "ROOT_MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--state-root", type=Path, default=Path(os.path.expanduser("~/.local/state/crispdm-data-foundation")))
    ap.add_argument("--master-entropy", required=False)
    ap.add_argument("--tape-out", type=Path)
    ap.add_argument("--materialize", type=Path, help="write-once reserve root from --tape")
    ap.add_argument("--tape", type=Path)
    a = ap.parse_args(argv)
    design = json.loads(a.design.read_text())
    if a.materialize:
        m = materialize_from_tape(json.loads(a.tape.read_text()), design, a.materialize)
        print(json.dumps({"units": len(m["units"]), "complete": m["complete"]}))
        return 0
    prior = scan_prior_seeds(a.state_root)
    tape = build_tape(design, a.master_entropy, prior)
    print(write_tape(tape, a.tape_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
