#!/usr/bin/env python3
"""Independent checks for the round-7 T2 ruling and population defect.

Run from predictor with a clean checkout of agent-multi commit 4bf38b6f:

    python docs/audits/evidence/repro_runs/\
      musashi_round7_additional_pre_2026_09_12.py \
      --t2-root <checkout-at-4bf38b6f>
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from fractions import Fraction
from pathlib import Path


EXPECTED_CORRECTED_DIGEST = (
    "1be80a0ab6d091794f7ce3ec97c2dbf88920911037c0415c383043bd95ca3a6d"
)


def sha_obj(value) -> str:
    # Match the committed T2 scientific-digest contract exactly. Its spaces
    # are part of the hashed byte representation.
    raw = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def exact_two_sided(k: int, n: int) -> Fraction:
    total = 2**n
    lower = sum(Fraction(math.comb(n, i), total) for i in range(k + 1))
    upper = sum(Fraction(math.comb(n, i), total) for i in range(k, n + 1))
    return min(Fraction(1), 2 * min(lower, upper))


def corrected_t2_digest(t2: Path) -> tuple[dict, str]:
    evidence = json.loads((
        t2 / "docs/audits/evidence/"
        "T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json"
    ).read_text())
    historical = evidence["screen_adjudication"]
    screen = json.loads(json.dumps(historical))
    table = [float(exact_two_sided(i, 6)) for i in range(7)]
    corrected = float(exact_two_sided(screen["signs_positive"], 6))
    screen["sign_test_exact_p_two_sided"] = corrected
    body = {
        "final_adjudication_counts": evidence["final_adjudication_counts"],
        "screen_adjudication": screen,
        "sign_test_corrected": {
            "corrected_table_0_to_6": table,
            "corrected_value": corrected,
            "signs_positive": historical["signs_positive"],
        },
    }
    return {
        "historical": historical["sign_test_exact_p_two_sided"],
        "corrected": corrected,
        "table": table,
    }, sha_obj(body)


def load_design_module(predictor: Path):
    path = predictor / "tools/per_variable_design_v4.py"
    spec = importlib.util.spec_from_file_location("design_v4", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def false_sufficient_bank(predictor: Path) -> dict:
    design = load_design_module(predictor)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        terminals = root / "terminals"
        terminals.mkdir()
        nodes = []
        contracts = []
        for panel in range(6):
            dataset = f"panel_{panel}"
            nodes.extend({
                "dataset_id": dataset,
                "column": f"x_{column}",
                "class": "CAUSAL_ACTIVE",
            } for column in range(5))
            contract = root / f"temporal_{panel}.json"
            contract.write_text(json.dumps({"dataset": {
                "dataset_id": dataset}}))
            contracts.append(contract)
        dag = root / "dag.json"
        dag.write_text(json.dumps({"nodes": nodes}))
        census = root / "census.json"
        census.write_text(json.dumps({"variables": [{
            "variable_id": "not_a_member",
            "semantics": "UNKNOWN",
            "role": "UNKNOWN",
            "license": "UNKNOWN",
            "unit": "UNKNOWN",
        }]}))
        return design.derive_population(
            terminals_v4=terminals,
            dag_v4=dag,
            temporal_contracts=contracts,
            census=census,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t2-root", required=True, type=Path)
    parser.add_argument(
        "--predictor-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    args = parser.parse_args()
    t2, digest = corrected_t2_digest(args.t2_root)
    assert digest == EXPECTED_CORRECTED_DIGEST
    bank = false_sufficient_bank(args.predictor_root)
    print("T2_P", json.dumps(t2, sort_keys=True))
    print("T2_CORRECTED_DIGEST", digest)
    print("FALSE_BANK", json.dumps({
        "verdict": bank["verdict"],
        "panels": bank["panel_count"],
        "members": bank["members"],
        "terminals": bank["terminals_independently_recomputed"],
        "semantic_variables":
            bank["census_variables_with_declared_semantics_role_unit_license"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
