"""Generate private, four-node DOIN configs for the H4 then H72 island sequence.

The generator consumes already-governed fold bindings.  It never discovers
credentials, copies data, starts services, or writes private values to git.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from tools.i6n_island_plan import build_plan, sequence_manifest


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _domain(source, horizon, predictor_root, python):
    matches = [d for d in source.get("domains", [])
               if d.get("optimization_config", {}).get("i6n_horizon") == horizon]
    if len(matches) != 1:
        raise ValueError(f"source must contain exactly one horizon-{horizon} domain")
    domain = deepcopy(matches[0])
    old = domain["optimization_config"]
    required = {"i6n_folds", "i6n_admitted_classes"}
    if not required <= set(old):
        raise ValueError(f"horizon-{horizon} source lacks {sorted(required - set(old))}")
    config = {
        "predictor_root": predictor_root,
        "optimizer_plugin": "i6n_optimizer",
        "i6n_island_mode": True,
        "i6n_horizon": horizon,
        "i6n_plan": build_plan(),
        "i6n_folds": deepcopy(old["i6n_folds"]),
        "i6n_admitted_classes": ["S1", "S2", "S3"],
        "i6n_evaluator_command": [python, "-m", "tools.i6n_historical_evaluator", "evaluate"],
        "i6n_evaluator_timeout": 14400,
        "population_size": 20,
        "n_generations": 100,
        "optimization_patience": 10,
        "min_relative_improvement": 1e-8,
        "mutation_probability": 0.8,
        "immigration_probability": 0.2,
        "i6n_s2_generation": 10,
        "i6n_s3_generation": 20,
        "i6n_min_generations": 30,
        "seed": 6102026,
        "ga_seed": 6102026,
        "higher_is_better": False,
        "shared_population": False,
        "optimization_resume": True,
        "optimization_resume_file": (
            f"/var/tmp/i6n-island-sequence/state/h{horizon}/{{island_id}}/checkpoint.json"
        ),
        "optimization_parameters_file": (
            f"/var/tmp/i6n-island-sequence/state/h{horizon}/{{island_id}}/champion.json"
        ),
        "stage_advance_threshold": 1e300,
    }
    domain.update(
        domain_id=f"i6n-island-sequence-h{horizon}",
        optimize=True,
        evaluate=True,
        optimization_plugin="predictor",
        higher_is_better=False,
        optimization_config=config,
    )
    return domain


def generate(source, nodes, output, predictor_root, python):
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise ValueError("nodes JSON must be a list of exactly four nodes")
    ids = [n.get("island_id") for n in nodes]
    if len(set(ids)) != 4 or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("four distinct nonempty island_id values are required")
    endpoints = [n.get("endpoint") for n in nodes]
    if len(set(endpoints)) != 4 or any(not isinstance(e, str) or ":" not in e for e in endpoints):
        raise ValueError("four distinct host:port endpoints are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = sequence_manifest()
    manifest["nodes"] = []
    for horizon in manifest["horizons_hours"]:
        common_domain = _domain(source, horizon, predictor_root, python)
        common_digest = canonical_sha256(common_domain["optimization_config"])
        for node in nodes:
            node_dir = output / f"h{horizon}" / node["island_id"]
            node_dir.mkdir(parents=True, exist_ok=True)
            port = int(node["endpoint"].rsplit(":", 1)[1])
            config = {
                "host": "0.0.0.0",
                "port": port,
                "data_dir": f"/var/tmp/i6n-island-sequence/doin/h{horizon}/{node['island_id']}",
                "bootstrap_peers": [e for e in endpoints if e != node["endpoint"]],
                "shared_min_peers": 3,
                "require_deterministic_seed": True,
                "dashboard_enabled": True,
                "domains": [deepcopy(common_domain)],
            }
            path = node_dir / "doin_node.json"
            path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest["nodes"].append({
                "horizon": horizon, "island_id": node["island_id"],
                "endpoint": node["endpoint"], "config": str(path),
                "optimization_config_sha256": common_digest,
            })
    manifest["optimization_config_sha256_by_horizon"] = {
        str(h): canonical_sha256(_domain(source, h, predictor_root, python)["optimization_config"])
        for h in manifest["horizons_hours"]
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-node-config", required=True)
    parser.add_argument("--nodes-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictor-root", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args(argv)
    source = json.loads(Path(args.source_node_config).read_text(encoding="utf-8"))
    nodes = json.loads(Path(args.nodes_json).read_text(encoding="utf-8"))
    manifest = generate(source, nodes, args.output, args.predictor_root, args.python)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
