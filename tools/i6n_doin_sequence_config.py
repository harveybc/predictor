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

from tools import i6n_contamination as contamination
from tools import i6n_island_plan as island_plan
from tools.i6n_contracts import feature_availability as availability
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
    admitted = old["i6n_admitted_classes"]
    if (not isinstance(admitted, list) or not admitted
            or len(set(admitted)) != len(admitted)
            or not set(admitted) <= {"S1", "S2", "S3"}):
        raise ValueError(f"horizon-{horizon} source has invalid admitted classes")
    config = {
        "predictor_root": predictor_root,
        "optimizer_plugin": "i6n_optimizer",
        "i6n_island_mode": True,
        "i6n_control_island_id": "omega-4070",
        "i6n_horizon": horizon,
        "i6n_plan": build_plan(),
        "i6n_folds": deepcopy(old["i6n_folds"]),
        "i6n_admitted_classes": deepcopy(admitted),
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
        # Historical-fold fitness is already verified by the governed
        # evaluator.  This domain has no synthetic generator, so use DOIN's
        # explicit reduced-weight auto-accept path instead of creating
        # verification tasks that no plugin can execute.
        has_synthetic_data=False,
        synthetic_data_validation=False,
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
    archive_urls = [n.get("candidate_archive_store_url") for n in nodes]
    if any(not isinstance(url, str) or not url.startswith(("http://", "https://"))
           for url in archive_urls):
        raise ValueError("every node requires an HTTP(S) candidate_archive_store_url")
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
                # Predictor improvements are relative-to-naive deltas and are
                # commonly much smaller than the generic DOIN default of 1.0.
                "initial_threshold": 1e-6,
                "data_dir": f"/var/tmp/i6n-island-sequence/doin/h{horizon}/{node['island_id']}",
                "bootstrap_peers": [e for e in endpoints if e != node["endpoint"]],
                "component_version_scope": [
                    "doin-node", "doin-core", "doin-plugins", "predictor"
                ],
                "shared_min_peers": 3,
                "require_deterministic_seed": True,
                "dashboard_enabled": True,
                "candidate_archive_mode": "external_payload",
                "candidate_archive_root": (
                    f"/var/tmp/i6n-island-sequence/archive/h{horizon}/{node['island_id']}"
                ),
                "candidate_archive_store_id": "doin_archive_cas",
                "candidate_archive_store_url": node["candidate_archive_store_url"],
                "candidate_archive_token_env": "DATA_GOV_LAKE_TOKEN",
                "candidate_archive_pump_interval": 30.0,
                "archive_reference_spool": (
                    f"/var/tmp/i6n-island-sequence/spool/h{horizon}/{node['island_id']}"
                ),
                "archive_spool_interval": 10.0,
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


SUPERSEDED = {
    "h4": "fb70328d8b057f110006e0713d304b5c42fec5419a8dcee35527add6ca19a4cf",
    "h72": "c22b67e456c9bcc7156024dd5318587d4c876f53220686c41e28e5c9d9aeb97a",
    "plan_sha256": "1b83490b54af927033a93105e3be85b8b7323c89672b147745bbeb1d3efee983",
}


def network_id(genesis_hash, domain_ids):
    """DOIN chain identity: sha256(canonical {genesis_hash, sorted domain ids})."""
    return canonical_sha256({"genesis_hash": genesis_hash, "domains": sorted(domain_ids)})


def _gate_fold_cells(folds, data_root, manifest, contract):
    """Every fold cell must be a v2 cell with exactly the plan manifest and a passing gate."""
    from tools.i6n_historical_cell import SCHEMA_V2, file_hash, read_config

    root = Path(data_root).resolve()
    reports = {}
    for fold, item in sorted(folds.items()):
        context = item.get("i6n_evaluation_context", {})
        contamination.assert_valid_skill({"experiment": context.get("experiment")})
        if context.get("experiment") != island_plan.EXPERIMENT_V3:
            raise ValueError(f"fold {fold} experiment must be {island_plan.EXPERIMENT_V3}")
        if context.get("schema") != "i6n.historical_evaluator.v3":
            raise ValueError(f"fold {fold} needs a portable v3 evaluation context")
        pins = {}
        for key in ("base_config", "frozen_support"):
            rel = Path(context[key]["relpath"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"fold {fold} {key} must be relative to the data root")
            path = root / rel
            if file_hash(path) != context[key]["sha256"]:
                raise ValueError(f"fold {fold} {key} digest differs from its pin")
            pins[key] = path
        raw = json.loads(pins["base_config"].read_text(encoding="utf-8"))
        if raw.get("schema") != SCHEMA_V2:
            raise ValueError(f"fold {fold} base cell must be {SCHEMA_V2}")
        availability.gate(raw.get("features"), contract)
        if raw["features"] != manifest["features"]:
            raise ValueError(f"fold {fold} features differ from the plan feature manifest")
        cfg = read_config(pins["base_config"])  # authenticates digests and the gate again
        if cfg["horizon_hours"] != 72:
            raise ValueError(f"fold {fold} base cell is not an H72 cell")
        frozen = json.loads(pins["frozen_support"].read_text(encoding="utf-8"))
        if frozen.get("support") != item.get("i6n_support"):
            raise ValueError(f"fold {fold} i6n_support differs from its frozen support")
        reports[fold] = raw["feature_manifest_sha256"]
    return reports


def _domain_v3(source, predictor_root, python, campaign_root, data_root, control_island_id):
    plan = island_plan.build_plan_v3()
    contract = availability.load_contract()
    manifest = plan["feature_manifest"]
    matches = [d for d in source.get("domains", [])
               if d.get("optimization_config", {}).get("i6n_horizon") == 72]
    if len(matches) != 1:
        raise ValueError("source must contain exactly one horizon-72 domain")
    old = matches[0]["optimization_config"]
    if not {"i6n_folds", "i6n_admitted_classes"} <= set(old):
        raise ValueError("horizon-72 source lacks i6n_folds or i6n_admitted_classes")
    admitted = old["i6n_admitted_classes"]
    if (not isinstance(admitted, list) or not admitted or len(set(admitted)) != len(admitted)
            or not set(admitted) <= {"S1", "S2", "S3"}):
        raise ValueError("horizon-72 source has invalid admitted classes")
    _gate_fold_cells(old["i6n_folds"], data_root, manifest, contract)
    tag = island_plan.RUN_TAG_V3
    config = {
        "predictor_root": predictor_root,
        "optimizer_plugin": "i6n_optimizer",
        "i6n_island_mode": True,
        "i6n_control_island_id": control_island_id,
        "i6n_horizon": 72,
        "i6n_plan": plan,
        "i6n_feature_manifest": deepcopy(manifest),
        "i6n_point_in_time_label_sha256": plan["point_in_time"]["contamination_label_sha256"],
        "i6n_folds": deepcopy(old["i6n_folds"]),
        "i6n_admitted_classes": deepcopy(admitted),
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
        "optimization_resume_file": f"{campaign_root}/state/{tag}/{{island_id}}/checkpoint.json",
        "optimization_parameters_file": f"{campaign_root}/state/{tag}/{{island_id}}/champion.json",
        "stage_advance_threshold": 1e300,
    }
    domain = deepcopy(matches[0])
    domain.update(
        domain_id=island_plan.DOMAIN_ID_V3,
        optimize=True,
        evaluate=True,
        optimization_plugin="predictor",
        higher_is_better=False,
        optimization_config=config,
    )
    return domain


def generate_h72_v3(source, nodes, output, predictor_root, python, *, campaign_root, data_root,
                    genesis_hash):
    """Write the four H72 successor configs (19 gated features) and their manifest.

    Nothing of the superseded identities is reused: new domain id, plan,
    experiment, folds, state/checkpoint/archive/spool/chain roots.  ``output``
    must not exist yet (never overwrite an earlier generation).
    """
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise ValueError("nodes JSON must be a list of exactly four nodes")
    ids = [n.get("island_id") for n in nodes]
    if len(set(ids)) != 4 or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("four distinct nonempty island_id values are required")
    endpoints = [n.get("endpoint") for n in nodes]
    if len(set(endpoints)) != 4 or any(not isinstance(e, str) or ":" not in e for e in endpoints):
        raise ValueError("four distinct host:port endpoints are required")
    if any(not str(n.get("candidate_archive_store_url", "")).startswith(("http://", "https://"))
           for n in nodes):
        raise ValueError("every node requires an HTTP(S) candidate_archive_store_url")
    if not isinstance(genesis_hash, str) or len(genesis_hash) != 64:
        raise ValueError("genesis_hash must be the 64-hex DOIN genesis block hash")
    # The first node hosts the governed fixed controls (seed and small alternative).
    domain = _domain_v3(source, predictor_root, python, campaign_root, data_root, ids[0])
    digest = canonical_sha256(domain["optimization_config"])
    label = contamination.load_label()
    contamination.assert_valid_skill({"scientific_config_sha256": digest,
                                      "network_id": network_id(genesis_hash, [domain["domain_id"]])}, label)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tag = island_plan.RUN_TAG_V3
    plan = domain["optimization_config"]["i6n_plan"]
    manifest = {
        "schema": "i6n.doin_island_sequence.v2",
        "plan_sha256": plan["plan_sha256"],
        "plan_version": plan["plan_version"],
        "horizons_hours": [72],
        "order": ["H72"],
        "seed": 6102026,
        "population_per_node": 20, "node_count": 4, "max_generations": 100,
        "generation_patience": 10, "min_generations_before_stop": 30,
        "min_relative_improvement": 1e-8,
        "domain_id": domain["domain_id"],
        "experiment": island_plan.EXPERIMENT_V3,
        "genesis_hash": genesis_hash,
        "network_id": network_id(genesis_hash, [domain["domain_id"]]),
        "feature_manifest_sha256": plan["feature_manifest"]["sha256"],
        "features": list(plan["feature_manifest"]["features"]),
        "excluded_features": list(plan["feature_manifest"]["excluded"]),
        "availability_contract_sha256": plan["feature_manifest"]["availability_contract_sha256"],
        "contamination_label_sha256": label["label_sha256"],
        "supersedes": dict(SUPERSEDED),
        "optimization_config_sha256_by_horizon": {"72": digest},
        "nodes": [],
    }
    for node in nodes:
        node_dir = output / "h72" / node["island_id"]
        node_dir.mkdir(parents=True, exist_ok=False)
        config = {
            "host": "0.0.0.0",
            "port": int(node["endpoint"].rsplit(":", 1)[1]),
            "data_dir": f"{campaign_root}/doin/{tag}/{node['island_id']}",
            "bootstrap_peers": [e for e in endpoints if e != node["endpoint"]],
            "component_version_scope": ["doin-node", "doin-core", "doin-plugins", "predictor"],
            "shared_min_peers": 3,
            "require_deterministic_seed": True,
            "dashboard_enabled": True,
            "candidate_archive_mode": "external_payload",
            "candidate_archive_root": f"{campaign_root}/archive/{tag}/{node['island_id']}",
            "candidate_archive_store_id": "doin_archive_cas",
            "candidate_archive_store_url": node["candidate_archive_store_url"],
            "candidate_archive_token_env": "DATA_GOV_LAKE_TOKEN",
            "candidate_archive_pump_interval": 30.0,
            "archive_reference_spool": f"{campaign_root}/spool/{tag}/{node['island_id']}",
            "archive_spool_interval": 10.0,
            "domains": [deepcopy(domain)],
        }
        path = node_dir / "doin_node.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["nodes"].append({
            "horizon": 72, "island_id": node["island_id"], "endpoint": node["endpoint"],
            "config": str(path), "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "optimization_config_sha256": digest,
        })
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def generate_h72_ds_in_v4(source, nodes, output, predictor_root, python, *, campaign_root,
                          data_root, genesis_hash):
    """Write the first H72 successor as an input-only downsampling experiment."""
    if not isinstance(nodes, list) or len(nodes) != 4:
        raise ValueError("nodes JSON must be a list of exactly four nodes")
    ids = [n.get("island_id") for n in nodes]
    endpoints = [n.get("endpoint") for n in nodes]
    if len(set(ids)) != 4 or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("four distinct nonempty island_id values are required")
    if len(set(endpoints)) != 4 or any(not isinstance(e, str) or ":" not in e for e in endpoints):
        raise ValueError("four distinct host:port endpoints are required")
    if any(not str(n.get("candidate_archive_store_url", "")).startswith(("http://", "https://"))
           for n in nodes):
        raise ValueError("every node requires an HTTP(S) candidate_archive_store_url")
    if not isinstance(genesis_hash, str) or len(genesis_hash) != 64:
        raise ValueError("genesis_hash must be the 64-hex DOIN genesis block hash")

    domain = _domain_v3(source, predictor_root, python, campaign_root, data_root, ids[0])
    plan = island_plan.build_plan_v4_ds_in()
    tag = island_plan.RUN_TAG_V4_DS_IN
    opt = domain["optimization_config"]
    opt.update(
        i6n_plan=plan,
        i6n_genotype_constraints=deepcopy(plan["search_constraints"]),
        optimization_resume_file=f"{campaign_root}/state/{tag}/{{island_id}}/checkpoint.json",
        optimization_parameters_file=f"{campaign_root}/state/{tag}/{{island_id}}/champion.json",
    )
    for fold in opt["i6n_folds"].values():
        fold["i6n_evaluation_context"]["experiment"] = island_plan.EXPERIMENT_V4_DS_IN
    domain["domain_id"] = island_plan.DOMAIN_ID_V4_DS_IN
    digest = canonical_sha256(opt)
    label = contamination.load_label()
    contamination.assert_valid_skill({
        "scientific_config_sha256": digest,
        "network_id": network_id(genesis_hash, [domain["domain_id"]]),
    }, label)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "i6n.doin_island_sequence.v2",
        "plan_sha256": plan["plan_sha256"],
        "plan_version": plan["plan_version"],
        "horizons_hours": [72],
        "order": ["H72-DS-IN"],
        "seed": 6102026,
        "population_per_node": 20,
        "node_count": 4,
        "max_generations": 100,
        "generation_patience": 10,
        "min_generations_before_stop": 30,
        "min_relative_improvement": 1e-8,
        "domain_id": domain["domain_id"],
        "experiment": island_plan.EXPERIMENT_V4_DS_IN,
        "genesis_hash": genesis_hash,
        "network_id": network_id(genesis_hash, [domain["domain_id"]]),
        "feature_manifest_sha256": plan["feature_manifest"]["sha256"],
        "features": list(plan["feature_manifest"]["features"]),
        "excluded_features": list(plan["feature_manifest"]["excluded"]),
        "availability_contract_sha256": plan["feature_manifest"]["availability_contract_sha256"],
        "contamination_label_sha256": label["label_sha256"],
        "genotype_constraints": deepcopy(plan["search_constraints"]),
        "supersedes": dict(SUPERSEDED),
        "optimization_config_sha256_by_horizon": {"72": digest},
        "nodes": [],
    }
    for node in nodes:
        node_dir = output / "h72-ds-in" / node["island_id"]
        node_dir.mkdir(parents=True, exist_ok=False)
        config = {
            "host": "0.0.0.0",
            "port": int(node["endpoint"].rsplit(":", 1)[1]),
            "data_dir": f"{campaign_root}/doin/{tag}/{node['island_id']}",
            "bootstrap_peers": [e for e in endpoints if e != node["endpoint"]],
            "component_version_scope": ["doin-node", "doin-core", "doin-plugins", "predictor"],
            "shared_min_peers": 3,
            "require_deterministic_seed": True,
            "dashboard_enabled": True,
            "candidate_archive_mode": "external_payload",
            "candidate_archive_root": f"{campaign_root}/archive/{tag}/{node['island_id']}",
            "candidate_archive_store_id": "doin_archive_cas",
            "candidate_archive_store_url": node["candidate_archive_store_url"],
            "candidate_archive_token_env": "DATA_GOV_LAKE_TOKEN",
            "candidate_archive_pump_interval": 30.0,
            "archive_reference_spool": f"{campaign_root}/spool/{tag}/{node['island_id']}",
            "archive_spool_interval": 10.0,
            "domains": [deepcopy(domain)],
        }
        path = node_dir / "doin_node.json"
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["nodes"].append({
            "horizon": 72,
            "island_id": node["island_id"],
            "endpoint": node["endpoint"],
            "config": str(path),
            "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "optimization_config_sha256": digest,
        })
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-node-config", required=True)
    parser.add_argument("--nodes-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictor-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--h72-v3-no-stlfsi4", action="store_true",
                        help="generate only the gated H72 successor identity (owner order 2026-10-12)")
    parser.add_argument("--h72-ds-in-v4", action="store_true",
                        help="generate the gated input-downsampled H72 successor identity")
    parser.add_argument("--campaign-root", help="runtime root for state/archive/spool/chain (v3)")
    parser.add_argument("--data-root", help="data root holding the fold cells and frozen supports (v3)")
    parser.add_argument("--genesis-hash", help="DOIN genesis block hash (v3)")
    args = parser.parse_args(argv)
    if args.h72_v3_no_stlfsi4 == args.h72_ds_in_v4:
        parser.error(
            "select exactly one of --h72-v3-no-stlfsi4 or --h72-ds-in-v4"
        )
    source = json.loads(Path(args.source_node_config).read_text(encoding="utf-8"))
    nodes = json.loads(Path(args.nodes_json).read_text(encoding="utf-8"))
    if not (args.campaign_root and args.data_root and args.genesis_hash):
        parser.error("--h72-v3-no-stlfsi4 requires --campaign-root, --data-root and --genesis-hash")
    generator = generate_h72_ds_in_v4 if args.h72_ds_in_v4 else generate_h72_v3
    manifest = generator(source, nodes, args.output, args.predictor_root, args.python,
                         campaign_root=args.campaign_root, data_root=args.data_root,
                         genesis_hash=args.genesis_hash)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
