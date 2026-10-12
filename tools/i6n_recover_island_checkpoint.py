"""Recover an in-progress I6N island checkpoint from its durable archive.

This utility is intentionally narrow.  It accepts only generation-zero archive
records that form an exact prefix of the deterministic initial population.  It
revalidates every archived payload and binds the recovered cache to the current
sealed optimizer contract before writing a checkpoint.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Iterator, Mapping

from optimizer_plugins.i6n_optimizer import Plugin, _V2, _digest, _json
from app.config import DEFAULT_VALUES
from tools import i6n_search_space as space
from tools.i6n_archive import records


RECOVERY_SCHEMA = "i6n.island.checkpoint.recovery.v1"


@contextlib.contextmanager
def _island_environment(island_id: str) -> Iterator[None]:
    previous = os.environ.get("I6N_ISLAND_ID")
    os.environ["I6N_ISLAND_ID"] = island_id
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("I6N_ISLAND_ID", None)
        else:
            os.environ["I6N_ISLAND_ID"] = previous


def _atomic_json(path: Path, payload: Mapping) -> None:
    Plugin._atomic_json(str(path), payload)


def _island_domain(node_config: Mapping) -> Mapping:
    domains = [
        domain for domain in node_config.get("domains", [])
        if isinstance(domain, Mapping)
        and isinstance(domain.get("optimization_config"), Mapping)
        and domain["optimization_config"].get("i6n_island_mode") is True
    ]
    if len(domains) != 1:
        raise ValueError("node config must contain exactly one I6N island domain")
    return domains[0]


def _predictor_runtime_config(optimization_config: Mapping) -> dict:
    """Reproduce doin-plugins' effective predictor config without callbacks."""
    config = deepcopy(DEFAULT_VALUES)
    config_file = optimization_config.get("load_config")
    if config_file:
        path = Path(config_file)
        predictor_root = optimization_config.get("predictor_root")
        if not path.is_absolute() and predictor_root:
            path = Path(predictor_root) / path
        config.update(json.loads(path.read_text(encoding="utf-8")))
    config.update({
        key: deepcopy(value)
        for key, value in optimization_config.items()
        if key not in ("predictor_root", "load_config", "optimization_callbacks")
    })
    config["disable_postfit_uncertainty"] = True
    config.setdefault("quiet", True)
    return config


def _read_payloads(outbox_path: Path) -> list[tuple[dict, str]]:
    connection = sqlite3.connect(f"file:{outbox_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT body, body_digest FROM items ORDER BY created_at, item_id"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("archive outbox contains no completed candidates")

    payloads = []
    for body, expected_digest in rows:
        body_bytes = body if isinstance(body, bytes) else body.encode("utf-8")
        observed_digest = hashlib.sha256(body_bytes).hexdigest()
        if observed_digest != expected_digest:
            raise ValueError("archive body digest mismatch")
        envelope = json.loads(body_bytes)
        if (set(envelope) != {"schema", "domain_id", "payload_schema", "payload",
                             "producer_peer_id"}
                or envelope["schema"] != "doin.external_payload_body.v1"):
            raise ValueError("invalid external archive envelope")
        payload = records.validate_island_candidate_payload(envelope["payload"])
        if envelope["payload_schema"] != payload["schema"]:
            raise ValueError("external archive payload schema mismatch")
        payloads.append((payload, observed_digest))
    return payloads


def _result_from_payload(payload: Mapping) -> dict:
    folds = payload["per_fold"]
    origins_bindings = {row["origins_binding"] for row in folds}
    if len(origins_bindings) != 1:
        raise ValueError("archived folds disagree on origins binding")
    metrics = {
        "relative_error_vs_naive": payload["fitness"],
        "per_fold": {row["fold"]: row["ratio_to_naive"] for row in folds},
        "receipts": {row["fold"]: row["receipt"] for row in folds},
        "origins_binding": next(iter(origins_bindings)),
        "origins_sha256": {row["fold"]: row["origins_sha256"] for row in folds},
        "aggregation": payload["aggregation"],
        "genotype_canonical_sha256": payload["genotype_canonical_sha256"],
        "effective_config_sha256": payload["effective_config_sha256"],
        "plan_sha256": payload["plan_sha256"],
        "contract_sha256": payload["contract_sha256"],
        "checkpoint_identity_sha256": payload["checkpoint_identity_sha256"],
    }
    aggregate = payload["aggregate_metrics"]
    return {
        "fitness": payload["fitness"],
        "hyper_dict": deepcopy(payload["effective_config"]),
        "genotype_canonical_sha256": payload["genotype_canonical_sha256"],
        "train_mae": aggregate["train_mae"],
        "train_naive_mae": aggregate["train_naive_mae"],
        "val_mae": aggregate["inner_mae"],
        "val_naive_mae": aggregate["inner_naive_mae"],
        "metrics": metrics,
    }


def recover_checkpoint(
    *, node_config_path: Path, outbox_path: Path, island_id: str,
    checkpoint_path: Path, report_path: Path,
) -> dict:
    node_config = json.loads(node_config_path.read_text(encoding="utf-8"))
    domain = _island_domain(node_config)
    optimization_config = deepcopy(domain["optimization_config"])
    runtime_config = _predictor_runtime_config(optimization_config)

    with _island_environment(island_id):
        plugin = Plugin()
        cfg = plugin._config(runtime_config)
        cfg.setdefault("i6n_plan", __import__(
            "tools.i6n_island_plan", fromlist=["build_plan"]
        ).build_plan())
        settings = plugin._island_settings(cfg)
        if settings["island_id"] != island_id:
            raise ValueError("runtime island identity mismatch")
        plan, contract = _V2(plugin)._contract(cfg)
        admitted = _V2._admitted(cfg)
        search_seed = int.from_bytes(hashlib.sha256(_json({
            "plan_sha256": plan["plan_sha256"],
            "horizon": settings["horizon"],
            "training_seed": settings["seed"],
            "island_id": settings["island_id"],
        }).encode("utf-8")).digest()[:8], "big")
        checkpoint_identity = plugin._checkpoint_identity(
            cfg, plan, contract, settings, admitted, search_seed
        )
        checkpoint_identity_sha256 = _digest(checkpoint_identity)
        rng = random.Random(search_seed)
        population = plugin._initial_island_population(settings, admitted, rng)

    configured_checkpoint = Path(plugin._runtime_path(
        cfg.get("optimization_resume_file"), settings
    ))
    if configured_checkpoint != checkpoint_path:
        raise ValueError("requested checkpoint path differs from optimizer runtime path")
    payloads = _read_payloads(outbox_path)
    population_ids = [space.genotype_sha256(genotype) for genotype in population]
    archived_ids = [payload["genotype_canonical_sha256"] for payload, _ in payloads]
    expected_positions = list(range(1, len(payloads) + 1))
    observed_positions = [payload["population_index"] for payload, _ in payloads]
    if observed_positions != expected_positions or archived_ids != population_ids[:len(payloads)]:
        raise ValueError("archived candidates are not an exact initial-population prefix")
    if len(set(archived_ids)) != len(archived_ids):
        raise ValueError("archive contains duplicate candidate identities")

    cache = {}
    for payload, _ in payloads:
        expected = {
            "experiment": domain["domain_id"],
            "horizon_hours": settings["horizon"],
            "seed": settings["seed"],
            "island_id": island_id,
            "generation": 0,
            "plan_sha256": plan["plan_sha256"],
            "contract_sha256": checkpoint_identity["contract_sha256"],
            "checkpoint_identity_sha256": checkpoint_identity_sha256,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValueError(f"archived candidate has mismatched {key}")
        cache[payload["genotype_canonical_sha256"]] = _result_from_payload(payload)

    champion_id = min(
        archived_ids,
        key=lambda identity: (float(cache[identity]["fitness"]), archived_ids.index(identity)),
    )
    champion = cache[champion_id]
    checkpoint = {
        "schema": "i6n.island.checkpoint.v2",
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "plan_sha256": plan["plan_sha256"],
        "horizon": settings["horizon"],
        "seed": settings["seed"],
        "island_id": island_id,
        "search_seed": search_seed,
        "generation": 0,
        "generation_complete": False,
        "population": population,
        "cache": cache,
        "no_improve": 0,
        "best_fitness": champion["fitness"],
        "best_params": champion["hyper_dict"],
        "best_identity": champion_id,
        "rng_state": rng.getstate(),
        "best_train_mae": champion["train_mae"],
        "best_train_naive_mae": champion["train_naive_mae"],
        "best_val_mae": champion["val_mae"],
        "best_val_naive_mae": champion["val_naive_mae"],
        "stop_reason": None,
    }
    if checkpoint_path.exists():
        current = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if current != checkpoint:
            raise ValueError("refusing to overwrite a different checkpoint")
    else:
        _atomic_json(checkpoint_path, checkpoint)

    report = {
        "schema": RECOVERY_SCHEMA,
        "island_id": island_id,
        "horizon": settings["horizon"],
        "generation": 0,
        "generation_complete": False,
        "recovered_candidate_count": len(cache),
        "population_size": len(population),
        "recovered_candidate_sha256": archived_ids,
        "archive_body_sha256": [digest for _, digest in payloads],
        "champion_sha256": champion_id,
        "champion_fitness": champion["fitness"],
        "plan_sha256": plan["plan_sha256"],
        "contract_sha256": checkpoint_identity["contract_sha256"],
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "checkpoint_sha256": _digest(checkpoint),
    }
    _atomic_json(report_path, report)
    return report


def patch_node_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    domain = _island_domain(config)
    for key in ("has_synthetic_data", "synthetic_data_validation"):
        if domain.get(key) not in (None, False):
            raise ValueError(f"refusing to replace true {key}")
        domain[key] = False
    if config.get("initial_threshold") not in (None, 1.0, 1e-6):
        raise ValueError("refusing to replace an unexpected initial_threshold")
    config["initial_threshold"] = 1e-6
    backup = path.with_suffix(path.suffix + ".pre-chain-recovery")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    _atomic_json(path, config)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "backup": str(backup),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--node-config", type=Path, required=True)
    recover.add_argument("--outbox", type=Path, required=True)
    recover.add_argument("--island-id", required=True)
    recover.add_argument("--checkpoint", type=Path, required=True)
    recover.add_argument("--report", type=Path, required=True)
    patch = subparsers.add_parser("patch-config")
    patch.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "recover":
        result = recover_checkpoint(
            node_config_path=args.node_config,
            outbox_path=args.outbox,
            island_id=args.island_id,
            checkpoint_path=args.checkpoint,
            report_path=args.report,
        )
    else:
        result = {"patched": [patch_node_config(path) for path in args.paths]}
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
