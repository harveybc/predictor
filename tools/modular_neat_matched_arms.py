"""Matched NEAT-vs-random driver over one governed modular campaign (D3, 2026-10-02).

Both arms use the same campaign tooling (durable queue, pinned bridge executor, exact
verification), the same declared base/space/data, the same restricted sub-space and the
same candidate counter.  ``init`` creates a NEW campaign root whose search space and base
are byte-identical to a source campaign (so a verified source cell keeps its config identity)
and imports the verified control cell(s) verbatim with provenance; nothing is retrained.

    init    --source-root R --source-queue DB --root NEW --seeds 2021 --control-flat F
    neat    --root NEW --restrict F --pop P --gens G --neat-seed S --host-role ROLE
    random  --root NEW --restrict F --n N --draw-seed S --host-role ROLE
    report  --root NEW
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import modular_doin_campaign as cm  # noqa: E402
from tools import modular_search_space as ss  # noqa: E402
from tools.modular_neat_policy import normalize_restrict, propose_restricted  # noqa: E402


def load_restrict(path):
    return json.loads(Path(path).read_text())["restrict"]


def init_campaign(source_root, source_queue, root, seeds, control_flat, campaign_id, note, caps=None, drop_caps=()):
    source = json.loads((Path(source_root) / "CAMPAIGN.json").read_text())
    declaration = copy.deepcopy(source)
    declaration["campaign_id"] = campaign_id
    declaration["paired_seeds"] = list(seeds)
    declaration["default_candidate"] = {k: v for k, v in control_flat.items() if k != "train.seed"}
    for role, (train_cap, verify_cap) in (caps or {}).items():  # caps = 1.25 x measured peak, never lowered
        res = declaration.setdefault("hosts", {}).setdefault(role, {}).setdefault("resources", {})
        for kind, cap in (("train", train_cap), ("verify", verify_cap)):
            res[kind] = {**declaration["resources"][kind], **res.get(kind, {}), "cap": cap}
    for dim in drop_caps:  # declared deviation: the dimension has no measurement path at enqueue time
        declaration["budget"]["caps"].pop(dim)
    declaration["continues"] = {"campaign": source["campaign_id"], "reason": note,
                                "search_space_sha256": ss.digest(source["search_space"]),
                                "base_sha256": ss.digest(source["base"])}
    campaign = cm.Campaign.create(root, declaration)
    imported = import_verified(campaign, source, source_queue, declaration["default_candidate"], seeds)
    return campaign, imported


def import_verified(campaign, source_declaration, source_queue, flat, seeds):
    """Copy the verified rows (candidate + attempts + budget) of ``flat`` x ``seeds`` verbatim."""
    for key in ("search_space", "base", "data", "executor"):
        if ss.digest(campaign.declaration[key]) != ss.digest(source_declaration[key]):
            raise ValueError(f"{key} differs from the source campaign: identities would not match")
    src = sqlite3.connect(f"file:{source_queue}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    imported = []
    campaign.db.execute("BEGIN IMMEDIATE")
    position = campaign.db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM candidates").fetchone()[0]
    for seed in seeds:
        nested = ss.from_flat({**flat, "train.seed": seed}, campaign.declaration["base"], campaign.space)
        cid = ss.digest(nested)
        row = src.execute("SELECT * FROM candidates WHERE cid=?", (cid,)).fetchone()
        if row is None or row["status"] != "verified":
            campaign.db.execute("ROLLBACK")
            raise ValueError(f"seed {seed}: no verified source row with cid {cid[:8]}")
        if campaign.db.execute("SELECT 1 FROM candidates WHERE cid=?", (cid,)).fetchone():
            continue
        values = list(row)
        values[1] = position
        campaign.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", values)
        for a in src.execute("SELECT * FROM attempts WHERE cid=?", (cid,)):
            campaign.db.execute("INSERT INTO attempts(" + ",".join(a.keys()) + ") VALUES(" +
                                ",".join("?" * len(a.keys())) + ")", list(a))
        for b in src.execute("SELECT * FROM budget WHERE cid=?", (cid,)):
            campaign.db.execute("INSERT INTO budget VALUES(?,?,?,?)", list(b))
        position += 1
        imported.append({"seed": seed, "cid": cid, "config_id": row["config_id"], "objective": row["objective"]})
    campaign.db.execute("COMMIT")
    return imported


def run_neat(root, restrict_path, pop, gens, neat_seed, host_role):
    from optimizer_plugins.modular_doin_optimizer import Plugin

    campaign = cm.Campaign(root)
    executor = cm.DoinBridgeExecutor(campaign.declaration, host_role)
    plugin = Plugin(executor=executor)
    plugin.set_params(modular_neat_population_size=pop, modular_neat_generations=gens,
                      modular_neat_seed=neat_seed, modular_neat_restrict=load_restrict(restrict_path),
                      modular_max_candidates=pop * gens * len(campaign.declaration["paired_seeds"]))
    plugin._run_neat(campaign, executor)


def run_random(root, restrict_path, n, draw_seed, host_role):
    campaign = cm.Campaign(root)
    restrict = normalize_restrict(campaign.space, load_restrict(restrict_path))
    rng = random.Random(draw_seed)
    control = ss.digest({k: v for k, v in campaign.declaration["default_candidate"].items()})
    seen, drawn = {control}, []
    while len(drawn) < n - 1:  # n counts the reused control, exactly as the NEAT arm's generation zero does
        flat = propose_restricted(campaign.space, campaign.declaration["base"], rng, restrict)
        if ss.digest(flat) in seen:
            continue
        seen.add(ss.digest(flat))
        drawn.append(flat)
    for i, flat in enumerate(drawn):
        campaign.enqueue(flat, f"random_{i:04d}")
    executor = cm.DoinBridgeExecutor(campaign.declaration, host_role)
    campaign.run(executor)


def report(root):
    campaign = cm.Campaign(root)
    out = []
    for r in campaign.db.execute("SELECT config_id, label, GROUP_CONCAT(seed) seeds, GROUP_CONCAT(status) st, "
                                 "GROUP_CONCAT(objective) obj, MIN(position) pos FROM candidates "
                                 "GROUP BY config_id ORDER BY pos"):
        att = campaign.db.execute(
            "SELECT kind, elapsed_seconds, cgroup_peak_bytes, observed_updates, stop_reason FROM attempts a "
            "JOIN candidates c ON c.cid=a.cid WHERE c.config_id=?", (r["config_id"],)).fetchall()
        out.append({**dict(r), "attempts": [dict(a) for a in att]})
    print(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--source-root", required=True)
    a.add_argument("--source-queue", required=True)
    a.add_argument("--root", required=True)
    a.add_argument("--seeds", type=int, nargs="+", required=True)
    a.add_argument("--control-flat", required=True)
    a.add_argument("--campaign-id", required=True)
    a.add_argument("--note", required=True)
    a.add_argument("--drop-budget-cap", action="append", default=[])
    a.add_argument("--cap", nargs=3, action="append", metavar=("ROLE", "TRAIN", "VERIFY"), default=[])
    a = sub.add_parser("neat")
    a.add_argument("--root", required=True)
    a.add_argument("--restrict", required=True)
    a.add_argument("--pop", type=int, required=True)
    a.add_argument("--gens", type=int, required=True)
    a.add_argument("--neat-seed", type=int, required=True)
    a.add_argument("--host-role", required=True)
    a = sub.add_parser("random")
    a.add_argument("--root", required=True)
    a.add_argument("--restrict", required=True)
    a.add_argument("--n", type=int, required=True)
    a.add_argument("--draw-seed", type=int, required=True)
    a.add_argument("--host-role", required=True)
    a = sub.add_parser("report")
    a.add_argument("--root", required=True)
    args = p.parse_args()
    if args.cmd == "init":
        _, imported = init_campaign(args.source_root, args.source_queue, args.root, args.seeds,
                                    json.loads(Path(args.control_flat).read_text()), args.campaign_id, args.note,
                                    {r: (t, v) for r, t, v in args.cap}, args.drop_budget_cap)
        print(json.dumps({"created": args.root, "imported_verified": imported}, indent=1))
    elif args.cmd == "neat":
        run_neat(args.root, args.restrict, args.pop, args.gens, args.neat_seed, args.host_role)
    elif args.cmd == "random":
        run_random(args.root, args.restrict, args.n, args.draw_seed, args.host_role)
    else:
        report(args.root)


if __name__ == "__main__":
    main()
