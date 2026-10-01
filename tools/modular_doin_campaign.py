#!/usr/bin/env python3
"""Persistent finite DOIN candidate queue and incumbent for modular candidates.

Every candidate is written to ``queue.sqlite`` (WAL, one transaction per state
change) BEFORE it executes. Each execution is an *attempt* with its own output
root, admission cap, observed cost (whole-cgroup peak from the heartbeat, wall,
updates, seconds/update), exit status and receipt. After a training attempt is
accepted, an independent checkpoint verification runs in a separate admitted
child; only ``verified`` candidates count for the incumbent.

Resume semantics: ``completed``/``verified``/``refuted``/``failed`` candidates are
never re-trained. A ``running`` row whose recorded launcher process is gone is
marked ``interrupted`` and returns to ``queued`` with a new attempt number; a
``completed`` row lacking verification resumes at verification only.

Incumbent: configurations (candidate minus seed) with ALL declared paired seeds
verified, ranked by the mean of the declared validation objective in its
declared direction. Changes are appended to ``incumbent_changes``; nothing is
overwritten. Architecture complexity never enters the ranking.

Executors: ``doin_bridge`` launches ``doin_node.predictor_bridge`` (the DOIN
OptimizationPlugin/InferencePlugin adapter) through ``crispdm-run``; ``local``
runs the predictor worker directly (tests, labelled local).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import modular_search_space as ss  # noqa: E402

TERMINAL = ("verified", "refuted", "failed")
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates(
  cid TEXT PRIMARY KEY, position INTEGER NOT NULL UNIQUE, config_id TEXT NOT NULL,
  seed INTEGER NOT NULL, label TEXT NOT NULL, flat TEXT NOT NULL, nested TEXT, status TEXT NOT NULL,
  blocked_reason TEXT, created REAL NOT NULL, objective REAL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS attempts(
  cid TEXT NOT NULL, attempt INTEGER NOT NULL, kind TEXT NOT NULL, launcher_pid INTEGER,
  started REAL NOT NULL, finished REAL, status TEXT NOT NULL, exit_code INTEGER, error TEXT,
  cap TEXT, wall TEXT, output_root TEXT NOT NULL, cgroup_peak_bytes INTEGER, elapsed_seconds REAL,
  observed_updates INTEGER, selected_epoch INTEGER, per_update_seconds REAL, stop_reason TEXT,
  receipt_path TEXT, objective REAL, model_sha256 TEXT, weights_sha256 TEXT, verdict TEXT, host TEXT,
  PRIMARY KEY(cid, attempt, kind));
CREATE TABLE IF NOT EXISTS incumbent_changes(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, config_id TEXT NOT NULL,
  mean_objective REAL NOT NULL, seeds TEXT NOT NULL, cids TEXT NOT NULL,
  previous_config_id TEXT, previous_mean REAL, reason TEXT NOT NULL);
"""


def now():
    return time.time()


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Campaign:
    def __init__(self, root):
        self.root = Path(root)
        self.declaration = json.loads((self.root / "CAMPAIGN.json").read_text())
        self.db = sqlite3.connect(self.root / "queue.sqlite", timeout=60, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA_SQL)
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(attempts)")}
        if "host" not in columns:  # queues created before the multi-host runner
            self.db.execute("ALTER TABLE attempts ADD COLUMN host TEXT")
        self.space = ss.validate_space(self.declaration["search_space"])

    # ------------------------------------------------------------ creation --
    @classmethod
    def create(cls, root, declaration):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        ss.validate_space(declaration["search_space"])
        objective = declaration["base"]["objective"]
        if objective.get("split") != "validation" or type(objective.get("higher_is_better")) is not bool:
            raise ValueError("campaign requires an explicit validation objective and direction")
        seeds = declaration["paired_seeds"]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("paired_seeds must be distinct")
        for split in ("train", "validation"):
            data = declaration["data"][split]
            if declaration.get("data_location") == "workers":
                # bytes live on the workers; every attempt's receipt must bind these digests
                if len(data["sha256"]) != 64:
                    raise ValueError(f"{split} data digest must be a full sha256")
            elif sha_file(data["path"]) != data["sha256"]:
                raise ValueError(f"{split} data digest mismatch at creation")
        text = json.dumps(declaration, indent=2, sort_keys=True, allow_nan=False) + "\n"
        (root / "CAMPAIGN.json").write_text(text)
        campaign = cls(root)
        campaign.db.execute("INSERT INTO meta VALUES('campaign_sha256', ?)", (hashlib.sha256(text.encode()).hexdigest(),))
        return campaign

    def enqueue(self, flat_without_seed, label):
        """Persist one configuration x every paired seed BEFORE any execution."""
        added = []
        self.db.execute("BEGIN IMMEDIATE")
        try:
            position = self.db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM candidates").fetchone()[0]
            for seed in self.declaration["paired_seeds"]:
                flat = {**flat_without_seed, "train.seed": seed}
                ss.validate_flat(flat, self.space)  # invalid combinations fail before fit and before queueing
                try:
                    nested = ss.from_flat(flat, self.declaration["base"], self.space)
                    status, reason = "queued", None
                    config_id, cid = ss.config_identity(nested), ss.digest(nested)
                    nested_text = ss.canonical(nested)
                except ss.SearchSpaceError as exc:
                    if "donor" not in str(exc):
                        raise
                    status, reason, nested_text = "blocked", str(exc), None
                    stripped = {k: v for k, v in flat.items() if k != "train.seed"}
                    config_id, cid = "flat:" + ss.digest(stripped), "flat:" + ss.digest(flat)
                if self.db.execute("SELECT 1 FROM candidates WHERE cid=?", (cid,)).fetchone():
                    continue
                self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                                (cid, position, config_id, seed, label, ss.canonical(flat), nested_text,
                                 status, reason, now(), None, now()))
                position += 1
                added.append(cid)
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return added

    def unblock(self):
        """Re-materialize blocked candidates once their donors are declared in base.donors."""
        released = []
        for row in self.db.execute("SELECT * FROM candidates WHERE status='blocked' ORDER BY position").fetchall():
            if (row["blocked_reason"] or "").startswith("HOLD:"):
                continue  # operational holds are released only by release_hold
            flat = json.loads(row["flat"])
            try:
                nested = ss.from_flat(flat, self.declaration["base"], self.space)
            except ss.SearchSpaceError:
                continue
            self.db.execute("UPDATE candidates SET cid=?, config_id=?, nested=?, status='queued', blocked_reason=NULL,"
                            " updated=? WHERE cid=?", (ss.digest(nested), ss.config_identity(nested),
                                                      ss.canonical(nested), now(), row["cid"]))
            released.append(ss.digest(nested))
        return released

    def hold(self, predicate, reason):
        """Block queued candidates whose flat parameters satisfy ``predicate`` (never drops them)."""
        held = []
        self.db.execute("BEGIN IMMEDIATE")
        for row in self.db.execute("SELECT cid, flat FROM candidates WHERE status='queued'").fetchall():
            if predicate(json.loads(row["flat"])):
                self.db.execute("UPDATE candidates SET status='blocked', blocked_reason=?, updated=? WHERE cid=?",
                                ("HOLD:" + reason, now(), row["cid"]))
                held.append(row["cid"])
        self.db.execute("COMMIT")
        return held

    def import_superseded(self, source_root, reason, status="SUPERSEDED_OLD_ARCH"):
        """Record every candidate of an older campaign as a terminal, never-dispatched row.

        The old cid/config_id/flat/objective are kept verbatim with their source campaign
        and source status, so old results stay citable as evidence of their own design.
        """
        source = sqlite3.connect(f"file:{Path(source_root) / 'queue.sqlite'}?mode=ro", uri=True)
        source.row_factory = sqlite3.Row
        campaign_id = json.loads((Path(source_root) / "CAMPAIGN.json").read_text())["campaign_id"]
        imported = 0
        self.db.execute("BEGIN IMMEDIATE")
        position = self.db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM candidates").fetchone()[0]
        for row in source.execute("SELECT * FROM candidates ORDER BY position"):
            cid = f"{campaign_id}:{row['cid']}"
            if self.db.execute("SELECT 1 FROM candidates WHERE cid=?", (cid,)).fetchone():
                continue
            self.db.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (cid, position, f"{campaign_id}:{row['config_id']}", row["seed"], row["label"],
                             row["flat"], row["nested"], status,
                             f"{reason} (source {campaign_id}, source status {row['status']})",
                             now(), row["objective"], now()))
            position += 1
            imported += 1
        self.db.execute("COMMIT")
        return imported

    def release_hold(self, reason):
        cur = self.db.execute("UPDATE candidates SET status='queued', blocked_reason=NULL, updated=? WHERE "
                              "status='blocked' AND blocked_reason=?", (now(), "HOLD:" + reason))
        return cur.rowcount

    # -------------------------------------------------------------- resume --
    def recover(self):
        """Interrupted launchers return their candidate to the queue; completed work is kept."""
        recovered = []
        for row in self.db.execute("SELECT * FROM attempts WHERE status='running'").fetchall():
            pid = row["launcher_pid"]
            if pid and Path(f"/proc/{pid}").exists():
                continue
            self.db.execute("UPDATE attempts SET status='interrupted', finished=? WHERE cid=? AND attempt=? AND kind=?",
                            (now(), row["cid"], row["attempt"], row["kind"]))
            back = "queued" if row["kind"] == "train" else "completed"
            self.db.execute("UPDATE candidates SET status=?, updated=? WHERE cid=?", (back, now(), row["cid"]))
            recovered.append((row["cid"], row["kind"]))
        return recovered

    def claim(self, host="local"):
        """Atomically pick the next work item for ``host`` and mark it running (one transaction).

        Verification is claimed only by the host that holds the candidate's checkpoint.
        Two runners (one per host) can share this queue: no candidate is dispatched twice.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute(
                "SELECT c.* FROM candidates c WHERE c.status='completed' AND (SELECT a.host FROM attempts a"
                " WHERE a.cid=c.cid AND a.kind='train' AND a.status='completed' ORDER BY a.attempt DESC LIMIT 1)"
                " IS ? ORDER BY c.position LIMIT 1", (host,)).fetchone()
            kind = "verify"
            if row is None:
                excluded = set(self.declaration.get("placement", {}).get("exclude", {}).get(host, []))
                row = next((r for r in self.db.execute(
                    "SELECT * FROM candidates WHERE status='queued' ORDER BY position").fetchall()
                    if r["config_id"] not in excluded), None)
                kind = "train"
            if row is None:
                self.db.execute("COMMIT")
                return None
            cid = row["cid"]
            attempt = self.db.execute("SELECT COALESCE(MAX(attempt), 0) + 1 FROM attempts WHERE cid=? AND kind=?",
                                      (cid, kind)).fetchone()[0]
            output_root = self.root / "attempts" / cid[:16] / f"{kind}-{attempt}"
            resources = self.resources(host)[kind]
            self.db.execute("INSERT INTO attempts(cid, attempt, kind, launcher_pid, started, status, cap, wall,"
                            " output_root, host) VALUES(?,?,?,?,?,?,?,?,?,?)",
                            (cid, attempt, kind, os.getpid(), now(), "running", resources["cap"], resources["wall"],
                             str(output_root), host))
            self.db.execute("UPDATE candidates SET status=?, updated=? WHERE cid=?",
                            ("running" if kind == "train" else "verifying", now(), cid))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return row, kind, attempt, output_root

    def resources(self, host):
        return {**self.declaration["resources"],
                **self.declaration.get("hosts", {}).get(host, {}).get("resources", {})}

    def next_work(self, host="local"):
        """Read-only preview of what claim() would pick (no state change)."""
        row = self.db.execute("SELECT * FROM candidates WHERE status='completed' ORDER BY position LIMIT 1").fetchone()
        if row:
            return row, "verify"
        row = self.db.execute("SELECT * FROM candidates WHERE status='queued' ORDER BY position LIMIT 1").fetchone()
        return (row, "train") if row else (None, None)

    # ------------------------------------------------------------ execution --
    def pinned(self):
        """A campaign dispatches only when its predictor revision is a full commit id."""
        revision = self.declaration.get("executor", {}).get("predictor_revision", "")
        return isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)

    def run(self, executor, max_candidates=None, stop_file=None):
        if self.declaration.get("require_pin", False) and not self.pinned():
            raise RuntimeError("campaign is not pinned to a full predictor commit; nothing is dispatched")
        done = 0
        host = getattr(executor, "host", "local")
        self.recover()
        while max_candidates is None or done < max_candidates:
            if stop_file and Path(stop_file).exists():
                break
            claimed = self.claim(host)
            if claimed is None:
                break
            row, kind, attempt, output_root = claimed
            self.execute(row, kind, executor, attempt, output_root)
            if kind == "verify":
                self.update_incumbent()
            done += kind == "train"
        return done

    def execute(self, row, kind, executor, attempt, output_root):
        cid = row["cid"]
        output_root.mkdir(parents=True, exist_ok=False)
        try:
            if kind == "train":
                outcome = executor.train(json.loads(row["nested"]), output_root, self.declaration)
            else:
                receipt = self.db.execute("SELECT receipt_path FROM attempts WHERE cid=? AND kind='train' AND"
                                          " status='completed' ORDER BY attempt DESC LIMIT 1", (cid,)).fetchone()[0]
                outcome = executor.verify(receipt, output_root, self.declaration)
        except Exception as exc:  # the executor normally reports failures in outcome
            outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if (kind == "verify" and outcome.get("verdict") == "VERIFIED" and outcome.get("exact_match") is not True
                and self.declaration.get("verification", {}).get("require_exact_match")):
            # deterministic campaigns accept only bitwise-equal rescoring; anything else is a finding
            outcome = {**outcome, "verdict": "FINDING_NOT_EXACT"}
        (output_root / "OUTCOME.json").write_text(json.dumps(outcome, indent=1, default=str) + "\n")
        if kind == "train" and outcome.get("status") == "completed":
            declared = {s: self.declaration["data"][s]["sha256"] for s in ("train", "validation")}
            if outcome.get("data_sha256") != declared:
                outcome = {**outcome, "status": "failed",
                           "error": f"receipt data digests {outcome.get('data_sha256')} != declared {declared}"}
        self._record(row, kind, attempt, outcome)
        return outcome

    def _record(self, row, kind, attempt, outcome):
        cid = row["cid"]
        fields = {k: outcome.get(k) for k in ("exit_code", "error", "cgroup_peak_bytes", "elapsed_seconds",
                                              "observed_updates", "selected_epoch", "per_update_seconds",
                                              "stop_reason", "receipt_path", "objective", "model_sha256",
                                              "weights_sha256", "verdict")}
        status = outcome["status"]
        self.db.execute("BEGIN IMMEDIATE")
        self.db.execute("UPDATE attempts SET finished=?, status=?, " + ", ".join(f"{k}=?" for k in fields) +
                        " WHERE cid=? AND attempt=? AND kind=?",
                        (now(), status, *fields.values(), cid, attempt, kind))
        if kind == "train":
            new = "completed" if status == "completed" else "failed"
            self.db.execute("UPDATE candidates SET status=?, objective=?, updated=? WHERE cid=?",
                            (new, outcome.get("objective"), now(), cid))
        else:
            new = {"VERIFIED": "verified", "REFUTED": "refuted",
                   "FINDING_NOT_EXACT": "finding"}.get(outcome.get("verdict"), "completed")
            if status != "completed":
                new = "completed" if attempt < 2 else "failed"  # one verification retry, then stop
            self.db.execute("UPDATE candidates SET status=?, updated=? WHERE cid=?", (new, now(), cid))
        self.db.execute("COMMIT")

    # ------------------------------------------------------------ incumbent --
    def standings(self):
        seeds = self.declaration["paired_seeds"]
        higher = self.declaration["base"]["objective"]["higher_is_better"]
        groups = {}
        for row in self.db.execute("SELECT * FROM candidates ORDER BY position"):
            groups.setdefault(row["config_id"], []).append(dict(row))
        table = []
        for config_id, rows in groups.items():
            verified = {r["seed"]: r for r in rows if r["status"] == "verified"}
            eligible = set(verified) == set(seeds) and all(
                r["objective"] is not None and math.isfinite(r["objective"]) for r in verified.values())
            values = [verified[s]["objective"] for s in seeds if s in verified]
            table.append({"config_id": config_id, "label": rows[0]["label"], "eligible": eligible,
                          "mean_objective": sum(values) / len(values) if values else None,
                          "per_seed": {str(s): verified[s]["objective"] for s in seeds if s in verified},
                          "cids": [verified[s]["cid"] for s in seeds if s in verified],
                          "statuses": {str(r["seed"]): r["status"] for r in rows}})
        ranked = sorted((t for t in table if t["eligible"]), key=lambda t: t["mean_objective"], reverse=higher)
        return table, ranked

    def incumbent(self):
        row = self.db.execute("SELECT * FROM incumbent_changes ORDER BY seq DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def update_incumbent(self):
        _, ranked = self.standings()
        if not ranked:
            return None
        best, current = ranked[0], self.incumbent()
        higher = self.declaration["base"]["objective"]["higher_is_better"]
        if current and current["config_id"] == best["config_id"]:
            return current
        if current:
            better = (best["mean_objective"] > current["mean_objective"] if higher
                      else best["mean_objective"] < current["mean_objective"])
            if not better:
                return current
        self.db.execute("INSERT INTO incumbent_changes(time, config_id, mean_objective, seeds, cids,"
                        " previous_config_id, previous_mean, reason) VALUES(?,?,?,?,?,?,?,?)",
                        (now(), best["config_id"], best["mean_objective"],
                         json.dumps(self.declaration["paired_seeds"]), json.dumps(best["cids"]),
                         current["config_id"] if current else None, current["mean_objective"] if current else None,
                         "first paired-seed verified configuration" if not current else
                         "lower mean validation objective across all paired seeds" if not higher else
                         "higher mean validation objective across all paired seeds"))
        return self.incumbent()

    def status(self):
        counts = dict(self.db.execute("SELECT status, COUNT(*) FROM candidates GROUP BY status").fetchall())
        attempts = [dict(r) for r in self.db.execute("SELECT * FROM attempts ORDER BY started")]
        table, ranked = self.standings()
        history = [dict(r) for r in self.db.execute("SELECT * FROM incumbent_changes ORDER BY seq")]
        return {"campaign": self.declaration["campaign_id"], "counts": counts,
                "total": sum(counts.values()), "attempts": attempts, "standings": table,
                "incumbent": self.incumbent(), "incumbent_history": history,
                "objective": self.declaration["base"]["objective"]}


# ------------------------------------------------------------------ proposals --

def _sample(spec, rng):
    if "choices" in spec:
        return rng.choice(spec["choices"])
    if spec["type"] == "int":
        return rng.randint(spec["low"], spec["high"])
    if spec.get("log"):
        return float(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"]))))
    return float(rng.uniform(spec["low"], spec["high"]))


def propose(space, rng, fixed=None, max_tries=2000):
    """One valid flat configuration (without seed) drawn uniformly within bounds."""
    fixed = fixed or {}
    for _ in range(max_tries):
        flat = {n: _sample(space["bounds"][n], rng) for n in ss.parameter_names()}
        flat.update(fixed)
        flat["train.seed"] = space["bounds"]["train.seed"]["choices"][0]
        flat = {n: flat[n] for n in ss.active_parameters(flat)}
        try:
            ss.validate_flat(flat, space)
        except ss.SearchSpaceError:
            continue
        flat.pop("train.seed")
        return flat
    raise ss.SearchSpaceError("no valid configuration found within max_tries")


def paired_loss_arms(flat, huber_delta):
    """The same architecture/training draw as a Huber arm and an MAE arm (equal budget)."""
    huber = {**{k: v for k, v in flat.items() if k != "train.huber_delta"}, "train.loss": "huber",
             "train.huber_delta": huber_delta}
    mae = {k: v for k, v in flat.items() if k != "train.huber_delta"}
    mae["train.loss"] = "mae"
    return huber, mae


# ------------------------------------------------------------------ executors --

def _last_heartbeat_peak(root):
    peaks = []
    for path in Path(root).rglob("heartbeat.jsonl"):
        for line in path.read_text().splitlines():
            try:
                peak = json.loads(line)["resources"]["cgroup"].get("peak_bytes")
            except (ValueError, KeyError, TypeError):
                continue
            if isinstance(peak, int):
                peaks.append(peak)
    return max(peaks) if peaks else None


class DoinBridgeExecutor:
    """Train through doin_node.predictor_bridge, verify through its checkpoint scorer path.

    ``host`` names a role in declaration["hosts"]; its settings (cuda_visible_devices,
    resources, extra_env) override the executor defaults. Runs on the host itself.
    """

    def __init__(self, declaration, host="local"):
        self.host = host
        self.e = {**declaration["executor"], **declaration.get("hosts", {}).get(host, {}).get("executor", {})}
        self.resources = {**declaration["resources"], **declaration.get("hosts", {}).get(host, {}).get("resources", {})}

    def _launch(self, argv, output_root, kind, declaration):
        res = self.resources[kind]
        env_prefix = ["env", f"CUDA_VISIBLE_DEVICES={self.e.get('cuda_visible_devices', '')}", f"M04_HOST_ROLE={self.host}",
                      "PYTHONUNBUFFERED=1", f"PYTHONPATH={self.e['doin_pythonpath']}",
                      *[f"{k}={v}" for k, v in sorted(self.e.get("extra_env", {}).items())]]
        command = [self.e["crispdm_run"], "-m", res["cap"], "-t", res["wall"], "-q", "-W",
                   str(res.get("queue_seconds", 3600)), "-n", f"m04-{kind}-{output_root.parent.name[:8]}",
                   "--", *env_prefix, *argv]
        (output_root / "command.json").write_text(json.dumps(command, indent=1) + "\n")
        started = time.monotonic()
        with open(output_root / "launcher.log", "w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        return code, time.monotonic() - started

    def _bridge_config(self, nested, output_root, declaration, timeout, revision=None):
        """Bridge settings; ``revision`` selects the pinned checkout that produced a receipt.

        Verification always runs under the SAME predictor revision as the training attempt
        (the bridge refuses otherwise); earlier pins stay reachable through
        executor.checkouts_by_revision.
        """
        revision = revision or self.e["predictor_revision"]
        checkout = (self.e["predictor_checkout"] if revision == self.e["predictor_revision"]
                    else self.e.get("checkouts_by_revision", {})[revision])
        return {"predictor_checkout": checkout, "predictor_python": self.e["predictor_python"],
                "predictor_revision": revision,
                "train_path": declaration["data"]["train"]["path"],
                "validation_path": declaration["data"]["validation"]["path"],
                "output_dir": str(output_root / "bridge"), "timeout_seconds": timeout,
                "cpu_threads": self.e.get("cpu_threads", 1),
                "cuda_visible_devices": self.e.get("cuda_visible_devices", ""),
                "candidate_config": nested}

    def train(self, nested, output_root, declaration):
        timeout = self.resources["train"]["timeout_seconds"]
        config = self._bridge_config(nested, output_root, declaration, timeout)
        (output_root / "bridge.json").write_text(json.dumps(config, indent=1) + "\n")
        argv = [self.e["doin_python"], "-u", "-m", "doin_node.predictor_bridge", "--config",
                str(output_root / "bridge.json")]
        code, elapsed = self._launch(argv, output_root, "train", declaration)
        return summarize_train(output_root, code, elapsed)

    def verify(self, receipt_path, output_root, declaration):
        timeout = self.resources["verify"]["timeout_seconds"]
        produced_by = json.loads(Path(receipt_path).read_text())["bridge"]["predictor_revision"]
        config = self._bridge_config({"objective": declaration["base"]["objective"]}, output_root,
                                     declaration, timeout, revision=produced_by)
        (output_root / "bridge.json").write_text(json.dumps(config, indent=1) + "\n")
        argv = [self.e["doin_python"], "-u", "-m", "doin_node.predictor_bridge", "--config",
                str(output_root / "bridge.json"), "--verify", receipt_path]
        code, elapsed = self._launch(argv, output_root, "verify", declaration)
        return summarize_verify(output_root, code, elapsed)


class RemoteExecutor:
    """Runs each attempt on a worker over ssh; the queue stays on the orchestrating host.

    The ssh alias is read from the environment variable M04_SSH_<role> at run time and is
    never written to any file; files name only the role. The remote side executes
    ``modular_doin_campaign.py attempt`` (a light launcher) which admits the heavy child
    through crispdm-run on that host and prints the outcome as its last stdout line.
    """

    def __init__(self, declaration, host, root):
        self.host, self.root = host, Path(root)
        self.alias = os.environ[f"M04_SSH_{host}"]
        spec = {**declaration["executor"], **declaration.get("hosts", {}).get(host, {}).get("executor", {})}
        self.python, self.checkout = spec["predictor_python"], spec["predictor_checkout"]

    def _remote(self, kind, output_root, payload):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", self.alias, self.python, "-u",
                   f"{self.checkout}/tools/modular_doin_campaign.py", "attempt", "--root", str(self.root),
                   "--host-role", self.host, "--kind", kind, "--output-root", str(output_root)]
        (output_root / "remote_command.json").write_text(json.dumps(
            ["ssh", f"<{self.host}>", *command[6:]], indent=1) + "\n")
        started = time.monotonic()
        done = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True)
        (output_root / "remote.log").write_text(done.stdout[-20000:] + "\n--- stderr ---\n" + done.stderr[-20000:])
        lines = [l for l in done.stdout.splitlines() if l.startswith("{")]
        if done.returncode != 0 or not lines:
            return {"status": "failed", "exit_code": done.returncode, "elapsed_seconds": time.monotonic() - started,
                    "error": (done.stderr or done.stdout)[-2000:]}
        return json.loads(lines[-1])

    def train(self, nested, output_root, declaration):
        return self._remote("train", output_root, nested)

    def verify(self, receipt_path, output_root, declaration):
        return self._remote("verify", output_root, {"receipt_path": receipt_path})


def summarize_train(output_root, code, elapsed):
    accepted = list(Path(output_root).rglob("accepted.json"))
    outcome = {"exit_code": code, "elapsed_seconds": elapsed,
               "cgroup_peak_bytes": _last_heartbeat_peak(output_root)}
    if code != 0 or len(accepted) != 1:
        logs = list(Path(output_root).rglob("worker.log")) + [Path(output_root) / "launcher.log"]
        tail = ""
        for log in logs:
            if log.exists():
                tail += log.read_text()[-1500:]
        return {**outcome, "status": "failed", "error": tail[-2000:] or f"exit {code}"}
    receipt = json.loads(accepted[0].read_text())
    training = receipt["training"]
    return {**outcome, "status": "completed", "receipt_path": str(accepted[0]),
            "objective": receipt["objective"]["value"], "observed_updates": training["observed_updates"],
            "selected_epoch": training["selected_epoch"], "stop_reason": training["stop_reason"],
            "per_update_seconds": training["elapsed_seconds"] / training["observed_updates"],
            "model_sha256": receipt["digests"]["model_sha256"], "weights_sha256": receipt["digests"]["weights_sha256"],
            "data_sha256": {"train": receipt["digests"]["train_sha256"],
                            "validation": receipt["digests"]["validation_sha256"]},
            "environment": receipt.get("environment"), "candidate": receipt.get("candidate")}


def summarize_verify(output_root, code, elapsed):
    found = list(Path(output_root).rglob("verification.json"))
    outcome = {"exit_code": code, "elapsed_seconds": elapsed}
    if len(found) != 1:
        log = Path(output_root) / "launcher.log"
        return {**outcome, "status": "failed", "error": (log.read_text()[-2000:] if log.exists() else f"exit {code}")}
    result = json.loads(found[0].read_text())
    peak = result.get("resources", {}).get("cgroup", {}).get("peak_bytes")
    outcome["cgroup_peak_bytes"] = peak if isinstance(peak, int) else None
    return {**outcome, "status": "completed", "verdict": result["verdict"], "receipt_path": str(found[0]),
            "exact_match": result.get("exact_match"), "batch_size": result.get("batch_size"),
            "objective": result["objective"]["rescored_value"], "model_sha256": result["digests"]["model_sha256"],
            "error": "; ".join(result["problems"]) or None}


# ----------------------------------------------------------------------- CLI --

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--root", required=True)
    p.add_argument("--declaration", required=True)
    p = sub.add_parser("enqueue-batch", help="default R0 first, then K paired Huber/MAE draws")
    p.add_argument("--root", required=True)
    p.add_argument("--draws", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("run")
    p.add_argument("--root", required=True)
    p.add_argument("--max", type=int)
    p.add_argument("--stop-file")
    p = sub.add_parser("materialize", help="print the nested candidate for a flat JSON (no seed added)")
    p.add_argument("--declaration", required=True)
    p.add_argument("--flat", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--huber-delta", type=float)
    p = sub.add_parser("attempt", help="(runs ON the worker) one train/verify attempt; payload on stdin")
    p.add_argument("--root", required=True)
    p.add_argument("--host-role", required=True)
    p.add_argument("--kind", choices=("train", "verify"), required=True)
    p.add_argument("--output-root", required=True)
    p = sub.add_parser("run-remote", help="(orchestrator) run the shared queue on one worker role")
    p.add_argument("--root", required=True)
    p.add_argument("--host-role", required=True)
    p.add_argument("--max", type=int)
    p.add_argument("--stop-file")
    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "init":
        Campaign.create(args.root, json.loads(Path(args.declaration).read_text()))
        print(json.dumps({"created": args.root}))
    elif args.command == "enqueue-batch":
        campaign = Campaign(args.root)
        decl = campaign.declaration
        added = []
        default = decl["default_candidate"]
        h, m = paired_loss_arms(default, decl["default_huber_delta"])
        added += campaign.enqueue(h, "default_R0_huber")
        added += campaign.enqueue(m, "default_R0_mae")
        rng = random.Random(args.seed)
        for k in range(args.draws):
            draw = propose(campaign.space, rng, fixed=decl.get("fixed_in_batch"))
            delta = _sample(campaign.space["bounds"]["train.huber_delta"], rng)
            h, m = paired_loss_arms(draw, delta)
            added += campaign.enqueue(h, f"draw{k}_huber")
            added += campaign.enqueue(m, f"draw{k}_mae")
        print(json.dumps({"enqueued": len(added)}))
    elif args.command == "run":
        campaign = Campaign(args.root)
        executor = DoinBridgeExecutor(campaign.declaration)
        print(json.dumps({"trained": campaign.run(executor, args.max, args.stop_file)}))
    elif args.command == "attempt":
        declaration = json.loads((Path(args.root) / "CAMPAIGN.json").read_text())
        executor = DoinBridgeExecutor(declaration, args.host_role)
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=False)
        payload = json.loads(sys.stdin.read())
        if args.kind == "train":
            outcome = executor.train(payload, output_root, declaration)
        else:
            outcome = executor.verify(payload["receipt_path"], output_root, declaration)
        outcome["host"] = args.host_role
        (output_root / "OUTCOME.json").write_text(json.dumps(outcome, indent=1, default=str) + "\n")
        print(json.dumps(outcome, default=str))
    elif args.command == "run-remote":
        campaign = Campaign(args.root)
        executor = RemoteExecutor(campaign.declaration, args.host_role, args.root)
        print(json.dumps({"trained": campaign.run(executor, args.max, args.stop_file), "host": args.host_role}))
    elif args.command == "materialize":
        decl = json.loads(Path(args.declaration).read_text())
        flat = {**json.loads(Path(args.flat).read_text()), "train.seed": args.seed}
        if args.huber_delta is not None:
            flat["train.huber_delta"] = args.huber_delta
        print(ss.canonical(ss.from_flat(flat, decl["base"], decl["search_space"])))
    else:
        print(json.dumps(Campaign(args.root).status(), indent=1, default=str))


if __name__ == "__main__":
    main()
