#!/usr/bin/env python3
"""I6N-AUD-01 addendum: reproducible provenance of the 20 RAW features used by I6-E/I6-N.

Read-only. It never reads a TEST row (decision time >= 2025-01-01): parquet files are read
column by column, the timestamp column first, and target columns only as finiteness masks of
rows dated before 2025-01-01; no label value is printed. No model is trained.

Usage (from the repository root of the audit worktree):
  STATE=~/.local/state python3 docs/audits/evidence/I6N_PLAN_AUDIT_20261011/addendum/provenance/provenance_check.py \
      --draft <audited draft worktree> --commit 269c2d13 --write-inputs <dir>/INPUTS.json

Every path printed is either $STATE/... or repository-relative; nothing host specific is printed.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

TEST_START = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)
INPUTS = []


def canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v):
    return hashlib.sha256(canon(v).encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Ctx:
    def __init__(self, state, repo, draft, commit):
        self.state, self.repo, self.draft, self.commit = state, repo, draft, commit

    def s(self, rel):  # state file
        p = self.state / rel
        INPUTS.append(dict(path="$STATE/" + rel, sha256=sha_file(p), role=None))
        return p

    def git(self, rel):  # git-tracked file at the audited commit
        b = subprocess.check_output(["git", "-C", str(self.repo), "show", f"{self.commit}:{rel}"])
        INPUTS.append(dict(path=f"{rel}@{self.commit}", sha256=hashlib.sha256(b).hexdigest(), role=None))
        return b

    def d(self, rel):  # uncommitted draft file in the audited worktree
        p = self.draft / rel
        INPUTS.append(dict(path=f"<audited draft worktree over {self.commit}>/{rel}", sha256=sha_file(p), role=None))
        return p


def role(text):
    INPUTS[-1]["role"] = text


def hdr(t):
    print("\n" + "=" * 8 + " " + t)


def iso(ts):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--commit", default="269c2d13")
    ap.add_argument("--write-inputs", type=Path)
    a = ap.parse_args()
    state = Path(os.environ.get("STATE", "~/.local/state")).expanduser()
    repo = Path(subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"], text=True).strip())
    c = Ctx(state, repo, a.draft.resolve(), a.commit)
    C = "canonical_20261003/"
    print("provenance_check.py  I6N-AUD-01 addendum  run_utc=%s  commit=%s" % (dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), a.commit))

    # ---------------------------------------------------------------- A. what the I6-E-M runner uses
    hdr("A. Authoritative 20-member list as read by the I6-E-M preparer")
    src = c.git("tools/i6m_prepare.py").decode().splitlines()
    role("I6-E-M preparer; lines 29-37 read parent/INPUT.npz (pinned sha) and parent/DESIGN.json['members']")
    for n in range(29, 38):
        print("  tools/i6m_prepare.py:%d: %s" % (n, src[n - 1]))
    pinned = re.search(r'expected = "([0-9a-f]{64})"', "\n".join(src)).group(1)
    parent_input = c.s(C + "i6e/statistical72_v1/INPUT.npz")
    role("parent INPUT.npz of I6-E-M (matches the constant pinned in i6m_prepare.py:30)")
    print("  pinned parent INPUT sha256 = %s ; $STATE/%si6e/statistical72_v1/INPUT.npz sha256 = %s ; match=%s"
          % (pinned, C, INPUTS[-1]["sha256"], pinned == INPUTS[-1]["sha256"]))
    parent = json.loads(c.s(C + "i6e/statistical72_v1/DESIGN.json").read_text())
    role("parent DESIGN.json read by i6m_prepare.py:33-35 (members)")
    seal_ok = parent["design_sha256"] == digest({k: v for k, v in parent.items() if k != "design_sha256"})
    print("  parent design_sha256=%s seal_recomputed_ok=%s test_read=%s train_years=%s" % (parent["design_sha256"], seal_ok, parent["test_read"], parent["train_years"]))
    m_parent = list(parent["members"])
    runners = {}
    for name in ("multihorizon_2024_v1", "multihorizon_2024_v2"):
        dsg = json.loads(c.s(C + f"i6e/{name}/DESIGN.json").read_text())
        role(f"I6-E-M runner design {name} (sealed with sha256)")
        ok = dsg["sha256"] == digest({k: v for k, v in dsg.items() if k != "sha256"})
        runners[name] = dsg
        print("  %s: sha256=%s seal_ok=%s parent_design_sha256==parent=%s parent_input_sha256==pinned=%s members==parent(order)=%s lineage=%s test_read=%s"
              % (name, dsg["sha256"], ok, dsg["parent_design_sha256"] == parent["design_sha256"], dsg["parent_input_sha256"] == pinned,
                 list(dsg["members"]) == m_parent, dsg.get("lineage"), dsg.get("test_read")))
    i6b = json.loads(c.s(C + "i6b/eurusd/train_selected20.manifest.json").read_text())
    role("I6-B EURUSD TRAIN manifest; binds the 20 selected_features to the FS4 TEST_FREEZE digest")
    print("  I6-B manifest schema=%s freeze_sha256=%s selected_features==parent(order)=%s train_end_exclusive=%s validation_read=%s test_read=%s"
          % (i6b["schema"], i6b["freeze_sha256"], list(i6b["selected_features"]) == m_parent, i6b["train_end_exclusive"], i6b["validation_read"], i6b["test_read"]))
    print("  NOTE: the manifest has no field naming how its selected_features were chosen; validation_read=false describes the bytes this build read, not the selection.")
    print("  20 members (parent order): %s" % m_parent)

    # ---------------------------------------------------------------- B. FS4 weekly winners
    hdr("B. FS4 weekly selection records (rule fields verbatim)")
    W = C + "fs4/preliminary/wave2_successor/weekly/"
    tf = json.loads(c.s(W + "TEST_FREEZE.json").read_text())
    role("FS4 TEST_FREEZE: winners per population/target and the frozen selection rule")
    fr_ok = tf["freeze_sha256"] == digest({k: v for k, v in tf.items() if k not in ("freeze_sha256", "frozen_utc")})
    print("  TEST_FREEZE freeze_sha256=%s recomputed_ok=%s frozen_utc=%s firewall_phase=%s test_weeks_sealed=%s"
          % (tf["freeze_sha256"], fr_ok, tf["frozen_utc"], tf["firewall_phase"], tf["test_weeks_sealed"]))
    print("  I6-B manifest freeze_sha256 == TEST_FREEZE.freeze_sha256: %s" % (i6b["freeze_sha256"] == tf["freeze_sha256"]))
    for k in ("evaluation_mode", "update_mode", "tie_rule"):
        print("  TEST_FREEZE.%s = %s" % (k, json.dumps(tf[k])))
    print("  TEST_FREEZE.selector.aggregate_rule = %s" % json.dumps(tf["selector"]["aggregate_rule"]))
    print("  TEST_FREEZE.selector.selection_arm = %s" % json.dumps(tf["selector"]["selection_arm"]))
    wp = json.loads(c.s(W + "WEEKLY_PLAN.json").read_text())
    role("FS4 WEEKLY_PLAN: validation year, weeks, candidate sets per target")
    pl_ok = wp["plan_sha256"] == digest({k: v for k, v in wp.items() if k not in ("plan_sha256", "built_utc")})
    print("  NOTE: WEEKLY_PLAN.json is rewritten in place (built_utc changes), so its FILE sha256 differs between runs; its content identity is plan_sha256.")
    print("  WEEKLY_PLAN plan_sha256=%s recomputed_ok=%s ==TEST_FREEZE.plan_sha256=%s built_utc=%s (built_utc is outside the plan digest)"
          % (wp["plan_sha256"], pl_ok, wp["plan_sha256"] == tf["plan_sha256"], wp["built_utc"]))
    for k in ("validation_year", "weeks_in_validation_year", "rolling_calendar_years", "input_modes", "selection_arm", "scored_rows", "support", "test"):
        print("  WEEKLY_PLAN.%s = %s" % (k, json.dumps(wp[k])))
    vw, tw = wp["weeks"], wp["test_weeks"]
    print("  WEEKLY_PLAN.weeks: n=%d first=%s last=%s" % (len(vw), json.dumps(vw[0]), json.dumps(vw[-1])))
    print("  WEEKLY_PLAN.test_weeks: n=%d first=%s" % (len(tw), json.dumps(tw[0])))
    print("  folds of the selection: one fit per VALIDATION week, fit rows = [fit_start, cutoff - horizon] (rolling 4 calendar years), "
          "scored rows = [week.start, week.end); inner_validation_weeks=%s" % wp["support"]["inner_validation_weeks"])
    eur = {s["set_id"]: s for s in wp["sets"] if s["population_id"] == "EURUSD"}
    win = tf["winners"]["EURUSD"]
    print("  EURUSD winners (target, horizon_hours, winner set_id[:12], methods, n_features, score, beats_naive, candidates on VALIDATION):")
    union, contain = set(), {}
    for t in sorted(win, key=lambda t: (t[:3], eur[win[t]["RAW"]["set_id"]]["horizon_hours"])):
        r = win[t]["RAW"]
        h = eur[r["set_id"]]["horizon_hours"]
        cands = [s for s in eur.values() if s["target_id"] == t]
        cdesc = ["%s:%s:k=%d" % (s["set_id"][:12], "+".join(s["methods"]), s["n_features"]) for s in sorted(cands, key=lambda s: s["set_id"])]
        print("    %-9s h=%-3d %s %-30s k=%-2d score=%+.5f beats_naive=%s weeks_completed=%s candidates=%d %s"
              % (t, h, r["set_id"][:12], "+".join(r["methods"]), r["n_features"], r["score"], r["beats_naive_on_aggregate"], r["weeks_completed"], len(cands), cdesc))
        union |= set(r["members"])
        for m in r["members"]:
            contain.setdefault(m, []).append("%s(h=%d,%s)" % (t, h, "+".join(r["methods"])))
    sel_targets = sorted({(t, eur[win[t]["RAW"]["set_id"]]["horizon_hours"]) for t in win}, key=lambda x: (x[1], x[0]))
    print("  selection targets/horizons: %s" % sel_targets)

    # ---------------------------------------------------------------- C. equality
    hdr("C. Set equality: 20 runner members vs union of the 14 EURUSD RAW winners")
    print("  runner members sorted (%d): %s" % (len(m_parent), sorted(m_parent)))
    print("  winner union sorted   (%d): %s" % (len(union), sorted(union)))
    print("  set_equal=%s  symmetric_difference=%s" % (set(m_parent) == union, sorted(set(m_parent) ^ union)))
    print("  per-feature winner sets containing it:")
    for m in sorted(m_parent):
        print("    %-42s %2d targets: %s" % (m, len(contain.get(m, [])), ", ".join(contain.get(m, []))))
    only_random = sorted(m for m in m_parent if all("RANDOM_K" in x for x in contain.get(m, [])))
    print("  features that enter only through RANDOM_K-method winner sets: %d %s" % (len(only_random), only_random))
    i6n_h = [2, 3, 4, 5, 6, 12, 18, 24, 36, 48, 60, 72]
    sh = {h for _, h in sel_targets}
    print("  I6-N horizons (proposal section 2) without a selection target of that horizon: %s" % [h for h in i6n_h if h not in sh])
    print("  selection-target horizons outside I6-N grid: %s" % sorted(h for h in sh if h not in i6n_h))

    # ---------------------------------------------------------------- D. what was chosen before/without VALIDATION
    hdr("D. Pre-VALIDATION stages (frontier, TRAIN rankings) and timeline")
    fs = json.loads(c.s(C + "fs4/preliminary/wave2_successor/EURUSD/FRONTIER_SEAL.json").read_text())
    role("FS4 EURUSD frontier seal: which candidate sets entered the VALIDATION comparison (TRAIN only)")
    print("  FRONTIER_SEAL seal_sha256=%s ==WEEKLY_PLAN.populations.EURUSD.frontier_seal_sha256=%s denominator=%s validation_read=%s"
          % (fs["seal_sha256"], fs["seal_sha256"] == wp["populations"]["EURUSD"]["frontier_seal_sha256"], fs["denominator"], json.dumps(fs["validation_read"])))
    print("  FRONTIER_SEAL.rule_text = %s" % json.dumps(fs["rule_text"]))
    print("  frontier set_ids == WEEKLY_PLAN EURUSD set_ids: %s" % (sorted(fs["frontier_set_ids"]) == sorted(eur)))
    fsm = json.loads(c.git("docs/audits/evidence/canonical_20261003/fs_closure/fs_close/FINAL_SELECTION_MANIFEST.json"))
    role("FS-CLOSE TRAIN manifest (inner folds of the TRAIN rankings that fed phase 3)")
    print("  FINAL_SELECTION_MANIFEST dataset_id=%s folds=%s generated_utc=%s closure_rule.declared_before_reading_validation=%s"
          % (fsm["dataset_id"], fsm["folds"], fsm["generated_utc"], fsm["selection"]["closure_rule"]["declared_before_reading_validation"]))
    s2 = json.loads(c.s(W + "STAGE2_LIST.json").read_text())
    role("FS4 stage-2 list (computed from stage-1 VALIDATION RAW aggregates)")
    sc = json.loads(c.s(W + "WEEKLY_SELECTION_COMPLETE.json").read_text())
    role("FS4 weekly selection closure")
    fw = json.loads(c.s(W + "FIREWALL.json").read_text())
    role("FS4 TEST firewall")
    db = c.s(C + "fs4_weekly/wave2/weekly_queue.sqlite")
    role("FS4 weekly task store (read-only: split, week_start, validation_first_read_utc)")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    q = con.execute("SELECT group_concat(DISTINCT split), min(week_start), max(week_start), count(*), "
                    "min(json_extract(result,'$.validation_first_read_utc')), max(json_extract(result,'$.validation_first_read_utc')) FROM tasks").fetchone()
    print("  task store: splits=%s week_start range=[%s, %s] tasks=%d validation_first_read_utc range=[%s, %s]" % q)
    print("  timeline: first VALIDATION read %s -> stage2 computed %s -> selection closed %s -> TEST_FREEZE %s"
          % (q[4], s2["computed_utc"], sc["closed_utc"], tf["frozen_utc"]))
    print("  closure: state=%s validation_year=%s test_opened=%s final_selection=%s ; FIREWALL phase=%s test_observations=%d selection_receipts=%d"
          % (sc["state"], sc["validation_year"], sc["test_opened"], sc["final_selection"], fw["phase"], len(fw["test_observations"]), len(fw["selection_receipts"])))

    # ---------------------------------------------------------------- E. TEST and boundary
    hdr("E. TEST 2025 touch check (timestamps and finiteness masks only)")
    import numpy as np
    for name in ("statistical72_v1", "multihorizon_2024_v1", "multihorizon_2024_v2"):
        for f in ("INPUT.npz", "TRAIN.npz"):
            p = c.s(C + f"i6e/{name}/{f}")
            role(f"I6-E {name} {f} (only the ts array is read)")
            with np.load(p, allow_pickle=False) as z:
                ts = z["ts"]
            print("  $STATE/%si6e/%s/%s rows=%d ts=[%s, %s] rows_ge_2025=%d" % (C, name, f, len(ts), iso(ts.min()), iso(ts.max()), int((ts >= TEST_START.timestamp()).sum())))
    import pyarrow.parquet as pq
    vt = C + "i6a/validation_2024/batch_001/targets_train.parquet"
    p = c.s(vt)
    role("VALIDATION 2024 targets (sha listed in I6-E source_digests and in lane A validation digests); only t_decision_utc and isfinite masks")
    print("  validation targets sha256 in statistical72 source_digests: %s" % (INPUTS[-1]["sha256"] == parent["source_digests"]["validation:targets_train.parquet"]))
    tcol = pq.read_table(p, columns=["t_decision_utc"]).column(0).to_pylist()
    tsec = np.array([x.timestamp() for x in tcol])
    pre = tsec < TEST_START.timestamp()
    print("  rows=%d t_decision=[%s, %s] rows_ge_2025=%d" % (len(tsec), iso(tsec.min()), iso(tsec.max()), int((~pre).sum())))
    last = tsec[pre].max()
    for col, h in (("Y_s_6h", 6), ("Y_l_24h", 24), ("Y_l_72h", 72), ("Y_l_144h", 144)):
        fin = np.isfinite(np.array(pq.read_table(p, columns=[col]).column(0).to_numpy(zero_copy_only=False), dtype=float))
        fin = fin & pre
        lf = tsec[fin].max()
        print("  %-9s finite=%d censored(non-finite)=%d last finite decision=%s -> its label time t+h=%s (last row of file %s) ; label_time_in_2025=%s"
              % (col, int(fin.sum()), int((~fin & pre).sum()), iso(lf), iso(lf + 3600 * h), iso(last), lf + 3600 * h >= TEST_START.timestamp()))
    v2 = runners["multihorizon_2024_v2"]
    print("  I6-E-M (multihorizon_2024_v2) segments=%d last=%s" % (len(v2["weeks"]), json.dumps(v2["weeks"][-1])))
    print("  FS4 sealed TEST week 0 = %s ; FIREWALL expected_test_week_ids[0]=%s" % (json.dumps(tw[0]), fw["expected_test_week_ids"][0]))
    print("  overlap: I6-E-M scores [2024-12-30, 2025-01-01) which FS4 assigns to its sealed TEST week starting 2024-12-30: %s"
          % (v2["weeks"][-1]["score_start"] == tw[0]["start"]))

    # ---------------------------------------------------------------- F. declarations
    hdr("F. Declarations in git-tracked docs and in the audited draft")
    pat = r"later validation|selected on validation|winners were chosen|RAW winners|20 rasgos RAW|20 features RAW"
    out = subprocess.run(["git", "-C", str(repo), "grep", "-n", "-i", "-E", pat, a.commit, "--", "docs/tres_temas_entrevista", "docs/handoffs"],
                         capture_output=True, text=True).stdout.splitlines()
    for line in out:
        print("  " + line[:260])
    prop = "docs/tres_temas_entrevista/program_v3/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md"
    lines = c.d(prop).read_text().splitlines()
    role("audited I6-N proposal draft (uncommitted)")
    for n, l in enumerate(lines, 1):
        if re.search(r"retenid|VALIDATION 2024|TEST 2025|semanas internas de 2023|VALIDATION escoge", l, re.I):
            print("  draft:%s:%d: %s" % (prop, n, l.strip()))
    print("  draft mentions FS4/TEST_FREEZE/freeze digest/selection year for the 20 features: %s"
          % bool(re.search(r"TEST_FREEZE|freeze_sha|FS4|seleccionad|selected", "\n".join(lines))))

    hdr("INPUTS")
    for i in INPUTS:
        print("  %s  %s  | %s" % (i["sha256"], i["path"], i["role"]))
    if a.write_inputs:
        a.write_inputs.write_text(json.dumps(INPUTS, indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
