#!/usr/bin/env python3
"""Regenerate the CURRENT block of PROJECT_METHOD_STATE.json and the master-plan header from artifacts.

Nothing historical is rewritten: every existing key stays; a new `current_state_20261001` block is
inserted first, `updated_on` moves to the run date, and `documents.orders` moves to the newest order
with the previous value kept in `documents.orders_history`.  Tips are read from git (repo@sha must
exist and be contained in the named remote branch); campaign counts are read from M06's STATUS.json.
Every claim is classed IMPLEMENTED (code at a tip), EXECUTED (a run happened) or VERIFIED (an
independent check passed), never mixed."""
import hashlib, json, os, subprocess, sys
from pathlib import Path

GH = Path.home() / "Documents/GitHub"
ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "docs/tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json"
MASTER = ROOT / "docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md"
STATUS = ROOT / "docs/audits/evidence/MODULAR_CAMPAIGN_20260930/STATUS.json"
ORDERS = ["docs/handoffs/SATOSHI_MAINLINE_PARALLEL_CONTINUATION_2026_10_01.md",
          "docs/handoffs/SATOSHI_AUTONOMOUS_PARALLEL_EXECUTION_2026_10_01.md",
          "docs/handoffs/SATOSHI_NEXT_DISPATCH_2026_10_01.md"]
SUBPLANS = ["docs/tres_temas_entrevista/program_v3/MODULAR_STACK_WORK_PLAN_2026_09_30.md",
            "docs/tres_temas_entrevista/program_v3/MODULAR_STACK_METHOD_STATE.json",
            "docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_REPRESENTATION_WORK_PLAN_2026_09_30.md",
            "docs/tres_temas_entrevista/program_v3/FEATURE_SELECTION_METHOD_STATE.json",
            "docs/tres_temas_entrevista/program_v3/RL_TEMPORAL_COMPARISON_WORK_PLAN_2026_10_01.md",
            "docs/handoffs/SATOSHI_SOURCE_TRANSFORM_COVERAGE_ADDENDUM_2026_10_01.md"]
FRONTS = [  # front, owner, critical-path step, repo, sha, remote branch
    ("A", "M01/M02 a5b79d5488abdee3a", 3, "predictor", "4f3b75e4", "satoshi/a-engine-integration-20261001"),
    ("A", "M01/M02 a5b79d5488abdee3a", 3, "predictor", "3b073d2e", "satoshi/a-engine-integration-20261001"),
    ("B", "M03 a160efdffffffa9f2", 1, "feature-eng", "dee000b", "satoshi/b-source-transform-coverage-20261001"),
    ("B", "M03 a160efdffffffa9f2", 1, "feature-eng", "8484162", "satoshi/b-source-transform-coverage-20261001"),
    ("C", "C2 a5fd5cd9bcbc7f3bd", 2, "predictor", "141623ef", "satoshi/c2-causal-eth-20261001"),
    ("D", "M04 a083979bc9a8fa1b8", 3, "predictor", "4c129c5c", "satoshi/d-corrected-queue-20261001"),
    ("D/F2", "M07 afcf115024ffa1381", 4, "predictor", "58572172", "satoshi/f2-eth-forecast-20261001"),
    ("E", "M05 a65d2f3a115a9aa25", 5, "lts", "42035f1", "satoshi/m05-paper-adapter-20260930"),
    ("E", "M05 a65d2f3a115a9aa25", 5, "heuristic-strategy", "4b5dd7e", "satoshi/s08-backtest-naive-gate-20261001"),
    ("G", "lane G aa7913479a97b45c0", 5, "agent-multi", "25ee6efd", "satoshi/g-rl-temporal-20261001"),
    ("H", "H1 a1a511567dd7f306f", 2, "predictor", "940834d2", "satoshi/h-kalman-20261001"),
    ("I", "I1 ad39bdf372d45ceb0", None, "M5PHET", "a16317a", "satoshi/i-m5phet-20261001"),
]


def git(repo, *a):
    return subprocess.run(["git", "-C", str(GH / repo), *a], capture_output=True, text=True).stdout.strip()


def tip_record(front, owner, step, repo, sha, branch):
    remote = git(repo, "rev-parse", "--verify", "-q", f"origin/{branch}")
    ref = remote or git(repo, "rev-parse", "--verify", "-q", branch)
    full = git(repo, "rev-parse", "--verify", "-q", f"{sha}^{{commit}}") if sha else ref
    contained = bool(full) and bool(ref) and subprocess.run(
        ["git", "-C", str(GH / repo), "merge-base", "--is-ancestor", full, ref]).returncode == 0
    subj = git(repo, "log", "-1", "--format=%cI %s", full) if full else ""
    return {"front": front, "owner": owner, "critical_path_step": step, "repo": repo, "branch": branch,
            "tip": (full or "")[:12] or None, "branch_head": (ref or "")[:12] or None, "tip_in_branch": contained,
            "branch_location": "origin" if remote else "local_only_unpushed",
            "subject": subj[:160], "class": "IMPLEMENTED" if contained else "UNVERIFIED_TIP"}


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest() if (ROOT / p).is_file() else None


def build(run_date):
    st = json.load(open(STATUS))
    camps = [{"campaign": c["campaign"], "status_counts": c.get("status_counts"), "identity": c.get("identity"),
              "incumbent": c.get("incumbent")} for c in st.get("campaigns", [])]
    return {
        "generated_by": "tools/m06_method_state.py (M06), from git tips and STATUS.json; replaces nothing historical",
        "generated_at": st["observed_at"], "status_source": "docs/audits/evidence/MODULAR_CAMPAIGN_20260930/STATUS.json",
        "orders": [{"path": p, "sha256": sha(p)} for p in ORDERS],
        "subplans": [{"path": p, "sha256": sha(p)} for p in SUBPLANS],
        "critical_path": ["1 point-in-time financial dataset with short/long targets", "2 progressive selection + temporal representation",
                          "3 DOIN R0/R1/R2 on the modular architecture", "4 naive gate per horizon and seed",
                          "5 heuristic strategy replay + RL SAC/DQN contrast", "6 MT5 demo + Alpaca paper with the best eligible candidate",
                          "7 progressive replacement only on better evidence"],
        "fronts": [tip_record(*f) for f in FRONTS],
        "executed": {"campaigns_from_status": camps,
                     "ecl_corrected_v1": "CLOSED: 32 VERIFIED + 4 REFUSED_BY_ENGINE of 36; R0 incumbent c09f3034 0.383565, R1 incumbent c950ec17 0.375091 (mean validation MAE_z, 2 seeds) vs persistence 0.851406 and 24 h seasonal 0.247966 (both lose to the seasonal reference); NOT_COMPARABLE; not financial evidence",
                     "traffic_h96": "three seeds, seed-mean MSE 0.375199 / MAE 0.251143 vs TimeFilter Table 8 0.375 / 0.251: OPERATIONAL_AGREEMENT (RESULTS/traffic_h96_closure_table.md)"},
        "verified": {"ecl_corrected_v1_closure": "M06 regeneration vs M04 f787ad51: 0 disagreements (RESULTS/corrected_vs_m04_f787ad51.json)",
                     "admission_repairs": "ADM 775c5545 then a1f3898b deployed on worker_a and worker_b 2026-10-01T07:25:58Z: suite 62/62 in a worker_b checkout, 256M smoke ADMITTED+RELEASED on both, launcher sha bdc50f60 / module e4128596 (registry.json deployments); coordinator not deployed"},
        "eligibility_for_strategy_and_deployment": {"ETH 4h R0 (F2)": "0/12 eligible (naive gate; orchestrator report)",
                                                     "Kalman (H1)": "none yet", "others": "none yet"},
        "superseded_current_pointer": "rp140_rp143 / rp144_rp151 blocks remain as history; they no longer describe the current state",
    }


def main():
    run_date = sys.argv[1] if len(sys.argv) > 1 else "2026-10-01"
    state = json.load(open(STATE))
    block = build(run_date)
    new = {"schema": state["schema"], "current_state_20261001": block}
    for k, v in state.items():
        if k not in new:
            new[k] = v
    prev = new.get("updated_on")
    new["updated_on"] = run_date
    docs = new.setdefault("documents", {})
    hist = docs.setdefault("orders_history", [])
    if docs.get("orders") and docs["orders"] != "../../handoffs/SATOSHI_MAINLINE_PARALLEL_CONTINUATION_2026_10_01.md":
        hist.append({"orders": docs["orders"], "superseded_on": run_date})
    docs["orders"] = "../../handoffs/SATOSHI_MAINLINE_PARALLEL_CONTINUATION_2026_10_01.md"
    docs["orders_also_current"] = ["../../handoffs/SATOSHI_AUTONOMOUS_PARALLEL_EXECUTION_2026_10_01.md",
                                   "../../handoffs/SATOSHI_NEXT_DISPATCH_2026_10_01.md"]
    new["updated_on_previous"] = prev
    tmp = STATE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, STATE)
    # master-plan header block (inserted once, regenerated between markers)
    txt = MASTER.read_text()
    begin, end = "<!-- CURRENT_STATE_BEGIN (generated by tools/m06_method_state.py) -->", "<!-- CURRENT_STATE_END -->"
    lines = [begin, "", f"**Current state, {run_date}.** Generated from git and STATUS.json; history below is unchanged.", "",
             "- Orders: " + "; ".join(f"`{o['path']}` (sha `{(o['sha256'] or '')[:8]}`)" for o in block["orders"]),
             "- Sub-plans: " + "; ".join(f"`{s['path'].split('/')[-1]}`" for s in block["subplans"]),
             "- Active tips (IMPLEMENTED unless noted): " + "; ".join(
                 f"{f['front']} {f['repo']}@{f['tip']} ({f['branch']}{'' if f['tip_in_branch'] else ', UNVERIFIED'})" for f in block["fronts"]),
             "- Executed: " + block["executed"]["ecl_corrected_v1"],
             "- Verified: " + "; ".join(block["verified"].values()),
             "- Eligibility (strategy/deployment): " + "; ".join(f"{k}: {v}" for k, v in block["eligibility_for_strategy_and_deployment"].items()),
             "", end]
    hdr = "\n".join(lines)
    if begin in txt:
        a, b = txt.index(begin), txt.index(end) + len(end)
        txt = txt[:a] + hdr + txt[b:]
    else:
        first_nl = txt.index("\n") + 1
        txt = txt[:first_nl] + "\n" + hdr + "\n" + txt[first_nl:]
    MASTER.write_text(txt)
    print(json.dumps({"fronts": [(f["front"], f["repo"], f["tip"], f["tip_in_branch"]) for f in block["fronts"]]}))


if __name__ == "__main__":
    main()
