#!/usr/bin/env bash
# I6-N audit addendum, reclassification lane: retained quote/digest check.
# Prints the sha256 of every document this lane quotes and the exact line
# ranges quoted in RECLASSIFICATION_DRAFT.{json,md}. Read-only; no data, no TEST.
# Usage: quote_check.sh <OBJ> <AUD>
#   OBJ = root of the audited object worktree (draft over 269c2d13)
#   AUD = docs/audits/evidence/I6N_PLAN_AUDIT_20261011 directory (commit 4d3af09d)
set -euo pipefail
OBJ="$1"; AUD="$2"; P="$OBJ/docs/tres_temas_entrevista/program_v3"
show() { # file first last
  local f="$1" a="$2" b="$3"
  printf '\n=== %s :%s-%s\n' "${f#"$OBJ"/}" "$a" "$b" | sed "s#${AUD}/##"
  sed -n "${a},${b}p" "$f" | nl -ba -v "$a"
}
echo "# sha256 of quoted inputs"
( cd "$OBJ" && sha256sum \
  docs/tres_temas_entrevista/program_v3/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md \
  docs/tres_temas_entrevista/program_v3/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md \
  docs/tres_temas_entrevista/program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md \
  docs/tres_temas_entrevista/program_v3/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md \
  docs/tres_temas_entrevista/program_v3/PROJECT_METHOD_STATE.json \
  docs/tres_temas_entrevista/program_v3/I6E_MULTI_HORIZON_RESOLUTION_WORK_PLAN.md \
  docs/tres_temas_entrevista/program_v3/I6A_ARCHITECTURE_CAMPAIGN_2026_10_08.md \
  docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md \
  tools/modular_candidate_evaluator.py tools/i6m_data.py tools/i6m_prepare.py )
( cd "$AUD" && sha256sum FINDINGS.json DICTAMEN.md CORRECTIONS_REQUIRED.md \
  EVIDENCE_MANIFEST.json PARAMETER_AUDIT.csv INTERACTION_AUDIT.csv \
  probes/a/strategy_counterexample.py probes/a/strategy_counterexample.out \
  probes/a/wmc_calendar.out probes/cde/check.py probes/cde/out.json )
echo; echo "# proposal (owner contract choices and self-imposed sealing)"
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 6 6
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 21 21
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 40 46
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 68 69
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 103 108
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 112 116
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 128 143
show "$P/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md" 163 168
echo; echo "# weekly walk-forward contract (selection year, update mode, checkpoint tail, fixed manifest)"
show "$P/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md" 45 47
show "$P/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md" 59 73
show "$P/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md" 83 87
show "$P/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md" 159 166
echo; echo "# market-week contract"
show "$P/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md" 14 19
show "$P/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md" 76 79
show "$P/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md" 96 104
show "$P/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md" 199 201
show "$P/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md" 209 213
echo; echo "# financial loss/optimizer policy and its registration"
show "$P/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md" 3 12
show "$P/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md" 23 28
show "$P/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md" 50 52
show "$P/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md" 77 103
show "$P/FINANCIAL_LOSS_OPTIMIZER_POLICY_2026_09_21.md" 118 122
grep -n '"financial_loss_policy"\|"id": "FIN-LOSS-OPT"' "$P/PROJECT_METHOD_STATE.json"
show "$P/PROJECT_METHOD_STATE.json" 618 628
echo; echo "# inherited fit recipe and inner definition"
show "$P/I6E_MULTI_HORIZON_RESOLUTION_WORK_PLAN.md" 117 121
show "$P/I6E_MULTI_HORIZON_RESOLUTION_WORK_PLAN.md" 131 136
show "$OBJ/tools/i6m_prepare.py" 63 65
show "$OBJ/tools/i6m_data.py" 48 55
show "$OBJ/tools/modular_candidate_evaluator.py" 393 396
echo; echo "# master window default; I6A admission"
show "$OBJ/docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md" 314 315
show "$P/I6A_ARCHITECTURE_CAMPAIGN_2026_10_08.md" 50 52
echo; echo "# original-audit evidence retention facts"
echo "X-A5 entry in EVIDENCE_MANIFEST.json:"; grep -n -A4 '"id": "X-A5"' "$AUD/EVIDENCE_MANIFEST.json"
echo "X-F1 entry in EVIDENCE_MANIFEST.json:"; grep -n -A4 '"id": "X-F1"' "$AUD/EVIDENCE_MANIFEST.json"
echo "occurrences of '1.6e6' in retained probe outputs: $(cat "$AUD"/probes/*/*.out "$AUD"/probes/cde/out.json | grep -c '1.6e6' || true)"
echo "strategy .out lines: $(wc -l < "$AUD/probes/a/strategy_counterexample.out")"
echo "strategy .out duplicated lines (count>1):"; sort "$AUD/probes/a/strategy_counterexample.out" | uniq -d | sed 's/^/  /'
echo "string 'has no short argument' in retained script: $(grep -c 'has no short argument' "$AUD/probes/a/strategy_counterexample.py" || true)"
echo "string 'has no short argument' in retained output: $(grep -c 'has no short argument' "$AUD/probes/a/strategy_counterexample.out" || true)"
echo "variants with close=True in BOTH scenarios (informative for time order):"; grep -E 'close=True; SL@2h/TP@12h -> close=True' "$AUD/probes/a/strategy_counterexample.out" | sed 's/^/  /'
echo "variants with close=False in BOTH scenarios (no early close triggered, uninformative):"; grep -c 'close=False; SL@2h/TP@12h -> close=False' "$AUD/probes/a/strategy_counterexample.out"
echo "cde probe: weights are random (seeded) in stride probe:"; grep -n 'set_random_seed\|Conv1D(4, k, strides=2' "$AUD/probes/cde/check.py"
echo "cde probe: ClockView origin position is arithmetic, not a call into the model code:"; grep -n 'def pos_t' "$AUD/probes/cde/check.py"
echo "market weeks with A in 2024 (assumed calendar):"; grep -n 'VALIDATION-2024 market weeks' "$AUD/probes/a/wmc_calendar.out"
