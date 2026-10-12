#!/bin/sh
# Integrated suite run on the integration branch (CPU only, one job at a time).
cd "$(dirname "$0")/../../../../.." || exit 2
OUT=docs/audits/evidence/I6N_SWARM_WEEKLY_20261011/integration/SUITES.txt
: > "$OUT"
run() { name=$1; shift
  echo "== $name" >> "$OUT"
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 120s -n i6n-int-$name -- python -m pytest -q -p no:cacheprovider "$@" 2>&1 | tail -4 >> "$OUT"
  echo "exit=$?" >> "$OUT"; }
run swarm tests/test_i6n_swarm_*.py
run search tests/test_i6n_search_space_v2.py tests/test_i6n_search_plan.py tests/test_i6n_optimizer_v2.py tests/test_i6n_optimizer.py tests/test_i6n_search.py
run controller tests/test_i6n_search_controller.py tests/test_i6n_search_follower.py tests/test_i6n_search_publish.py
run contracts tests/test_i6n_eligibility.py tests/test_i6n_paired_metrics.py tests/test_i6n_paired_metrics_r3.py tests/test_i6n_paired_metrics_r4.py tests/test_i6n_business_metrics.py tests/test_i6n_temporal_ops.py
run weekly tests/test_i6n_weekly_campaign.py
run weekly2 tests/test_i6n_weekly_rollup.py tests/test_i6n_weekly_data.py tests/test_i6n_weekly_history_extension.py tests/test_i6n_external_clock_audit_v2_resume.py tests/test_i6n_external_clock_audit_v2.py tests/test_i6n_external_clock_audit_v2_verify.py tests/test_i6n_external_clock_audit.py tests/test_i6n_clock_successor.py
run archive tests/test_i6n_archive.py
run evaluator tests/test_i6n_trajectory_contract.py tests/test_i6n_historical_evaluator.py tests/test_i6n_historical_cell.py tests/test_i6n_publish.py tests/test_i6n_verify.py
run model tests/test_i6n_trajectory_model.py tests/test_i6n_model.py tests/test_i6n_attention_mask.py tests/test_i6n_training.py
run fit_h12 tests/test_i6n_trajectory_fit_h12.py
run fit_h72 tests/test_i6n_trajectory_fit_h72.py
echo DONE >> "$OUT"
