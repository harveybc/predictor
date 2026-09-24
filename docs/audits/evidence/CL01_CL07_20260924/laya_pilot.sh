#!/usr/bin/env bash
# The bounded EURUSD relevance pilot: direct SDK, then the wrapper through the installed registry, then a restart.
# One admission, one GPU, no duplicate installation and no training.
set -uo pipefail
B=$HOME/.local/state/crispdm-data-foundation/laya-pilot-20260924
NS=$HOME/work/news-signal
UUID=GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8
OUT=${1:-$B/pilot}
mkdir -p "$OUT"
exec > >(tee "$OUT/pilot.log") 2>&1
echo "START $(date -u +%FT%TZ) host=$(uname -n)"

export CUDA_VISIBLE_DEVICES=$UUID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEWS_SIGNAL_CHECKPOINT=$B/checkpoint
export NEWS_SIGNAL_MANIFEST=$B/manifest.json
export NEWS_SIGNAL_DEVICE=cuda:0
export NEWS_SIGNAL_GPU_UUID=$UUID
PY=$B/venv/bin/python
cd "$NS"

echo "=== 1/6 direct SDK reference (imports laya, never news_signal) ==="
$PY tools/direct_sdk_reference.py --checkpoint "$B/checkpoint" --device cuda:0 --gpu-uuid "$UUID" \
    --batch --permute --out "$OUT/direct.json" --input examples/eurusd/events/*.json
echo "rc_direct=$?"
grep -c news_signal tools/direct_sdk_reference.py

echo "=== 2/6 wrapper through the installed entry point, first process ==="
$PY tools/run_pilot.py --corpus examples/eurusd/corpus.json --out "$OUT/first" --store "$OUT/shadow" --label first
echo "rc_first=$?"

echo "=== 3/6 the same path in a NEW process: restart identity and duplicate handling ==="
$PY tools/run_pilot.py --corpus examples/eurusd/corpus.json --out "$OUT/restart" --store "$OUT/shadow" --label restart
echo "rc_restart=$?"

echo "=== 4/6 replay from disk: no model, no inference ==="
$PY -m news_signal replay --store "$OUT/shadow" > "$OUT/replay.json"
echo "rc_replay=$?"; cat "$OUT/replay.json"

echo "=== 5/6 parity: the native reference against the wrapper's receipts ==="
$PY tools/parity_report.py --direct "$OUT/direct.json" --out "$OUT/parity.json" \
    --wrapper $(ls "$OUT"/first/receipts/*.json | grep -v _refusal) --wrapper-b $(ls "$OUT"/restart/receipts/*.json | grep -v _refusal) > /dev/null
echo "rc_parity=$?"
$PY -c "
import json;p=json.load(open('$OUT/parity.json'))
print(json.dumps({k:p[k] for k in ('compared','equal','mismatches','process_reload')},indent=1)[:2500])"

echo "=== 6/6 the documented single-shot command, and the corpus score ==="
$PY -m news_signal classify-registry --input examples/eurusd/events/00_relevant.json \
    --task news_relevance_eurusd.v1 --as-of 2026-09-24T12:00:00Z --store "$OUT/shadow" > "$OUT/single_shot.json"
echo "rc_single=$?"
$PY tools/score_corpus.py --corpus examples/eurusd/corpus.json --direct "$OUT/direct.json" \
    --wrapper "$OUT/first/receipts/*.json" --out "$OUT/score.json" > /dev/null
echo "rc_score=$?"
$PY -c "
import json;s=json.load(open('$OUT/score.json'))
for k in ('wrapper','direct_sdk'):
    if k in s: print(k, json.dumps({x:s[k][x] for x in ('rows_scored','coverage','macro_f1','accuracy')}))
print('identical_labels', s.get('identical_labels'))"
echo "=== 7/7 the SDK's own batched path, reported separately ==="
$PY tools/batch_experiment.py --direct "$OUT/direct.json" --out "$OUT/batch_experiment.json" | head -40
nvidia-smi --query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
echo "FINISH $(date -u +%FT%TZ)"
