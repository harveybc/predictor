#!/usr/bin/env bash
# CL09 parity for user-authored questions + CL10 diagnosis. One admission, the external 5090, no training.
set -uo pipefail
B=$HOME/.local/state/crispdm-data-foundation/laya-pilot-20260924
NS=$HOME/work/news-signal
UUID=GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8
OUT=${1:-$B/questions}
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
echo "START $(date -u +%FT%TZ) host=$(uname -n)"
export CUDA_VISIBLE_DEVICES=$UUID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEWS_SIGNAL_CHECKPOINT=$B/checkpoint NEWS_SIGNAL_MANIFEST=$B/manifest.json
export NEWS_SIGNAL_DEVICE=cuda:0 NEWS_SIGNAL_GPU_UUID=$UUID
PY=$B/venv/bin/python
cd "$NS"

echo "=== 1/4 the wrapper: two user questions over all 13 items, plus the preset for the error table ==="
$PY tools/run_questions.py --corpus examples/eurusd/corpus.json --questions examples/eurusd/questions.json \
    --out "$OUT/wrapper" --store "$OUT/shadow"
echo "rc_wrapper=$?"

echo "=== 2/4 the independent witness: the SAME user questions, no import of news_signal ==="
$PY - <<'PYEOF'
import json, pathlib
specs = json.loads(pathlib.Path("examples/eurusd/questions.json").read_text())
for spec in specs:
    pathlib.Path(f"/tmp/question_{spec['name']}.json").write_text(json.dumps(spec))
print(" ".join(s["name"] for s in specs))
PYEOF
for Q in rate_cut_risk who_is_affected; do
  $PY tools/direct_sdk_reference.py --checkpoint "$B/checkpoint" --device cuda:0 --gpu-uuid "$UUID" \
      --question-file "/tmp/question_$Q.json" --out "$OUT/direct_$Q.json" --input examples/eurusd/events/*.json
  echo "rc_direct_$Q=$?"
done

echo "=== 3/4 parity per question, exact fields, no tolerance ==="
$PY - "$OUT" <<'PYEOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
summary = json.loads((out / "wrapper/questions_pilot.json").read_text())
report = {"schema": "laya_question_parity.v1", "tolerance": "NONE", "per_question": {}}
for question in summary["cl09"]["questions"]:
    name = question["name"]
    direct = json.loads((out / f"direct_{name}.json").read_text())
    native = {r["input_sha256"]: r["response"]["answers"] for r in direct["single"]}
    rows = [r for r in summary["cl09"]["rows"] if r["question"] == name and r["status"] == "SHADOW_ONLY"]
    wrapped = {r["input_sha256"]: {name: r["sdk_answer"]} for r in rows}
    shared = sorted(set(native) & set(wrapped))
    equal = [i for i in shared if json.dumps(native[i]) == json.dumps(wrapped[i])]
    report["per_question"][name] = {
        "task_id": question["task_id"], "question": question["question"],
        "option_order": question["option_order"],
        "questions_sent_match": json.dumps(direct.get("questions_as_sent"), sort_keys=True) ==
                                json.dumps({name: summary["cl09"]["questions"] and
                                            json.loads((out / "wrapper/receipts" /
                                                        f"{name}__00_relevant.json").read_text())["question"]["questions"][name]},
                                           sort_keys=True),
        "compared": len(shared), "equal": len(equal),
        "mismatches": [{"input_sha256": i, "direct": native[i], "wrapper": wrapped[i]} for i in shared if i not in equal],
        "labels": {i: native[i][name]["choice"] for i in shared},
    }
identities = {q["task_id"] for q in summary["cl09"]["questions"]}
report["distinct_task_identities"] = len(identities)
report["same_news_different_questions"] = True
(out / "question_parity.json").write_text(json.dumps(report, indent=1, sort_keys=True))
print(json.dumps({k: {"compared": v["compared"], "equal": v["equal"], "mismatches": len(v["mismatches"]),
                      "questions_sent_match": v["questions_sent_match"]}
                  for k, v in report["per_question"].items()}, indent=1))
print("distinct task identities:", report["distinct_task_identities"])
PYEOF
echo "rc_parity=$?"

echo "=== 4/4 the error table and the token accounting cross-check ==="
$PY - "$OUT" <<'PYEOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
summary = json.loads((out / "wrapper/questions_pilot.json").read_text())
cl10 = summary["cl10"]
print("scored", cl10["scored_rows"], "errors", len(cl10["errors"]))
for row in cl10["errors"]:
    print(f"  {row['event_id']:<24} gold={row['gold']:<9} pred={row['predicted']:<9} "
          f"p={row['probabilities']} conf={row['confidence_field']}")
budgets = [r["token_budget"]["relevance"] for r in cl10["rows"] if r.get("token_budget")]
if budgets:
    print("token budget: head", budgets[0]["head_tokens"], "kept", budgets[0]["head_tokens_kept"],
          "| options", budgets[0]["option_tokens"], "| state max",
          max(b["state_tokens"] for b in budgets), "room", budgets[0]["state_room"],
          "| all fit:", all(b["fits"] for b in budgets))
PYEOF
nvidia-smi --query-gpu=uuid,temperature.gpu,utilization.gpu,memory.used --format=csv,noheader
echo "FINISH $(date -u +%FT%TZ)"
