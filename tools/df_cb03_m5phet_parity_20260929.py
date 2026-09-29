"""CB03: the same 400 rows through OUR framework, against the native reference.

The native reference is invoked independently -- it is
`tools/df_cb03_native_agnews_20260929.py`, run in its own process, in its own
environment, under the author's own harness, and its artifact is read here as
data.  Nothing in this file recomputes it, and nothing here can move it.

What "through M5PHET" means, exactly: the request goes through
`m5phet.runtime.run` with the `laya_news` provider registered the way the
distribution registers it, so the runtime's own capability checks, fitted-state
binding, output-schema contract and classification payload validation all run.
The question is built by the framework's own `ad_hoc_task`, from the author's
instruction string and the author's four options in the author's order.

Three things the framework imposes that the author's recipe does not, all
recorded rather than worked around:

  * its classification entry point is news-shaped.  `provider.infer` serialises
    `canonical({asset, headline, body})` as the state, and every one of those
    fields must be a non-empty string.  The author's state is `{"article": ...}`.
    The two strings are different, so the model is not given identical bytes.
  * `LayaBackend.cfg` hard-codes `max_len=512, head_max_len=192`, while this
    checkpoint's own `rl_agent_config.json` declares 1024 and 256.
  * the backend pins the SDK to one VCS commit (laya 0.3.11) and refuses any
    other install, while the published cell was produced under laya 0.2.1.

Each is attributed separately by the native runner's `--max-len`,
`--head-max-len` and `--state-envelope` options, so a difference is explained
rather than averaged away.

  python tools/df_cb03_m5phet_parity_20260929.py --native native_published400.json --out parity.json
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone

GOVERNED_TEST = "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb"
PINNED_400 = "b4c5f991060bcefcc69fac9339b32086dcd0674ac15e44f41eeeeb7c9e782324"
CHECKPOINT_WEIGHTS_SHA256 = "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"

LAYA_CRITERIA = {"world": "world news and international politics", "sports": "sports",
                 "business": "business and economy", "sci_tech": "science and technology"}
LAYA_INSTRUCTIONS = "What is the topic of `article`?"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def population_digest(rows):
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def governed_rows(cache):
    import pandas as pd
    path = os.path.join(cache, "%s.parquet" % GOVERNED_TEST)
    if sha256_file(path) != GOVERNED_TEST:
        raise SystemExit("REFUSED: AG News test bytes are not the delivered bytes")
    frame = pd.read_parquet(path)
    return [{"text": str(t), "label": int(l)} for t, l in zip(frame["text"], frame["label"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="the independently produced native artifact")
    ap.add_argument("--cache", default=os.path.expanduser("~/cb03_20260929/governed"))
    ap.add_argument("--models", default=os.path.expanduser("~/laya_models"))
    ap.add_argument("--asset", default="AGNEWS")
    ap.add_argument("--headline", default="AG News item")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    native = json.load(open(args.native))
    if native["population_sha256"] != PINNED_400:
        raise SystemExit("REFUSED: the native artifact is not the pinned 400-row population")

    checkpoint = os.path.join(args.models, "laya-typed-decisions")
    if sha256_file(os.path.join(checkpoint, "model.safetensors")) != CHECKPOINT_WEIGHTS_SHA256:
        raise SystemExit("REFUSED: the checkpoint weights are not the pinned bytes")

    from news_signal.backends import seal_checkpoint
    manifest = seal_checkpoint(checkpoint)
    manifest_path = os.path.join(os.path.dirname(args.out), "cb03_checkpoint_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)

    os.environ["NEWS_SIGNAL_BACKEND"] = "laya"
    os.environ["NEWS_SIGNAL_CHECKPOINT"] = checkpoint
    os.environ["NEWS_SIGNAL_MANIFEST"] = manifest_path
    os.environ["NEWS_SIGNAL_DEVICE"] = "cpu"

    from m5phet.runtime import Registry, run
    from news_signal.provider import LayaNewsProvider, request_for, state_ref_for
    from news_signal.question import ad_hoc_task

    task = ad_hoc_task(name="topic", question=LAYA_INSTRUCTIONS, options=dict(LAYA_CRITERIA))
    spec = {"name": "topic", "question": LAYA_INSTRUCTIONS, "options": dict(LAYA_CRITERIA)}
    keys = list(LAYA_CRITERIA)
    # the framework's own builder must not have reordered or reworded the options
    built = task["questions"]["topic"]
    if built["instructions"] != LAYA_INSTRUCTIONS or list(built["criteria"]) != keys \
            or built["criteria"] != LAYA_CRITERIA:
        raise SystemExit("REFUSED: the framework rebuilt the author's question into a different one")

    provider = LayaNewsProvider()
    registry = Registry()
    registry.register(provider)
    state_ref = state_ref_for(manifest["sha256"])

    rows = governed_rows(args.cache)[:400]
    if args.limit:
        rows = rows[:args.limit]
        digest = None
    else:
        digest = population_digest([[i, r["text"], r["label"]] for i, r in enumerate(rows)])
        if digest != PINNED_400:
            raise SystemExit("REFUSED: population digest is not the pinned 400-row identity")

    native_rows = {r["i"]: r for r in native["per_row"]}
    results, started = [], time.time()
    for i, row in enumerate(rows):
        now = datetime.now(timezone.utc)
        published = (now - timedelta(seconds=30)).isoformat()
        event = {"schema": "news_event.v1",
                 "event_id": "agnews:test:%d" % i,
                 "source": "AGNEWS_ZHANG2015_TEST",
                 "asset": args.asset,
                 "language": "en",
                 "published_at": published,
                 "received_at": published,
                 "headline": args.headline,
                 "body": row["text"]}
        request = request_for(event, task_id=task["task_id"], state_ref=state_ref,
                              request_id="cb03-agnews-%04d" % i, as_of=now.isoformat(),
                              max_age_seconds=3600, question_spec=spec)
        answer = run(request, registry)
        outputs = ((answer.get("result") or {}).get("outputs")
                   or (answer.get("outputs") or {}))
        entry = {"i": i, "gold": row["label"], "status": answer.get("status")}
        topic = outputs.get("topic") if isinstance(outputs, dict) else None
        if isinstance(topic, dict) and topic.get("status") == "OK":
            payload = topic["payload"]
            probs = payload["uncalibrated_probabilities"]
            entry.update({"label": payload["label"],
                          "predicted": keys.index(payload["label"]),
                          "probabilities": [float(probs[k]) for k in keys],
                          "refused": False})
        else:
            entry.update({"label": None, "predicted": None, "probabilities": None,
                          "refused": True,
                          "why": (topic or {}).get("why") if isinstance(topic, dict)
                                 else answer.get("why")})
        results.append(entry)
    wall = time.time() - started

    # ---- parity, per row, against the independently produced native artifact ----
    label_agree = prob_rows = 0
    max_abs = 0.0
    disagreements, refusals = [], []
    for entry in results:
        reference = native_rows.get(entry["i"])
        if entry["refused"]:
            refusals.append({"i": entry["i"], "why": entry.get("why"),
                             "native_answered": reference is not None and not reference["dropped"]})
            continue
        if reference is None or reference["dropped"]:
            continue
        if entry["predicted"] == reference["predicted"]:
            label_agree += 1
        else:
            disagreements.append({"i": entry["i"], "gold": entry["gold"],
                                  "framework": entry["predicted"], "native": reference["predicted"],
                                  "framework_probabilities": entry["probabilities"],
                                  "native_probabilities": reference["probabilities"]})
        prob_rows += 1
        max_abs = max(max_abs, max(abs(a - b) for a, b in
                                   zip(entry["probabilities"], reference["probabilities"])))

    answered = [e for e in results if not e["refused"]]
    correct = sum(1 for e in answered if e["predicted"] == e["gold"])
    out = {
        "schema": "cb03_m5phet_parity.v1",
        "produced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "population_sha256": digest,
        "rows": len(rows),
        "native_artifact": os.path.basename(args.native),
        "native_artifact_sha256": sha256_file(args.native),
        "native_accuracy": native["metrics"]["accuracy"],
        "framework": {
            "provider": provider.name,
            "task_id": task["task_id"],
            "questions_as_built": task["questions"],
            "backend_identity": provider.identity_for(state_ref),
            "checkpoint_manifest_sha256": manifest["sha256"],
            "backend_declared_budget": {"max_len": 512, "head_max_len": 192},
            "state_envelope_fields": {"asset": args.asset, "headline": args.headline},
        },
        "framework_answered": len(answered),
        "framework_refused": len(results) - len(answered),
        "framework_accuracy_on_answered": round(correct / max(1, len(answered)), 4),
        "coverage": round(len(answered) / max(1, len(results)), 4),
        "parity": {
            "rows_compared": prob_rows,
            "label_agreement": label_agree,
            "label_agreement_fraction": round(label_agree / max(1, prob_rows), 6),
            "max_abs_probability_difference": max_abs,
            "exact_label_parity": label_agree == prob_rows and prob_rows == len(results),
            "population_coverage_parity": len(answered) == len(
                [r for r in native["per_row"] if not r["dropped"]]),
            "disagreements": disagreements[:50],
            "disagreement_count": len(disagreements),
            "refusals": refusals[:50],
        },
        "wall_seconds": round(wall, 1),
        "per_row": results,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_row"}, indent=1)[:6000])


if __name__ == "__main__":
    main()
