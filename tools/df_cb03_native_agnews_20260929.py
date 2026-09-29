"""CB03: the NATIVE AG News reproduction, under the author's own recipe.

This is not an approximation through our framework.  The scoring engine, the
option rendering, the temperature lookup and every metric below are the author's
own functions, imported unmodified from their committed benchmark harness
(`research/scripts/bench_local.py`), and the case construction is copied verbatim
from their `research/scripts/bench_apps.py` `jev.ag_news` block.  Both files are
pinned by sha256 and refused if they are not those bytes.

One deliberate substitution, and only one: the author's harness calls
`load_dataset("fancyzhx/ag_news", split="test")`, which is a fresh network fetch
against an unpinned revision.  This reads the GOVERNED delivery instead -- the
same distributor parquet CB02 registered -- re-hashes it in this process, and
then proves the substitution is not a change of population by recomputing the
CB01 population digest and refusing unless it equals the pinned value.  Same
rows, same order, named revision.

Populations are kept apart on purpose:

  published400  the first 400 rows of the official test split in file order,
                which is the population of the author's published cell.  It is
                NOT balanced: the paired train-derived majority naive on these
                rows is 0.180000, not the full split's 0.250000.
  full_test     all 7,600 official test rows.  A SEPARATE experiment.  Its
                denominator may never be attached to the published 400-row score.
  train_smoke   the first rows of the official TRAIN split, for the bounded cost
                smoke only.  It is never a quality claim.

  python tools/df_cb03_native_agnews_20260929.py --population train_smoke --limit 32
"""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))

#: The author's committed harness, at the earliest revision that carries it
#: (NandhaKishorM/laya ee760389dc69e28c66893b717fa87c84c0b6063a, 2026-09-20) --
#: the closest committed form of the recipe to the published run of 2026-09-19.
AUTHOR_HARNESS_SHA256 = "08862fb4dba102db873e3e3ad428dedf92c12832b4e8ef71b35e433cce4eea22"
AUTHOR_APPS_SHA256 = "b25e7fb1228fb7d4d51d121ea08b3c3471b9c9af67d5a40fe415a2f33983e960"

GOVERNED = {
    "test": "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb",
    "train": "fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee",
}
#: CB01, docs/contracts/classification_populations.v1.json
PINNED_POPULATION = {
    "published400": "b4c5f991060bcefcc69fac9339b32086dcd0674ac15e44f41eeeeb7c9e782324",
    "full_test": "f36e986861c885e494224c96e6af814e7a16af3e659208cdf2db54bb74f5dc5d",
}
CHECKPOINT_WEIGHTS_SHA256 = "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e"
CHECKPOINT_REVISION = "1a793eb568e6718f15941d08f85432581df534e3"

#: Verbatim from bench_apps.py.  Changing any of it makes a new variant, not a
#: replication: the instruction string, the four option keys IN THIS ORDER, and
#: their glosses.
LAYA_CRITERIA = {"world": "world news and international politics", "sports": "sports",
                 "business": "business and economy", "sci_tech": "science and technology"}
LAYA_INSTRUCTIONS = "What is the topic of `article`?"
HF_LABEL_ORDER = ["World", "Sports", "Business", "Sci/Tech"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def population_digest(rows):
    """CB01's identity over ordered (position, text, label) triples."""
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def governed_rows(cache, split):
    import pandas as pd
    digest = GOVERNED[split]
    path = os.path.join(cache, "%s.parquet" % digest)
    if not os.path.isfile(path):
        raise SystemExit("REFUSED: no governed delivery on disk for AG News %s" % split)
    observed = sha256_file(path)
    if observed != digest:
        raise SystemExit("REFUSED: AG News %s bytes are not the delivered bytes" % split)
    frame = pd.read_parquet(path)
    return [{"text": str(t), "label": int(l)} for t, l in zip(frame["text"], frame["label"])]


def framework_state(text, asset, headline):
    """The state string our own framework would hand the SDK for the same article.

    `news_signal.provider.infer` serialises `canonical({asset, headline, body})`,
    which is a different string from the author's `{"article": ...}`.  It is
    reproduced here, verbatim, only so the envelope's contribution to any parity
    difference can be measured separately from the wrapper's."""
    return json.dumps({"asset": asset, "body": text, "headline": headline},
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def build_cases(rows, envelope="author", asset=None, headline=None):
    """Verbatim from bench_apps.py, `jev.ag_news`, with `d` bound to governed rows."""
    crit = dict(LAYA_CRITERIA)
    keys = list(crit)
    cases, gold = [], []
    for r in rows:
        if envelope == "framework":
            state = framework_state(r["text"], asset, headline)
        else:
            state = {"article": r["text"]}
        cases.append((state,
                      {"topic": {"type": "choice", "instructions": LAYA_INSTRUCTIONS,
                                 "criteria": dict(crit)}}))
        gold.append(keys.index(keys[int(r["label"])]))
    return cases, gold, keys


def ece_main_revision(conf, corr, bins=15):
    """The later committed ece_score (first bin closed on the left).  The two
    committed revisions of the harness differ in this one function and nothing
    else; both are reported so the difference is visible instead of chosen."""
    import numpy as np
    conf, corr = np.asarray(conf, float), np.asarray(corr, float)
    if not len(conf):
        return float("nan")
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        s = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(conf[s].mean() - corr[s].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True,
                    choices=("train_smoke", "published400", "full_test"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default=os.path.expanduser("~/cb03_20260929/governed"))
    ap.add_argument("--harness", default=os.path.expanduser("~/cb03_20260929"))
    ap.add_argument("--models", default=os.path.expanduser("~/laya_models"))
    ap.add_argument("--checkpoint", default="typed-decisions")
    # Defaults reproduce the published cell: the checkpoint's OWN declared budget
    # and the author's own state envelope.  The overrides exist only to attribute
    # a parity difference to one cause at a time; they are recorded in the output.
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--state-envelope", choices=("author", "framework"), default="author")
    ap.add_argument("--asset", default="AGNEWS")
    ap.add_argument("--headline", default="AG News item")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # The author's harness, unmodified, refused if it is not those bytes.
    harness = os.path.join(args.harness, "bench_local.py")
    apps = os.path.join(args.harness, "bench_apps_author.py")
    for path, expected in ((harness, AUTHOR_HARNESS_SHA256), (apps, AUTHOR_APPS_SHA256)):
        observed = sha256_file(path)
        if observed != expected:
            raise SystemExit("REFUSED: %s is not the author's bytes" % os.path.basename(path))
    sys.path.insert(0, args.harness)
    os.environ["LAYA_MODELS"] = args.models

    import numpy as np
    import torch
    import laya
    import bench_local as B
    from laya.common import render_options, build_sequence
    B.ROOT = args.models
    B.MODELS = {k: os.path.join(args.models, v) for k, v in
                (("english", "laya"), ("multilingual", "laya-multilingual"),
                 ("typed-decisions", "laya-typed-decisions"))}

    weights = os.path.join(B.MODELS[args.checkpoint], "model.safetensors")
    weights_sha = sha256_file(weights)
    if weights_sha != CHECKPOINT_WEIGHTS_SHA256:
        raise SystemExit("REFUSED: the checkpoint weights are not the pinned bytes")

    split = "train" if args.population == "train_smoke" else "test"
    rows = governed_rows(args.cache, split)
    if args.population == "published400":
        rows = rows[:400]
    elif args.population == "train_smoke":
        rows = rows[:(args.limit or 32)]
    elif args.limit:
        rows = rows[:args.limit]

    triples = [[i, r["text"], r["label"]] for i, r in enumerate(rows)]
    digest = population_digest(triples)
    pinned = PINNED_POPULATION.get(args.population)
    if args.limit and args.population == "full_test":
        pinned = None
    if pinned and digest != pinned:
        raise SystemExit("REFUSED: population digest %s is not the pinned %s" % (digest, pinned))

    cases, gold, keys = build_cases(rows, args.state_envelope, args.asset, args.headline)

    ag = B.load(args.checkpoint)
    checkpoint_budget = {"max_len": ag.cfg.get("max_len", 512),
                         "head_max_len": ag.cfg.get("head_max_len", 192)}
    if args.max_len is not None:
        ag.cfg["max_len"] = args.max_len
    if args.head_max_len is not None:
        ag.cfg["head_max_len"] = args.head_max_len
    max_len = ag.cfg.get("max_len", 512)
    hml = ag.cfg.get("head_max_len", 192)

    # Every label option must fit untruncated.  The author's own drop rule is
    # `len(markers) != len(render_options(q))`; this records the check per row
    # rather than trusting the aggregate `dropped` count.
    fit = {"rows": 0, "options_expected": len(keys), "rows_with_all_options": 0,
           "max_sequence_tokens": 0, "max_len": max_len, "head_max_len": hml,
           "rows_truncated_in_options": []}
    for ci, (state, questions) in enumerate(cases):
        q = B.to_internal(questions["topic"])
        ids, mk = build_sequence(ag.tok, state, q, max_len, hml)
        fit["rows"] += 1
        fit["max_sequence_tokens"] = max(fit["max_sequence_tokens"], len(ids))
        if len(mk) == len(render_options(q)):
            fit["rows_with_all_options"] += 1
        else:
            fit["rows_truncated_in_options"].append(ci)

    t0 = time.time()
    logits, index, secs, dropped = B.score_cases(ag, cases, tag=args.population)
    wall = time.time() - t0

    per_row, scored = [], []
    for (ci, qid, qt, k), z in zip(index, logits):
        if z is None:
            per_row.append({"i": ci, "gold": gold[ci], "predicted": None,
                            "probabilities": None, "dropped": True})
            continue
        t = B.temp_for(ag, qt, k)
        p = B.softmax_t(z, t)
        pred = int(np.argmax(p))
        per_row.append({"i": ci, "gold": gold[ci], "predicted": pred,
                        "probabilities": [float(v) for v in p],
                        "logits": [float(v) for v in np.asarray(z, float)],
                        "temperature": float(t), "dropped": False})
        scored.append((gold[ci], p))

    metrics = B.metrics(scored)
    conf = [float(np.max(p)) for _, p in scored]
    corr = [float(int(np.argmax(p)) == g) for g, p in scored]
    metrics["ece_later_harness_revision"] = round(ece_main_revision(conf, corr), 4)
    metrics["seconds"] = round(secs, 1)
    metrics["ms_per_case"] = round(1000 * secs / max(1, len(cases)), 1)
    metrics["dropped"] = dropped

    confusion = [[0] * (len(keys) + 1) for _ in keys]
    for row in per_row:
        if row["dropped"]:
            confusion[row["gold"]][len(keys)] += 1
        else:
            confusion[row["gold"]][row["predicted"]] += 1

    out = {
        "schema": "cb03_native_agnews.v1",
        "produced": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "population": args.population,
        "rows": len(rows),
        "population_sha256": digest,
        "pinned_population_sha256": pinned,
        "governed_resource_sha256": GOVERNED[split],
        "checkpoint": "convaiinnovations/laya-%s" % args.checkpoint,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_weights_sha256": weights_sha,
        "device": str(ag.device),
        "laya_version": laya.__version__,
        "laya_module": laya.__file__,
        "torch_version": torch.__version__,
        "harness_sha256": AUTHOR_HARNESS_SHA256,
        "apps_sha256": AUTHOR_APPS_SHA256,
        "prompt": {"instructions": LAYA_INSTRUCTIONS, "criteria": LAYA_CRITERIA,
                   "option_keys": keys},
        "label_order_huggingface": HF_LABEL_ORDER,
        "state_envelope": args.state_envelope,
        "framework_envelope_fields": ({"asset": args.asset, "headline": args.headline}
                                      if args.state_envelope == "framework" else None),
        "checkpoint_declared_budget": checkpoint_budget,
        "budget_used": {"max_len": max_len, "head_max_len": hml},
        "option_fit": fit,
        "metrics": metrics,
        "confusion_reference_by_predicted_plus_abstained": confusion,
        "wall_seconds": round(wall, 1),
        "per_row": per_row,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_row"}, indent=1))


if __name__ == "__main__":
    main()
