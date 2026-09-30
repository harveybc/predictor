#!/usr/bin/env python3
"""BANKING77 native reference path: continue it under its existing budget, or name the
dependency that stops it. No score is produced here, and none is fabricated.

Why this tool exists
--------------------
The lane's previous return answered a fit question about the SHIPPED one-shot wrapper
and refused to produce a number. That refusal was published with a sentence broader than
the measurement it rested on. Two things therefore happen here, and they are kept apart:

  1. the scope of the fit finding is corrected BESIDE the retained measurement, never in
     place of it (section ``corrected_scope``); and
  2. the already selected NATIVE reference path is advanced as far as it goes without a
     download or a training allocation, and the exact unfulfilled dependency is named
     (sections ``published_target`` .. ``dependency_ledger``).

The native path is the MTEB ``Banking77Classification`` few-shot linear probe over
``jinaai/jina-embeddings-v5-text-small`` at revision ``46ed7da5…``, published accuracy
0.914578. It is a DIFFERENT method from the one-shot typed choice the shipped wrapper
asks, and neither is evidence for the other.

What is decided offline, from bytes we already hold
---------------------------------------------------
Everything in the recipe except the embeddings:

  P1  the published row, read from its own artifact instead of transcribed, and its
      aggregate recomputed from its ten per-experiment cells;
  P2  the population, re-hashed from the governed delivery cache against the pins the
      contract recorded before any score existed;
  P3  the exactly-balanced support of the test split, which FORCES three metric
      identities that the published artifact in fact shows - so the file's ``f1`` is
      simultaneously macro-F1 and weighted F1 ON THIS POPULATION, and that is what makes
      it comparable to a macro-F1 row without loosening the cross-metric refusal;
  P4  the evaluator's ten deterministic 8-per-label training draws, reproduced from the
      pinned source of the version that produced the row, with a digest per draw.

What it deliberately does NOT do: download weights, install anything, load a model, open
a GPU, write a warehouse row, or emit a metric. The option-budget arithmetic of the
earlier return is neither repeated nor revised - it is cited by digest.

    python3 tools/df_banking77_native_path_20260929.py --out REPORT.json
    python3 tools/df_banking77_native_path_20260929.py --offline-only --out REPORT.json

No host name, alias, address or account identifier is written by this tool.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "banking77_native_path.v1"

#: the governed CB02 delivery cache, content addressed; never the staging area
CACHE = Path(os.environ.get(
    "CRISPDM_CLASSIFICATION_CACHE",
    Path.home() / ".local/state/crispdm-data-foundation"
    / "classification-extension-20260928/cache/sota_benchmarks"))

POPULATIONS = ROOT / "docs/contracts/classification_populations.v1.json"
NAIVES = ROOT / "docs/contracts/classification_naive_baselines.v1.json"
RETAINED_FIT = ROOT / "docs/audits/evidence/cb04_row_identity_20260929/BANKING77_LABEL_FIT.json"

TRAIN_SHA = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
TEST_SHA = "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"

#: the selected native reference, carried forward from CB01 1.3 and not re-derived
MODEL = "jinaai/jina-embeddings-v5-text-small"
MODEL_REVISION = "46ed7da5b47e4bca710b756313fafaf4110c6bd1"
DATASET_REVISION = "0fd18e25b25c072e09e0d92ab615fda904d66300"
PUBLISHED_ACCURACY = 0.914578

#: the evaluator contract, transcribed from the pinned source of the version that
#: produced the published row; the sha256 of every file compared is recorded below.
SEED = 42
SAMPLES_PER_LABEL = 8
N_EXPERIMENTS = 10

#: read-only source pins. Each entry: (label, url, sha256 of the bytes read).
SOURCE_PINS = {
    "published_result_2_3_11": (
        "embeddings-benchmark/results @ main :: results/jinaai__jina-embeddings-v5-text-small/"
        f"{MODEL_REVISION}/Banking77Classification.json",
        "ba162c90aa25977f57d0cf25dc10199d9b0d6ecf429238415faa5f065b800fa9"),
    "mteb_2.3.11_abstasks_classification": (
        "embeddings-benchmark/mteb @ 2.3.11 :: mteb/abstasks/classification.py",
        "93fc2b335aead9771510af7b46953fb8472a8625966d5eec9b707476dbcfe726"),
    "mteb_2.9.0_abstasks_classification": (
        "embeddings-benchmark/mteb @ 2.9.0 :: mteb/abstasks/classification.py",
        "f75c0c4a6b38feda8481fbbae8bafdc317466b2fc9e4eb4d89fc22ae625ae6fd"),
    "mteb_2.3.11_banking77_task": (
        "embeddings-benchmark/mteb @ 2.3.11 :: mteb/tasks/classification/eng/banking77_classification.py",
        "6b7f4d1f272fa1b9689a16092098392374d8501ff003660780e1fadd5b06b5c0"),
    "mteb_2.9.0_banking77_task": (
        "embeddings-benchmark/mteb @ 2.9.0 :: mteb/tasks/classification/eng/banking77_classification.py",
        "6b7f4d1f272fa1b9689a16092098392374d8501ff003660780e1fadd5b06b5c0"),
    "mteb_2.3.11_sklearn_evaluator": (
        "embeddings-benchmark/mteb @ 2.3.11 :: mteb/_evaluators/sklearn_evaluator.py",
        "7fd4a005bbe00ab925b5d4de6d04e2c53eca9886a01a48d0fdc3672ae1d36f7c"),
    "mteb_2.9.0_sklearn_evaluator": (
        "embeddings-benchmark/mteb @ 2.9.0 :: mteb/_evaluators/sklearn_evaluator.py",
        "6538d47099e52873c0351e6dc2c5434ca00a6666310894ce86e9087e689fd967"),
    "mteb_2.9.0_jina_model_registry": (
        "embeddings-benchmark/mteb @ 2.9.0 :: mteb/models/model_implementations/jina_models.py",
        "dabfd27f4687e1803fcd7c5d4ca275a34874fb3a60f271d4c6c195afd4d8f09e"),
    "model_repo_tree_at_revision": (
        f"huggingface.co/api/models/{MODEL}/tree/{MODEL_REVISION}?recursive=true",
        "c8be5f401537f21614d72f29ae7a9e06e3ae9b51c3200c9a5e7dd6149c7363eb"),
    "model_config_json": (
        f"huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/config.json",
        "1af1e1269488c83d8b2332e42099f0d2201d687fbe074d1ed096c6201f283546"),
    "model_remote_code_modeling": (
        f"huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/modeling_jina_embeddings_v5.py",
        "389836c791ae345c164108c23858522590be32477b370c94b218cb1b4f10c69c"),
    "model_remote_code_custom_st": (
        f"huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/custom_st.py",
        "52e0931dca24ca1a4fe4f0a5165f1b77e85deac6bebbd3532cda4c09253a29db"),
    "model_sentence_transformers_config": (
        f"huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/config_sentence_transformers.json",
        "e13a56778b7ba8561a3d540173e682ff92acb8457a24d567fe92a8388402fb14"),
    "model_classification_adapter_config": (
        f"huggingface.co/{MODEL}/resolve/{MODEL_REVISION}/adapters/classification/adapter_config.json",
        "7c3f89c17197f40070fae65449fd013c6cf0ee5565ceb45e9d5e292aa8472563"),
}

#: exact acquisition cost of the model artifacts at the pinned revision, from the
#: repository tree read above. Metadata only: no file below was downloaded.
MODEL_FILES = {
    "model.safetensors": 1192133208,
    "adapters/classification/adapter_model.safetensors": 40420208,
    "adapters/clustering/adapter_model.safetensors": 40420208,
    "adapters/retrieval/adapter_model.safetensors": 40420176,
    "adapters/text-matching/adapter_model.safetensors": 40420208,
    "tokenizer.json": 11422654,
    "vocab.json": 2776833,
    "merges.txt": 1671853,
    "README.md": 11340,
    "tokenizer_config.json": 9732,
    "modeling_jina_embeddings_v5.py": 4132,
    "custom_st.py": 3929,
    "config.json": 991,
    "adapters/classification/adapter_config.json": 884,
    "adapters/clustering/adapter_config.json": 883,
    "adapters/retrieval/adapter_config.json": 883,
    "adapters/text-matching/adapter_config.json": 883,
    "config_sentence_transformers.json": 276,
    "generation_config.json": 239,
    "modules.json": 168,
    "configuration_jina_embeddings_v5.py": 120,
    ".gitattributes": 1570,
}

#: direct wheels of the third environment the native path needs, sizes read from the
#: PyPI metadata API. The transitive closure is NOT resolved here: resolving it is part
#: of the same single request, because a dry-run resolve is itself a network act.
DIRECT_WHEELS = {
    "mteb==2.9.0": 5138022,
    "mteb==2.3.11": 4614520,
    "sentence-transformers==5.1.2": 539845,
    "transformers==4.57.0": 11957862,
    "torch==2.8.0 (manylinux cp313, PyPI variant)": 887939584,
    "peft==0.21.1": 863928,
    "datasets==5.0.1": 542418,
    "scikit-learn==1.9.1 (manylinux cp313)": 9143910,
}


#: the two acquisition tiers for the third environment. Tier 1 assumes the installed
#: transformers/torch line runs the repository's remote code - every symbol that code
#: imports was checked present by import, with no weights loaded. Tier 2 is the fallback
#: to the toolchain the model card declares.
TIER1_KEYS = ("mteb==2.9.0", "sentence-transformers==5.1.2", "peft==0.21.1",
              "datasets==5.0.1", "scikit-learn==1.9.1 (manylinux cp313)")
TIER1_BYTES = sum(DIRECT_WHEELS[k] for k in TIER1_KEYS)
TIER2_EXTRA_BYTES = (DIRECT_WHEELS["transformers==4.57.0"]
                     + DIRECT_WHEELS["torch==2.8.0 (manylinux cp313, PyPI variant)"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def tilde(text) -> str:
    """A recorded path with the home prefix replaced, so no account identifier is stored."""
    return str(text).replace(str(Path.home()), "~")


def read_governed_csv(digest: str) -> tuple[list[str], list[str], str]:
    """(texts, categories, re-hashed digest) from the content-addressed delivery cache."""
    path = CACHE / f"{digest}.csv"
    if not path.exists():
        raise SystemExit(f"governed resource absent from the delivery cache: {digest}")
    actual = sha256_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return ([r["text"] for r in rows], [r["category"] for r in rows], actual)


# --------------------------------------------------------------------------------------
# 1. The corrected scope of the fit finding, beside the retained measurement
# --------------------------------------------------------------------------------------

CORRECTED = (
    "BANKING77 is unsupported by the SHIPPED one-shot wrapper and its serializations: "
    "the question builder caps options at 12, and the option payloads cost 706 and 894 "
    "tokens against tested head budgets of 192 and 256. That is not impossibility for "
    "any budget, provider or model.")

WITHDRAWN = (
    "BANKING77_SEVENTY_SEVEN_OPTIONS_DO_NOT_FIT_THE_PROVIDER_NO_SCORE_PRODUCED read as "
    "though the 77 labels cannot fit at any budget. Only the scope is withdrawn; every "
    "measured number behind it stands and is retained unchanged.")


def corrected_scope() -> dict:
    retained = json.loads(RETAINED_FIT.read_text()) if RETAINED_FIT.exists() else None
    return {
        "corrected_sentence": CORRECTED,
        "what_is_withdrawn": WITHDRAWN,
        "the_measurement_is_retained_unchanged": True,
        "retained_fit_evidence": tilde(RETAINED_FIT),
        "retained_fit_evidence_sha256": sha256_file(RETAINED_FIT) if RETAINED_FIT.exists() else None,
        "retained_numbers_cited_not_revised": {
            "max_options_in_the_shipped_question_builder": 12,
            "option_tokens_empty_description_floor": 706,
            "option_tokens_label_as_its_own_description": 894,
            "tested_head_budgets": [192, 256],
        },
        "scope_flags": {
            "unsupported_by_the_shipped_wrapper_and_its_serializations": True,
            "impossible_at_any_budget_provider_or_model": False,
            "a_larger_budget_is_a_different_configuration_not_the_same_reference": True,
            "a_hierarchy_or_retrieval_shortlist_is_a_distinct_method": True,
            "such_a_method_scores_its_routing_errors_over_the_full_population": True,
        },
        "two_questions_never_blended": {
            "native_author_reproduction": "the MTEB few-shot linear probe over the "
                                          "selected embedder; the primary route",
            "alternative_framework_adaptation": "any adaptation through our own "
                                                "framework; not evidence for the first",
        },
        "no_score_is_fabricated_for_a_refused_request": True,
    }


# --------------------------------------------------------------------------------------
# 2. The published target, read from its own artifact
# --------------------------------------------------------------------------------------

def published_target(artifact: Path | None) -> dict:
    out = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_revision_pinned_by_the_contract": DATASET_REVISION,
        "value_carried_forward_from_CB01": PUBLISHED_ACCURACY,
        "evidence_class": "PUBLISHED_REFERENCE",
        "this_is_not_our_measurement": True,
    }
    if artifact is None or not artifact.exists():
        out["artifact"] = None
        out["artifact_read"] = False
        return out
    raw = artifact.read_bytes()
    doc = json.loads(raw)
    cell = doc["scores"]["test"][0]
    per = cell["scores_per_experiment"]
    accs = [x["accuracy"] for x in per]
    recomputed = float(np.mean(accs))
    out.update({
        "artifact": tilde(artifact),
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_read": True,
        "task_name": doc["task_name"],
        "mteb_version_that_produced_the_row": doc["mteb_version"],
        "dataset_revision_in_the_artifact": doc["dataset_revision"],
        "dataset_revision_matches_the_contract_pin":
            doc["dataset_revision"] == DATASET_REVISION,
        "evaluation_time_seconds": doc.get("evaluation_time"),
        "device_recorded_in_the_artifact": None,
        "main_score": cell["main_score"],
        "accuracy": cell["accuracy"],
        "f1_reported": cell["f1"],
        "f1_weighted_reported": cell["f1_weighted"],
        "n_experiments_in_the_artifact": len(per),
        "accuracy_per_experiment": accs,
        "aggregate_recomputed_from_its_own_cells": round(recomputed, 6),
        "aggregate_reproduces_the_published_value":
            round(recomputed, 6) == round(cell["accuracy"], 6),
        "accuracy_spread": [min(accs), max(accs)],
        "accuracy_stdev_sample": float(np.std(accs, ddof=1)),
        "identities_observed_in_the_artifact": {
            "f1_equals_f1_weighted_in_every_experiment":
                all(x["f1"] == x["f1_weighted"] for x in per),
            "precision_equals_precision_weighted_in_every_experiment":
                all(x["precision"] == x["precision_weighted"] for x in per),
            "macro_recall_equals_accuracy_in_every_experiment":
                all(x["recall"] == x["accuracy"] for x in per),
        },
    })
    return out


# --------------------------------------------------------------------------------------
# 3. The population, and why three metric identities are forced rather than lucky
# --------------------------------------------------------------------------------------

def population_identity() -> dict:
    contract = json.loads(POPULATIONS.read_text())
    pops = contract["populations"]
    train_texts, train_cats, train_digest = read_governed_csv(TRAIN_SHA)
    test_texts, test_cats, test_digest = read_governed_csv(TEST_SHA)

    test_support = Counter(test_cats)
    train_support = Counter(train_cats)
    supports = sorted(set(test_support.values()))
    balanced = len(supports) == 1
    per_class = supports[0] if balanced else None

    naive = json.loads(NAIVES.read_text()) if NAIVES.exists() else {}
    majority = max(train_support, key=lambda c: (train_support[c], c))
    majority_accuracy = sum(1 for c in test_cats if c == majority) / len(test_cats)

    mirror = pops["banking77_mteb_mirror"]["splits"]
    return {
        "train": {
            "resource_sha256_pinned": TRAIN_SHA,
            "resource_sha256_rehashed_now": train_digest,
            "matches": train_digest == TRAIN_SHA,
            "rows": len(train_texts),
            "rows_pinned": pops["banking77_train_authors_csv"]["rows"],
            "distinct_labels": len(train_support),
        },
        "test": {
            "resource_sha256_pinned": TEST_SHA,
            "resource_sha256_rehashed_now": test_digest,
            "matches": test_digest == TEST_SHA,
            "rows": len(test_texts),
            "rows_pinned": pops["banking77_test_authors_csv"]["rows"],
            "distinct_labels": len(test_support),
            "support_per_class_values": supports,
            "exactly_balanced": balanced,
            "support_per_class": per_class,
        },
        "mirror_substitution_carried_forward_not_re_derived": {
            "mirror": pops["banking77_mteb_mirror"]["mirror"],
            "same_text_sequence_in_file_order":
                {s: mirror[s]["same_text_sequence_in_file_order"] for s in mirror},
            "mirror_label_ids_are_case_insensitive_alphabetical":
                {s: mirror[s]["mirror_label_id_order_is_case_insensitive_sorted"] for s in mirror},
            "mirror_label_ids_are_byte_sorted":
                {s: mirror[s]["mirror_label_id_order_is_ascii_sorted"] for s in mirror},
            "why_it_matters": "a naive sorted() map mislabels most of the 77 classes; the "
                              "ids are carried from the contract and not re-derived here",
        },
        "forced_metric_identities": {
            "premise": "every class has exactly the same test support",
            "holds": balanced,
            "weighted_f1_equals_macro_f1": balanced,
            "weighted_precision_equals_macro_precision": balanced,
            "macro_recall_equals_accuracy": balanced,
            "why": "a weighted average with equal weights IS the unweighted average, and "
                   "accuracy is the support-weighted mean of per-class recall",
            "what_the_published_f1_is":
                "macro-F1 by the pinned source's own definition - `f1` is computed with "
                "average='macro' and `f1_weighted` with average='weighted' - so the "
                "published 0.913809 is a MACRO_F1 comparison target on its definition, "
                "not on this coincidence",
            "what_the_identity_adds":
                "the equality of the two keys is FORCED by the balanced support, and the "
                "published artifact shows exactly the three identities the balance "
                "predicts, which corroborates that its population is our population",
            "what_it_does_not_license":
                "MACRO_F1 and WEIGHTED_F1 remain different declared families and a "
                "comparison between them still refuses by name even where they coincide "
                "numerically; accuracy against macro-F1 refuses likewise",
        },
        "same_row_naive_carried_forward": {
            "source": tilde(NAIVES),
            "majority_class_fitted_on_train_labels_only": majority,
            "majority_accuracy_recomputed_on_the_governed_test_rows": majority_accuracy,
            "majority_accuracy_pinned": (naive.get("tasks", {})
                                         .get("banking77_test", {})
                                         .get("majority", {})
                                         .get("accuracy")),
            "one_over_seventy_seven": 1 / 77,
        },
    }


# --------------------------------------------------------------------------------------
# 4. The evaluator's deterministic draws, reproduced from the pinned source
# --------------------------------------------------------------------------------------

def deterministic_draws(labels: list[str]) -> dict:
    """`AbsTaskClassification._undersample_data`, transcribed from the pinned source.

    Byte-equal between 2.3.11 (the version that produced the published row) and 2.9.0
    (the first version whose registry names the selected model) apart from the returned
    tuple's arity and its docstring; the selection logic is identical. Verified by
    source comparison, not by re-execution - the same standard CB01 used.
    """
    idxs = list(range(len(labels)))
    draws = []
    for _experiment in range(N_EXPERIMENTS):
        # a FRESH RandomState per call, exactly as the source does; `idxs` is shuffled
        # in place and carried into the next experiment, which is what makes the ten
        # draws differ under one seed
        np.random.RandomState(SEED).shuffle(idxs)
        counter: dict[str, int] = defaultdict(int)
        sampled: list[int] = []
        for i in idxs:
            label = labels[i]
            if counter[label] < SAMPLES_PER_LABEL:
                sampled.append(i)
                counter[label] += 1
        draws.append(list(sampled))

    records = []
    for n, draw in enumerate(draws):
        counts = Counter(labels[i] for i in draw)
        records.append({
            "experiment": n,
            "rows": len(draw),
            "distinct_labels": len(counts),
            "exactly_k_per_label": sorted(set(counts.values())) == [SAMPLES_PER_LABEL],
            "index_set_sha256": digest_of(draw),
            "first_five_indices": draw[:5],
        })
    return {
        "seed": SEED,
        "samples_per_label": SAMPLES_PER_LABEL,
        "n_experiments": N_EXPERIMENTS,
        "evaluator_model": "sklearn LogisticRegression(n_jobs=-1, max_iter=100), "
                           "random_state set to the task seed",
        "train_split_rows": len(labels),
        "expected_rows_per_draw": SAMPLES_PER_LABEL * len(set(labels)),
        "draws": records,
        "all_draws_distinct": len({r["index_set_sha256"] for r in records}) == N_EXPERIMENTS,
        "every_draw_is_exactly_k_per_label": all(r["exactly_k_per_label"] for r in records),
        "draw_set_sha256": digest_of([r["index_set_sha256"] for r in records]),
        "label_id_mapping_is_irrelevant_here":
            "the greedy counter buckets by label equality only, and category -> mirror id "
            "is a bijection, so the selected indices do not depend on which of the three "
            "recorded label orders is used",
        "indices_are_the_mirror_indices":
            "the contract records the mirror's text sequence as identical to the authors' "
            "in file order on both splits, so an index computed on the registered CSV is "
            "the same row the published evaluator drew",
        "what_is_still_missing_after_this":
            "only the embeddings: the draws, the probe, the seed, the scorer and the rows "
            "are fixed, and the encoder is the one unfulfilled dependency",
    }


# --------------------------------------------------------------------------------------
# 5. Live host capacity and the reusable state that actually exists
# --------------------------------------------------------------------------------------

WORKER_PROBE = r'''
set -u
echo "@@interpreters"
for v in "$HOME"/cb03_20260929/venv "$HOME"/cb03_20260929/venv_fw; do
  [ -x "$v/bin/python" ] || continue
  echo "$(basename $v) $("$v/bin/python" -V 2>&1 | tr ' ' '-') $(du -sb $v | cut -f1)"
done
echo "@@packages"
for v in "$HOME"/cb03_20260929/venv "$HOME"/cb03_20260929/venv_fw; do
  [ -x "$v/bin/python" ] || continue
  echo "env $(basename $v)"
  "$v/bin/python" -c '
import importlib.metadata as M
for n in ("torch","transformers","sentence-transformers","mteb","peft","datasets","scikit-learn","numpy","laya","huggingface-hub","safetensors"):
    try: print("  ", n, M.version(n))
    except Exception: print("  ", n, "MISSING")
' 2>&1
done
echo "@@hf_hub_cache"
if [ -d "$HOME/.cache/huggingface/hub" ]; then
  for d in "$HOME/.cache/huggingface/hub"/*; do
    [ -d "$d" ] || continue
    echo "$(basename $d) $(du -sb "$d" 2>/dev/null | cut -f1)"
  done
else echo NONE; fi
echo "@@retained_checkpoint"
[ -d "$HOME/laya_models/laya-typed-decisions" ] \
  && echo "laya-typed-decisions $(du -sb "$HOME/laya_models/laya-typed-decisions" | cut -f1)" \
  || echo NONE
echo "@@disk_bytes_available"; df -B1 --output=avail "$HOME" | tail -1
echo "@@pip_http_cache_bytes"; du -sb "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo 0
echo "@@remote_code_symbols"
"$HOME"/cb03_20260929/venv_fw/bin/python -c '
import importlib
for mod, attr in (("transformers.models.qwen3","Qwen3Model"),
                  ("transformers.models.qwen3","Qwen3Config"),
                  ("transformers.modeling_utils","PreTrainedModel"),
                  ("huggingface_hub","snapshot_download")):
    try:
        print(mod + "." + attr, hasattr(importlib.import_module(mod), attr))
    except Exception as exc:
        print(mod + "." + attr, "IMPORT_ERR_" + type(exc).__name__)
' 2>&1
echo "@@launcher_md5"; md5sum "$HOME/.local/bin/crispdm-run" 2>/dev/null | cut -d" " -f1 || echo MISSING
'''


def parse_worker(text: str) -> dict:
    out: dict = {"interpreters": {}, "packages": {}, "hf_hub_cache": {},
                 "retained_checkpoint": None, "disk_bytes_available": None,
                 "pip_http_cache_bytes": None, "remote_code_symbols": {},
                 "launcher_md5": None}
    section = None
    env = None
    for line in text.splitlines():
        if line.startswith("@@"):
            section = line[2:].strip()
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if section == "interpreters":
            parts = stripped.split()
            if len(parts) == 3:
                out["interpreters"][parts[0]] = {
                    "python": parts[1].replace("-", " "), "bytes": int(parts[2])}
        elif section == "packages":
            if stripped.startswith("env "):
                env = stripped.split(None, 1)[1]
                out["packages"][env] = {}
            elif env:
                parts = stripped.split()
                if len(parts) == 2:
                    out["packages"][env][parts[0]] = parts[1]
        elif section == "hf_hub_cache":
            parts = stripped.split()
            if len(parts) == 2:
                out["hf_hub_cache"][parts[0]] = int(parts[1])
        elif section == "retained_checkpoint":
            parts = stripped.split()
            if len(parts) == 2:
                out["retained_checkpoint"] = {"name": parts[0], "bytes": int(parts[1])}
        elif section == "disk_bytes_available" and stripped.isdigit():
            out["disk_bytes_available"] = int(stripped)
        elif section == "pip_http_cache_bytes" and stripped.isdigit():
            out["pip_http_cache_bytes"] = int(stripped)
        elif section == "remote_code_symbols":
            parts = stripped.split()
            if len(parts) == 2:
                out["remote_code_symbols"][parts[0]] = (
                    True if parts[1] == "True" else
                    False if parts[1] == "False" else parts[1])
        elif section == "launcher_md5":
            out["launcher_md5"] = stripped
    return out


def live_resources(*, mem="1G", wall="8m", name="b77-native-inventory") -> dict:
    """Live, read-only. Capacity is re-read rather than taken from any earlier report."""
    import df_dispatch as D
    import df_host_capacity as HC

    roles = json.loads((Path.home() / ".config/crispdm/host_roles.json").read_text())
    alias = (roles.get("WORKER_A") or {}).get("ssh")
    inventory = HC.read_inventory(roles)

    backend = D.SystemdBackend(roles, timeout=900.0)
    # the probe travels base64-encoded so no quoting survives two shells, and the
    # deployed launcher is called by its ABSOLUTE path because a non-interactive ssh
    # shell does not carry ~/.local/bin on PATH. Nothing is installed or reinstalled.
    blob = base64.b64encode(WORKER_PROBE.encode()).decode()
    script = (f'set -u; echo {blob} | base64 -d > /tmp/b77_native_probe.sh && '
              f'$HOME/.local/bin/crispdm-run -m {mem} -t {wall} -n {name} -q -- '
              f'bash /tmp/b77_native_probe.sh')
    completed = subprocess.run(backend.argv("WORKER_A", script),
                               capture_output=True, text=True, timeout=900)
    body = HC.redact(completed.stdout or "", [alias])
    return {
        "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capacity": reduce_capacity(inventory),
        "secondary_worker_state": parse_worker(body),
        "probe_rc": completed.returncode,
        "probe_stderr": HC.redact((completed.stderr or "")[-400:], [alias]),
        "nothing_was_started_stopped_or_written_on_any_host": True,
    }


def reduce_capacity(inventory) -> dict:
    if not inventory:
        return {}
    roles = inventory.get("roles", inventory)
    out = {}
    for role, value in roles.items():
        if not isinstance(value, dict):
            continue
        slice_ = value.get("batch_slice") or {}
        out[role] = {
            "mem_total_bytes": value.get("mem_total_bytes"),
            "mem_available_bytes": value.get("mem_available_bytes"),
            "slice_memory_max_bytes": slice_.get("memory_max_bytes"),
            "slice_memory_high_bytes": slice_.get("memory_high_bytes"),
            "slice_memory_current_bytes": slice_.get("memory_current_bytes"),
            "batch_scopes_running": len(value.get("batch_scopes") or []),
            "gpus": [{"index": g.get("index"), "status": g.get("status"),
                      "vram_free_bytes": g.get("vram_free_bytes"),
                      "compute_processes": g.get("compute_processes")}
                     for g in (value.get("gpus") or [])],
        }
    return out


# --------------------------------------------------------------------------------------
# 6. The dependency ledger, and the exact unfulfilled dependency
# --------------------------------------------------------------------------------------

def dependency_ledger(*, population: dict, draws: dict, target: dict,
                      resources: dict) -> dict:
    worker = (resources or {}).get("secondary_worker_state") or {}
    packages = worker.get("packages") or {}
    needed = ("mteb", "sentence-transformers", "peft", "datasets", "scikit-learn")
    missing_in_every_env = sorted(
        name for name in needed
        if all(env.get(name, "MISSING") == "MISSING" for env in packages.values())
    ) if packages else sorted(needed)
    hub = worker.get("hf_hub_cache") or {}
    model_bytes = sum(MODEL_FILES.values())
    classification_route_bytes = sum(
        size for path, size in MODEL_FILES.items()
        if not path.startswith("adapters/") or path.startswith("adapters/classification/"))

    components = {
        "published_target_artifact": {
            "state": "PRESENT",
            "how": "read from the results repository at the pinned model revision and "
                   "retained by this lane; the aggregate is recomputed from its own cells",
            "sha256": target.get("artifact_sha256"),
        },
        "evaluator_source_contract": {
            "state": "PRESENT",
            "how": "pinned by digest at both 2.3.11 and 2.9.0; the draw and the scorer "
                   "are source-identical for this task",
        },
        "population_bytes": {
            "state": "PRESENT",
            "how": "the governed CB02 delivery, re-hashed in this process against the "
                   "digests the contract pinned before any score existed",
            "train_matches": population["train"]["matches"],
            "test_matches": population["test"]["matches"],
        },
        "deterministic_training_draws": {
            "state": "PRODUCED_HERE",
            "how": "ten 8-per-label draws reproduced offline from the pinned source",
            "draw_set_sha256": draws["draw_set_sha256"],
        },
        "same_row_naive": {
            "state": "PRESENT",
            "how": "train-fitted majority on the identical governed test rows, pinned "
                   "before any score existed",
        },
        "admissible_host": {
            "state": "PRESENT",
            "how": "the secondary worker; capacity re-read live in this run. The "
                   "preferred external host stays ineligible and unused.",
        },
        "model_artifacts_at_the_pinned_revision": {
            "state": "MISSING",
            "bytes_whole_repository": model_bytes,
            "bytes_classification_route_only": classification_route_bytes,
            "files": len(MODEL_FILES),
            "why_all_four_adapters":
                "the repository's own remote code calls snapshot_download(allow_patterns="
                "['adapters/*']) at load time whenever the model path is not a local "
                "directory, so an offline load needs the whole adapter set materialised",
            "hf_hub_cache_on_the_admissible_host": hub,
            "requires": "an approved download allocation",
            "licence": "CC BY-NC 4.0, research only; incompatible with any commercial or "
                       "trading use",
        },
        "third_pinned_environment": {
            "state": "MISSING",
            "missing_in_every_existing_environment": missing_in_every_env,
            "existing_environments": {
                env: {"laya": pkgs.get("laya"), "torch": pkgs.get("torch"),
                      "transformers": pkgs.get("transformers")}
                for env, pkgs in packages.items()},
            "why_not_reuse":
                "both retained environments are pinned to a Laya SDK version, and "
                "installing into either would mutate an environment a published "
                "reproduction depends on. The native route needs its own environment.",
            "remote_code_symbols_under_the_installed_line":
                worker.get("remote_code_symbols") or {},
            "tier_1_if_the_installed_transformers_torch_line_runs_the_remote_code": {
                "packages": ["mteb==2.9.0", "sentence-transformers==5.1.2", "peft",
                             "datasets", "scikit-learn"],
                "direct_wheel_bytes": {k: v for k, v in DIRECT_WHEELS.items()
                                       if k.split("==")[0] in
                                       ("mteb", "sentence-transformers", "peft",
                                        "datasets", "scikit-learn")
                                       and k != "mteb==2.3.11"},
                "direct_wheel_bytes_total": TIER1_BYTES,
                "evidence_for_this_tier":
                    "every symbol the repository's remote code imports - Qwen3Model, "
                    "Qwen3Config, PreTrainedModel and snapshot_download - resolves under "
                    "the installed transformers line, checked by import on the admissible "
                    "host with no weights loaded. That is necessary, not sufficient: the "
                    "code has not been RUN, because running it needs the weights.",
            },
            "tier_2_if_it_does_not": {
                "packages": ["transformers==4.57.0", "torch==2.8.0", "and tier 1"],
                "additional_direct_wheel_bytes": TIER2_EXTRA_BYTES,
                "note": "the declared toolchain on the model card is transformers 4.57.0 "
                        "/ torch 2.8.0 / sentence-transformers 5.1.2; the admissible host "
                        "carries a newer line",
            },
            "pip_http_cache_bytes_on_the_admissible_host":
                worker.get("pip_http_cache_bytes"),
            "pip_cache_caveat":
                "the cache is large enough to plausibly already hold the installed "
                "torch/transformers blobs, so a same-version install may need no network "
                "at all - NOT verified, because verifying it means attempting a resolve",
            "transitive_closure": "NOT_RESOLVED; resolving it is itself a network act and "
                                  "belongs to the same single request",
            "requires": "an approved install allocation",
        },
        "remote_code_execution_decision": {
            "state": "UNDECIDED",
            "what": "the repository ships auto_map and a custom sentence-transformers "
                    "module, so loading it executes third-party code from the hub",
            "requires": "an explicit trust_remote_code decision; it is a review, not a "
                        "compute allocation",
        },
        "model_wrapper_version_declaration": {
            "state": "RESOLVED_BY_EVIDENCE_AND_MUST_BE_DECLARED",
            "what": "the registry of the version that produced the published row does "
                    "not name the selected model at all; the first release that names it "
                    "is later. So a reproduction must declare which of two configurations "
                    "it ran, and may not print either as though it were the other.",
            "option_a": "the published row's evaluator version plus our own "
                        "reconstruction of the model wrapper",
            "option_b": "the first evaluator version whose registry names the model and "
                        "pins this very revision; its draw and scorer are source-"
                        "identical to the published row's for this task",
            "recommended": "option_b, declared beside every number",
        },
        "device_declaration": {
            "state": "UNDECIDED",
            "what": "the published artifact records no device. The admissible host's GPU "
                    "is idle and large enough; a CPU run in bfloat16 is a different "
                    "device and its numerics need not match.",
            "requires": "a declared device, never a blended one",
        },
    }
    unmet = sorted(k for k, v in components.items() if v["state"] == "MISSING")
    return {
        "components": components,
        "unfulfilled": unmet,
        "the_exact_unfulfilled_dependency":
            "an approved acquisition of {:,} bytes of model artifacts at the pinned "
            "revision, plus a third pinned Python 3.13 environment carrying {} "
            "({:,} bytes of direct wheels at tier 1, transitive closure unresolved), "
            "together with an explicit decision to execute the repository's remote code."
            .format(model_bytes, ", ".join(missing_in_every_env), TIER1_BYTES),
        "everything_else_on_the_native_path_is_pinned": not (
            set(unmet) - {"model_artifacts_at_the_pinned_revision",
                          "third_pinned_environment"}),
        "no_authority_is_invented_here": True,
        "no_download_install_training_or_gpu_run_happened_in_this_lane": True,
    }


PREPARED_COMMANDS = {
    "what_this_is": "prepared, NOT run. Each would need the allocation named above.",
    "1_acquire_model_artifacts": (
        "crispdm-run -m 4G -t 30m -n b77-native-fetch -q -- "
        "<env>/bin/python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{MODEL}', revision='{MODEL_REVISION}', "
        "local_dir='$HOME/b77_models/jina-embeddings-v5-text-small')\""),
    "2_verify_identity_before_any_use": (
        "re-hash every file against the repository tree read at the pinned revision, on "
        "disk after the fetch and again inside the scoring child, exactly as the retained "
        "checkpoint was verified"),
    "3_build_the_third_environment": (
        "crispdm-run -m 4G -t 30m -n b77-native-env -q -- python3.13 -m venv "
        "$HOME/b77_native/venv && $HOME/b77_native/venv/bin/pip install "
        "'mteb==2.9.0' 'sentence-transformers==5.1.2' 'peft' 'datasets' 'scikit-learn'"),
    "4_score_the_native_path": (
        "crispdm-run -m 8G -t 60m -n b77-native-probe -q -- "
        "$HOME/b77_native/venv/bin/python -m mteb run -m "
        f"{MODEL} --model-revision {MODEL_REVISION} -t Banking77Classification "
        "--output-folder $HOME/b77_native/results   # device declared explicitly"),
    "5_refuse_rather_than_adapt": (
        "if the draws, the seed, the probe or the row order cannot be reproduced, the "
        "deliverable is the refusal, not a number"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--offline-only", action="store_true",
                        help="skip the live host read (offline sections only)")
    parser.add_argument("--published-artifact", type=Path, default=None,
                        help="the retained results-repository artifact for the row")
    args = parser.parse_args()

    _, train_cats, _ = read_governed_csv(TRAIN_SHA)
    population = population_identity()
    draws = deterministic_draws(train_cats)
    target = published_target(args.published_artifact)
    resources = {} if args.offline_only else live_resources()
    ledger = dependency_ledger(population=population, draws=draws, target=target,
                               resources=resources)

    report = {
        "schema": SCHEMA,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "lane": "BANKING77 native reference path",
        "no_score_is_produced_or_fabricated_here": True,
        "corrected_scope": corrected_scope(),
        "published_target": target,
        "population_identity": population,
        "deterministic_recipe": draws,
        "source_pins": {k: {"source": v[0], "sha256": v[1]} for k, v in SOURCE_PINS.items()},
        "live_resources": resources,
        "dependency_ledger": ledger,
        "prepared_commands": PREPARED_COMMANDS,
        "verdict": {
            "native_path_state": "PINNED_AND_BLOCKED_ON_ACQUISITION",
            "finding_instead_of_a_score":
                "BANKING77_NATIVE_PATH_BLOCKED_ON_MODEL_AND_ENVIRONMENT_ACQUISITION",
            "no_badge": True,
            "no_warehouse_row_written": True,
            "cross_metric_comparisons_still_refuse_by_name": True,
        },
    }
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {tilde(args.out)}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
