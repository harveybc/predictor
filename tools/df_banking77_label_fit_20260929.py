"""BANKING77: do all 77 label options fit untruncated? Answered BEFORE any score exists.

Why this gate runs first
------------------------
BANKING77 is the most informative unrun cell in this family because it is the only
one where two known defects can actually bite. At four options both are harmless; at
seventy-seven they are not:

* the framework hard-codes a **512 / 192** sequence budget in `LayaBackend.cfg`, below
  the checkpoint's own declared **1024 / 256**; and
* the pinned SDK **clamps** the checkpoint's `choice:11+` temperature, the bucket that
  seventy-seven options select.

A truncated question is a different question, and a score computed on a truncated
question is a measurement of something nobody asked. So the fit is decided first, and
if the labels do not fit **that is the finding and the score is not the deliverable**.

What decides it, and what is deliberately not needed
----------------------------------------------------
`news_signal.backends.sequence_budget` makes the option budget independent of the
news text: `option_budget = head_max_len - option_tokens`, and the question does not
fit when the budget falls below 16 or when any single option exceeds
`OPTION_TOKEN_LIMIT`. So the answer needs exactly two things - the 77 label strings
and the checkpoint's own tokenizer - and needs **no dataset, no model weights, no
GPU and no inference**. Nothing is downloaded and no score is produced here.

Three gates are reported separately, because they fail for different reasons:

  G1  the shipped question builder: `MAX_OPTIONS = 12`, so a 77-option choice is
      refused before any encoder is reached, and every option additionally requires a
      non-empty description that BANKING77 does not ship.
  G2  the option budget under the framework's hard-coded 512/192.
  G3  the option budget under the checkpoint's own declared 1024/256, which answers
      whether repairing the hard-code would be enough.

Plus the label-order audit: the mirror's ids are case-insensitive alphabetical, NOT
byte-sorted, so a naive `sorted()` map mislabels most of the classes. That is counted
here rather than asserted.

The tokenizer lives on the secondary worker with the digest-verified checkpoint, so
G2/G3 run there, read-only, inside a fresh capped admission through the deployed
`crispdm-run`. No host name, alias or address is written by this tool.

    python3 tools/df_banking77_label_fit_20260929.py --out REPORT.json
    python3 tools/df_banking77_label_fit_20260929.py --local-only --out REPORT.json
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

POPULATIONS = ROOT / "docs/contracts/classification_populations.v1.json"
#: the shipped provider's checkout, beside this one. `NEWS_SIGNAL_SRC` may be overridden
#: by the environment; no host name, alias or account identifier is written anywhere.
CHECKOUTS = Path(os.environ.get("CRISPDM_CHECKOUTS", Path.home() / "Documents/GitHub"))
NEWS_SIGNAL_SRC = Path(os.environ.get("NEWS_SIGNAL_SRC", CHECKOUTS / "news-signal/src"))
NEWS_SIGNAL_QUESTION = NEWS_SIGNAL_SRC / "news_signal/question.py"
M5PHET_PATHS = (CHECKOUTS / "M5PHET", CHECKOUTS / "M5PHET/src")


def tilde(text) -> str:
    """Any recorded path with the home prefix replaced, so no account identifier is stored."""
    return str(text).replace(str(Path.home()), "~")

#: the instruction head this gate measures with, declared rather than discovered
INSTRUCTIONS = "Which banking intent does this customer message express?"

#: Runs on the worker, read-only. It loads the checkpoint's tokenizer and the SHIPPED
#: `sequence_budget`, and touches neither weights nor data.
REMOTE = r'''
import base64, hashlib, json, os, sys
from pathlib import Path
HOME = Path(os.environ["HOME"])
LANE = HOME / "cb03_20260929"
MODEL = HOME / "laya_models/laya-typed-decisions"
payload = json.loads(base64.b64decode(os.environ["B77_PAYLOAD"]).decode())
labels, instructions = payload["labels"], payload["instructions"]

out = {"schema": "banking77_label_fit_worker.v1"}

# 1. the checkpoint the tokenizer belongs to, verified against the retained manifest
manifest = json.loads((LANE / "cb03_checkpoint_manifest.json").read_text())
files = manifest["files"]
verified, mismatched = [], []
for name in sorted(files):
    path = MODEL / name
    if not path.is_file():
        mismatched.append({"file": name, "why": "absent"})
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size < (1 << 28) \
        else None
    if digest is None:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 20):
                h.update(chunk)
        digest = h.hexdigest()
    (verified if digest == files[name]["sha256"] else mismatched).append(
        {"file": name, "sha256": digest})
out["checkpoint"] = {"files_in_manifest": len(files), "verified": len(verified),
                     "mismatched": mismatched,
                     "tokenizer_files_verified": sorted(
                         v["file"] for v in verified if v["file"].startswith("tokenizer/"))}
if mismatched:
    out["refused"] = "CHECKPOINT_CHANGED"
    print(json.dumps(out)); raise SystemExit(0)

declared = json.loads((MODEL / "rl_agent_config.json").read_text())
out["checkpoint_declared_budget"] = {"max_len": declared["max_len"],
                                     "head_max_len": declared["head_max_len"]}
out["checkpoint_temperature_by_options"] = declared.get("temperature_by_options")

sys.path.insert(0, str(LANE / "src_fw/news-signal/src"))
from news_signal.backends import sequence_budget, OPTION_TOKEN_LIMIT, LayaBackend
from news_signal import backends as B
out["shipped"] = {"backends_file": B.__file__,
                  "framework_hard_coded_budget": dict(LayaBackend.cfg),
                  "option_token_limit": OPTION_TOKEN_LIMIT}

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(MODEL / "tokenizer"))
out["tokenizer"] = {"class": type(tok).__name__,
                    "vocab_size": int(getattr(tok, "vocab_size", 0))}

def renderings(labels):
    return {
        "label_as_its_own_description":
            {label: label.replace("_", " ") for label in labels},
        "empty_description_floor": {label: "" for label in labels},
    }

out["fit"] = {}
for rendering, criteria in renderings(labels).items():
    question = {"banking77_intent": {"type": "choice", "instructions": instructions,
                                     "criteria": criteria}}
    per_budget = {}
    for name, (max_len, head_max_len) in {
            "framework_hard_coded_512_192": (LayaBackend.cfg["max_len"],
                                             LayaBackend.cfg["head_max_len"]),
            "checkpoint_declared_1024_256": (declared["max_len"],
                                             declared["head_max_len"])}.items():
        report = sequence_budget(tok, "", question, max_len=max_len,
                                 head_max_len=head_max_len)["banking77_intent"]
        asked = report.pop("option_tokens_asked")
        kept = report.pop("option_tokens_kept")
        report["options"] = len(asked)
        report["option_tokens_min"] = min(asked.values())
        report["option_tokens_max"] = max(asked.values())
        report["option_tokens_mean"] = sum(asked.values()) / len(asked)
        report["options_over_the_per_option_limit"] = sorted(
            k for k, n in asked.items() if n > OPTION_TOKEN_LIMIT)
        report["room_left_for_the_news_text"] = report["state_room"]
        report["head_fully_kept"] = report["head_tokens_kept"] >= report["head_tokens"]
        per_budget[name] = report
    out["fit"][rendering] = per_budget

# 2. the temperature the 77-option bucket selects, and what the pinned SDK does to it
sys.path.insert(0, str(LANE / "venv_fw/lib/python3.13/site-packages"))
import laya.common as C
shipped = declared["temperature_by_options"]
bucket = C.temp_bucket(0, len(labels))
out["temperature"] = {
    "bucket_for_77_options": bucket,
    "shipped_value": shipped.get(bucket),
    "clamp_bounds": [getattr(C, "TEMP_MIN", None), getattr(C, "TEMP_MAX", None)],
    "value_after_clamp": (C.clamp_temperature(shipped[bucket]) if bucket in shipped
                          and hasattr(C, "clamp_temperature") else None),
    "clamp_fires": (bucket in shipped and hasattr(C, "clamp_temperature")
                    and C.clamp_temperature(shipped[bucket]) != shipped[bucket]),
    "sdk_version": __import__("laya").__version__,
}
print(json.dumps(out))
'''


def _load_question_module():
    """The SHIPPED `news_signal.question`, imported, not reimplemented.

    `news_signal.core` needs M5PHET beside it, so both checkouts join the path.
    """
    for path in (NEWS_SIGNAL_SRC, *M5PHET_PATHS):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from news_signal import question, core
    if Path(question.__file__) != NEWS_SIGNAL_QUESTION:
        raise SystemExit(f"REFUSED: imported {question.__file__}, not the shipped module")
    return question, core


def label_order_audit(labels) -> dict:
    """The mirror's ids are case-insensitive alphabetical. A naive sort mislabels most."""
    naive = sorted(labels)
    case_insensitive = sorted(labels, key=lambda s: (s.casefold(), s))
    wrong = [{"id": index, "mirror_label": labels[index], "naive_sorted_label": naive[index]}
             for index in range(len(labels)) if labels[index] != naive[index]]
    return {
        "labels": len(labels),
        "mirror_order_is_case_insensitive_alphabetical": labels == case_insensitive,
        "mirror_order_is_byte_sorted": labels == naive,
        "ids_a_naive_sorted_map_would_mislabel": len(wrong),
        "fraction_mislabelled": len(wrong) / len(labels),
        "first_five_mislabelled": wrong[:5],
        "carried_forward_not_re_derived": (
            "the order is read from docs/contracts/classification_populations.v1.json, "
            "which pins the mirror revision; this function only checks it and counts what "
            "a naive sorted() map would get wrong"),
    }


def question_builder_gate(labels) -> dict:
    """G1: what the SHIPPED question builder does with 77 options."""
    module, core = _load_question_module()
    attempts = {}
    for name, options in (
            ("seventy_seven_with_descriptions",
             {label: label.replace("_", " ") for label in labels}),
            ("seventy_seven_bare_labels", {label: "" for label in labels}),
            ("twelve_with_descriptions",
             {label: label.replace("_", " ") for label in labels[:12]}),
            ("thirteen_with_descriptions",
             {label: label.replace("_", " ") for label in labels[:13]})):
        try:
            built = module.build_question(name="banking77_intent",
                                          question=INSTRUCTIONS, options=options)
            criteria = built["banking77_intent"]["criteria"]
            attempts[name] = {"built": True, "options": len(criteria),
                              "total_characters": len(INSTRUCTIONS) + sum(
                                  len(k) + len(v) for k, v in criteria.items())}
        except core.Refusal as refusal:
            attempts[name] = {"built": False, "refusal": str(refusal).split(":")[0],
                              "detail": str(refusal)}
    return {"module": tilde(NEWS_SIGNAL_QUESTION),
            "max_options": module.MAX_OPTIONS,
            "max_total_question_characters": module.MAX_TOTAL_QUESTION_CHARACTERS,
            "attempts": attempts,
            "all_77_options_can_be_asked": attempts[
                "seventy_seven_with_descriptions"]["built"]}


def run_on_the_worker(labels, *, mem="3G", wall="15m", name="b77-label-fit-20260929"):
    """G2/G3 where the digest-verified tokenizer is: read-only, capped, no weights loaded."""
    import df_dispatch as D
    import df_host_capacity as HC
    roles = json.loads((Path.home() / ".config/crispdm/host_roles.json").read_text())
    backend = D.SystemdBackend(roles, timeout=1800.0)
    alias = (roles.get("WORKER_A") or {}).get("ssh")
    payload = base64.b64encode(json.dumps(
        {"labels": labels, "instructions": INSTRUCTIONS}).encode()).decode()
    remote = base64.b64encode(REMOTE.encode()).decode()
    script = (f"set -u; export B77_PAYLOAD={payload}; "
              f"echo {remote} | base64 -d > /tmp/b77_label_fit.py && "
              # a non-interactive ssh shell does not have ~/.local/bin on PATH: the
              # deployed launcher is called by its absolute path, never reinstalled
              f"$HOME/.local/bin/crispdm-run -m {mem} -t {wall} -n {name} -q -- "
              f"$HOME/cb03_20260929/venv_fw/bin/python /tmp/b77_label_fit.py")
    argv = backend.argv("WORKER_A", script)
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    stdout = completed.stdout or ""
    line = next((l for l in reversed(stdout.splitlines()) if l.startswith("{")), None)
    record = json.loads(line) if line else None
    if isinstance(record, dict):
        for section, field in (("shipped", "backends_file"),):
            if isinstance(record.get(section), dict) and field in record[section]:
                record[section][field] = tilde(record[section][field])
    return {"rc": completed.returncode, "record": record,
            "admission": HC.redact(
                "\n".join(l for l in stdout.splitlines() if not l.startswith("{"))[-600:],
                [alias]),
            "stderr": HC.redact((completed.stderr or "")[-600:], [alias])}


def verdict(local, worker) -> dict:
    record = (worker or {}).get("record") or {}
    fit = record.get("fit") or {}
    gates = {}
    for rendering, budgets in fit.items():
        for budget, report in budgets.items():
            gates[f"{rendering}/{budget}"] = {
                "fits": report["fits"],
                "option_tokens": report["option_tokens"],
                "head_max_len": report["head_max_len"],
                "option_budget": report["head_max_len"] - report["option_tokens"],
                "options_capped": report["options_capped"],
                "options_truncated": report["options_truncated"],
                "head_fully_kept": report["head_fully_kept"],
                "room_left_for_the_news_text": report["room_left_for_the_news_text"]}
    answered = bool(gates)
    all_fit = answered and all(g["fits"] for g in gates.values())
    return {
        "question": "do all 77 BANKING77 label options fit untruncated?",
        "answered": answered,
        "all_77_labels_fit_untruncated": all_fit if answered else None,
        "the_shipped_question_builder_can_ask_them":
            local["question_builder_gate"]["all_77_options_can_be_asked"],
        "gates": gates,
        "score_is_the_deliverable": bool(
            all_fit and local["question_builder_gate"]["all_77_options_can_be_asked"]),
        "finding_instead_of_a_score": (
            None if (all_fit and local["question_builder_gate"]["all_77_options_can_be_asked"])
            else "THE_SEVENTY_SEVEN_LABELS_DO_NOT_FIT_NO_SCORE_IS_PRODUCED"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--mem", default="3G")
    args = parser.parse_args()

    populations = json.loads(POPULATIONS.read_text())
    banking = next(v for k, v in populations.items()
                   if isinstance(v, dict) and "mirror_label_id_order" in v
                   and len(v["mirror_label_id_order"]) == 77) \
        if any(isinstance(v, dict) and "mirror_label_id_order" in v
               for v in populations.values()) else None
    if banking is None:
        banking = _find_label_order(populations)
    labels = banking["mirror_label_id_order"]

    report = {"schema": "banking77_label_fit.v1",
              "what_this_is": ("the acceptance gate that decides whether a BANKING77 score "
                               "may exist at all: do all 77 label options fit untruncated"),
              "what_this_is_not": ("a score, a benchmark run, a model load, a dataset "
                                   "download, a badge, or any claim about accuracy"),
              "labels_source": {"file": str(POPULATIONS.relative_to(ROOT)),
                                "mirror": banking.get("mirror"),
                                "labels": len(labels)},
              "instructions_head": INSTRUCTIONS,
              "label_order_audit": label_order_audit(labels),
              "question_builder_gate": question_builder_gate(labels)}
    worker = None
    if not args.local_only:
        worker = run_on_the_worker(labels, mem=args.mem)
        report["worker"] = worker
    report["verdict"] = verdict(report, worker)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({"label_order_audit": report["label_order_audit"],
                      "question_builder": {
                          k: v for k, v in report["question_builder_gate"].items()
                          if k != "attempts"},
                      "question_builder_attempts":
                          report["question_builder_gate"]["attempts"],
                      "verdict": report["verdict"], "written": args.out},
                     indent=1, sort_keys=True))
    return 0


def _find_label_order(document):
    """The 77-label order, wherever the populations contract carries it."""
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            order = node.get("mirror_label_id_order")
            if isinstance(order, list) and len(order) == 77:
                return node
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    raise SystemExit("REFUSED: no 77-label mirror order in the populations contract")


if __name__ == "__main__":
    raise SystemExit(main())
