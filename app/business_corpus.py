"""The held-out business news corpus: sealed, and opened only on the record.

`tools/build_cb04_business_corpus.py` wrote this corpus after the metric contract
existed. This module is the only way the evaluation path reads it, and it exists
to make two things hard rather than merely discouraged.

**Reading the labels leaves a trace.** Items are handed over freely; labels are
handed over only against an appended entry in `USE_LEDGER.jsonl`, naming who
opened them and why. A corpus scored once is validation. The same corpus scored
eleven times while the prompt is adjusted between attempts is development data,
and the ledger is what tells the two apart afterwards. Nothing here stops a
second use - it records it.

**A baseline may not be fitted on the held-out labels.** A majority-class naive
whose prior came from the corpus it is scored on is not a baseline. The prior
must be supplied from development material, and
:func:`refuse_prior_fitted_on_held_out` refuses a prior whose counts are the
corpus's own.

Both seals are checked on every read. A corpus whose bytes changed is not the
corpus the manifest describes, and it is refused rather than used.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
from pathlib import Path

CORPUS_DIR = (Path(__file__).resolve().parents[1]
              / "docs/audits/evidence/cb04_business_corpus_20260928")
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.json"
ITEMS_PATH = CORPUS_DIR / "items.jsonl"
LABELS_PATH = CORPUS_DIR / "labels.jsonl"
LEDGER_PATH = CORPUS_DIR / "USE_LEDGER.jsonl"

QUESTIONS = ("relevance", "novelty", "window")


class CorpusRefused(Exception):
    """The corpus was not handed over. Every refusal here is named."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise CorpusRefused(f"no corpus manifest at {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text())


def verify_seals() -> dict:
    """Both digests, checked against the manifest. Refuses a corpus that moved."""
    recorded = manifest()
    actual = {"items_sha256": _sha256(ITEMS_PATH), "labels_sha256": _sha256(LABELS_PATH)}
    for name, value in actual.items():
        if recorded[name] != value:
            raise CorpusRefused(
                f"{name} is {value}, the manifest records {recorded[name]}: this is not the "
                f"frozen corpus the manifest describes")
    return actual


def _read_jsonl(path: Path) -> list:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_items() -> list:
    """The items, with no labels. Free to read; nothing is recorded."""
    verify_seals()
    return _read_jsonl(ITEMS_PATH)


def ledger_entries() -> list:
    if not LEDGER_PATH.exists():
        return []
    return _read_jsonl(LEDGER_PATH)


def open_labels(*, actor: str, reason: str, provider: str, receipt_id=None) -> list:
    """The labels, against an appended ledger entry. Both arguments are required.

    The returned list is the labels. The ledger entry is written first, so a
    process that crashes after reading still leaves the use on the record.
    """
    if not isinstance(actor, str) or not actor.strip():
        raise CorpusRefused("actor is required: the ledger records who opened the labels")
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise CorpusRefused(
            "reason is required and must say what the labels are being opened for; a held-out "
            "corpus opened without a stated reason cannot later be described as untouched")
    if not isinstance(provider, str) or not provider.strip():
        raise CorpusRefused("provider is required: the ledger records what was scored")
    verify_seals()
    entry = {
        "opened_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "actor": actor.strip(),
        "provider": provider.strip(),
        "reason": reason.strip(),
        "receipt_id": receipt_id,
        "use_index": len(ledger_entries()) + 1,
        "labels_sha256": manifest()["labels_sha256"],
    }
    with LEDGER_PATH.open("a", encoding="ascii") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=True) + "\n")
    return _read_jsonl(LABELS_PATH)


def label_counts(question: str) -> dict:
    """The corpus's own label distribution, from the manifest, without opening labels."""
    if question not in QUESTIONS:
        raise CorpusRefused(f"{question!r} is not a corpus question: {', '.join(QUESTIONS)}")
    return dict(manifest()["questions"][question]["counts"])


def refuse_prior_fitted_on_held_out(question: str, prior_counts: dict) -> dict:
    """Refuse a naive prior that was fitted on the corpus it will be scored on.

    A majority-class baseline is only a baseline when its prior comes from
    somewhere else. If the supplied counts are the held-out corpus's own counts,
    the "baseline" has seen the answers.
    """
    own = label_counts(question)
    supplied = {str(key): int(value) for key, value in prior_counts.items()}
    if supplied == {str(k): int(v) for k, v in own.items()}:
        raise CorpusRefused(
            f"the prior offered for {question!r} is the held-out corpus's own label "
            f"distribution ({own}); a naive fitted on the evaluation labels is not a baseline. "
            f"Supply a prior from development material.")
    return supplied


def evaluation_population_sha256(question: str, item_ids) -> str:
    """The digest that binds a receipt to exactly these rows, for this question."""
    if question not in QUESTIONS:
        raise CorpusRefused(f"{question!r} is not a corpus question")
    ordered = list(item_ids)
    known = {item["item_id"] for item in load_items()}
    unknown = [identifier for identifier in ordered if identifier not in known]
    if unknown:
        raise CorpusRefused(f"these rows are not in the corpus: {', '.join(sorted(unknown))}")
    body = json.dumps({"corpus_id": manifest()["corpus_id"],
                       "items_sha256": manifest()["items_sha256"],
                       "question": question, "item_ids": ordered},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("ascii")).hexdigest()
