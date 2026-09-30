"""Close a local, offline load path and publish a synthetic metric record.

This module does not import the Jina remote modules, does not set
trust_remote_code, and does not contact the hub. A missing snapshot file is a
refusal. snapshot_download without a pinned local revision cannot fetch.

Absence of subprocess, eval, or exec in the reviewed remote Python is not a
security certification. This harness does not certify that source either.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PINNED_REVISION = "46ed7da5b47e4bca710b756313fafaf4110c6bd1"
LICENCE = "CC BY-NC 4.0"
WEIGHT_TREE_BYTES = 1369721378

# Same roles as the 22-file revision tree. The synthetic fixture fills them
# with inert bytes; those bytes are not the revision's blobs.
MODEL_ROLES = (
    ".gitattributes",
    "README.md",
    "adapters/classification/adapter_config.json",
    "adapters/classification/adapter_model.safetensors",
    "adapters/clustering/adapter_config.json",
    "adapters/clustering/adapter_model.safetensors",
    "adapters/retrieval/adapter_config.json",
    "adapters/retrieval/adapter_model.safetensors",
    "adapters/text-matching/adapter_config.json",
    "adapters/text-matching/adapter_model.safetensors",
    "config.json",
    "config_sentence_transformers.json",
    "configuration_jina_embeddings_v5.py",
    "custom_st.py",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "modeling_jina_embeddings_v5.py",
    "modules.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
REVISION_FILE = "REVISION"
TASK_NAMES = ("classification", "clustering", "retrieval", "text-matching")

# Constructed labels. Not BANKING77 intents, and not a model output.
GOLD = ("red", "blue", "red", "green", "blue", "green", "red", "blue")
PREDICTED = ("red", "blue", "green", "green", "red", "green", "red", "blue")


class LocalSnapshotRefusal(Exception):
    """A missing or mismatched local file. Nothing was fetched."""

    def __init__(self, missing):
        self.missing = tuple(missing)
        joined = ", ".join(self.missing)
        super().__init__(f"refusal, not a download: {joined}")


class UnpinnedSnapshotDownloadRefusal(Exception):
    """snapshot_download cannot fetch a revision. bytes_fetched is always 0."""


class CredentialRefusal(Exception):
    """The process environment still carries a hub token or an API key."""


def is_credential_key(key: str) -> bool:
    upper = key.upper()
    if upper.startswith("HF_"):
        return True
    if upper in {
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    }:
        return True
    if upper.endswith("_TOKEN") or upper.endswith("_API_KEY") or upper.endswith("_SECRET"):
        return True
    if "SECRET" in upper or "PASSWORD" in upper or "PASSWD" in upper:
        return True
    return False


def require_clean_process_env(environ=None) -> None:
    env = os.environ if environ is None else environ
    found = sorted(key for key in env if is_credential_key(key))
    if found:
        raise CredentialRefusal("credential or hub environment is set: " + ", ".join(found))


class DownloadLedger:
    def __init__(self):
        self.events = []

    def refuse(self, **fields):
        event = dict(fields)
        event["bytes_fetched"] = 0
        event["fetched"] = False
        self.events.append(event)

    @property
    def bytes_fetched(self) -> int:
        return sum(event["bytes_fetched"] for event in self.events)


def snapshot_download_without_revision(repo_id, *, allow_patterns=None, revision=None, ledger):
    """The reviewed else-branch, closed. There is no fetch implementation."""
    if ledger is None:
        ledger = DownloadLedger()
    patterns = None if allow_patterns is None else list(allow_patterns)
    ledger.refuse(repo_id=str(repo_id), revision=revision, allow_patterns=patterns)
    raise UnpinnedSnapshotDownloadRefusal(
        "offline local-only: snapshot_download cannot fetch a new revision"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_json(role: str):
    if role == "config.json":
        return {
            "task_names": list(TASK_NAMES),
            "auto_map": {
                "AutoConfig": "configuration_jina_embeddings_v5.JinaEmbeddingsV5Config",
                "AutoModel": "modeling_jina_embeddings_v5.JinaEmbeddingsV5Model",
            },
            "synthetic": True,
        }
    if role == "modules.json":
        return [{"type": "sentence_transformers.models.Transformer", "path": "", "module": "custom_st"}]
    if role == "config_sentence_transformers.json":
        return {"synthetic": True, "__version__": {"sentence_transformers": "not-a-real-export"}}
    if role == "generation_config.json":
        return {"synthetic": True}
    if role == "tokenizer_config.json":
        return {"tokenizer_class": "SyntheticTokenizer", "synthetic": True}
    if role.endswith("adapter_config.json"):
        return {"peft_type": "LORA", "revision": None, "synthetic": True}
    raise ValueError(role)


def synthetic_bytes(role: str) -> bytes:
    structured = role.endswith("adapter_config.json") or role in {
        "config.json",
        "modules.json",
        "config_sentence_transformers.json",
        "generation_config.json",
        "tokenizer_config.json",
    }
    if structured:
        return json.dumps(_synthetic_json(role), separators=(",", ":")).encode() + b"\n"
    if role.endswith(".py"):
        return f"INERT {role}\nnot executable and not the remote module\n".encode()
    if role == "model.safetensors" or role.endswith("adapter_model.safetensors"):
        return f"INERT {role}\nnot weights\n".encode()
    return f"INERT {role}\n".encode()


def build_synthetic_snapshot(root, *, revision=PINNED_REVISION) -> dict:
    """Write inert role files. Does not copy or import the remote Python."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    written = {}
    for role in MODEL_ROLES:
        target = root / role
        target.parent.mkdir(parents=True, exist_ok=True)
        data = synthetic_bytes(role)
        target.write_bytes(data)
        written[role] = {"bytes": len(data), "sha256": _sha256(data)}
    revision_bytes = (revision + "\n").encode()
    (root / REVISION_FILE).write_bytes(revision_bytes)
    written[REVISION_FILE] = {"bytes": len(revision_bytes), "sha256": _sha256(revision_bytes)}
    return written


def _read_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise LocalSnapshotRefusal((str(path), "symlink"))
    return path.read_bytes()


def close_load_path(root, *, pinned_revision, ledger=None, expected_sha256=None):
    """Walk a local snapshot. Never calls out, never imports a role file.

    A path that is not a directory is the reviewed branch that would call
    snapshot_download with allow_patterns adapters/* and no revision.
    That call is refused here. A missing file on a local directory is a
    refusal and does not enter that branch.
    """
    if ledger is None:
        ledger = DownloadLedger()
    require_clean_process_env()
    root_path = Path(root)
    if not root_path.is_dir():
        snapshot_download_without_revision(
            root_path,
            allow_patterns=["adapters/*"],
            revision=None,
            ledger=ledger,
        )
    if pinned_revision != PINNED_REVISION:
        snapshot_download_without_revision(
            root_path,
            allow_patterns=None,
            revision=pinned_revision,
            ledger=ledger,
        )

    missing = []
    revision_path = root_path / REVISION_FILE
    if not revision_path.is_file() or revision_path.is_symlink():
        missing.append(REVISION_FILE)
    else:
        recorded = revision_path.read_text(encoding="utf-8").strip()
        if recorded != pinned_revision:
            missing.append(REVISION_FILE + ":revision-mismatch")
    for role in MODEL_ROLES:
        candidate = root_path / role
        if not candidate.is_file() or candidate.is_symlink():
            missing.append(role)
    if missing:
        raise LocalSnapshotRefusal(missing)

    for path in sorted(root_path.rglob("*")):
        if path.is_symlink():
            raise LocalSnapshotRefusal((str(path.relative_to(root_path)), "symlink"))

    config = json.loads(_read_bytes(root_path / "config.json"))
    task_names = config.get("task_names")
    if not isinstance(task_names, list) or not task_names:
        raise LocalSnapshotRefusal(("config.json:task_names",))
    for name in task_names:
        for leaf in ("adapter_config.json", "adapter_model.safetensors"):
            relative = f"adapters/{name}/{leaf}"
            candidate = root_path / relative
            if not candidate.is_file() or candidate.is_symlink():
                raise LocalSnapshotRefusal((relative,))

    modules_bytes = _read_bytes(root_path / "modules.json")
    if b"custom_st" not in modules_bytes:
        raise LocalSnapshotRefusal(("modules.json:custom_st",))
    # Present as bytes. Not imported.
    _read_bytes(root_path / "custom_st.py")
    _read_bytes(root_path / "modeling_jina_embeddings_v5.py")
    _read_bytes(root_path / "configuration_jina_embeddings_v5.py")

    files = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        data = _read_bytes(path)
        relative = str(path.relative_to(root_path))
        digest = _sha256(data)
        if expected_sha256 is not None and relative in expected_sha256:
            if expected_sha256[relative] != digest:
                raise LocalSnapshotRefusal((relative + ":hash-mismatch",))
        files.append({
            "path": relative,
            "bytes": len(data),
            "sha256": digest,
            "revision": pinned_revision,
            "bytes_are_the_pinned_revision_blobs": False,
        })
    if expected_sha256 is not None:
        absent = sorted(set(expected_sha256) - {item["path"] for item in files})
        if absent:
            raise LocalSnapshotRefusal(absent)

    return {
        "revision": pinned_revision,
        "offline": True,
        "local_only": True,
        "trust_remote_code": False,
        "imported_remote_modules": False,
        "snapshot_download_invoked": False,
        "bytes_fetched": ledger.bytes_fetched,
        "files": files,
        "security_certification": False,
    }


def _class_counts(gold, predicted):
    labels = tuple(sorted(set(gold) | set(predicted)))
    rows = {}
    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)
        support = sum(1 for g in gold if g == label)
        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return labels, rows


def classification_scores(gold, predicted) -> dict:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("gold and predicted must be the same non-empty length")
    labels, rows = _class_counts(gold, predicted)
    total = len(gold)
    correct = sum(1 for g, p in zip(gold, predicted) if g == p)
    accuracy = correct / total
    macro = {key: sum(rows[label][key] for label in labels) / len(labels) for key in ("precision", "recall", "f1")}
    weight = sum(rows[label]["support"] for label in labels)
    weighted = {
        key: sum(rows[label][key] * rows[label]["support"] for label in labels) / weight
        for key in ("precision", "recall", "f1")
    }
    one = {
        "accuracy": accuracy,
        "f1": macro["f1"],
        "f1_weighted": weighted["f1"],
        "precision": macro["precision"],
        "precision_weighted": weighted["precision"],
        "recall": macro["recall"],
        "recall_weighted": weighted["recall"],
        "ap": None,
        "ap_weighted": None,
    }
    return {
        "labels": list(labels),
        "per_class": rows,
        "correct": correct,
        "total": total,
        "experiment": one,
        "accuracy": accuracy,
        "f1": macro["f1"],
        "f1_weighted": weighted["f1"],
        "precision": macro["precision"],
        "precision_weighted": weighted["precision"],
        "recall": macro["recall"],
        "recall_weighted": weighted["recall"],
        "main_score": accuracy,
    }


def synthetic_metric_record() -> dict:
    """The metric record a later trial would fill. These numbers are not a score of BANKING77."""
    scored = classification_scores(GOLD, PREDICTED)
    experiment = dict(scored["experiment"])
    summary = dict(scored["experiment"])
    summary["main_score"] = scored["main_score"]
    summary["hf_subset"] = "synthetic"
    summary["languages"] = ["und"]
    summary["scores_per_experiment"] = [experiment]
    return {
        "schema": "mteb_classification_metric_record.v1",
        "status": "SYNTHETIC_PREP",
        "real_trial_status": "NOT_RUN",
        "is_banking77_score": False,
        "corpus": "synthetic_constructed_labels",
        "model": None,
        "model_explicitly_not": "jinaai/jina-embeddings-v5-text-small",
        "task_name": None,
        "task_explicitly_not": "Banking77Classification",
        "mteb_version_executed": None,
        "mteb_version_the_real_trial_would_use": "2.9.0",
        "label_source": "constructed_by_this_lane",
        "gold": list(GOLD),
        "predicted": list(PREDICTED),
        "per_class": scored["per_class"],
        "metric_definition": (
            "One constructed experiment. accuracy is correct/n. "
            "precision, recall, and f1 are the unweighted mean over the sorted union of labels. "
            "The weighted keys use gold support. ap is null. "
            "A later trial replaces these numbers; it does not extend them."
        ),
        "scores": {"test": [summary]},
        "evaluation_time": None,
        "kg_co2_emissions": None,
        "main_score": scored["main_score"],
    }


def main(argv=None) -> int:
    del argv
    raise SystemExit(
        "NOT_RUN: this module does not load the model, does not set trust_remote_code, and does not contact the hub"
    )


if __name__ == "__main__":
    main()
