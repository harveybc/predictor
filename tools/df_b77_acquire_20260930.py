"""Download the pinned model tree and the BANKING77 corpus as bytes.

This module does not import the remote model, does not import TensorFlow, and
does not execute a dataset script. ``json.loads`` on a JSON line is a data
read. A file whose bytes do not match the hub declaration for the pinned
revision is deleted and is not accepted.

Absence of a later execution is not a security certification of the three
remote Python files. They are stored as bytes and not imported.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUEST_PATH = ROOT / "docs/audits/evidence/banking77_request_20260930/REQUEST.json"

MODEL_REPO = "jinaai/jina-embeddings-v5-text-small"
MODEL_REVISION = "46ed7da5b47e4bca710b756313fafaf4110c6bd1"
MODEL_LICENCE = "CC BY-NC 4.0"
CORPUS_REPO = "mteb/banking77"
CORPUS_REVISION = "0fd18e25b25c072e09e0d92ab615fda904d66300"

# Git blob oids declared by the hub for this corpus revision. The live tree
# must name these same oids. A disagreement is a refusal, not a download.
CORPUS_GIT_OIDS = {
    ".gitattributes": "a770de78287b5353fb380cf8774b1e6caf2881d9",
    "prepare_data.py": "4e850b687bbd014f6edf58028069f956cd1eeb54",
    "test.jsonl": "8a8da8d3e3ad9a0575d97e5085afb1b70f01a1ce",
    "train.jsonl": "d85ae1d60f5768140eceaa5fc7c567d939408df3",
}

USER_AGENT = "retsu-b77-acquire/1.0"
CHUNK = 1024 * 1024


class AcquisitionRefusal(Exception):
    """A pinned file did not match the hub. Nothing from that file is kept."""


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _sha256_and_blob(path: Path) -> tuple[int, str, str]:
    if path.is_symlink():
        raise AcquisitionRefusal(f"symlink refused: {path}")
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            sha.update(chunk)
            size += len(chunk)
    blob = hashlib.sha1()
    blob.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            blob.update(chunk)
    return size, sha.hexdigest(), blob.hexdigest()


def _hub_tree_url(kind: str, repo: str, revision: str) -> str:
    quoted = urllib.parse.quote(repo, safe="/")
    return f"https://huggingface.co/api/{kind}/{quoted}/tree/{revision}?recursive=true"


def _resolve_url(kind: str, repo: str, revision: str, path: str) -> str:
    quoted_repo = urllib.parse.quote(repo, safe="/")
    quoted_path = urllib.parse.quote(path, safe="/")
    prefix = "https://huggingface.co/datasets" if kind == "datasets" else "https://huggingface.co"
    return f"{prefix}/{quoted_repo}/resolve/{revision}/{quoted_path}?download=true"


def _opener_open(opener, url: str):
    if opener is None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        return urllib.request.urlopen(request, timeout=120)
    return opener(url)


def fetch_tree(kind: str, repo: str, revision: str, opener=None) -> list[dict]:
    """The hub's file list for one revision. Directory entries are omitted."""
    with _opener_open(opener, _hub_tree_url(kind, repo, revision)) as response:
        payload = json.loads(response.read().decode())
    if not isinstance(payload, list):
        raise AcquisitionRefusal("hub tree was not a file list")
    files = []
    for item in payload:
        if item.get("type") != "file":
            continue
        files.append({
            "path": item["path"],
            "bytes": int(item["size"]),
            "git_oid": item.get("oid"),
            "lfs_oid": (item.get("lfs") or {}).get("oid"),
            "lfs": bool(item.get("lfs")),
        })
    return files


def _stream_download(url: str, dest: Path, opener=None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    transferred = 0
    try:
        with _opener_open(opener, url) as response, temporary.open("wb") as handle:
            status = getattr(response, "status", 200)
            if status != 200:
                raise AcquisitionRefusal(f"hub status {status} for {dest.name}")
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                transferred += len(chunk)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.replace(temporary, dest)
    except Exception:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        raise
    return transferred


def _declared_digest(entry: dict, sha256: str, blob: str) -> None:
    if entry["lfs"]:
        lfs_oid = str(entry["lfs_oid"] or "")
        lfs_oid = lfs_oid.removeprefix("sha256:")
        if lfs_oid != sha256:
            raise AcquisitionRefusal(
                f"{entry['path']}: sha256 does not match the hub lfs oid")
    else:
        if entry["git_oid"] != blob:
            raise AcquisitionRefusal(
                f"{entry['path']}: git blob does not match the hub oid")


def _reject(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def place_file(entry: dict, dest: Path, *, url: str, opener=None) -> dict:
    """Download one file and keep it only when the hub declaration matches."""
    transferred = _stream_download(url, dest, opener)
    try:
        size, sha256, blob = _sha256_and_blob(dest)
        if size != entry["bytes"] or transferred != entry["bytes"]:
            raise AcquisitionRefusal(
                f"{entry['path']}: size {size} transferred {transferred} "
                f"hub {entry['bytes']}")
        _declared_digest(entry, sha256, blob)
    except Exception:
        _reject(dest)
        raise
    return {
        "path": entry["path"],
        "bytes": size,
        "sha256": sha256,
        "git_oid": entry["git_oid"],
        "lfs": entry["lfs"],
        "lfs_oid": entry["lfs_oid"],
        "executed": False,
        "python_executed": False,
    }


def _cross_check_model(entries: list[dict], request: dict) -> None:
    expected = {item["path"]: item for item in request["model"]["files"]}
    live = {item["path"]: item for item in entries}
    if set(expected) != set(live):
        raise AcquisitionRefusal("model tree paths differ from REQUEST.json")
    for path, prior in expected.items():
        current = live[path]
        if prior.get("git_oid") != current["git_oid"]:
            raise AcquisitionRefusal(f"{path}: git oid differs from REQUEST.json")
        if int(prior["bytes"]) != int(current["bytes"]):
            raise AcquisitionRefusal(f"{path}: size differs from REQUEST.json")
        prior_lfs = prior.get("lfs_oid")
        if prior_lfs and prior_lfs != current["lfs_oid"]:
            raise AcquisitionRefusal(f"{path}: lfs oid differs from REQUEST.json")


def _cross_check_corpus(entries: list[dict]) -> None:
    live = {item["path"]: item for item in entries}
    if set(live) != set(CORPUS_GIT_OIDS):
        raise AcquisitionRefusal("corpus tree paths differ from the pinned oid list")
    for path, oid in CORPUS_GIT_OIDS.items():
        if live[path]["git_oid"] != oid:
            raise AcquisitionRefusal(f"{path}: corpus git oid is not the pinned oid")
        if live[path]["lfs"]:
            raise AcquisitionRefusal(f"{path}: corpus file was declared lfs; not accepted")


def count_jsonl(path: Path) -> dict:
    """Row count of a JSON-lines file. This does not execute Python."""
    rows = 0
    blank = 0
    labels: dict[str, int] = {}
    keys: dict[str, int] = {}
    label_field = None
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                blank += 1
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise AcquisitionRefusal(f"{path.name}: a row is not a JSON object")
            rows += 1
            for key in item:
                keys[key] = keys.get(key, 0) + 1
            if label_field is None:
                for candidate in ("label", "label_text", "intent"):
                    if candidate in item:
                        label_field = candidate
                        break
            if label_field is not None:
                label = str(item.get(label_field))
                labels[label] = labels.get(label, 0) + 1
    supports = list(labels.values())
    return {
        "rows": rows,
        "blank_lines": blank,
        "label_field": label_field,
        "distinct_labels": len(labels),
        "min_support": min(supports) if supports else None,
        "max_support": max(supports) if supports else None,
        "keys": sorted(keys),
        "reader": "json.loads per line",
        "executed_python": False,
    }


def _write_revision(directory: Path, revision: str) -> None:
    target = directory / "REVISION"
    if target.is_symlink():
        raise AcquisitionRefusal("REVISION is a symlink")
    target.write_text(revision + "\n", encoding="utf-8")


def _acquire_one(kind: str, repo: str, revision: str, directory: Path, entries: list[dict],
                 opener=None) -> dict:
    started = time.perf_counter()
    files = []
    downloaded = 0
    for entry in sorted(entries, key=lambda item: item["path"]):
        dest = directory / entry["path"]
        url = _resolve_url(kind, repo, revision, entry["path"])
        placed = place_file(entry, dest, url=url, opener=opener)
        files.append(placed)
        downloaded += placed["bytes"]
    _write_revision(directory, revision)
    return {
        "repository": repo,
        "revision": revision,
        "files": files,
        "hub_bytes": sum(item["bytes"] for item in files),
        "bytes_downloaded": downloaded,
        "seconds": round(time.perf_counter() - started, 6),
        "python_executed": False,
        "revision_file": "REVISION",
    }


def acquire(dest, *, request_path=REQUEST_PATH, opener=None) -> dict:
    """Download both pinned trees. A mismatch deletes the offending file."""
    dest = Path(dest)
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    model_entries = fetch_tree("models", MODEL_REPO, MODEL_REVISION, opener)
    corpus_entries = fetch_tree("datasets", CORPUS_REPO, CORPUS_REVISION, opener)
    _cross_check_model(model_entries, request)
    _cross_check_corpus(corpus_entries)
    model_dir = dest / "model" / "jina-embeddings-v5-text-small"
    corpus_dir = dest / "corpus" / "mteb-banking77"
    model = _acquire_one("models", MODEL_REPO, MODEL_REVISION, model_dir, model_entries, opener)
    corpus = _acquire_one("datasets", CORPUS_REPO, CORPUS_REVISION, corpus_dir, corpus_entries, opener)
    splits = {}
    for name in ("train.jsonl", "test.jsonl"):
        splits[name] = count_jsonl(corpus_dir / name)
    corpus["splits"] = splits
    corpus["prepare_data_py_executed"] = False
    corpus["data_files_are_jsonl"] = True
    corpus["dataset_script_was_required"] = False
    model["licence"] = MODEL_LICENCE
    model["licence_is_not_commercial_approval"] = True
    model["security_certification"] = False
    return {
        "schema": "banking77_acquisition.v1",
        "status": "ACQUIRED_NOT_EXECUTED",
        "trial_status": "NOT_RUN",
        "trust_remote_code_set_by_this_lane": False,
        "model": model,
        "corpus": corpus,
        "bytes_downloaded": model["bytes_downloaded"] + corpus["bytes_downloaded"],
        "seconds": round(model["seconds"] + corpus["seconds"], 6),
    }


def check_tree(dest, record) -> dict:
    """Re-hash a local tree against an acquisition record. Does not download."""
    dest = Path(dest)
    if isinstance(record, (str, Path)):
        record = json.loads(Path(record).read_text(encoding="utf-8"))
    mismatches = []
    checked = 0
    for key, folder in (
        ("model", dest / "model" / "jina-embeddings-v5-text-small"),
        ("corpus", dest / "corpus" / "mteb-banking77"),
    ):
        section = record[key]
        revision_path = folder / "REVISION"
        if not revision_path.is_file() or revision_path.is_symlink():
            mismatches.append(f"{key}/REVISION")
        elif revision_path.read_text(encoding="utf-8").strip() != section["revision"]:
            mismatches.append(f"{key}/REVISION")
        for item in section["files"]:
            path = folder / item["path"]
            checked += 1
            if not path.is_file() or path.is_symlink():
                mismatches.append(item["path"])
                continue
            size, sha256, _blob = _sha256_and_blob(path)
            if size != item["bytes"] or sha256 != item["sha256"]:
                mismatches.append(item["path"])
    if mismatches:
        raise AcquisitionRefusal("local tree does not match the acquisition record: "
                                 + ", ".join(mismatches))
    return {"status": "MATCH", "files_checked": checked, "bytes_fetched": 0}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Acquire pinned BANKING77 bytes. Does not execute them.")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--check", action="store_true",
                        help="re-hash the local tree; do not download")
    args = parser.parse_args(argv)
    record_path = Path(args.record)
    try:
        if args.check:
            result = check_tree(args.dest, record_path)
        else:
            result = acquire(args.dest)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (AcquisitionRefusal, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(json.dumps({
        "status": result["status"],
        "bytes_downloaded": result.get("bytes_downloaded", 0),
        "bytes_fetched_by_check": result.get("bytes_fetched", None),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
