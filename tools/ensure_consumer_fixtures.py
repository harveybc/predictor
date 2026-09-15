#!/usr/bin/env python3
"""Make the declared fixtures available, or say exactly why they are not.

R2 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "Repair fixture regeneration: manifest-only checkout, missing artifact, mismatched
     artifact and fresh checkout must each yield regeneration into temporary storage or an
     explicit diagnostic, never accidental FileNotFoundError. Verify generated identities
     against the published manifest; never overwrite original evidence."

`MANIFEST.json` is committed as evidence; the CSV files it describes live in a lake root that
a fresh checkout does not have. Anything reading the manifest and opening the paths straight
away dies with a FileNotFoundError that says nothing about what to do.

This resolves the four situations by name:

  present_and_verified  the file is there and its sha256 matches the manifest — used as is;
  regenerated           it is absent, so it is rebuilt from the manifest's own seed into a
                        TEMPORARY directory and its sha256 is checked against the manifest;
  mismatched            it is there and its bytes differ. The original is never touched: the
                        regenerated copy goes to temporary storage and both digests are
                        reported, because which one is right is not this tool's to decide;
  unreproducible        the generator itself no longer matches `generator_sha256`, so a
                        regenerated file proves nothing about the published identity.

usage:
  ensure_consumer_fixtures.py --manifest MANIFEST.json [--lake-root DIR] [--work-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "make_consumer_fixtures.py"


class FixtureError(RuntimeError):
    """The fixtures cannot be made available, and the reason is named."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generator_matches(manifest: dict) -> bool:
    """Whether the generator on disk is the one that produced the published identities."""
    declared = manifest.get("generator_sha256")
    if not declared or not GENERATOR.is_file():
        return False
    return hashlib.sha256(GENERATOR.read_bytes()).hexdigest() == declared


def regenerate(manifest: dict, work_dir: Path) -> Path:
    """Rebuild every fixture from the manifest's own seed, into `work_dir`.

    Never writes next to the originals: the caller owns `work_dir`, and this tool only ever
    creates files inside it.
    """
    sys.path.insert(0, str(HERE))
    import make_consumer_fixtures  # noqa: E402 — resolved from this tools directory

    work_dir.mkdir(parents=True, exist_ok=True)
    code = make_consumer_fixtures.main([
        "--out", str(work_dir),
        "--seed", str(manifest.get("seed", 20260914)),
        "--rows", str(manifest.get("rows_per_series", 900)),
    ])
    if code != 0:
        raise FixtureError(f"the generator exited {code}; no fixture was produced")
    return work_dir


def ensure(manifest_path: Path, lake_root: Path | None = None,
           work_dir: Path | None = None) -> dict:
    """Resolve every declared fixture and return one verdict per file."""
    if not manifest_path.is_file():
        raise FixtureError(
            f"no fixture manifest at {manifest_path}: publish MANIFEST.json, or run "
            "make_consumer_fixtures.py --out DIR to produce one")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    if not files:
        raise FixtureError(f"{manifest_path} declares no files")

    root = Path(lake_root) if lake_root else manifest_path.parent
    reproducible = generator_matches(manifest)
    rebuilt: Path | None = None
    resolved, verdicts = {}, {}

    for name, record in sorted(files.items()):
        published = record.get("sha256")
        original = root / name
        if original.is_file() and digest(original) == published:
            verdicts[name] = "present_and_verified"
            resolved[name] = str(original)
            continue

        if not reproducible:
            verdicts[name] = "unreproducible"
            continue

        if rebuilt is None:
            rebuilt = regenerate(manifest, Path(work_dir) if work_dir else
                                 Path(tempfile.mkdtemp(prefix="consumer-fixtures-")))
        candidate = rebuilt / name
        if not candidate.is_file():
            verdicts[name] = "not_produced_by_the_generator"
            continue
        if digest(candidate) != published:
            verdicts[name] = "regenerated_but_different"
            continue

        if original.is_file():
            # the bytes on disk are NOT the published ones: keep them, use the rebuilt copy
            verdicts[name] = "mismatched_original_preserved"
        else:
            verdicts[name] = "regenerated"
        resolved[name] = str(candidate)

    return {"manifest": str(manifest_path), "generator_reproducible": reproducible,
            "work_dir": str(rebuilt) if rebuilt else None,
            "verdicts": verdicts, "resolved": resolved,
            "missing": sorted(name for name in files if name not in resolved)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lake-root", type=Path, default=None,
                        help="where the published fixtures should be (default: beside the "
                             "manifest)")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="temporary directory for regenerated copies; a fresh temporary "
                             "directory is used when omitted")
    parser.add_argument("--keep", action="store_true",
                        help="keep the temporary directory (it is kept by default when "
                             "--work-dir is given)")
    args = parser.parse_args(argv)
    try:
        report = ensure(args.manifest, args.lake_root, args.work_dir)
    except FixtureError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, indent=1))
        return 2
    home = str(Path.home())
    printable = json.loads(json.dumps(report).replace(home, "~"))
    print(json.dumps(printable, indent=1))
    if report["missing"] and not args.keep and args.work_dir is None and report["work_dir"]:
        shutil.rmtree(report["work_dir"], ignore_errors=True)
    return 0 if not report["missing"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
