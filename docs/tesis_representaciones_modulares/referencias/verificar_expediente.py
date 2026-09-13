#!/usr/bin/env python3
"""Verify bibliography numbering, local filenames, and local file identities."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MANIFEST = ROOT / "MANIFEST.json"
README = ROOT / "README.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    references = manifest["references"]
    expected_numbers = list(range(1, len(references) + 1))
    numbers = [item["number"] for item in references]
    if numbers != expected_numbers:
        raise SystemExit(f"manifest numbering mismatch: {numbers}")

    proposal = REPO / manifest["proposal_source"]
    tex = proposal.read_text(encoding="utf-8")
    bibkeys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    expected_bibkeys = [item["bibkey"] for item in references]
    if bibkeys != expected_bibkeys:
        raise SystemExit("proposal bibliography order differs from MANIFEST.json")
    cited_keys = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", tex)
        for key in group.split(",")
    }
    if cited_keys != set(expected_bibkeys):
        missing = sorted(set(expected_bibkeys) - cited_keys)
        unknown = sorted(cited_keys - set(expected_bibkeys))
        raise SystemExit(f"citation coverage mismatch; missing={missing}, unknown={unknown}")

    readme = README.read_text(encoding="utf-8")
    entries = re.findall(r"^### \[(\d+)\] `([^`]+)`$", readme, re.MULTILINE)
    expected_entries = [(str(item["number"]), item["file"]) for item in references]
    if entries != expected_entries:
        raise SystemExit("README numbering or filenames differ from MANIFEST.json")

    numbered_files = sorted(
        path.name for path in ROOT.iterdir() if re.match(r"^\d{2}_", path.name)
    )
    expected_files = sorted(item["file"] for item in references)
    if numbered_files:
        if numbered_files != expected_files:
            missing = sorted(set(expected_files) - set(numbered_files))
            extra = sorted(set(numbered_files) - set(expected_files))
            raise SystemExit(f"local bundle mismatch; missing={missing}, extra={extra}")
        for item in references:
            observed = sha256(ROOT / item["file"])
            if observed != item["sha256"]:
                raise SystemExit(
                    f"digest mismatch for [{item['number']}] {item['file']}: {observed}"
                )
        local_status = "35/35 local artifacts match their digests"
    else:
        local_status = "local artifacts absent by policy; structure verified"

    print(
        "REFERENCE_BUNDLE_OK: "
        f"{len(references)} cited bibliography entries, {len(entries)} guide entries; "
        f"{local_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
