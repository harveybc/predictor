"""Read-only bounded audit of retained Huber identity and local resource state."""
import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[3]
EXPECTED = "be2e776e5c64a8422a6411447a4cbeba6e5607c7158456f9a64f89a4244b6965"


def git_blob(rev, path):
    return subprocess.check_output(["git", "show", f"{rev}:{path}"], cwd=REPO)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def collect():
    retained = git_blob("76650ce8", "docs/audits/evidence/HUBER_ADAMW_2026_09_21/DESIGN.json")
    design = json.loads(retained)
    canonical = json.dumps({k: v for k, v in design.items() if k != "design_sha256"},
                           sort_keys=True, separators=(",", ":"), default=str).encode()
    root = Path.home() / ".local/state/crispdm-data-foundation/huber_adamw_v2"
    report = json.loads(git_blob("3bb65960", "docs/audits/evidence/HUBER_ADAMW_2026_09_21/REPORT.json"))
    expected_cells = {c["cell_id"] for c in design["cells"]}
    cells = {}
    for name in sorted(expected_cells):
        p = root / "attempts" / name / "cell.json"
        cells[name] = json.loads(p.read_text())["design_sha256"] == EXPECTED if p.exists() else None
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "retained_raw_sha256": digest(retained), "canonical_sha256": digest(canonical),
        "canonical_equals_previous_digest": digest(canonical) == EXPECTED,
        "runroot_byte_identical": (root / "DESIGN.json").read_bytes() == retained,
        "historical_report_revision": "3bb65960", "historical_report_digest_matches": report["design_sha256"] == EXPECTED,
        "sources_at_execution": {name: digest(git_blob("73f3bab", "tools/" + name)) == value
                                 for name, value in design["source_code"].items()},
        "cell_identity_matches": cells,
        "limits": ["No new inference, fitting or holdout read", "Filesystem timestamps do not prove immutable chronology",
                   "This verifies content identity against prior recorded digest, not unique physical-file provenance"],
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
