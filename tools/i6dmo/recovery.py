"""Authenticate a completed fit for a new campaign after a reporting-only repair.

The imported fit keeps its original identity and code pin. Reports and all new
fits use the successor's code pin; the failed campaign remains intact.
"""
import json
import shutil
import subprocess
from pathlib import Path

from tools.i6n_swarm.common import file_sha256


def reporting_only_changes(source_commit, target_commit):
    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run(["git", "diff", "--name-only", source_commit, target_commit], cwd=repo,
                         capture_output=True, text=True, check=True)
    allowed = {"tools/i6dmo/reporting.py", "tools/i6dmo/recovery.py",
               "tools/i6dmo/campaign.py", "tools/i6dmo/cli.py", "tools/i6dmo/worker.py"}
    changed = out.stdout.splitlines()
    rejected = [p for p in changed if p not in allowed and not p.startswith("tests/")
                and not (p.startswith("docs/") and p.endswith(".md"))]
    if rejected:
        raise ValueError(f"FIT_REUSE_TRAINING_CODE_CHANGED:{rejected}")
    return changed


def prepare(source_root, target):
    from tools.i6dmo import campaign as C, worker
    source_root = Path(source_root).resolve()
    source = C.load_effective(source_root)
    if C.mode_of(source) != C.MODE_MARKET_TIME or C.mode_of(target) != C.MODE_MARKET_TIME:
        raise C.Refused("FIT_REUSE_REQUIRES_MARKET_TIME_PAIR")
    for key in ("design", "adapter_effective", "long", "data_pins", "data_sha256", "seed", "cadence", "entrypoints"):
        if source.get(key) != target.get(key):
            raise C.Refused(f"FIT_REUSE_CONTRACT_DIFFERS:{key}")
    unit = next(u for u in C.load_plan(source_root)["units"] if u["unit_id"] == "short.fit")
    if unit["identity"]["code_commit"] != source["code_commit"]:
        raise C.Refused("FIT_REUSE_SOURCE_CODE_PIN_DIFFERS")
    folder = source_root / "terminals" / unit["identity_sha256"]
    stored = worker._verify_stored(folder, unit)
    if stored["terminal"]["status"] != "COMPLETE":
        raise C.Refused("FIT_REUSE_SOURCE_NOT_COMPLETE")
    q = C.open_queue(source_root)
    try:
        if q.unit("short.fit")["state"] != "COMPLETE":
            raise C.Refused("FIT_REUSE_SOURCE_QUEUE_NOT_COMPLETE")
    finally:
        q.close()
    changed = reporting_only_changes(source["code_commit"], target["code_commit"])
    return {"schema": "i6dmo.completed_fit_reuse.v1", "source_root": str(source_root),
            "source_campaign": C.load_plan(source_root)["campaign_id"],
            "unit": json.loads(json.dumps(unit)), "source_terminal_sha256": file_sha256(folder / "TERMINAL.json"),
            "source_code_commit": source["code_commit"], "reporting_code_commit": target["code_commit"],
            "changed_files": changed, "training_repeated": False}


def install(root, receipt, q):
    from tools.i6dmo import worker
    unit = receipt["unit"]
    source = Path(receipt["source_root"]) / "terminals" / unit["identity_sha256"]
    if file_sha256(source / "TERMINAL.json") != receipt["source_terminal_sha256"]:
        raise ValueError("FIT_REUSE_SOURCE_CHANGED")
    worker._verify_stored(source, unit)
    target = Path(root) / "terminals" / unit["identity_sha256"]
    shutil.copytree(source, target)
    stored = worker._verify_stored(target, unit)
    q.adopt("short.fit", stored["terminal"], stored["child_terminal_sha256"],
            identity_sha256=unit["identity_sha256"], source_detail=receipt)
    q.event("COMPLETED_FIT_REUSED", "short.fit", receipt)
