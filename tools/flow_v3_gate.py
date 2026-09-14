#!/usr/bin/env python3
"""Flow v3 work-plan gate for dispatchers.

A decision campaign (classification GOVERNING) may only be dispatched with a
campaign manifest (`governed_campaign.v1`, the body data-gov registers) that
declares its deliveries and its terminal destination. Synthetic and mechanics
campaigns stay NON_GOVERNING and say so in the dispatch root. The gate is
offline: it validates the manifest's shape and seals its digest; registration
and delivery verification happen in the runner (`tools/governed_run.py` or
`data-gov/tools/governed_exec.py`).

Refusal is typed (DispatchRefusal, exit 4) and, when a dispatch root is given,
written once to DISPATCH_REFUSAL.json; acceptance is written once to
DISPATCH_GATE.json so the root records under which classification it ran.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

KEY_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
CLASSIFICATIONS = ("GOVERNING", "NON_GOVERNING")
MANIFEST_KEYS = {
    "schema", "campaign_key", "classification", "project", "code_identity",
    "config_sha256", "input_mode", "synthetic_spec_sha256", "units", "datasets",
    "terminal_lake",
}
GATE_FILE = "DISPATCH_GATE.json"
REFUSAL_FILE = "DISPATCH_REFUSAL.json"


class DispatchRefusal(SystemExit):
    """Typed refusal: exit 4, reason on the exception."""

    def __init__(self, reason: str):
        super().__init__(4)
        self.reason = reason

    def __str__(self) -> str:
        return self.reason


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def write_once(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def validate_manifest(manifest) -> dict:
    """Shape of governed_campaign.v1 as data-gov normalises it; returns the
    manifest with its canonical digest under `manifest_sha256`."""
    if not isinstance(manifest, dict) or set(manifest) - {"manifest_sha256"} != MANIFEST_KEYS:
        raise ValueError("campaign manifest must carry exactly the governed_campaign.v1 keys")
    if manifest["schema"] != "governed_campaign.v1":
        raise ValueError("campaign manifest schema must be governed_campaign.v1")
    for key in ("campaign_key", "project"):
        if not isinstance(manifest[key], str) or not KEY_RE.match(manifest[key]):
            raise ValueError(f"invalid campaign manifest {key}")
    classification = manifest["classification"]
    if classification not in CLASSIFICATIONS:
        raise ValueError("invalid campaign manifest classification")
    identity = manifest["code_identity"]
    if not isinstance(identity, dict) or set(identity) != {"kind", "value"}:
        raise ValueError("invalid campaign manifest code_identity")
    if identity["kind"] == "git_commit":
        if classification == "GOVERNING" and not (isinstance(identity["value"], str) and HEX40_RE.match(identity["value"])):
            raise ValueError("a governing manifest needs a 40-hex git commit")
    elif identity["kind"] == "file_manifest":
        if not (isinstance(identity["value"], str) and HEX64_RE.match(identity["value"])):
            raise ValueError("invalid file_manifest identity")
    else:
        raise ValueError("invalid campaign manifest code_identity kind")
    if not (isinstance(manifest["config_sha256"], str) and HEX64_RE.match(manifest["config_sha256"])):
        raise ValueError("invalid campaign manifest config_sha256")
    if not isinstance(manifest["terminal_lake"], str) or not manifest["terminal_lake"]:
        raise ValueError("campaign manifest declares no Flow v3 terminal destination")
    units = manifest["units"]
    if not isinstance(units, list) or not units or any(not isinstance(u, str) or not KEY_RE.match(u) for u in units):
        raise ValueError("campaign manifest declares no valid units")
    if len(set(units)) != len(units):
        raise ValueError("duplicate unit in campaign manifest")
    datasets = manifest["datasets"]
    if not isinstance(datasets, list):
        raise ValueError("invalid campaign manifest datasets")
    for item in datasets:
        if not isinstance(item, dict) or set(item) != {"lake", "resource", "role", "from", "to"}:
            raise ValueError("invalid campaign manifest dataset")
        for key in ("lake", "resource", "role"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"campaign manifest dataset lacks {key}")
    mode = manifest["input_mode"]
    if mode == "DATASETS":
        if not datasets or manifest["synthetic_spec_sha256"] is not None:
            raise ValueError("a DATASETS manifest declares its deliveries and no synthetic spec")
    elif mode == "SYNTHETIC":
        if datasets or not (isinstance(manifest["synthetic_spec_sha256"], str) and HEX64_RE.match(manifest["synthetic_spec_sha256"])):
            raise ValueError("a SYNTHETIC manifest declares a synthetic spec and no deliveries")
    else:
        raise ValueError("invalid campaign manifest input_mode")
    body = {key: manifest[key] for key in MANIFEST_KEYS}
    digest = sha256_text(canonical(body))
    declared = manifest.get("manifest_sha256")
    if declared is not None and declared != digest:
        raise ValueError("campaign manifest digest does not match its body")
    return {**body, "manifest_sha256": digest}


def load_campaign_manifest(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return validate_manifest(json.load(handle))


def require_governed_dispatch(classification: str, manifest_path=None, *, root=None,
                              jobs_sha256=None, non_governing_reason=None) -> dict:
    """Admit a dispatch. GOVERNING needs a valid manifest whose classification is
    GOVERNING; NON_GOVERNING needs a stated reason. Both are sealed in the root."""
    try:
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown classification {classification!r}")
        gate = {"schema": "flow_v3_dispatch_gate.v1", "classification": classification,
                "jobs_sha256": jobs_sha256}
        if classification == "GOVERNING":
            if not manifest_path:
                raise ValueError("a GOVERNING dispatch requires --campaign-manifest")
            manifest = load_campaign_manifest(manifest_path)
            if manifest["classification"] != "GOVERNING":
                raise ValueError("the campaign manifest is not GOVERNING")
            gate.update(campaign_key=manifest["campaign_key"], project=manifest["project"],
                        manifest_sha256=manifest["manifest_sha256"], terminal_lake=manifest["terminal_lake"],
                        units=len(manifest["units"]), datasets=len(manifest["datasets"]),
                        input_mode=manifest["input_mode"])
        else:
            if not non_governing_reason:
                raise ValueError("a NON_GOVERNING dispatch states its reason (--non-governing-reason)")
            gate["non_governing_reason"] = str(non_governing_reason)
            if manifest_path:
                manifest = load_campaign_manifest(manifest_path)
                gate.update(campaign_key=manifest["campaign_key"], manifest_sha256=manifest["manifest_sha256"])
    except (OSError, ValueError) as exc:
        reason = str(exc)
        if root is not None:
            try:
                write_once(Path(root) / REFUSAL_FILE, {"schema": "flow_v3_dispatch_refusal.v1",
                                                       "classification": classification, "reason": reason})
            except FileExistsError:
                pass
        raise DispatchRefusal(reason) from exc
    if root is not None:
        try:
            write_once(Path(root) / GATE_FILE, gate)
        except FileExistsError:
            existing = json.loads((Path(root) / GATE_FILE).read_text(encoding="utf-8"))
            if existing != gate:
                raise DispatchRefusal("dispatch root already sealed under a different gate") from None
    return gate


def add_gate_arguments(parser) -> None:
    parser.add_argument("--classification", choices=CLASSIFICATIONS, default="NON_GOVERNING",
                        help="GOVERNING requires --campaign-manifest; default NON_GOVERNING")
    parser.add_argument("--campaign-manifest", type=Path,
                        help="governed_campaign.v1 JSON registered (or to be registered) with data-gov")
    parser.add_argument("--non-governing-reason", default=None,
                        help="why this dispatch does not govern (synthetic, mechanics, replay)")
