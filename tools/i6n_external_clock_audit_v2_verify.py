"""Offline preparation gate for complete, hash-pinned V2 clock audit evidence.

SHA-256 validates local evidence consistency, not HTTP provider attestation.
"""

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from tools import i6n_external_clock_audit_v2 as v2

v1, source = v2.v1, v2.source


def _hash(value):
    return source.digest(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode())


def _links(report):
    return sorted({report[k] for k in ("prior_v2_report_sha256", "v1_report_sha256") if k in report})


def _sealed(root):
    raw = v1._read(root / "REPORT.json", 2_000_000)
    digest = source.digest(raw)
    if (root / "REPORT.sha256").read_text().strip() != digest:
        raise ValueError("report SHA-256 mismatch")
    return raw, digest


def bundle_chain(output, roots):
    directory = output / "report_chain"
    directory.mkdir()
    content = {}
    for root in roots:
        root = Path(root)
        raw, digest = _sealed(root)
        content[digest] = raw
        existing = root / "report_chain"
        if existing.exists():
            for path in existing.glob("*.json"):
                raw = v1._read(path, 2_000_000)
                if path.stem != source.digest(raw):
                    raise ValueError("ancestor report SHA-256 mismatch")
                content[path.stem] = raw
    for digest, raw in content.items():
        v1._write(directory / f"{digest}.json", raw)


def add_evidence_pins(report):
    groups = {role: [c["date"] for c in report["cases"] if c["evidence_role"] == role]
              for role in ("replay", "holdout")}
    report["case_groups"] = groups
    report["case_groups_sha256"] = _hash(groups)
    traces = [dict(date=c["date"], evidence_role=c["evidence_role"],
                   dukascopy_sha256=c.get("dukascopy_sha256"),
                   transport_trace=c.get("transport_trace"), url=c.get("url")) for c in report["cases"]]
    report["transport_chain_sha256"] = _hash(dict(ancestors=_links(report), cases=traces))


def verify_audit(report_root, annual_manifest):
    """Raise ValueError on a failed gate; return detached evidence on success.

    report_root is the final successor directory, containing report_chain.
    annual_manifest is the exact recovered annual MANIFEST.json file.
    No network, re-fitting or acceptance-policy changes occur here.
    """
    from tools.i6n_external_clock_audit_v2_resume import redirect_target, transport_failure
    root = Path(report_root)
    raw, report_hash = _sealed(root)
    report = json.loads(raw)
    manifest_raw = v1._read(Path(annual_manifest), v1.MAX_BYTES)
    manifest_hash = source.digest(manifest_raw)
    if (report.get("status") != "VERIFIED" or report.get("clock_policy") != v2.CLOCK_POLICY
            or report.get("input_manifest_sha256") != manifest_hash):
        raise ValueError("clock status, policy or annual manifest pin failed")
    replay = [d.isoformat() for d in v1.planned_dates() if d.isoformat() != "2019-01-21"]
    holdouts = [d.isoformat() for d in v2.holdout_dates()]
    expected = replay + holdouts
    if (report.get("planned_dates") != expected
            or [c.get("date") for c in report["cases"]] != expected
            or report.get("case_groups") != dict(replay=replay, holdout=holdouts)):
        raise ValueError("case completeness or evidence group mismatch")
    copied = json.loads(raw)
    add_evidence_pins(copied)
    if any(copied[k] != report.get(k) for k in ("case_groups_sha256", "transport_chain_sha256")):
        raise ValueError("group/transport chain SHA-256 mismatch")
    archives = {int(a["month"]): a["zip_sha256"] for a in json.loads(manifest_raw)["attempts"]
                if a.get("status") == "DOWNLOADED" and a.get("month") in {str(y) for y in range(2019, 2025)}}
    mapping = report["annual_mapping"]
    if ([m["year"] for m in mapping] != list(range(2019, 2025))
            or any(m["status"] != "PASS" or m.get("never_moves_earlier") is not True
                   or m.get("unique") is not True or m.get("ordered") is not True
                   or m.get("source_zip_sha256") != archives.get(m["year"]) for m in mapping)):
        raise ValueError("annual mapping evidence failed")
    chain, pending = {}, _links(report)
    while pending:
        digest = pending.pop()
        if digest in chain:
            continue
        if len(chain) >= 16 or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid or excessive ancestor chain")
        ancestor_raw = v1._read(root / "report_chain" / f"{digest}.json", 2_000_000)
        if source.digest(ancestor_raw) != digest:
            raise ValueError("ancestor report SHA-256 mismatch")
        ancestor = json.loads(ancestor_raw)
        if ancestor.get("schema", "").startswith("i6n.external_clock_audit.v2"):
            if (ancestor.get("input_manifest_sha256") != manifest_hash
                    or ancestor.get("planned_dates") != expected
                    or [c.get("date") for c in ancestor["cases"]] != expected):
                raise ValueError("ancestor manifest pin or case completeness mismatch")
        chain[digest] = ancestor
        pending.extend(_links(ancestor))
    for current in [report, *chain.values()]:
        parent_hash = current.get("prior_v2_report_sha256")
        if not parent_hash:
            continue
        parent = chain[parent_hash]
        if current["planned_dates"] != parent["planned_dates"] or current["acceptance"] != parent["acceptance"]:
            raise ValueError("successor changed the frozen acceptance plan")
        for previous, case in zip(parent["cases"], current["cases"]):
            if previous["status"] == "PASS" and previous != case:
                raise ValueError("successful case changed across transport chain")
            if previous["status"] != "PASS" and not transport_failure(previous):
                raise ValueError("numeric/nontransport failure retried")
    for i, case in enumerate(report["cases"]):
        count = case.get("available_endpoints")
        day = date.fromisoformat(case["date"])
        if (case["status"] != "PASS" or case["evidence_role"] != ("replay" if i < 17 else "holdout")
                or type(count) is not int or not 12 <= count <= 24
                or case.get("matched_exact") != count or case.get("mean_abs_error") != 0
                or case.get("max_abs_error") != 0 or case.get("source_zip_sha256") != archives.get(day.year)):
            raise ValueError("case numeric acceptance or source hash failed")
        missing = case["absent_endpoints_utc"]
        endpoints = {(datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=h)).isoformat()
                     for h in range(1, 25)}
        if (len(missing) != 24 - count or case["absent_endpoint_count"] != len(missing)
                or len(set(missing)) != len(missing) or not set(missing) <= endpoints):
            raise ValueError("missing endpoint accounting failed")
        body = v1._read(root / f"{day.isoformat()}.bi5", v1.MAX_BYTES)
        if source.digest(body) != case["dukascopy_sha256"]:
            raise ValueError("case artifact SHA-256 mismatch")
        local = json.loads(v1._read(root / f"{day.isoformat()}.json", v1.MAX_BYTES))
        if local != case:
            raise ValueError("case receipt differs from terminal report")
        canonical = f"{v1.BASE}/{day.year}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
        trace = case.get("transport_trace") or [dict(url=case.get("url"), status=200)]
        if not 1 <= len(trace) <= 2 or trace[0]["url"] != canonical or trace[-1].get("status") != 200:
            raise ValueError("invalid successful transport trace")
        if len(trace) == 2 and (trace[0].get("status") not in (301, 302, 307, 308)
                or trace[1]["url"] != redirect_target(canonical, trace[0].get("location", ""))):
            raise ValueError("invalid redirect chain")
    return dict(status="VERIFIED", clock_policy=v2.CLOCK_POLICY, report_sha256=report_hash,
                input_manifest_sha256=manifest_hash, case_count=23,
                case_groups_sha256=report["case_groups_sha256"],
                transport_chain_sha256=report["transport_chain_sha256"])
