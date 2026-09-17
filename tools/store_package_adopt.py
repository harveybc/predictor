#!/usr/bin/env python3
"""Adopt a new build of `predictor-olap-store` into the production warehouse host (K4).

The existing procedure, as the 2026-09-14 adoption recorded it: inventory before, backup,
rehearsal on a disposable stack, the change, restart, inventory after, a bounded post-check
that writes nothing new, and a write-once receipt. This tool does exactly that for the store
package the DuckDB warehouse consumes (`write_foundation_envelope` imports it at call time), and
nothing else: no schema change, no data change, no other service touched.

    --rehearse   inventory + backup + rehearsal (the real service on a disposable cube with the
                 candidate package on PYTHONPATH); changes nothing in production
    --adopt      the above, then install into the venv, restart the warehouse unit, post-check

Run under the service environment file so the token comes from the environment:
    systemd-run --user --wait --pipe -p EnvironmentFile=... python tools/store_package_adopt.py ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
#: (schema, table): the envelope tables live in `public`, the governed terminals in `main`.
TABLES = (("public", "fact_campaign_unit"), ("public", "fact_campaign_consumption"),
          ("public", "dim_campaign"), ("public", "dim_campaign_run"),
          ("main", "gov_terminal"), ("main", "gov_terminal_metric"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(argv, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=kw.pop("timeout", 600),
                          **kw)


def http(url: str, token: str, *, method="GET", body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except ValueError:
            return exc.code, {}
    except Exception as exc:                       # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def query(url: str, token: str, sql: str):
    status, body = http(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}), token)
    return status, (body or {}).get("rows")


def service_state(unit: str) -> dict:
    out = run(["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p", "MainPID",
               "-p", "NRestarts", "-p", "ExecMainStartTimestamp"])
    return dict(line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line)


def installed(venv: Path, dist: str) -> dict:
    pip = venv / "bin" / "pip"
    out = run([str(pip), "show", "-f", dist])
    info = {"present": out.returncode == 0, "raw": out.stdout[:2000]}
    version, location, files = None, None, []
    for line in out.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
        elif line.startswith("Location:"):
            location = line.split(":", 1)[1].strip()
        elif line.startswith("  ") and line.strip():
            files.append(line.strip())
    info.update(version=version, location=location, files=files)
    envelope = next((f for f in files if f.endswith("campaign_envelope.py")), None)
    if location and envelope:
        info["envelope_sha256"] = sha_file(Path(location) / envelope)
    return info


def inventory(args, token: str) -> dict:
    counts = {}
    for schema, table in TABLES:
        status, rows = query(args.url, token,
                             f'SELECT count(*) AS n FROM "{schema}"."{table}" LIMIT 1')
        counts[f"{schema}.{table}"] = rows[0]["n"] if status == 200 and rows else f"http {status}"
    health_status, health = http(f"{args.url}/healthz", token)
    return {"at": now_iso(), "package": installed(args.venv, args.dist),
            "service": service_state(args.unit), "healthz": health_status,
            "table_counts": counts}


def backup(args, adoption: Path, before: dict) -> dict:
    pkg = before["package"]
    if not pkg.get("present"):
        return {"backed_up": False, "why": "package not installed"}
    location = Path(pkg["location"])
    dest = adoption / "backup"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for entry in sorted({f.split("/")[0] for f in pkg["files"]}):
        src = location / entry
        if src.is_dir():
            shutil.copytree(src, dest / entry, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dest / entry)
        copied.append(entry)
    manifest = {p.relative_to(dest).as_posix(): sha_file(p) for p in dest.rglob("*")
                if p.is_file()}
    (adoption / "backup.MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return {"backed_up": True, "entries": copied, "files": len(manifest),
            "restore": f"pip install --no-deps --force-reinstall <previous build> or copy "
                       f"{dest} back into {location}"}


def rehearse(args, adoption: Path) -> dict:
    """The real warehouse service on a disposable DuckDB cube with the candidate package
    ahead of site-packages: the K4 rules, verbatim, as the rehearsal."""
    log = adoption / "rehearsal.log"
    out = run([str(args.test_python), "-m", "pytest", "-q", "-p", "no:cacheprovider",
               str(REPO / "tests" / "test_olap_ingest_diagnostics.py"), "-k", "real_service"],
              cwd=str(REPO), env=dict(os.environ, STORE_HOSTS_PYTHON=str(args.venv / "bin" / "python"),
                                      K4_EXTRA_PYTHONPATH=os.pathsep.join(
                                          str(Path(p).resolve()) for p in args.candidate_pythonpath)),
              timeout=900)
    log.write_text(out.stdout + out.stderr)
    passed = out.returncode == 0 and "passed" in out.stdout and "skipped" not in out.stdout.split("\n")[-2]
    return {"passed": passed, "returncode": out.returncode,
            "tail": (out.stdout + out.stderr)[-600:]}


def wheel_matches_source(wheel: Path, source: Path) -> list:
    """Every .py in the wheel must be byte-identical to the same path under the source tree
    (src/ layout or flat): a wheel is never trusted to be what the commit says."""
    import zipfile
    problems = []
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if not name.endswith(".py") or ".dist-info/" in name:
                continue
            candidates = [source / name, source / "src" / name]
            match = next((c for c in candidates if c.is_file()), None)
            if match is None:
                problems.append(f"{name}: not in source")
            elif match.read_bytes() != z.read(name):
                problems.append(f"{name}: differs from source")
    return problems


def install(args, adoption: Path) -> dict:
    """Build the wheel with the test interpreter, install ONLY that wheel into the host venv:
    the production venv carries no build backend and compiles nothing."""
    wheels = adoption / "wheel"
    wheels.mkdir(exist_ok=True)
    source = args.package
    if args.package_ref:
        # the wheel is built from the named commit exported clean, never from a working tree
        # that may carry someone's uncommitted change
        export = adoption / "export"
        export.mkdir(exist_ok=True)
        archived = subprocess.run(["git", "-C", str(args.package_repo or args.package), "archive",
                                   "--format=tar", args.package_ref], capture_output=True,
                                  timeout=300)
        if archived.returncode != 0:
            return {"returncode": archived.returncode, "stage": "export",
                    "tail": archived.stderr.decode(errors="replace")[-400:]}
        (export / "src.tar").write_bytes(archived.stdout)
        run(["tar", "-xf", str(export / "src.tar"), "-C", str(export)], timeout=300)
        rel = Path(args.package).resolve().relative_to(Path(args.package_repo or args.package).resolve())
        source = export / rel
        # a tracked build/ or egg-info in the export is stale packaging state: setuptools
        # would ship it instead of the source (the 2026-09-17 warehouse adoption crash-looped
        # on exactly that). Only the source may become the wheel.
        for stale in list(source.glob("build")) + list(source.glob("*.egg-info")):
            shutil.rmtree(stale, ignore_errors=True)
    built = run([str(args.test_python), "-m", "pip", "wheel", "--no-deps", "--no-cache-dir",
                 "-w", str(wheels), str(source)], timeout=900)
    (adoption / "wheel.log").write_text(built.stdout + built.stderr)
    wheel = sorted(wheels.glob("*.whl"))
    if built.returncode != 0 or not wheel:
        return {"returncode": built.returncode or 1, "stage": "wheel",
                "tail": (built.stdout + built.stderr)[-400:]}
    mismatch = wheel_matches_source(wheel[-1], source)
    if mismatch:
        return {"returncode": 1, "stage": "wheel-parity", "wheel": wheel[-1].name,
                "tail": f"wheel modules differ from the exported source: {mismatch[:10]}"}
    pip = args.venv / "bin" / "pip"
    out = run([str(pip), "install", "--no-deps", "--force-reinstall", "--no-cache-dir",
               str(wheel[-1])], timeout=900)
    (adoption / "install.log").write_text(out.stdout + out.stderr)
    return {"returncode": out.returncode, "stage": "install", "wheel": wheel[-1].name,
            "wheel_sha256": sha_file(wheel[-1]), "tail": (out.stdout + out.stderr)[-400:]}


def restart(args, token: str = "") -> dict:
    out = run(["systemctl", "--user", "restart", args.unit], timeout=120)
    deadline = time.monotonic() + 90
    healthy = None
    while time.monotonic() < deadline:
        status, _ = http(f"{args.url}/healthz", token)
        if status is not None:                    # any HTTP answer: the process is up
            healthy = status
            break
        time.sleep(1.0)
    return {"returncode": out.returncode, "stderr": out.stderr[-300:], "healthz": healthy,
            "service": service_state(args.unit)}


def idempotency_probe(url: str, token: str, *, schema: str, outbox: Path) -> dict:
    """Post ONE envelope the cube already holds, chosen by identity (its envelope_sha256 is in
    fact_campaign_unit), and compare content: the store must skip every unit and open no run.
    With no held envelope in loaded/, nothing is posted — an old file is never a substitute.
    (The first adoption's probe posted the last loaded/ file by name and loaded a foreign
    DEVELOPMENT envelope; that receipt stands.)"""
    _, held = query(url, token, f'SELECT DISTINCT envelope_sha256 FROM "{schema}"'
                                '."fact_campaign_unit" LIMIT 5000')
    held = {r["envelope_sha256"] for r in (held or [])}
    known = None
    for path in sorted(outbox.joinpath("loaded").glob("envelope-*.json")):
        doc = json.loads(path.read_text())["document"]
        if doc.get("envelope_sha256") in held:
            known = doc
            break
    if known is None:
        return {"skipped": "no loaded envelope is held by this cube; nothing posted", "ok": True}
    _, runs_before = query(url, token, f'SELECT count(*) AS n FROM "{schema}"."dim_campaign_run" LIMIT 1')
    _, units_before = query(url, token, f'SELECT count(*) AS n FROM "{schema}"."fact_campaign_unit" '
                                        f"WHERE envelope_sha256 = '{known['envelope_sha256']}' LIMIT 1")
    status, body = http(f"{url}/api/v2/foundation-envelopes", token, method="POST",
                        body={"document": known})
    _, runs_after = query(url, token, f'SELECT count(*) AS n FROM "{schema}"."dim_campaign_run" LIMIT 1')
    _, units_after = query(url, token, f'SELECT count(*) AS n FROM "{schema}"."fact_campaign_unit" '
                                       f"WHERE envelope_sha256 = '{known['envelope_sha256']}' LIMIT 1")
    same_content = (units_before and units_after and units_before[0]["n"] == units_after[0]["n"]
                    == len(known.get("units", [])))
    return {"envelope_sha256": known["envelope_sha256"], "status": status,
            "answer": {k: v for k, v in (body or {}).items() if isinstance(v, int)},
            "runs_before": runs_before[0]["n"] if runs_before else None,
            "runs_after": runs_after[0]["n"] if runs_after else None,
            "content_matches": bool(same_content),
            # the store reports the campaign it touched even when it wrote nothing; what
            # proves idempotency is measured: no new run, no new unit, every unit skipped
            "ok": status == 201 and (body or {}).get("units", 0) == 0
            and (body or {}).get("runs", 0) == 0
            and (body or {}).get("skipped_existing", 0) == len(known.get("units", []))
            and runs_before == runs_after and bool(same_content)}


def postcheck(args, token: str, before: dict) -> dict:
    """Writes nothing new: the malformed document must be a 400; a document the cube already
    holds must be a 201 with everything skipped; table counts unchanged."""
    checks = {}
    malformed = json.loads((REPO / "tests" / "fixtures" / "k4_malformed_envelope.json").read_text()) \
        if (REPO / "tests" / "fixtures" / "k4_malformed_envelope.json").is_file() else None
    if malformed is None:
        sys.path.insert(0, str(REPO / "tests"))
        import test_olap_ingest_diagnostics as k4   # noqa: E402
        malformed, known = k4.malformed(), None
    status, body = http(f"{args.url}/api/v2/foundation-envelopes", token, method="POST",
                        body={"document": malformed})
    checks["malformed_is_400"] = {"status": status, "error": str(body.get("error", ""))[:200],
                                  "ok": status == 400}
    checks["known_envelope_is_idempotent"] = idempotency_probe(
        args.url, token, schema=args.schema, outbox=Path(args.outbox).expanduser())
    after = inventory(args, token)
    checks["table_counts_unchanged"] = {"before": before["table_counts"],
                                        "after": after["table_counts"],
                                        "ok": before["table_counts"] == after["table_counts"]}
    checks["package"] = after["package"]
    checks["service"] = after["service"]
    checks["ok"] = all(c.get("ok") for c in checks.values() if isinstance(c, dict) and "ok" in c)
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venv", type=Path, default=Path.home() / ".venvs/store-hosts-duckdb-prod")
    parser.add_argument("--package", type=Path, default=REPO / "olap" / "store")
    parser.add_argument("--dist", default="predictor-olap-store")
    parser.add_argument("--package-repo", type=Path, help="git repository holding --package")
    parser.add_argument("--package-ref", help="commit to export and build (never the tree)")
    parser.add_argument("--candidate-pythonpath", action="append", default=[],
                        help="paths put ahead of site-packages for the rehearsal service")
    parser.add_argument("--unit", default="crispdm-data-warehouse-olap.service")
    parser.add_argument("--url", default="http://127.0.0.1:5057")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--outbox", default="~/.local/share/predictor/olap_outbox")
    parser.add_argument("--test-python", type=Path,
                        default=Path.home() / "anaconda3/envs/trading-stack/bin/python")
    parser.add_argument("--adoption-dir", type=Path, required=True)
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rehearse", action="store_true")
    mode.add_argument("--adopt", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"REFUSED: {args.token_env} is not set; run under the service "
                         "environment file")
    adoption = args.adoption_dir
    adoption.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "store_package_adoption.v1", "started_at": now_iso(),
               "mode": "adopt" if args.adopt else "rehearse", "dist": args.dist,
               "candidate": {"path": str(args.package), "ref": args.package_ref,
                             "candidate_pythonpath": [str(p) for p in args.candidate_pythonpath]}}
    envelope_copy = args.package / "src" / "predictor_olap_store" / "campaign_envelope.py"
    if envelope_copy.is_file():
        receipt["candidate"]["envelope_sha256"] = sha_file(envelope_copy)
    receipt["before"] = inventory(args, token)
    receipt["backup"] = backup(args, adoption, receipt["before"])
    receipt["rehearsal"] = rehearse(args, adoption)
    outcome = "REHEARSED" if receipt["rehearsal"]["passed"] else "REHEARSAL_FAILED"
    if args.adopt and receipt["rehearsal"]["passed"]:
        receipt["install"] = install(args, adoption)
        if receipt["install"]["returncode"] == 0:
            receipt["restart"] = restart(args, token)
            receipt["postcheck"] = postcheck(args, token, receipt["before"])
            active = receipt["restart"].get("service", {}).get("ActiveState") == "active"
            outcome = "ADOPTED" if receipt["postcheck"]["ok"] and active \
                else "ADOPTED_POSTCHECK_FAILED"
        else:
            outcome = "INSTALL_FAILED"
    receipt["outcome"] = outcome
    receipt["finished_at"] = now_iso()
    (adoption / "RECEIPT.json").write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"outcome": outcome, "before_version": receipt["before"]["package"].get("version"),
                      "candidate_envelope_sha256": (receipt["candidate"].get("envelope_sha256") or "")[:16],
                      "rehearsal": receipt["rehearsal"]["passed"],
                      "postcheck": receipt.get("postcheck", {}).get("ok"),
                      "adoption_dir": str(adoption)}, indent=1))
    return 0 if outcome in ("REHEARSED", "ADOPTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
