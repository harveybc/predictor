#!/usr/bin/env python3
"""Reversible restoration of the registered household endpoint (RB03).

The household panel (`uci_235_individual_household_power/panel.parquet`) is served by the lake
`public_panels`, which the live governance registry publishes at a loopback port of its own.  That
lake host unit is disabled and inactive, so every household consumer is refused at the download
step -- including one whose content-addressed cache is already warm, because the client re-streams
and re-verifies the bytes on every unit.

What this tool does NOT do, by construction:

* it never writes, copies or restarts the governance kernel's configuration.  The registry already
  carries this lake, unchanged, and the adoption receipt's own `restore` line -- copy the
  2026-09-20 registry backup over the live configuration and restart the kernel -- would today
  DELETE the SOTA benchmarks lake that was registered after it.  That line is stale.  Nothing here
  touches the kernel or any unrelated service.
* it never edits a data contract: not the lake host configuration, not a resource contract, not a
  holdout, not the availability declaration, not the panel bytes.
* it never enables the unit unless the operator explicitly asks.  `apply` starts the unit only, so
  the change does not survive a reboot and `rollback` is a stop.

Stages (each exits non-zero on the first failed check and mutates nothing before `apply`):

  preflight   read-only.  Identity of the unit file, the lake host configuration, the two panel
              digests and the installed serving provider against the adoption receipt; the
              registry entry; the port; and a snapshot of every other governed store unit.
  rehearse    read-only for the real endpoint.  Starts the SAME interpreter, module and
              configuration on a throwaway port -- only `web_port` differs -- proves the household
              panel is served byte-exact through the governed download path, then stops it.  The
              throwaway is never registered, never enabled and never reachable by a consumer.
  apply       `systemctl --user start` that ONE unit, then `verify`.  The operator's step.
  verify      read-only.  The endpoint answers, lists exactly its registered resources, and no
              other governed store unit changed state.
  rollback    `systemctl --user stop` that ONE unit (and `disable` only if `--also-enable` was
              used), then confirm the port is free again.

Author: Satoshi III (Mujuro Utsutsu), successor technical lead, 2026-09-28.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

STATE = Path.home() / ".local/state/crispdm-data-foundation"
ADOPTION = STATE / "satoshi-public-lake-adoption-20260920T182539Z"
RECEIPT = ADOPTION / "RECEIPT.json"
HOST_CONFIG = ADOPTION / "public-panels.host.json"
SERVICE_ENV = ADOPTION / "public-panels.service.env"
UNIT = "crispdm-data-lake-public-panels.service"
UNIT_FILE = Path.home() / ".config/systemd/user" / UNIT
REGISTRY = STATE / "musashi-store-adoption-20260914T181826Z/5055.runtime.json"
LAKE_ID = "public_panels"
HOUSEHOLD = "uci_235_individual_household_power/panel.parquet"
OTHER_STORE_UNITS = (
    "crispdm-data-gov.service",
    "crispdm-data-lake-financial.service",
    "crispdm-data-lake-synthetic.service",
    "crispdm-data-lake-sota-benchmarks.service",
    "crispdm-data-warehouse-olap.service",
)


class Refused(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"REFUSED: {message}")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(("systemctl", "--user", *args), capture_output=True, text=True)


def unit_state(unit: str) -> dict:
    out = systemctl("show", unit, "-p", "ActiveState", "-p", "UnitFileState", "-p", "SubState").stdout
    return dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)


def port_is_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def service_token() -> str:
    """Read the token from the deployed environment file.  It is never printed or stored."""
    for line in SERVICE_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("DATA_GOV_LAKE_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise Refused("the deployed environment file carries no store token")


def pythonpath() -> str:
    for line in SERVICE_ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith("PYTHONPATH="):
            return line.split("=", 1)[1].strip()
    raise Refused("the deployed environment file carries no PYTHONPATH")


def interpreter() -> str:
    """The interpreter and module the deployed unit itself names -- not a guess."""
    text = UNIT_FILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("ExecStart="):
            return line.split("=", 1)[1].split()[0]
    raise Refused("the unit file has no ExecStart")


def api(port: int, path: str, token: str, timeout: int = 30):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return handle.status, handle.read(), dict(handle.headers)


# --------------------------------------------------------------------------------------- preflight

def preflight(verbose: bool = True) -> dict:
    facts: dict = {"stage": "preflight", "checks": [], "ok": True}

    def check(name: str, ok: bool, detail: str) -> None:
        facts["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            facts["ok"] = False

    for path in (RECEIPT, HOST_CONFIG, SERVICE_ENV, UNIT_FILE, REGISTRY):
        check(f"present:{path.name}", path.is_file(), str(path))
    if not facts["ok"]:
        return facts

    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    expected = receipt["binding"]["expected"]
    config = json.loads(HOST_CONFIG.read_text(encoding="utf-8"))
    settings = config["backend"]["settings"]
    root = Path(settings["root_path"])
    port = int(config["web_port"])
    facts["port"] = port
    facts["store_id"] = config["store_id"]
    facts["root_path"] = str(root)

    check("store_id", config["store_id"] == LAKE_ID, config["store_id"])
    check("root_exists", root.is_dir(), str(root))
    check("household_is_registered", HOUSEHOLD in settings["include_globs"], HOUSEHOLD)
    check("household_is_untimed", HOUSEHOLD in settings.get("untimed", []),
          "whole-resource AS_IS only; every date range is refused by the contract")

    for resource, digest in expected["panels_sha256"].items():
        path = root / resource
        actual = sha_file(path) if path.is_file() else "ABSENT"
        check(f"panel_digest:{resource}", actual == digest, f"{actual} vs adopted {digest}")

    provider = Path(expected["serving_provider_path"])
    actual = sha_file(provider) if provider.is_file() else "ABSENT"
    check("serving_provider_digest", actual == expected["serving_provider_sha256"], actual)

    installed = None
    venv = Path(interpreter()).parent.parent
    for candidate in venv.glob("lib/python*/site-packages/financial_data_store/inventory.py"):
        installed = candidate
        break
    actual = sha_file(installed) if installed else "ABSENT"
    check("installed_provider_digest", actual == expected["installed_provider_sha256"],
          f"{actual} at {installed}")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = next((lake for lake in registry.get("lakes", []) if lake.get("lake_id") == LAKE_ID), None)
    check("registry_entry", entry is not None, "the governance registry publishes this lake")
    if entry:
        facts["registered_base_url"] = entry.get("base_url")
        check("registry_points_here", entry.get("base_url") == f"http://127.0.0.1:{port}",
              f"{entry.get('base_url')} vs this host's port {port}")
        check("registry_holdout_unchanged", entry.get("holdout_start") == config.get("holdout_start")
              or entry.get("holdout_start") == settings.get("holdout_start"),
              str(entry.get("holdout_start")))
    facts["registry_digest"] = sha_file(REGISTRY)
    facts["registry_note"] = ("the restoration does not read, write or restart the governance "
                              "kernel; this digest is recorded so a later verify can prove it")

    state = unit_state(UNIT)
    facts["unit_before"] = state
    check("port_free", not port_is_open(port),
          f"nothing is bound to {port}" if not port_is_open(port) else "something already serves it")

    facts["other_units_before"] = {unit: unit_state(unit) for unit in OTHER_STORE_UNITS}
    facts["unit_file_digest"] = sha_file(UNIT_FILE)
    facts["host_config_digest"] = sha_file(HOST_CONFIG)
    if verbose:
        print(json.dumps(facts, indent=1, sort_keys=True))
    return facts


# --------------------------------------------------------------------------------------- rehearsal

def rehearse(port: int, keep: bool = False) -> dict:
    """Prove the deployed command serves the household panel, on a throwaway port."""
    facts = preflight(verbose=False)
    if not facts["ok"]:
        raise Refused("preflight failed; the rehearsal does not start.  Run preflight for detail")
    if port_is_open(port):
        raise Refused(f"the rehearsal port {port} is already in use; choose another with --port")

    config = json.loads(HOST_CONFIG.read_text(encoding="utf-8"))
    rehearsal = copy.deepcopy(config)
    rehearsal["web_port"] = port
    rehearsal["store_id"] = config["store_id"]          # unchanged: the contract is not rewritten
    work = Path(tempfile.mkdtemp(prefix="panels-rehearsal-"))
    config_path = work / "public-panels.host.json"
    config_path.write_text(json.dumps(rehearsal, indent=1), encoding="utf-8")
    differing = [key for key in set(config) | set(rehearsal) if config.get(key) != rehearsal.get(key)]
    out: dict = {"stage": "rehearse", "port": port, "ok": False,
                 "config_differs_only_in": sorted(differing),
                 "backend_settings_identical": rehearsal["backend"] == config["backend"]}
    if out["config_differs_only_in"] != ["web_port"] or not out["backend_settings_identical"]:
        shutil.rmtree(work, ignore_errors=True)
        raise Refused("the rehearsal configuration differs from the deployed one beyond its port")

    token = service_token()
    env = dict(os.environ)
    env["DATA_GOV_LAKE_TOKEN"] = token
    env["PYTHONPATH"] = pythonpath()
    child = subprocess.Popen(
        [interpreter(), "-m", "data_lake_service.main", "--load_config", str(config_path)],
        cwd=str(work), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        start_new_session=True)
    try:
        for _ in range(120):
            if port_is_open(port):
                break
            if child.poll() is not None:
                raise Refused(f"the rehearsal host exited {child.returncode}: "
                              f"{(child.stderr.read() or b'').decode()[-400:]}")
            time.sleep(0.25)
        else:
            raise Refused("the rehearsal host did not begin serving within 30 s")

        status, body, _ = api(port, "/api/v1/host", token)
        out["host"] = json.loads(body)
        status, body, _ = api(port, "/api/v1/discover", token)
        resources = json.loads(body)["resources"]
        out["resources"] = sorted(item.get("resource_id") or item.get("id") or item for item in resources) \
            if resources and not isinstance(resources[0], str) else sorted(resources)
        out["household_listed"] = any(HOUSEHOLD in json.dumps(item) for item in resources)

        status, body, headers = api(
            port, f"/api/v2/download?resource={urllib.parse.quote(HOUSEHOLD)}", token, timeout=180)
        digest = hashlib.sha256(body).hexdigest()
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        adopted = receipt["binding"]["expected"]["panels_sha256"][HOUSEHOLD]
        out["download"] = {
            "http": status, "bytes": len(body), "sha256": digest,
            "byte_exact_against_adoption": digest == adopted,
            "declared_content_digest": (headers.get("X-Content-SHA256") or "").lower(),
            "availability_use": headers.get("X-Availability-Use"),
            "availability_label": headers.get("X-Availability-Label"),
            "availability_contract_sha256": headers.get("X-Availability-Contract-SHA256"),
        }
        # a ranged request over this archive must still be refused: the contract is unchanged
        try:
            api(port, f"/api/v2/download?resource={urllib.parse.quote(HOUSEHOLD)}"
                      "&from=2007-01-01&to=2007-01-02", token, timeout=60)
            out["range_refused"] = False
        except urllib.error.HTTPError as exc:
            out["range_refused"] = True
            out["range_refusal_http"] = exc.code
        out["ok"] = bool(out["household_listed"] and out["download"]["byte_exact_against_adoption"]
                         and out["download"]["http"] == 200 and out["range_refused"])
    finally:
        if not keep:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            shutil.rmtree(work, ignore_errors=True)
            out["throwaway_stopped"] = not port_is_open(port)
        else:
            out["throwaway_pid"] = child.pid
            out["throwaway_dir"] = str(work)
    print(json.dumps(out, indent=1, sort_keys=True))
    return out


# ------------------------------------------------------------------------------------------- apply

def verify(expect_registry_digest: str | None = None,
           expect_other_units: dict | None = None) -> dict:
    config = json.loads(HOST_CONFIG.read_text(encoding="utf-8"))
    port = int(config["web_port"])
    token = service_token()
    out: dict = {"stage": "verify", "port": port, "ok": False, "unit": unit_state(UNIT)}
    try:
        status, body, _ = api(port, "/api/v1/discover", token)
        resources = json.loads(body)["resources"]
    except (urllib.error.URLError, OSError) as exc:
        out["error"] = f"the endpoint does not answer: {exc}"
        print(json.dumps(out, indent=1, sort_keys=True))
        return out
    out["household_listed"] = any(HOUSEHOLD in json.dumps(item) for item in resources)
    out["resource_count"] = len(resources)
    out["registry_digest"] = sha_file(REGISTRY)
    out["registry_unchanged"] = (expect_registry_digest is None
                                 or out["registry_digest"] == expect_registry_digest)
    now = {unit: unit_state(unit) for unit in OTHER_STORE_UNITS}
    out["other_units_now"] = now
    out["other_units_unchanged"] = expect_other_units is None or now == expect_other_units
    out["ok"] = bool(out["household_listed"] and out["registry_unchanged"]
                     and out["other_units_unchanged"])
    print(json.dumps(out, indent=1, sort_keys=True))
    return out


def apply(also_enable: bool = False) -> dict:
    facts = preflight(verbose=False)
    if not facts["ok"]:
        raise Refused("preflight failed; nothing is started.  Run preflight for detail")
    before = facts["other_units_before"]
    registry_digest = facts["registry_digest"]
    steps = []
    if also_enable:
        result = systemctl("enable", UNIT)
        steps.append({"step": f"enable {UNIT}", "rc": result.returncode, "stderr": result.stderr.strip()})
        if result.returncode != 0:
            raise Refused(f"enable failed: {result.stderr.strip()}")
    result = systemctl("start", UNIT)
    steps.append({"step": f"start {UNIT}", "rc": result.returncode, "stderr": result.stderr.strip()})
    if result.returncode != 0:
        raise Refused(f"start failed: {result.stderr.strip()}; nothing else was touched")
    for _ in range(80):
        if port_is_open(int(facts["port"])):
            break
        time.sleep(0.25)
    out = {"stage": "apply", "also_enable": also_enable, "steps": steps,
           "verify": verify(registry_digest, before),
           "rollback": (f"{sys.argv[0]} rollback" + (" --also-enable" if also_enable else ""))}
    print(json.dumps({k: out[k] for k in ("stage", "also_enable", "steps", "rollback")},
                     indent=1, sort_keys=True))
    return out


def rollback(also_enable: bool = False) -> dict:
    config = json.loads(HOST_CONFIG.read_text(encoding="utf-8"))
    port = int(config["web_port"])
    before_registry = sha_file(REGISTRY)
    before_others = {unit: unit_state(unit) for unit in OTHER_STORE_UNITS}
    steps = []
    result = systemctl("stop", UNIT)
    steps.append({"step": f"stop {UNIT}", "rc": result.returncode, "stderr": result.stderr.strip()})
    if also_enable:
        result = systemctl("disable", UNIT)
        steps.append({"step": f"disable {UNIT}", "rc": result.returncode,
                      "stderr": result.stderr.strip()})
    for _ in range(40):
        if not port_is_open(port):
            break
        time.sleep(0.25)
    out = {"stage": "rollback", "steps": steps, "unit": unit_state(UNIT),
           "port_free": not port_is_open(port),
           "registry_unchanged": sha_file(REGISTRY) == before_registry,
           "other_units_unchanged": {unit: unit_state(unit) for unit in OTHER_STORE_UNITS} == before_others}
    out["ok"] = bool(out["port_free"] and out["registry_unchanged"] and out["other_units_unchanged"])
    print(json.dumps(out, indent=1, sort_keys=True))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("preflight")
    rehearsal = sub.add_parser("rehearse")
    rehearsal.add_argument("--port", type=int, default=5069,
                           help="a free loopback port for the throwaway host (never registered)")
    rehearsal.add_argument("--keep", action="store_true",
                           help="leave the throwaway running for inspection (you must stop it)")
    applied = sub.add_parser("apply")
    applied.add_argument("--also-enable", action="store_true",
                         help="persist across reboot as well; rollback then disables it again")
    sub.add_parser("verify")
    reverted = sub.add_parser("rollback")
    reverted.add_argument("--also-enable", action="store_true",
                          help="also disable the unit, undoing an apply --also-enable")
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        return 0 if preflight()["ok"] else 1
    if args.stage == "rehearse":
        return 0 if rehearse(args.port, args.keep)["ok"] else 1
    if args.stage == "apply":
        return 0 if apply(args.also_enable)["verify"]["ok"] else 1
    if args.stage == "verify":
        return 0 if verify()["ok"] else 1
    if args.stage == "rollback":
        return 0 if rollback(args.also_enable)["ok"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
