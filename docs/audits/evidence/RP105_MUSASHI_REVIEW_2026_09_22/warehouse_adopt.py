"""Narrow BIGINT maintenance: inspect, held snapshot, copy rehearsal, restart, compare.

No training. Receipts/backups are private. Never prints environment credentials.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


UNIT = "crispdm-data-warehouse-olap.service"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--env-file", type=Path, required=True)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--adopt", action="store_true")
    a = p.parse_args()
    token = None
    for line in a.env_file.read_text().splitlines():
        if line.startswith("DATA_GOV_LAKE_TOKEN="):
            token = shlex.split(line.split("=", 1)[1])[0]
    assert token, "service token unavailable"
    config = json.loads(a.config.read_text())
    settings = config["backend"]["settings"]
    schema = settings.get("schema", "main")
    source = Path(settings["duckdb_path"]).expanduser()
    import predictor_duckdb_store.provider as provider
    import duckdb
    from sqlalchemy import text
    installed = Path(provider.__file__)
    expected = a.repo / "olap/duckdb_store/src/predictor_duckdb_store/provider.py"
    assert installed.read_bytes() == expected.read_bytes(), "installed provider/source differ"
    freeze = module("freeze_live", a.repo / "tools/olap_freeze_live_evidence.py")
    migrate = module("duck_migrate", a.repo / "tools/olap_duckdb_migrate.py")
    os.environ["MUSASHI_WAREHOUSE_TOKEN"] = token
    service = freeze.Service("http://127.0.0.1:5057", "MUSASHI_WAREHOUSE_TOKEN", schema)

    def inspect_live():
        tables = {}
        for relation in migrate.GOVERNANCE:
            cols = service.columns(relation)
            tables[relation] = {"count": service.count(relation), "digest": service.digest(relation, cols)}
        return {"tables": tables, "bytes_type": service.query(
            "SELECT data_type FROM information_schema.columns WHERE "
            f"table_schema='{schema}' AND table_name='gov_terminal_artifact' AND column_name='bytes' LIMIT 1")[0]["data_type"]}

    before = inspect_live()
    print(json.dumps(before), flush=True)
    if not a.adopt:
        return
    a.state_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(a.state_dir, 0o700)
    receipt = {"before": before, "provider_sha256": hashlib.sha256(installed.read_bytes()).hexdigest(),
               "state": "STARTED", "started_unix": time.time()}
    stopped = False
    activated = False
    backup = a.state_dir / "snapshot.duckdb"
    original_provider = subprocess.check_output(["git", "-C", str(a.repo), "show",
        "fff9575:olap/duckdb_store/src/predictor_duckdb_store/provider.py"])
    (a.state_dir / "provider.previous.py").write_bytes(original_provider)
    (a.state_dir / "provider.candidate.py").write_bytes(installed.read_bytes())

    def ctl(action):
        subprocess.run(["systemctl", "--user", action, UNIT], check=True, timeout=45)

    def wait_live():
        for _ in range(25):
            try:
                return inspect_live()
            except Exception:
                time.sleep(1)
        raise RuntimeError("warehouse did not return verified queries")

    try:
        ctl("stop")
        stopped = True
        receipt["snapshot"] = migrate.snapshot_database(str(source), str(backup), schema=schema,
            expect_terminals=before["tables"]["gov_terminal"]["count"], owner_stopped=True)
        assert receipt["snapshot"]["verified"], "snapshot verification failed"
        rehearsal = a.state_dir / "rehearsal.duckdb"
        shutil.copy2(backup, rehearsal)
        store = provider.PredictorDuckdbStore()
        store.set_params(**{**settings, "duckdb_path": str(rehearsal), "threads": 1, "memory_limit": "512MB"})
        store.engine()
        with store.engine().connect() as conn:
            kind = conn.execute(text("SELECT data_type FROM information_schema.columns WHERE "
                f"table_schema='{schema}' AND table_name='gov_terminal_artifact' AND column_name='bytes'")).scalar()
            assert kind == "BIGINT", kind
            conn.rollback()
            with conn.begin():
                conn.execute(text(f'INSERT INTO "{schema}".gov_terminal_artifact '
                    f'SELECT terminal_sha256, role || :suffix, sha256, :bytes FROM "{schema}".gov_terminal_artifact LIMIT 1'),
                    {"suffix": "_musashi_bigint_probe", "bytes": 4198064038})
                assert conn.execute(text(f'SELECT bytes FROM "{schema}".gov_terminal_artifact WHERE role LIKE :role'),
                                    {"role": "%_musashi_bigint_probe"}).scalar() == 4198064038
                conn.rollback()
        store.engine().dispose()
        con = duckdb.connect(str(rehearsal), read_only=True, config={"threads": 1, "memory_limit": "512MB"})
        counts, digests = migrate._governance_digests(con, schema)
        con.close()
        assert counts == receipt["snapshot"]["source_counts"], "rehearsal changed counts"
        assert digests == receipt["snapshot"]["source_digests"], "rehearsal changed content"
        receipt["rehearsal"] = {"type": kind, "big_artifact_bytes_roundtrip": 4198064038,
                                "counts_equal": True, "contents_equal": True}
        activated = True
        ctl("start")
        stopped = False
        after = wait_live()
        assert after["bytes_type"] == "BIGINT", after["bytes_type"]
        assert after["tables"] == before["tables"], "live history changed during maintenance"
        receipt.update(state="ADOPTED_VERIFIED", after=after)
        rehearsal.unlink()
    except Exception as exc:
        receipt.update(state="FAILED", error=f"{type(exc).__name__}: {exc}")
        ctl("stop")
        stopped = True
        if receipt.get("snapshot", {}).get("verified") and not activated:
            # The single writer is stopped. Preserve failed candidate files before restoring.
            for suffix in ("", ".wal"):
                path = Path(str(source) + suffix)
                if path.exists():
                    shutil.copy2(path, a.state_dir / ("failed-cube.duckdb" + suffix))
                    path.unlink()
                saved = Path(str(backup) + suffix)
                if saved.exists():
                    shutil.copy2(saved, path)
        elif activated:
            # New accepted writes may already exist. Never replace them with a pre-start snapshot.
            receipt["database_rollback"] = "NOT_REPLACED: candidate service may have accepted new writes"
        installed.write_bytes(original_provider)
        for bytecode in (installed.parent / "__pycache__").glob("provider.*.pyc"):
            bytecode.unlink()
        ctl("start")
        stopped = False
        receipt["rollback"] = wait_live()
        receipt["rollback_verified"] = receipt["rollback"]["tables"] == before["tables"]
        raise
    finally:
        if stopped:
            ctl("start")
        receipt["finished_unix"] = time.time()
        (a.state_dir / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"state": receipt["state"], "elapsed_seconds": receipt["finished_unix"] - receipt["started_unix"],
                      "type": receipt["after"]["bytes_type"], "history_unchanged": True}), flush=True)


if __name__ == "__main__":
    main()
