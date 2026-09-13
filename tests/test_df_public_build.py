"""C126: one dataset's refusal never stops the others; every outcome is in a
write-once receipt; a backwards series refuses with its anomalies listed."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_public_build", ROOT / "tools/df_public_build.py")
B = importlib.util.module_from_spec(spec)
sys.modules["df_public_build"] = B
spec.loader.exec_module(B)

PAGE = """<html><body><p>This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.</p>
<p>The data set is at 10 min for about 4.5 months.</p><p>Appliances, energy use in Wh</p></body></html>"""
HEADER = '"date","Appliances"\n'


def raw_root(tmp_path, rows):
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    (raw / "license_evidence").mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("energydata_complete.csv", HEADER + "".join(f'"{t}","{v}"\n' for t, v in rows))
    zb = buf.getvalue()
    (raw / "data" / "a.zip").write_bytes(zb)
    ev = raw / "license_evidence" / "page.html"
    ev.write_text(PAGE)
    entry = {"logical_id": "uci_374_appliances_energy", "citation": "c", "doi": "d", "upstream_owner": "o",
             "files": [{"name": "a.zip", "bytes": len(zb), "sha256": hashlib.sha256(zb).hexdigest(),
                        "retrieved_at_utc": "2026-09-13T00:00:00Z"}],
             "license": {"url": "u", "evidence": {"file": "license_evidence/page.html",
                                                  "sha256": hashlib.sha256(ev.read_bytes()).hexdigest(),
                                                  "fetched_from": "f", "statement": "CC BY 4.0"}}}
    (raw / "PUBLIC_RAW_MANIFEST.json").write_text(json.dumps({"datasets": [entry]}))
    return raw


GOOD = [("2016-01-11 17:00:00", "60"), ("2016-01-11 17:10:00", "50"), ("2016-01-11 17:20:00", "40")]


def test_a_refusal_or_failure_never_stops_the_others(tmp_path):
    raw = raw_root(tmp_path, GOOD)
    r = B.build_all(raw, tmp_path / "panels", ["uci_235_individual_household_power", "uci_374_appliances_energy"])
    status = {x["logical_id"]: x["status"] for x in r["results"]}
    assert status == {"uci_235_individual_household_power": "FAILED", "uci_374_appliances_energy": "BUILT"}
    assert r["counts"] == {"BUILT": 1, "REFUSED": 0, "FAILED": 1}
    assert (tmp_path / "panels/BUILD_RECEIPT.json").is_file()


def test_backwards_time_refuses_with_every_anomaly_listed(tmp_path):
    rows = [("2016-01-11 17:00:00", "60"), ("2016-01-11 17:10:00", "50"), ("2016-01-11 16:50:00", "40"),
            ("2016-01-11 17:00:00", "30")]
    raw = raw_root(tmp_path, rows)
    r = B.build_all(raw, tmp_path / "panels", ["uci_374_appliances_energy"])
    res = r["results"][0]
    assert res["status"] == "REFUSED" and "go backwards at 1 rows" in res["problems"][0]
    anomalies = json.loads(res["problems"][1].split("anomalies: ", 1)[1])
    assert anomalies[0]["row"] == 2 and anomalies[0]["jump_seconds"] == -1200.0
    assert not (tmp_path / "panels/uci_374_appliances_energy").exists()


def test_the_receipt_is_write_once(tmp_path):
    raw = raw_root(tmp_path, GOOD)
    B.build_all(raw, tmp_path / "panels", ["uci_374_appliances_energy"])
    with pytest.raises(SystemExit, match="write-once"):
        B.build_all(raw, tmp_path / "panels", ["uci_374_appliances_energy"])
