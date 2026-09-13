"""C126: the two parser defects of the first build stay fixed, and a build
that fails after it started leaves nothing behind."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_public_contract_parsers", ROOT / "tools/df_public_contract.py")
P = importlib.util.module_from_spec(spec)
sys.modules["df_public_contract_parsers"] = P
spec.loader.exec_module(P)

LICENSE = "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license."


def root_with(tmp_path, logical_id, zip_bytes, page):
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    (raw / "license_evidence").mkdir()
    (raw / "data" / "a.zip").write_bytes(zip_bytes)
    ev = raw / "license_evidence" / "page.html"
    ev.write_text(page)
    entry = {"logical_id": logical_id, "citation": "c", "doi": "d", "upstream_owner": "o",
             "files": [{"name": "a.zip", "bytes": len(zip_bytes), "sha256": hashlib.sha256(zip_bytes).hexdigest(),
                        "retrieved_at_utc": "2026-09-13T00:00:00Z"}],
             "license": {"url": "u", "evidence": {"file": "license_evidence/page.html",
                                                  "sha256": hashlib.sha256(ev.read_bytes()).hexdigest(),
                                                  "fetched_from": "f", "statement": "CC BY 4.0"}}}
    (raw / "PUBLIC_RAW_MANIFEST.json").write_text(json.dumps({"datasets": [entry]}))
    return raw


def zipped(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_comma_decimals_parse_as_numbers(tmp_path):
    text = ('"";"MT_001";"MT_002"\n'
            '"2011-01-01 00:15:00";0;0\n'
            '"2011-01-01 00:30:00";3,80710659898477;22,7596017069701\n'
            '"2011-01-01 00:45:00";2,5;1\n')
    page = (f"<p>{LICENSE}</p><p>This data set contains electricity consumption of 370 points/clients.</p>"
            "<p>Values are in kW of each 15 min.</p><p>All time labels report to Portuguese hour.</p>")
    raw = root_with(tmp_path, "uci_321_electricityloaddiagrams20112014", zipped({"LD2011_2014.txt": text}), page)
    c = P.build("uci_321_electricityloaddiagrams20112014", raw, tmp_path / "panels")
    t = pq.read_table(tmp_path / "panels/uci_321_electricityloaddiagrams20112014/panel.parquet")
    assert t.column("MT_001").to_pylist() == [0.0, 3.80710659898477, 2.5]
    assert c["variables"][0]["unit"]["value"] == "kW"
    assert c["time"]["timezone"].startswith("Europe/Lisbon")


def test_text_columns_stay_text_under_the_pandas_string_dtype(tmp_path):
    header = '"No","year","month","day","hour","PM2.5","wd","WSPM","station"\n'
    a = header + '1,2013,3,1,0,4,"NNW",4.4,"A"\n2,2013,3,1,1,NA,NA,4.7,"A"\n3,2013,3,1,2,8,"N",1.0,"A"\n'
    b = header + '1,2013,3,1,0,5,"E",1.1,"B"\n2,2013,3,1,1,6,"SE",1.2,"B"\n3,2013,3,1,2,7,"S",1.3,"B"\n'
    inner = zipped({"PRSA/PRSA_Data_A.csv": a, "PRSA/PRSA_Data_B.csv": b})
    page = (f"<p>{LICENSE}</p><p>This data set includes hourly air pollutants data from 12 nationally-controlled "
            "air-quality monitoring sites.</p><p>PM2.5: PM2.5 concentration (ug/m^3)</p><p>Missing data are denoted as NA.</p>")
    raw = root_with(tmp_path, "uci_501_beijing_multisite_air_quality", zipped({"PRSA2017.zip": inner}), page)
    c = P.build("uci_501_beijing_multisite_air_quality", raw, tmp_path / "panels")
    t = pq.read_table(tmp_path / "panels/uci_501_beijing_multisite_air_quality/panel.parquet")
    assert t.column("A__wd").to_pylist() == ["NNW", None, "N"]
    assert t.column("A__PM2.5").to_pylist() == [4.0, None, 8.0]
    v = {x["name"]: x for x in c["variables"]}
    assert v["A__wd"]["physical_type"] == "string" and v["A__wd"]["license_state"] == "TERMS_REQUIRE_REVIEW"
    assert v["A__PM2.5"]["unit"]["value"] == "ug/m3"


def test_a_failure_after_the_build_started_leaves_nothing(tmp_path, monkeypatch):
    text = '"";"MT_001"\n"2011-01-01 00:15:00";1,5\n"2011-01-01 00:30:00";2,5\n"2011-01-01 00:45:00";3,5\n'
    raw = root_with(tmp_path, "uci_321_electricityloaddiagrams20112014", zipped({"LD2011_2014.txt": text}),
                    f"<p>{LICENSE}</p>")

    def boom(doc):
        raise P.C.ContractRefusal(["forced failure after the panel was written"])

    monkeypatch.setattr(P.C, "seal", boom)
    with pytest.raises(P.C.ContractRefusal, match="forced failure"):
        P.build("uci_321_electricityloaddiagrams20112014", raw, tmp_path / "panels")
    assert list((tmp_path / "panels").iterdir()) == []
