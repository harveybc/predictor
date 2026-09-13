"""C126 x C129: public archives become canonical panels and sealed contracts
only from verified bytes, and units come only from found statements."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_public_contract", ROOT / "tools/df_public_contract.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

PAGE_374 = """<html><body><p>This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.</p>
<p>The data set is at 10 min for about 4.5 months.</p><p>Appliances, energy use in Wh</p>
<p>T1, Temperature in kitchen area, in Celsius</p><p>rv1, Random variable 1, nondimensional</p>
<p>downloaded from a public data set from Reliable Prognosis (rp5.ru)</p></body></html>"""
CSV_374 = ('"date","Appliances","T1","T_out","rv1"\n'
           '"2016-01-11 17:00:00","  60","19.89","7.0","13.27"\n'
           '"2016-01-11 17:10:00","  60","19.89","6.8","18.60"\n'
           '"2016-01-11 17:20:00","  50","19.89","","28.64"\n' * 1)


def make_root(tmp_path, csv_text=CSV_374, page=PAGE_374, corrupt=False):
    raw = tmp_path / "raw"
    (raw / "data").mkdir(parents=True)
    (raw / "license_evidence").mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("energydata_complete.csv", csv_text)
    zb = buf.getvalue()
    name = "uci_374_appliances_energy__appliances.zip"
    (raw / "data" / name).write_bytes(zb)
    ev = raw / "license_evidence" / "uci_374_page.html"
    ev.write_text(page)
    entry = {"logical_id": "uci_374_appliances_energy", "citation": "Candanedo (2017)", "doi": "10.24432/C5VC8G",
             "upstream_owner": "Candanedo", "files": [{"name": name, "bytes": len(zb),
             "sha256": ("0" * 64) if corrupt else hashlib.sha256(zb).hexdigest(), "retrieved_at_utc": "2026-09-13T00:00:00Z"}],
             "license": {"url": "https://creativecommons.org/licenses/by/4.0/legalcode",
                         "evidence": {"file": "license_evidence/uci_374_page.html", "sha256": hashlib.sha256(ev.read_bytes()).hexdigest(),
                                      "fetched_from": "https://archive.ics.uci.edu/dataset/374", "statement": "CC BY 4.0"}}}
    (raw / "PUBLIC_RAW_MANIFEST.json").write_text(json.dumps({"datasets": [entry]}))
    return raw


def test_a_public_dataset_becomes_a_panel_and_a_sealed_contract(tmp_path):
    raw = make_root(tmp_path)
    c = P.build("uci_374_appliances_energy", raw, tmp_path / "panels")
    assert P.C.validate(c) == []
    v = {x["name"]: x for x in c["variables"]}
    assert v["Appliances"]["unit"]["value"] == "Wh" and "Appliances, energy use in Wh" in v["Appliances"]["unit"]["evidence"][0]["source"]
    assert v["T1"]["unit"]["value"] == "degC"
    # stated nowhere in this page: no unit is invented from the column name
    assert v["T_out"]["unit"]["value"] == "UNKNOWN"
    assert v["T_out"]["license_state"] == "TERMS_REQUIRE_REVIEW"
    assert v["rv1"]["role"] == "EXCLUDED"
    table = pq.read_table(tmp_path / "panels/uci_374_appliances_energy/panel.parquet")
    assert table.num_rows == 3 and table.column("T_out").to_pylist()[2] is None
    roles = {f["role"] for f in c["files"]}
    assert roles == {"RAW_ARCHIVE", "LICENSE_EVIDENCE", "DERIVED_CANONICAL_PANEL", "PARSE_RECEIPT"}
    assert c["original_fields"]["parse_receipt"]["rows_filled"] == 0


def test_bytes_that_differ_from_custody_refuse(tmp_path):
    raw = make_root(tmp_path, corrupt=True)
    with pytest.raises(P.C.ContractRefusal, match="differ from the custody manifest"):
        P.build("uci_374_appliances_energy", raw, tmp_path / "panels")


def test_a_page_without_the_license_statement_refuses(tmp_path):
    raw = make_root(tmp_path, page=PAGE_374.replace("Creative Commons Attribution 4.0 International", "something else"))
    with pytest.raises(P.C.ContractRefusal, match="license statement is not in the kept evidence page"):
        P.build("uci_374_appliances_energy", raw, tmp_path / "panels")


def test_backwards_time_refuses_and_nothing_is_reordered(tmp_path):
    lines = CSV_374.splitlines()
    raw = make_root(tmp_path, csv_text="\n".join([lines[0], lines[2], lines[1]]) + "\n")
    with pytest.raises(P.C.ContractRefusal, match="go backwards"):
        P.build("uci_374_appliances_energy", raw, tmp_path / "panels")


def test_panels_are_write_once(tmp_path):
    raw = make_root(tmp_path)
    P.build("uci_374_appliances_energy", raw, tmp_path / "panels")
    with pytest.raises(P.C.ContractRefusal, match="write-once"):
        P.build("uci_374_appliances_energy", raw, tmp_path / "panels")
