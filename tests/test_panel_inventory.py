"""C120: the panel inventory is reproducible, takes the successor from the
join, never counts what the join did not admit, and downloads nothing."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("panel_inventory", ROOT / "tools/panel_inventory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


I = _load()


def _roots_present() -> bool:
    return (Path.home() / ".local/share/agent-multi/t2_public_raw").is_dir() and I.LEDGER.is_file()


pytestmark = pytest.mark.skipif(not _roots_present(), reason="custodied data roots not on this host")


@pytest.fixture(scope="module")
def doc():
    return I.build()


def test_the_inventory_is_reproducible_and_path_free(doc):
    assert "generated_at_utc" not in doc
    assert json.dumps(I.build(), sort_keys=True, default=str) == json.dumps(doc, sort_keys=True, default=str)
    assert "/home/" not in json.dumps(doc, default=str)


def test_the_successor_comes_from_the_join(doc):
    succ = next(p for p in doc["panels"] if p["panel_id"] == "eth_h4_successor")
    facts = succ["artifacts_present_for_join"]
    assert facts["source"]["sha256"] == hashlib.sha256(I.LEDGER.read_bytes()).hexdigest()
    assert facts["candidates"] == 89 and facts["eligible_variables"] == 0
    assert "in_progress_artifacts_seen_not_read" not in succ
    assert succ["counts_toward_six"] is False and succ["license"] == "UNKNOWN"
    assert any("member-by-member join" in w for w in succ["why_not"])


def test_bank_is_insufficient_with_the_exact_deficit_and_no_download(doc):
    bank = doc["bank"]
    assert doc["counts_toward_six_total"] == 0 == bank["qualifying_panels"]
    assert bank["verdict"] == "BANK_INSUFFICIENT"
    assert bank["deficit"] == {"panels_required": 6, "panels_qualifying": 0, "panels_missing": 6,
                               "eligible_variables_required": 30, "eligible_variables_in_qualifying_panels": 0}
    assert bank["download_authorization"]["used"] is False and bank["download_authorization"]["bytes_downloaded"] == 0


def test_shortlist_licenses_are_read_from_the_listings(doc):
    sl = doc["bank"]["shortlist_not_downloaded"]
    assert [s["record"] for s in sl] == list(I.SHORTLIST_IDS) and not doc["bank"]["shortlist_ids_not_found_on_disk"]
    for s in sl:
        assert s["downloaded"] is False and s["license"] == "cc-by-4.0" and s["bytes"] > 0
        listing = Path.home() / ".local/share/agent-multi" / s["listing"]["file"]["relative_path"]
        assert hashlib.sha256(listing.read_bytes()).hexdigest() == s["listing"]["sha256"]


def test_a_modified_scan_refuses(monkeypatch):
    monkeypatch.setattr(I, "SCAN_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="not the reviewed draft"):
        I.build()


def test_the_inventory_never_counts_what_the_join_did_not_admit(monkeypatch):
    scan = I.load_scan()
    real_build = scan.build

    def promoted():
        d = real_build()
        next(p for p in d["panels"] if p["panel_id"] == "monash_pedestrian_counts")["counts_toward_six"] = True
        return d

    monkeypatch.setattr(scan, "build", promoted)
    monkeypatch.setattr(I, "load_scan", lambda: scan)
    with pytest.raises(SystemExit, match="did not admit"):
        I.build()


def test_a_ledger_for_another_census_refuses(tmp_path):
    pop = json.loads(I.LEDGER.read_text())
    pop["inputs"]["census"]["sha256"] = "0" * 64
    forged = tmp_path / "ledger.json"
    forged.write_text(json.dumps(pop))
    with pytest.raises(SystemExit, match="today's successor census"):
        I.build(forged)


def test_write_is_write_once(tmp_path):
    out = tmp_path / "PANEL_INVENTORY.v1.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="write-once"):
        I.main(["--out", str(out)])
