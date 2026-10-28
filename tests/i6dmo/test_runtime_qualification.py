"""Runtime qualification of the closed MG2/MG12 baseline and the public producer derivative.

MUSASHI_AUDIT_20261015 finding 1 / MUSASHI_TO_SATOSHI_MATCHED_AUDIT_FOLLOWUP_20261015 section 3.
Pure file checks: no fit, no GPU, no TEST, no network. The retained-receipt check runs only when
``I6DMO_CAMPAIGN_ROOT`` points at the private campaign root (it is skipped otherwise).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.i6dmo import design as D, provenance
from tools.i6dmo import runtime_qualification as RQ

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs/i6n/direct_multi_output_static"
SCAN_ROOTS = (DOCS, REPO / "docs/tres_temas_entrevista")
SUFFIXES = (".md", ".json", ".txt", ".csv", ".py", ".tex")
ORIGINAL = REPO / "tools/i6dmo/producer_recovery.v1.json"
DERIVATIVE = DOCS / "producer_recovery.v1.public.json"
#: sha256 the closed MG2/MG12 bundles (21400c51 / c3867b5c) record in code.files_sha256.
CLOSED_BUNDLE_PIN = "8d2289de0b7e6dfbb46d304b91adb1bf6f5f85e36d3c5001942e4b558985633c"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_files():
    return [p for root in SCAN_ROOTS for p in root.rglob("*") if p.is_file() and p.suffix in SUFFIXES]


# ------------------------------------------------------------------ qualification record

def test_record_is_sealed_and_labelled():
    rec = RQ.load_record()
    assert rec["label"] == "RUNTIME_SEAL_NONCONFORMANT_OBSERVED_TF_2_21"
    assert any("cannot attest compliance" in c for c in rec["consequences"])
    assert any("runtime-matched donor" in c for c in rec["consequences"])
    assert "no reseal of EXPERIMENT.mg2.json/EXPERIMENT.mg12.json" in rec["not_done"]
    for fit in ("short.fit", "long.fit"):
        assert rec["observed_runtime"][fit]["tensorflow_version"] == "2.21.0"


def test_record_cites_the_untouched_closed_design_seals():
    rec = RQ.load_record()
    for fam, name in (("short", "EXPERIMENT.mg2.json"), ("long", "EXPERIMENT.mg12.json")):
        declared = rec["declared_runtime"][fam]
        doc = D.load_design(DOCS / name)
        assert _sha(DOCS / name) == declared["design_file_sha256"]
        assert D.design_sha256(doc) == declared["design_sha256"]
        assert doc["seal_sha256"] == declared["seal_sha256"]
        assert doc["sealed_identities"]["runtime"] == declared["runtime"] == \
            {"tensorflow": "2.18.0", "keras": "3.9.0"}
    assert rec["observed_runtime"]["short.fit"]["receipt_design_sha256"] == \
        rec["declared_runtime"]["short"]["design_sha256"]
    assert rec["observed_runtime"]["long.fit"]["receipt_design_sha256"] == \
        rec["declared_runtime"]["long"]["design_sha256"]


def test_tampered_record_is_refused(tmp_path):
    rec = json.loads(RQ.RECORD_PATH.read_text())
    rec["observed_runtime"]["short.fit"]["tensorflow_version"] = "2.18.0"
    bad = tmp_path / RQ.RECORD_NAME
    bad.write_text(json.dumps(rec))
    with pytest.raises(RQ.QualificationError) as err:
        RQ.load_record(bad)
    assert err.value.code == "QUALIFICATION_SEAL"


@pytest.mark.skipif(not os.environ.get("I6DMO_CAMPAIGN_ROOT"),
                    reason="private campaign root not provided (I6DMO_CAMPAIGN_ROOT)")
def test_retained_receipts_match_the_record_byte_for_byte():
    root = Path(os.environ["I6DMO_CAMPAIGN_ROOT"])
    rec = RQ.load_record()
    for fit in ("short.fit", "long.fit"):
        obs = rec["observed_runtime"][fit]
        path = root / obs["receipt"].replace("<campaign-root>/", "")
        assert _sha(path) == obs["receipt_sha256"]
        receipt = json.loads(path.read_text())
        assert receipt["resources"]["tensorflow_version"] == obs["tensorflow_version"] == "2.21.0"
        assert receipt["identity"]["design_sha256"] == obs["receipt_design_sha256"]
        assert receipt["scientific"] is True and receipt["device"] == "gpu"
        terminal = path.parents[2] / "TERMINAL.json"
        assert _sha(terminal) == obs["terminal_sha256"]
    source = root / rec["observed_runtime"]["short.fit"]["same_bytes_as_source_attempt"].split(" ")[0] \
        .replace("<campaign-root>/", "")
    assert _sha(source) == rec["observed_runtime"]["short.fit"]["receipt_sha256"]


def test_matched_reseal_is_a_distinct_conforming_case():
    rec = RQ.load_record()
    distinct = rec["distinct_from_matched_reseal"]
    assert "BEFORE any execution" in distinct["statement"]
    for arm, unit in (("mg1", "mg1.fit"), ("mg2m", "mg2m.fit")):
        doc = D.load_design(DOCS / f"matched_short/EXPERIMENT.{arm}.json")
        cur = distinct["matched_current_receipts"][unit]
        assert D.design_sha256(doc) == cur["design_sha256"]
        assert doc["sealed_identities"]["runtime"]["tensorflow"] == cur["declared"] == \
            cur["tensorflow_version"] == "2.21.0"
        assert doc["seal_sha256"] not in D.MATCHED_SUPERSEDED_SEALS
    assert (REPO / distinct["matched_equality_witness"]).is_file()
    assert (REPO / distinct["matched_supersession_note"]).is_file()


# ------------------------------------------------------------------ citation gate

def test_every_citing_document_carries_the_qualification():
    rec = RQ.load_record()
    assert RQ.citation_problems(_scan_files(), rec) == []
    index = (DOCS / "RESULTS_INDEX.md").read_text()
    assert RQ.RECORD_NAME in index and "RUNTIME_SEAL_NONCONFORMANT_OBSERVED_TF_2_21" in index


def test_a_comparison_citing_the_closed_baseline_without_the_qualification_fails(tmp_path):
    rec = RQ.load_record()
    bare = tmp_path / "NEW_COMPARISON.md"
    bare.write_text("Baseline: closed campaign i6dmo-9e1ebad89ff2, MG2 6h/12h 0.049615%/0.093170%.\n")
    assert RQ.citation_problems([bare], rec, root=tmp_path) == ["NEW_COMPARISON.md"]
    bare.write_text(bare.read_text() + f"Qualified: {RQ.RECORD_NAME}\n")
    assert RQ.citation_problems([bare], rec, root=tmp_path) == []
    unrelated = tmp_path / "OTHER.md"
    unrelated.write_text("Matched MG1 vs MG2m on population ba01f87a.\n")
    assert RQ.citation_problems([unrelated], rec, root=tmp_path) == []


# ------------------------------------------------------------------ producer record

def test_producer_original_is_byte_identical_and_still_the_pinned_object():
    assert _sha(ORIGINAL) == CLOSED_BUNDLE_PIN
    assert provenance.PRODUCERS_PATH.name == "producer_recovery.v1.json"
    try:
        historic = subprocess.run(["git", "show", "0245c1f8:tools/i6dmo/producer_recovery.v1.json"],
                                  cwd=REPO, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git history unavailable")
    assert hashlib.sha256(historic).hexdigest() == CLOSED_BUNDLE_PIN


def test_public_derivative_points_at_the_original_and_never_claims_its_digest():
    pub = json.loads(DERIVATIVE.read_text())
    assert pub["original_sha256"] == _sha(ORIGINAL)
    assert pub["original_path"] == "tools/i6dmo/producer_recovery.v1.json"
    assert _sha(DERIVATIVE) != pub["original_sha256"]
    assert "0245c1f8" in pub["original_retention"] and "NOT sanitized" in pub["original_retention"]
    assert "never be presented" in pub["digest_warning"]
    # the derivative equals the original except where a role alias replaced a host token
    original = json.loads(ORIGINAL.read_text())

    def leaves(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from leaves(v, f"{path}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                yield from leaves(v, f"{path}[{i}]")
        else:
            yield path, x
    got, want = dict(leaves(pub["record"])), dict(leaves(original))
    assert set(got) == set(want)
    aliases = re.compile(r"<host-preferred>|<host-secondary>|<coordinator>|<host>|<home>")
    replaced = 0
    for path, value in want.items():
        if not isinstance(value, str):
            assert got[path] == value, path
            continue
        parts = re.split(f"({aliases.pattern})", got[path])
        pattern = "".join(r"[A-Za-z0-9._-]+" if aliases.fullmatch(part) else re.escape(part)
                          for part in parts)
        assert re.fullmatch(pattern, value), path
        replaced += len(aliases.findall(got[path]))
    assert replaced == pub["redaction"]["replacements"] > 0


def test_no_code_reads_the_derivative():
    hits = [p for p in (REPO / "tools").rglob("*.py")
            if "producer_recovery.v1.public" in p.read_text(errors="ignore")]
    assert hits == []
