"""The four situations a fixture manifest can be found in, each with a named outcome.

R2 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "manifest-only checkout, missing artifact, mismatched artifact and fresh checkout must
     each yield regeneration into temporary storage or an explicit diagnostic, never
     accidental FileNotFoundError. Verify generated identities against the published
     manifest; never overwrite original evidence."

Every rule here runs the real generator and the real resolver; the fixtures are small
(24 rows) so the whole file is a fraction of a second.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "tools"))

from ensure_consumer_fixtures import FixtureError, ensure  # noqa: E402
import make_consumer_fixtures  # noqa: E402

ROWS = 24


@pytest.fixture
def published(tmp_path):
    """A published fixture set: the CSVs and the manifest that describes them."""
    out = tmp_path / "lake"
    make_consumer_fixtures.main(["--out", str(out), "--rows", str(ROWS)])
    return out


def manifest_of(root: Path) -> dict:
    return json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))


def test_a_complete_checkout_verifies_every_file_without_regenerating(published):
    report = ensure(published / "MANIFEST.json")
    assert report["work_dir"] is None, "nothing needed rebuilding"
    assert set(report["verdicts"].values()) == {"present_and_verified"}
    assert not report["missing"]


def test_a_manifest_only_checkout_regenerates_into_temporary_storage(tmp_path, published):
    """The case that used to end in FileNotFoundError: evidence committed, data absent."""
    alone = tmp_path / "checkout"
    alone.mkdir()
    shutil.copy(published / "MANIFEST.json", alone / "MANIFEST.json")

    work = tmp_path / "work"
    report = ensure(alone / "MANIFEST.json", work_dir=work)
    assert set(report["verdicts"].values()) == {"regenerated"}
    assert not report["missing"], "every declared file must be resolvable"
    assert report["work_dir"] == str(work), "regeneration goes to temporary storage"
    for name, path in report["resolved"].items():
        assert Path(path).parent == work
        assert Path(path).is_file()


def test_the_regenerated_bytes_are_verified_against_the_published_digests(tmp_path,
                                                                          published):
    """Regeneration is worthless unless the identity is checked, so it is checked."""
    import hashlib

    alone = tmp_path / "checkout"
    alone.mkdir()
    shutil.copy(published / "MANIFEST.json", alone / "MANIFEST.json")
    report = ensure(alone / "MANIFEST.json", work_dir=tmp_path / "work")
    declared = manifest_of(published)["files"]
    for name, path in report["resolved"].items():
        rebuilt = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert rebuilt == declared[name]["sha256"], f"{name} was rebuilt with other bytes"


def test_one_missing_artifact_is_rebuilt_and_the_others_are_left_alone(tmp_path, published):
    victim = "synthetic_ohlc_1h.csv"
    (published / victim).unlink()
    report = ensure(published / "MANIFEST.json", work_dir=tmp_path / "work")
    assert report["verdicts"][victim] == "regenerated"
    others = {name: verdict for name, verdict in report["verdicts"].items() if name != victim}
    assert set(others.values()) == {"present_and_verified"}


def test_a_mismatched_artifact_is_never_overwritten(tmp_path, published):
    """The original bytes are evidence of something, even when they are wrong."""
    victim = published / "synthetic_ohlc_1h.csv"
    tampered = victim.read_text(encoding="utf-8") + "9999,1,1,1,1\n"
    victim.write_text(tampered, encoding="utf-8")

    report = ensure(published / "MANIFEST.json", work_dir=tmp_path / "work")
    assert report["verdicts"]["synthetic_ohlc_1h.csv"] == "mismatched_original_preserved"
    assert victim.read_text(encoding="utf-8") == tampered, "the original was modified"
    resolved = Path(report["resolved"]["synthetic_ohlc_1h.csv"])
    assert resolved.parent == tmp_path / "work", "the good copy lives in temporary storage"


def test_a_manifest_that_is_not_there_is_a_named_refusal_not_a_traceback(tmp_path):
    with pytest.raises(FixtureError, match="no fixture manifest"):
        ensure(tmp_path / "nope" / "MANIFEST.json")


def test_a_manifest_with_no_files_is_refused(tmp_path):
    empty = tmp_path / "MANIFEST.json"
    empty.write_text(json.dumps({"schema": "consumer_fixtures.v1", "files": {}}),
                     encoding="utf-8")
    with pytest.raises(FixtureError, match="declares no files"):
        ensure(empty)


def test_a_changed_generator_makes_regeneration_unreproducible_and_says_so(tmp_path,
                                                                          published):
    """If the generator is not the one that produced the identities, rebuilding proves nothing.

    The manifest's `generator_sha256` is what decides this, so the rule edits that field
    rather than the tool: no file of this repository is touched by the test.
    """
    alone = tmp_path / "checkout"
    alone.mkdir()
    manifest = manifest_of(published)
    manifest["generator_sha256"] = "0" * 64
    (alone / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = ensure(alone / "MANIFEST.json", work_dir=tmp_path / "work")
    assert set(report["verdicts"].values()) == {"unreproducible"}
    assert report["generator_reproducible"] is False
    assert report["missing"], "an unreproducible fixture is not resolved, and is reported"
