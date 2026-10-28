"""Superseded results remain inspectable and cannot be overwritten."""

import json

import pytest

from tools.i6a_archive_revision import main


def test_archive_retains_exact_bytes_and_refuses_second_move(tmp_path):
    source = tmp_path / "ARCH_C_val2024_week7.json"
    source.write_bytes(b'{"old":true}\n')
    assert main(["--results-dir", str(tmp_path)]) == 0
    archive = tmp_path / "archive" / "ARCH_C_pre_origin_alignment_v1"
    assert (archive / source.name).read_bytes() == b'{"old":true}\n'
    assert json.loads((archive / "MANIFEST.json").read_text())["files"][0]["name"] == source.name
    with pytest.raises(ValueError, match="ARCHIVE_ALREADY_EXISTS"):
        main(["--results-dir", str(tmp_path)])
