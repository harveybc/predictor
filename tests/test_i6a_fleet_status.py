"""A stale controller must not be mistaken for a finished campaign."""

import json

from tools.i6a_fleet_status import campaign_status


def test_campaign_status_distinguishes_worker_and_publication(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "STATUS.json").write_text(json.dumps({"state": "WAITING", "target": "Y_s_2h",
                                                 "workers": {"dragon4090": {"completed": 8,
                                                                             "total": 208,
                                                                             "eta_seconds": 100}}}))
    row = campaign_status("Y_s_2h", root)
    assert row["state"] == "WAITING"
    assert row["completed"] == 8
    assert row["eta_seconds"] == 100
    (root / "STATUS.json").write_text(json.dumps({"state": "PUBLISHED", "target": "Y_s_2h",
                                                 "reports": 208, "closure_sha256": "a" * 64}))
    (root / "CLOSURE.json").write_text(json.dumps({"state": "COMPLETE", "sha256": "a" * 64,
                                                  "verified_cells": 208}))
    assert campaign_status("Y_s_2h", root)["state"] == "PUBLISHED"


def test_campaign_status_rejects_mismatched_target(tmp_path):
    tmp_path.joinpath("STATUS.json").write_text(json.dumps({"state": "PUBLISHED",
                                                        "target": "Y_l_24h"}))
    assert campaign_status("Y_s_2h", tmp_path)["state"] == "INVALID_STATUS"
