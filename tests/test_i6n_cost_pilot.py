"""Pilot accounting must use the released cgroup of the exact workload."""
import json
import pytest
from tools.i6n_swarm.cost_pilot import retained_peak


def test_exact_released_scope(tmp_path):
    p = tmp_path / "ledger"
    row = dict(event="LEASE_RELEASED", label="wanted", peak_scope="cgroup", observed_tree_peak_bytes=100)
    p.write_text(json.dumps(dict(row, label="another", observed_tree_peak_bytes=1000))+"\n"+json.dumps(row)+"\n")
    assert retained_peak(p, "wanted")[0] == 100


@pytest.mark.parametrize("change", [{"peak_scope":"rss"}, {"observed_tree_peak_bytes":0}, {"event":"LEASE_ARMED"}])
def test_missing_or_wrong_measurement_rejected(tmp_path, change):
    p = tmp_path / "ledger"
    row = dict(event="LEASE_RELEASED", label="wanted", peak_scope="cgroup", observed_tree_peak_bytes=100)
    p.write_text(json.dumps(dict(row, **change))+"\n")
    with pytest.raises(ValueError):
        retained_peak(p, "wanted")
