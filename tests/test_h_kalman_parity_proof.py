"""Lane H: the parity/restart/leak proof tool produces digests that the tests can re-derive independently."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


proof = _load("h_kalman_parity_proof")
kf = proof.kf


def panel(R=1200, V=3, seed=0):
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.standard_normal((R, V)), axis=0) + rng.standard_normal((R, V))


def test_proof_on_a_synthetic_panel_reports_equal_digests_and_a_leak_probe_that_can_fail(tmp_path):
    Z = panel()
    out = proof.run_proof(Z, n_train=800, names=["a", "b", "c"], split_row=500, workdir=tmp_path, spec_kinds=[kf.LOCAL_LEVEL, kf.LOCAL_LINEAR_TREND])
    for kind, r in out["kinds"].items():
        assert r["batch_digest"] == r["tick_digest"] == r["restart_digest"] == r["chunk_digest"], kind
        assert r["restart_used_separate_process"] is True
        assert r["leak_probe"]["rows_checked"] > 0 and r["leak_probe"]["max_abs_move_before_t"] == 0.0
        assert r["leak_probe"]["oracle_detects_leak"] is True               # the smoother moves earlier rows
        assert r["smoother_refused_as_input"] is True
        assert r["state_digest_end"]
    assert out["artifact_digests"] and out["environment"]["single_thread_ok"] in (True, False)


def test_proof_digest_is_reproducible_in_a_second_process(tmp_path):
    Z = panel(seed=3)
    np.save(tmp_path / "Z.npy", Z)
    code = ("import sys,json,importlib.util;import numpy as np;"
            "spec=importlib.util.spec_from_file_location('h_kalman_parity_proof',sys.argv[1]);m=importlib.util.module_from_spec(spec);"
            "sys.modules['h_kalman_parity_proof']=m;spec.loader.exec_module(m);"
            "from pathlib import Path;Z=np.load(sys.argv[2]);"
            "o=m.run_proof(Z,n_train=800,names=['a','b','c'],split_row=500,workdir=Path(sys.argv[3]),spec_kinds=[m.kf.LOCAL_LEVEL]);"
            "print(json.dumps(o['digest_summary']))")
    w1, w2 = tmp_path / "w1", tmp_path / "w2"
    w1.mkdir(); w2.mkdir()
    outs = []
    for w in (w1, w2):
        r = subprocess.run([sys.executable, "-c", code, str(_TOOLS / "h_kalman_parity_proof.py"), str(tmp_path / "Z.npy"), str(w)],
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr
        outs.append(json.loads(r.stdout))
    assert outs[0] == outs[1]
