"""§4.A item 2 on a REAL cell (the pinned ETH 4h file, TRAIN rows only) through the real evaluator.

R1: every frozen component (each branch, temporal_core) hashes equal to its donor before and after
training, with no trainable tensors. R2: starts from the same donors (equal hashes at build) and every
component is updated. Tiny budget: the hashes are the claim, not the forecast numbers.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

import pytest

from tools import a2_regime_hash_proof as proof

DATA = Path(__file__).resolve().parents[1] / "examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv"


@pytest.mark.skipif(not DATA.is_file(), reason="pinned ETH 4h file absent from this checkout")
def test_r1_frozen_and_r2_updated_on_a_real_cell(tmp_path):
    r = proof.run_proof(DATA, tmp_path / "proof", features=("log_return_1", "rsi_14"), horizons=(1,),
                        fit=dict(max_epochs=2, patience=2, batch_size=128, loss="mae", learning_rate=1e-3))
    v = r["verdict"]
    assert v["R1_frozen_bit_for_bit"], r["regimes"]["R1"]["after_equals_donor"]
    assert v["R2_starts_from_same_donors"]
    assert v["R2_updates_every_component"], r["regimes"]["R2"]["after_equals_donor"]
    assert v["R1_R2_same_initial_model"]
    assert r["regimes"]["R1"]["observed_updates"] > 0                    # the head still trains under R1
    cost = r["cost_seconds"]
    assert cost["pretraining_total"] > 0 and cost["fit_R1"] > 0 and cost["fit_R2"] > 0
    assert set(cost["pretraining_branches"]) == {"branch_0", "branch_1"}
    assert r["cell"]["validation_rows"] > 0 and r["cell"]["forecast_train_rows"] > r["cell"]["validation_rows"]
