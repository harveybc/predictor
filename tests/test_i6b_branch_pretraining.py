import inspect
import json

import numpy as np
import pytest

from predictor_plugins.modular_temporal import build_modular
from predictor_plugins.modular_temporal.artifacts import load_donor, weights_hash
from tools import i6b_branch_pretraining as I6B


def _bundle():
    return build_modular(
        {
            "feature_names": ["signal", "hour_sin"],
            "sample_hours": 1,
            "window": 24,
            "branch_steps": 24,
            "branches": [
                {
                    "name": "signal_branch",
                    "features": ["signal", "hour_sin"],
                    "plugin": "causal_conv1d",
                    "params": {"channels": 16, "kernel_size": 3},
                    "regime": "R0",
                    "donor": None,
                }
            ],
            "output_steps": 6,
            "output_channels": 8,
            "horizons": [1],
            "target_count": 1,
            "alignment_probe": False,
        }
    )


def _windows(n=20):
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(n, 24, 1)).astype("float32")
    hour = np.sin(np.arange(24, dtype="float32") * np.pi / 12.0)
    seasonal = np.broadcast_to(hour[None, :, None], (n, 24, 1))
    return np.concatenate([signal, seasonal], axis=2)


def _settings(**overrides):
    return {
        "seed": 19,
        "max_epochs": 2,
        "patience": 2,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "min_delta": 0.0,
        "inner_tail_fraction": 0.25,
        "purge_rows": 1,
        "decoder_channels": 8,
        **overrides,
    }


def test_training_updates_the_exact_production_branch_and_saves_that_donor(tmp_path):
    bundle = _bundle()
    branch = bundle.branch_models["signal_branch"]
    before = weights_hash(branch)

    report = I6B.train_branch_donors(
        bundle=bundle,
        train_windows={"signal_branch": _windows()},
        output_dir=tmp_path,
        corpus={"dataset_id": "eurusd.train", "support": "2019-01-01/2023-12-31"},
        settings=_settings(),
    )

    assert report["status"] == "COMPLETE"
    assert weights_hash(branch) != before
    donor_path = tmp_path / "signal_branch.keras"
    loaded = load_donor(
        donor_path,
        bundle.donor_manifest("branch", "signal_branch"),
        require_contract="OPERATIONAL",
    )
    assert weights_hash(loaded) == weights_hash(branch)


def test_mean_train_tail_early_stop_restores_the_selected_weights():
    callback = I6B.MeanTrainTailEarlyStopping(patience=1, min_delta=0.0)

    class Model:
        def __init__(self):
            self.stop_training = False
            self.weights = [np.array([0.0], dtype="float32")]

        def get_weights(self):
            return [value.copy() for value in self.weights]

        def set_weights(self, values):
            self.weights = [value.copy() for value in values]

    model = Model()
    callback.set_model(model)
    callback.on_train_begin()
    for epoch, (weight, train, tail) in enumerate(
        [(1.0, 0.8, 0.6), (2.0, 0.4, 0.2), (3.0, 0.5, 0.3)]
    ):
        model.weights = [np.array([weight], dtype="float32")]
        callback.on_epoch_end(epoch, {"loss": train, "val_loss": tail})
    callback.on_train_end()

    assert callback.best_epoch == 2
    assert callback.best_score == pytest.approx(0.3)
    assert callback.restored_best_weights is True
    assert model.weights[0].item() == pytest.approx(2.0)


def test_sidecar_records_train_only_corpus_reconstruction_and_objective(tmp_path):
    bundle = _bundle()
    I6B.train_branch_donors(
        bundle=bundle,
        train_windows={"signal_branch": _windows()},
        output_dir=tmp_path,
        corpus={"dataset_id": "eurusd.train", "support": {"start": "2019", "end": "2024"}},
        settings=_settings(),
    )

    sidecar = json.loads((tmp_path / "signal_branch.manifest.json").read_text())
    provenance = sidecar["provenance"]
    assert provenance["conditioning_contract"] == "OPERATIONAL"
    assert provenance["learned_corpus"]["kind"] == "TRAIN_ONLY"
    assert provenance["learned_corpus"]["dataset_id"] == "eurusd.train"
    assert len(provenance["learned_corpus"]["data_sha256"]) == 64
    assert provenance["reconstruction"]["state"] == "MEASURED"
    assert provenance["objective"]["monitor"] == "mean(train_loss,val_loss)"
    assert provenance["objective"]["validation_semantics"] == "PURGED_ORDERED_INNER_TRAIN_TAIL"
    assert provenance["objective"]["restore_best_weights"] is True


@pytest.mark.parametrize("mutation", ["archive", "manifest"])
def test_status_rejects_tampered_archive_or_manifest_mismatch(tmp_path, mutation):
    bundle = _bundle()
    I6B.train_branch_donors(
        bundle=bundle,
        train_windows={"signal_branch": _windows()},
        output_dir=tmp_path,
        corpus={"dataset_id": "eurusd.train", "support": "train"},
        settings=_settings(),
    )
    if mutation == "archive":
        with (tmp_path / "signal_branch.keras").open("ab") as stream:
            stream.write(b"tamper")
    else:
        sidecar_path = tmp_path / "signal_branch.manifest.json"
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["manifest"]["params"]["channels"] = 15
        sidecar_path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="mismatch"):
        I6B.pretraining_status(tmp_path, bundle)


def test_public_training_interface_cannot_receive_validation_or_test_and_bundle_is_exact(tmp_path):
    parameters = set(inspect.signature(I6B.train_branch_donors).parameters)
    assert parameters == {"bundle", "train_windows", "output_dir", "corpus", "settings"}
    assert not any("validation" in name or "test" in name for name in parameters)

    bundle = _bundle()
    bundle._branch_manifests["signal_branch"]["plugin"]["version"] = "1.0.0"
    with pytest.raises(ValueError, match="causal_conv1d v2"):
        I6B.train_branch_donors(
            bundle=bundle,
            train_windows={"signal_branch": _windows()},
            output_dir=tmp_path,
            corpus={"dataset_id": "eurusd.train", "support": "train"},
            settings=_settings(),
        )
    assert list(tmp_path.iterdir()) == []


def test_nonfinite_train_input_fails_before_writing_artifacts(tmp_path):
    bundle = _bundle()
    windows = _windows()
    windows[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        I6B.train_branch_donors(
            bundle=bundle,
            train_windows={"signal_branch": windows},
            output_dir=tmp_path,
            corpus={"dataset_id": "eurusd.train", "support": "train"},
            settings=_settings(),
        )
    assert list(tmp_path.iterdir()) == []


def test_completed_branch_terminal_is_reused_after_process_interruption(tmp_path, monkeypatch):
    bundle = _bundle()
    first = I6B.train_branch_donors(
        bundle=bundle,
        train_windows={"signal_branch": _windows()},
        output_dir=tmp_path,
        corpus={"dataset_id": "eurusd.train", "support": "train"},
        settings=_settings(),
    )
    (tmp_path / "REPORT.json").unlink()
    (tmp_path / "STATUS.json").unlink()
    terminal = tmp_path / "signal_branch.terminal.json"
    assert terminal.is_file()

    def must_not_fit(*args, **kwargs):
        raise AssertionError("a committed branch must not be trained twice")

    monkeypatch.setattr(I6B, "_fit_branch", must_not_fit)
    resumed = I6B.train_branch_donors(
        bundle=_bundle(),
        train_windows={"signal_branch": _windows()},
        output_dir=tmp_path,
        corpus={"dataset_id": "eurusd.train", "support": "train"},
        settings=_settings(),
    )
    assert resumed["branches"] == first["branches"]
