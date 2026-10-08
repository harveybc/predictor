"""Architecture acceptance checks for the selected-feature RAW comparison."""

import numpy as np
import pytest

from tools import fs4_temporal_predictor as P
from tools.i6a_architectures import ARMS, build_model


def test_all_arms_preserve_temporal_grid_and_share_core_head():
    pytest.importorskip("tensorflow")
    spec = P.PredictorSpec()
    for arm in ARMS:
        model = build_model(spec, 4, "RAW", None, arm)
        assert model.input_shape == (None, 24, 8)
        assert model.get_layer("channel_fusion").output.shape[1] == 6
        assert model.get_layer(P.CORE_OUTPUT_NAME).output.shape[1] == 6
        assert model.get_layer(P.HEAD_INPUT_NAME).output.shape[-1] == spec.core_filters
        assert model.output_shape == (None, 1)
        got = model(np.zeros((2, 24, 8), dtype="float32"), training=False).numpy()
        assert np.isfinite(got).all()


def test_refuses_unknown_or_encoder_arm():
    with pytest.raises(P.Refusal, match="UNKNOWN_ARCHITECTURE"):
        build_model(P.PredictorSpec(), 4, "RAW", None, "OTHER")
    with pytest.raises(P.Refusal, match="I6A_RAW_ONLY"):
        build_model(P.PredictorSpec(), 4, "TRAINED_ENCODER", 8, "ARCH_A")


def test_fixed_downsample_reads_origin_and_never_future():
    pytest.importorskip("tensorflow")
    model = build_model(P.PredictorSpec(), 4, "RAW", None, "ARCH_0")
    import keras

    probe = keras.Model(model.input, model.get_layer("fixed_causal_downsample").output)
    x = np.arange(24 * 8, dtype="float32").reshape(1, 24, 8)
    np.testing.assert_array_equal(probe(x).numpy(), x[:, 3::4, :])
