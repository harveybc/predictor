"""Contract tests for the unsealed I5-P reconstructed-input diagnostic."""
import json
import shutil

import numpy as np
import pytest

from tools import i5p_reconstructed_inputs as I


@pytest.fixture(scope="module")
def extractor():
    return I.load_pinned_extractor(I.PINNED_EXTRACTOR_ROOT)


@pytest.fixture
def case(tmp_path, extractor):
    X, U = extractor
    hp = X.Hyper(filters=4, latent_dim=2, max_epochs=1)
    encoder, decoder, model = X.build_origin_covering_models(hp)
    # Positive kernels make the origin-response assertion independent of a random draw.
    for layer in model.layers:
        for weight in layer.weights:
            weight.assign(np.ones(weight.shape, dtype=np.float32) * 0.03)
    weights = tmp_path / "chosen.weights.h5"
    model.save_weights(weights)
    feature = "px.close_loc"
    identity = "phase1-eurusd-final:94d20c038d55e152"
    source = {"role": "eurusd_ps1_batch_001", "file_sha256": X.CORPORA[identity]["files"],
              "column_sha256": "a" * 64, "bar_seconds": 3600, "grid_seconds": 3600}
    fold = {"fold_id": "inner_2023", "fit": [1, 2], "val": [3, 4]}
    claim = {"schema": "fs4.extractibility.task.v1", "population_id": "EURUSD",
             "identity": identity, "feature_id": feature, "fold_id": "inner_2023",
             "arm": "TRAINED_ENCODER_V2", "seed": 0}
    chosen = X.weights_digest([encoder, decoder])
    record = {**claim, "schema": "fs4.extractibility.result.v1", "status": "COMPLETE",
              "task_id": X.task_digest(claim), "code_commit": I.PINNED_EXTRACTOR_COMMIT,
              "code_sha256": I.extractor_code_sha256(I.PINNED_EXTRACTOR_ROOT),
              "input_sha256": X.digest({"identity": identity, "population_id": "EURUSD",
                                         "feature_id": feature, "file_sha256": source["file_sha256"],
                                         "column_sha256": source["column_sha256"], "grid_seconds": 3600,
                                         "window": 24, "fold": fold}),
              "model_sha256": chosen, "source": source, "fold": fold,
              "architecture": {"id": X.ARCHITECTURE_ID_V2, "last_latent_lag_rows": 0,
                               "calendar_context": list(U.CALENDAR_SPEC), "target_input": False},
              "hyper": hp.to_dict(), "weights": {"chosen_weights_sha256": chosen, "updates": 1},
              "normalization": {"mean": 0.0, "std": 1.0, "n": 100, "constant": False},
              "artifacts": {"chosen_weights_file_sha256": U.sha256_file(str(weights))}}
    terminal = tmp_path / "result.json"
    terminal.write_text(json.dumps(record))
    t = np.arange(40, dtype=np.int64) * 3600 + 1_700_000_000 // 3600 * 3600
    values = np.linspace(0.1, 0.9, 40, dtype=np.float64)[:, None]
    rows = np.array([27, 30], dtype=np.int64)
    row_ids = np.arange(40, dtype=np.int64) + 1000
    target = np.array([0.4, 0.5])
    naive = np.array([0.3, 0.4])
    spec = {feature: {"terminal": terminal, "sha256": I.sha256_file(terminal)}}
    args = dict(timestamps=t, row_ids=row_ids, values=values, columns=(feature,), scored_rows=rows,
                target=target, naive=naive, checkpoints=spec, extractor_root=I.PINNED_EXTRACTOR_ROOT)
    args["expected_data_sha256"] = I.data_sha256(t, row_ids, values, (feature,), rows, target, naive)
    return args, record, terminal


def test_reconstruction_is_causal_origin_covering_and_preserves_support(case):
    args, _, _ = case
    out = I.reconstruct_selected_inputs(**args)
    np.testing.assert_array_equal(out.row_ids, args["row_ids"][args["scored_rows"]])
    np.testing.assert_array_equal(out.target, args["target"])
    np.testing.assert_array_equal(out.naive, args["naive"])
    assert out.values.shape == (2, 1)
    assert out.receipt["diagnostic_only"] is True
    changed_future = dict(args, values=args["values"].copy())
    changed_future["values"][35, 0] += 10
    changed_future["expected_data_sha256"] = I.data_sha256(
        args["timestamps"], args["row_ids"], changed_future["values"], args["columns"], args["scored_rows"],
        args["target"], args["naive"])
    np.testing.assert_array_equal(out.values, I.reconstruct_selected_inputs(**changed_future).values)
    changed_origin = dict(args, values=args["values"].copy())
    changed_origin["values"][30, 0] += 2
    changed_origin["expected_data_sha256"] = I.data_sha256(
        args["timestamps"], args["row_ids"], changed_origin["values"], args["columns"], args["scored_rows"],
        args["target"], args["naive"])
    assert out.values[1, 0] != I.reconstruct_selected_inputs(**changed_origin).values[1, 0]


def test_missing_historical_value_uses_mask_but_missing_origin_refuses(case):
    args, _, _ = case
    with_gap = dict(args, values=args["values"].copy())
    with_gap["values"][5, 0] = np.nan
    with_gap["expected_data_sha256"] = I.data_sha256(
        args["timestamps"], args["row_ids"], with_gap["values"], args["columns"],
        args["scored_rows"], args["target"], args["naive"])
    assert np.isfinite(I.reconstruct_selected_inputs(**with_gap).values).all()
    with_gap["values"][30, 0] = np.nan
    with_gap["expected_data_sha256"] = I.data_sha256(
        args["timestamps"], args["row_ids"], with_gap["values"], args["columns"],
        args["scored_rows"], args["target"], args["naive"])
    with pytest.raises(I.Refusal, match="ORIGIN_SUPPORT_MISMATCH"):
        I.reconstruct_selected_inputs(**with_gap)


def test_each_selected_column_uses_its_own_authenticated_terminal(case, tmp_path):
    args, first, terminal = case
    second_name = "tech.rsi"
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    shutil.copy2(terminal.parent / "chosen.weights.h5", second_dir / "chosen.weights.h5")
    second = json.loads(json.dumps(first))
    second["feature_id"] = second_name
    X, _ = I.load_pinned_extractor(I.PINNED_EXTRACTOR_ROOT)
    claim = {key: second[key] for key in ("population_id", "identity", "feature_id", "fold_id", "arm", "seed")}
    second["task_id"] = X.task_digest({"schema": "fs4.extractibility.task.v1", **claim})
    second["input_sha256"] = X.digest({"identity": second["identity"], "population_id": "EURUSD",
                                        "feature_id": second_name, "file_sha256": second["source"]["file_sha256"],
                                        "column_sha256": second["source"]["column_sha256"],
                                        "grid_seconds": 3600, "window": 24, "fold": second["fold"]})
    second_terminal = second_dir / "result.json"
    second_terminal.write_text(json.dumps(second))
    two = dict(args)
    two["columns"] = (second_name, first["feature_id"])
    two["values"] = np.column_stack((args["values"][:, 0] * 2, args["values"][:, 0]))
    two["checkpoints"] = {second_name: {"terminal": second_terminal, "sha256": I.sha256_file(second_terminal)},
                          first["feature_id"]: args["checkpoints"][first["feature_id"]]}
    two["expected_data_sha256"] = I.data_sha256(two["timestamps"], two["row_ids"], two["values"],
                                                 two["columns"], two["scored_rows"], two["target"], two["naive"])
    out = I.reconstruct_selected_inputs(**two)
    assert out.values.shape == (2, 2)
    assert [r["feature_id"] for r in out.receipt["models"]] == list(two["columns"])
    np.testing.assert_array_equal(out.row_ids, args["row_ids"][args["scored_rows"]])


@pytest.mark.parametrize("mutation,code", [
    ("v1", "ARCHITECTURE_MISMATCH"), ("wrong_feature", "FEATURE_MISMATCH"),
    ("wrong_columns", "COLUMN_MISMATCH"), ("missing_weights", "CHECKPOINT_MISSING"),
    ("nonfinite", "NONFINITE_INPUT"), ("wrong_data", "DATA_IDENTITY_MISMATCH"),
    ("wrong_terminal", "TERMINAL_IDENTITY_MISMATCH"), ("bad_weights", "CHECKPOINT_DIGEST_MISMATCH"),
    ("short_history", "ORIGIN_SUPPORT_MISMATCH"), ("wrong_clock", "ROW_SUPPORT_MISMATCH"),
])
def test_fail_closed(case, mutation, code):
    args, rec, terminal = case
    args = dict(args)
    if mutation in ("v1", "wrong_feature"):
        altered = json.loads(json.dumps(rec))
        if mutation == "v1":
            altered["architecture"]["id"] = "fs4_causal_conv_24_12_6_v1"
            altered["architecture"]["last_latent_lag_rows"] = 3
        else:
            altered["feature_id"] = "other"
        terminal.write_text(json.dumps(altered))
        args["checkpoints"] = {args["columns"][0]: {"terminal": terminal, "sha256": I.sha256_file(terminal)}}
    elif mutation == "wrong_columns":
        args["columns"] = ("other",)
    elif mutation == "missing_weights":
        (terminal.parent / "chosen.weights.h5").unlink()
    elif mutation == "bad_weights":
        (terminal.parent / "chosen.weights.h5").write_bytes(b"not h5")
    elif mutation == "nonfinite":
        args["target"] = args["target"].copy()
        args["target"][0] = np.nan
    elif mutation == "wrong_data":
        args["expected_data_sha256"] = "b" * 64
    elif mutation == "short_history":
        args["scored_rows"] = np.array([10, 27], dtype=np.int64)
        args["expected_data_sha256"] = I.data_sha256(args["timestamps"], args["row_ids"], args["values"],
                                                      args["columns"], args["scored_rows"], args["target"], args["naive"])
    elif mutation == "wrong_clock":
        args["timestamps"] = args["timestamps"].copy()
        args["timestamps"][4] += 1
    else:
        args["checkpoints"] = {args["columns"][0]: {"terminal": terminal, "sha256": "b" * 64}}
    with pytest.raises(I.Refusal, match=code):
        I.reconstruct_selected_inputs(**args)
