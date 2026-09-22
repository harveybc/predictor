"""Read-only, bounded independent reduction of retained ECL cells; no author loader."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    design = json.loads((args.root / "DESIGN.json").read_text())
    delivered = json.loads((args.root / "DELIVERIES.json").read_text())
    source = Path(next(iter(delivered["units"].values()))["path"])
    source_digest = digest(source)
    assert source_digest == design["source_data"]["sha256"]
    frame = pd.read_csv(source)
    dates = pd.to_datetime(frame.pop("date"))
    assert ((dates.diff().dropna() / pd.Timedelta(hours=1)) == 1).all()
    columns = [c for c in frame.columns if c != "OT"] + ["OT"]
    values = frame[columns].to_numpy()
    N, C = values.shape
    train, test = int(N * .7), int(N * .2)
    # Share only the published scaler implementation; derive all row indices here.
    scaler = StandardScaler().fit(values[:train])
    values = scaler.transform(values).astype(np.float32)
    first_target = N - test
    rows = []
    for cell in design["cells"]:
        folder = args.root / "attempts" / cell["cell_id"]
        if not (folder / "arrays.npz").is_file():
            continue
        record = json.loads((folder / "cell.json").read_text())
        H = cell["horizon"]
        W = test - H + 1
        array_digest = digest(folder / "arrays.npz")
        assert array_digest == record["arrays_sha256"]
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            prediction = archive["pred"]
        assert prediction.shape == (W, H, C)
        sums = np.zeros(6, dtype=np.float64)
        step_abs = np.zeros(H)
        channel_abs = np.zeros(C)
        pred_hash, target_hash = hashlib.sha256(), hashlib.sha256()
        for begin in range(0, W, 16):
            end = min(W, begin + 16)
            origins = first_target + np.arange(begin, end)
            target = values[origins[:, None] + np.arange(H)[None, :]]
            pred = prediction[begin:end]
            assert np.isfinite(pred).all() and np.isfinite(target).all()
            pred_hash.update(memoryview(np.ascontiguousarray(pred)).cast("B"))
            target_hash.update(memoryview(np.ascontiguousarray(target)).cast("B"))
            delta = pred.astype(np.float64) - target.astype(np.float64)
            persistence = values[origins - 1, None, :].astype(np.float64)
            seasonal = values[origins[:, None] - 24 + np.arange(H)[None, :] % 24].astype(np.float64)
            naive_delta = persistence - target
            seasonal_delta = seasonal - target
            sums += [np.abs(delta).sum(), np.square(delta).sum(),
                     np.abs(naive_delta).sum(), np.square(naive_delta).sum(),
                     np.abs(seasonal_delta).sum(), np.square(seasonal_delta).sum()]
            step_abs += np.abs(delta).sum(axis=(0, 2))
            channel_abs += np.abs(delta).sum(axis=(0, 1))
        assert pred_hash.hexdigest() == record["pred_sha256"]
        assert target_hash.hexdigest() == record["true_sha256"]
        metrics = dict(zip(["mae", "mse", "naive_mae", "naive_mse", "seasonal24_mae", "seasonal24_mse"], (sums / (W * H * C)).tolist()))
        vault = json.loads((folder / "METRICS_VAULT.json").read_text())
        differences = {k: metrics[k] - vault["global"][k] for k in metrics}
        rows.append({"unit": cell["cell_id"], "shape": list(prediction.shape),
                     "arrays_sha256": array_digest, "record_sha256": digest(folder / "cell.json"),
                     "vault_sha256": digest(folder / "METRICS_VAULT.json"),
                     "pred_sha256": pred_hash.hexdigest(), "true_sha256": target_hash.hexdigest(),
                     "metrics": metrics, "difference_vs_vault": differences,
                     "per_step_mae_max_difference": float(np.max(np.abs(step_abs / (W * C) - vault["per_step"]["mae"]))),
                     "per_channel_mae_max_difference": float(np.max(np.abs(channel_abs / (W * H) - vault["per_channel"]["mae"]))),
                     "author_metric_difference": {k: metrics[k] - record["author_metric_float32"][k] for k in ("mae", "mse")}})
        del prediction
        print(json.dumps(rows[-1]), flush=True)
    result = {"schema": "musashi_rp97_independent_reduction.v1", "source_sha256": source_digest,
              "population": {"rows": N, "channels": C, "train_rows": train, "test_rows": test,
                             "first_test_target_index": first_target}, "rows": rows,
              "scope": "independent indices/reductions against bytes on worker, hashes match cell records; no live warehouse query; extended metrics not certified",
              "elapsed_seconds": time.monotonic() - started,
              "cpu_seconds": time.process_time(), "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
