"""ECL per-group typed AE (2026-10-02). Same as `ladder_ae_forecast.py ae` but the per-channel-mean naive diagnostic is computed in
float64 CHUNKS (the single-shot version needs ~4 float64 copies of the group tensor, 253 channels -> >3.4 GB) and the group tensor is
released before the adapter reloads it. Same quantities, bounded memory. Use with an adapter whose reconstruction metric is chunked too."""
import argparse, json, os, sys
import numpy as np

def main(a):
    sys.path.insert(0, a.adapter_dir)
    from app.npz_encoder_adapter import train_encoder
    spec = json.load(open(a.groups))["groups"][a.group]
    z = np.load(a.data_dir + "/train.npz", allow_pickle=False)
    names = [str(n) for n in z["feature_names"]]; idx = [names.index(f) for f in spec]
    x = np.ascontiguousarray(z["windows"][:, :, idx], dtype=np.float32); rid = z["row_ids"]
    os.makedirs(a.out_dir, exist_ok=True); tmp = os.path.join(a.out_dir, "group_train.npz")
    np.savez(tmp, x=x, row_ids=rid, split=np.array("train"))
    n, C = x.shape[0] * x.shape[1], x.shape[2]
    s = np.zeros(C)
    for i in range(0, x.shape[0], 1024): s += x[i:i + 1024].reshape(-1, C).astype(np.float64).sum(0)
    cm = s / n; sa = sq = 0.0
    for i in range(0, x.shape[0], 1024):
        e = x[i:i + 1024].astype(np.float64) - cm; sa += float(np.abs(e).sum()); sq += float((e ** 2).sum())
    nmae, nmse = sa / x.size, sq / x.size
    del x, z
    m = train_encoder(tmp, a.out_dir, seed=a.seed, latent_dim=a.latent_dim, filters=a.filters, epochs=a.epochs, batch_size=128)
    os.remove(tmp)
    res = {"group": a.group, "channels": len(idx), "seed": a.seed, "latent_dim": a.latent_dim, "recon_mae": m.reconstruction_mae,
           "recon_mse": m.reconstruction_mse, "naive_channel_mean_mae": nmae, "naive_channel_mean_mse": nmse,
           "mae_skill": 1 - m.reconstruction_mae / nmae, "mse_skill": 1 - m.reconstruction_mse / nmse,
           "weight_sha256": m.weight_sha256, "features": spec}
    json.dump(res, open(os.path.join(a.out_dir, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "features"}))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("groups", "group", "data_dir", "out_dir", "adapter_dir"): p.add_argument("--" + k)
    p.add_argument("--seed", type=int, default=2021); p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--filters", type=int, default=32); p.add_argument("--epochs", type=int, default=20); main(p.parse_args())
