"""ECL representation ladder forecast cell (2026-10-02). Modes: ae (frozen per-group typed-AE latents) | raw (flattened standardized windows).
Reads train.npz (fit + train-tail holdout for early stopping) and validation.npz (evaluated once, never used to fit or stop). No test rows exist.
Target: all 24 horizons x 321 channels (z_train); metric = MAE over all cells. Naives are computed here from the same validation rows:
seasonal-24 (prediction for horizon h = window[:, h-1, :]), persistence (window[:, -1, :]) and the train-median intercept.
"""
import argparse, json, os
import numpy as np

def main(a):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    # arrays were extracted once (tools/ladder_ecl_extract.py) from train.npz / validation.npz to .npy; memory-mapped here so the
    # 565 MB train windows/targets are page cache, not anonymous memory
    tr = {k: np.load(os.path.join(a.npy_dir, "train_%s.npy" % k), mmap_mode="r") for k in ("windows", "targets")}
    Wva = np.load(os.path.join(a.npy_dir, "validation_windows.npy")); Tva = np.load(os.path.join(a.npy_dir, "validation_targets.npy"))
    names = [str(n) for n in np.load(a.data_dir + "/train.npz", allow_pickle=False)["feature_names"]]
    if a.mode == "ae":
        groups = json.load(open(a.groups))["groups"]
        encs = [(tf.keras.models.load_model(os.path.join(a.ae_root, g, "encoder.keras"), compile=False), [names.index(f) for f in groups[g]]) for g in sorted(groups)]
        def lat(W):  # chunked: never materializes a whole group tensor
            return np.concatenate([np.concatenate([e(np.ascontiguousarray(W[i:i + 256][:, :, idx]), training=False).numpy() for e, idx in encs], 1)
                                   for i in range(0, len(W), 256)]).astype(np.float32)
        Xtr = lat(tr["windows"]); Xva = lat(Wva)
    else:
        Xtr = tr["windows"].reshape(tr["windows"].shape[0], -1); Xva = Wva.reshape(len(Wva), -1)  # views
    n = len(Xtr); cut = int(n * 0.85); purge = 48
    nf = cut - purge  # fit rows are [0, nf), holdout rows are [cut, n): contiguous slices, no index copies
    mu = np.zeros(Xtr.shape[1]); m2 = np.zeros(Xtr.shape[1])
    for i in range(0, nf, 2048):
        c = Xtr[i:min(i + 2048, nf)].astype(np.float64); mu += c.sum(0); m2 += (c ** 2).sum(0)
    mu /= nf; sd = np.sqrt(np.maximum(m2 / nf - mu ** 2, 0)) + 1e-6
    mu32, sd32 = mu.astype(np.float32), sd.astype(np.float32)
    def nz(X): return (X - mu32) / sd32  # standardize on the fly (fit-row statistics); a no-op copy for the small latent matrix
    if a.mode == "ae": Xtr = nz(Xtr); Xva = nz(Xva); nz = lambda X: X
    ytr = tr["targets"].reshape(n, -1)
    med = np.concatenate([np.median(ytr[:nf, j:j + 1024], 0) for j in range(0, ytr.shape[1], 1024)]).astype(np.float32)
    Xh, yh = Xtr[cut:], ytr[cut:]  # views
    class Seq(tf.keras.utils.PyDataset):  # batch slices from the numpy arrays; no whole-array tensor copies
        def __init__(self):
            super().__init__(); self.perm = np.random.permutation(nf)
        def __len__(self): return int(np.ceil(nf / 64))
        def __getitem__(self, i):
            ix = np.sort(self.perm[i * 64:(i + 1) * 64]); return nz(Xtr[ix]), np.asarray(ytr[ix])
        def on_epoch_end(self): self.perm = np.random.permutation(nf)
    L = tf.keras.layers
    inp = L.Input(shape=(Xtr.shape[1],)); h = L.Dense(64, activation="relu")(inp)
    out = L.Dense(ytr.shape[1], bias_initializer=tf.keras.initializers.Constant(med))(h)
    m = tf.keras.Model(inp, out); m.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))
    def mae_on(X, Y):
        t = 0.0
        for i in range(0, len(X), 256): t += float(np.abs(m.predict(nz(X[i:i + 256]), batch_size=256, verbose=0) - Y[i:i + 256]).sum())
        return t / Y.size
    best, bw, bad, hist = 1e9, None, 0, []
    for ep in range(30):
        m.fit(Seq(), epochs=1, verbose=0)
        hm = mae_on(Xh, yh); hist.append(hm)
        if hm < best: best, bw, bad = hm, m.get_weights(), 0
        else:
            bad += 1
            if bad >= 5: break
    m.set_weights(bw)
    yva = Tva.reshape(len(Xva), -1); ph = np.zeros(24); tot = 0.0
    for i in range(0, len(Xva), 256):
        e = np.abs(m.predict(nz(Xva[i:i + 256]), batch_size=256, verbose=0) - yva[i:i + 256]).reshape(-1, 24, Tva.shape[2])
        ph += e.sum((0, 2)); tot += float(e.sum())
    ph /= (len(Xva) * Tva.shape[2]); mm = tot / yva.size
    T = Tva
    nv = {"seasonal_24": float(np.abs(Wva - T).mean()), "persistence_last_value": float(np.abs(Wva[:, -1:, :] - T).mean()),
          "intercept_train_median": float(np.abs(med.reshape(1, 24, -1) - T).mean())}
    res = {"mode": a.mode, "seed": a.seed, "input_dim": int(Xtr.shape[1]), "epochs_run": len(hist), "best_holdout_mae": best, "holdout_hist": hist,
           "val_mae_mean": mm, "val_mae_per_horizon": {str(i + 1): float(x) for i, x in enumerate(ph)}, "naive_mean": nv,
           "skill_vs": {k: 1 - mm / v for k, v in nv.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("holdout_hist", "val_mae_per_horizon")}))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["ae", "raw"], required=True); p.add_argument("--groups"); p.add_argument("--data_dir"); p.add_argument("--npy_dir")
    p.add_argument("--ae_root"); p.add_argument("--out"); p.add_argument("--seed", type=int, default=2021); main(p.parse_args())
