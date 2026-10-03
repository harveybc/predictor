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
    tr = np.load(a.data_dir + "/train.npz", allow_pickle=False); va = np.load(a.data_dir + "/validation.npz", allow_pickle=False)
    names = [str(n) for n in tr["feature_names"]]; assert names == [str(n) for n in va["feature_names"]]
    Wtr, Wva = tr["windows"], va["windows"]
    if a.mode == "ae":
        groups = json.load(open(a.groups))["groups"]
        def lat(W):
            out = []
            for g in sorted(groups):
                enc = tf.keras.models.load_model(os.path.join(a.ae_root, g, "encoder.keras"), compile=False)
                idx = [names.index(f) for f in groups[g]]
                out.append(enc.predict(np.ascontiguousarray(W[:, :, idx]), batch_size=256, verbose=0))
            return np.concatenate(out, 1).astype(np.float32)
        Xtr, Xva = lat(Wtr), lat(Wva)
    else:
        Xtr, Xva = Wtr.reshape(len(Wtr), -1).astype(np.float32), Wva.reshape(len(Wva), -1).astype(np.float32)
    n = len(Xtr); cut = int(n * 0.85); purge = 48
    fit_i, hold_i = np.arange(0, cut - purge), np.arange(cut, n)
    mu, sd = Xtr[fit_i].mean(0), Xtr[fit_i].std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    ytr = tr["targets"].reshape(n, -1).astype(np.float32); yva = va["targets"].reshape(len(Xva), -1).astype(np.float32)
    med = np.median(ytr[fit_i], 0)
    L = tf.keras.layers
    inp = L.Input(shape=(Xtr.shape[1],)); h = L.Dense(64, activation="relu")(inp)
    out = L.Dense(ytr.shape[1], bias_initializer=tf.keras.initializers.Constant(med))(h)
    m = tf.keras.Model(inp, out); m.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))
    best, bw, bad, hist = 1e9, None, 0, []
    for ep in range(30):
        m.fit(Xtr[fit_i], ytr[fit_i], batch_size=64, epochs=1, shuffle=True, verbose=0)
        hm = float(np.abs(m.predict(Xtr[hold_i], batch_size=512, verbose=0) - ytr[hold_i]).mean()); hist.append(hm)
        if hm < best: best, bw, bad = hm, m.get_weights(), 0
        else:
            bad += 1
            if bad >= 5: break
    m.set_weights(bw)
    pv = m.predict(Xva, batch_size=512, verbose=0)
    err = np.abs(pv - yva).reshape(len(Xva), 24, -1)
    ph = err.mean((0, 2)); mm = float(err.mean())
    T = va["targets"]
    nv = {"seasonal_24": float(np.abs(Wva - T).mean()), "persistence_last_value": float(np.abs(Wva[:, -1:, :] - T).mean()),
          "intercept_train_median": float(np.abs(med.reshape(1, 24, -1) - T).mean())}
    res = {"mode": a.mode, "seed": a.seed, "input_dim": int(Xtr.shape[1]), "epochs_run": len(hist), "best_holdout_mae": best, "holdout_hist": hist,
           "val_mae_mean": mm, "val_mae_per_horizon": {str(i + 1): float(x) for i, x in enumerate(ph)}, "naive_mean": nv,
           "skill_vs": {k: 1 - mm / v for k, v in nv.items()}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("holdout_hist", "val_mae_per_horizon")}))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--mode", choices=["ae", "raw"], required=True); p.add_argument("--groups"); p.add_argument("--data_dir")
    p.add_argument("--ae_root"); p.add_argument("--out"); p.add_argument("--seed", type=int, default=2021); main(p.parse_args())
