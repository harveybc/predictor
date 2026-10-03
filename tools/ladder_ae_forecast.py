"""Representation ladder (2026-10-02): table-derived feature groups -> per-group typed AE -> frozen-latent forecast.

Subcommands: groups | ae | forecast.  Only train.npz is read by groups/ae; forecast reads train.npz
(fit + train-tail holdout for early stopping) and validation.npz (evaluated once, never used to fit or stop).
No test rows exist in these files.  Grouping uses ONLY the H2 train-only per-feature table columns
adf_pvalue, acf_lag24, n_unique (no target-association columns).
"""
import argparse, csv, json, os, sys
import numpy as np

def parse_groups(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    g = {}
    for r in rows:
        f = r["feature"]; p = float(r["adf_pvalue"]); a24 = float(r["acf_lag24"]); nu = int(float(r["n_unique"]))
        if nu <= 2: k = "binary_regime"
        elif a24 >= 0.85: k = "level_nonstationary"
        elif p < 0.05 and abs(a24) < 0.10: k = "fast_stationary"
        elif a24 >= 0.45: k = "slow_state"
        else: k = "mid_state"
        g.setdefault(k, []).append(f)
    return g

def cmd_groups(a):
    g = parse_groups(a.table)
    json.dump({"rule": "n_unique<=2 -> binary_regime; acf_lag24>=0.85 -> level_nonstationary; adf_p<0.05 & |acf_lag24|<0.10 -> fast_stationary; acf_lag24>=0.45 -> slow_state; else mid_state",
               "source": os.path.basename(a.table), "groups": g}, open(a.out, "w"), indent=1)
    for k, v in g.items(): print(k, len(v), v)

def _train(npz):
    z = np.load(npz, allow_pickle=False)
    return z

def cmd_ae(a):
    sys.path.insert(0, a.adapter_dir)
    from app.npz_encoder_adapter import train_encoder
    spec = json.load(open(a.groups))["groups"][a.group]
    z = _train(a.data_dir + "/train.npz")
    names = [str(n) for n in z["feature_names"]]
    idx = [names.index(f) for f in spec]
    x = np.ascontiguousarray(z["windows"][:, :, idx], dtype=np.float32)
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = os.path.join(a.out_dir, "group_train.npz")
    np.savez(tmp, x=x, row_ids=z["row_ids"], split=np.array("train"))
    cm = x.reshape(-1, x.shape[2]).astype(np.float64).mean(0)
    err = x.astype(np.float64) - cm
    nmae, nmse = float(np.abs(err).mean()), float((err ** 2).mean())
    m = train_encoder(tmp, a.out_dir, seed=a.seed, latent_dim=a.latent_dim, filters=a.filters, epochs=a.epochs, batch_size=128)
    os.remove(tmp)
    res = {"group": a.group, "channels": len(idx), "seed": a.seed, "latent_dim": a.latent_dim, "recon_mae": m.reconstruction_mae,
           "recon_mse": m.reconstruction_mse, "naive_channel_mean_mae": nmae, "naive_channel_mean_mse": nmse,
           "mae_skill": 1 - m.reconstruction_mae / nmae, "mse_skill": 1 - m.reconstruction_mse / nmse,
           "weight_sha256": m.weight_sha256, "features": spec}
    json.dump(res, open(os.path.join(a.out_dir, "result.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "features"}))

def cmd_forecast(a):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    groups = json.load(open(a.groups))["groups"]
    tr = _train(a.data_dir + "/train.npz"); va = _train(a.data_dir + "/validation.npz")
    names = [str(n) for n in tr["feature_names"]]
    assert names == [str(n) for n in va["feature_names"]]
    def lat(z):
        out = []
        for g in sorted(groups):
            enc = tf.keras.models.load_model(os.path.join(a.ae_root, g, "encoder.keras"), compile=False)
            idx = [names.index(f) for f in groups[g]]
            out.append(enc.predict(np.ascontiguousarray(z["windows"][:, :, idx]), batch_size=256, verbose=0))
        return np.concatenate(out, 1).astype(np.float32)
    Ltr, Lva = lat(tr), lat(va)
    mu, sd = Ltr.mean(0), Ltr.std(0) + 1e-6
    Ltr, Lva = (Ltr - mu) / sd, (Lva - mu) / sd
    ytr, yva = tr["targets"][:, :, 0].astype(np.float32), va["targets"][:, :, 0].astype(np.float32)
    n = len(Ltr); cut = int(n * 0.85); purge = 40
    fit_i, hold_i = np.arange(0, cut - purge), np.arange(cut, n)
    med = np.median(ytr[fit_i], 0)
    L = tf.keras.layers
    inp = L.Input(shape=(Ltr.shape[1],)); h = L.Dense(64, activation="relu")(inp)
    out = L.Dense(6, bias_initializer=tf.keras.initializers.Constant(med))(h)
    m = tf.keras.Model(inp, out)
    m.compile(tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber(delta=1.0))
    best, bw, bad, hist = 1e9, None, 0, []
    for ep in range(30):
        m.fit(Ltr[fit_i], ytr[fit_i], batch_size=64, epochs=1, shuffle=True, verbose=0)
        hm = float(np.abs(m.predict(Ltr[hold_i], verbose=0) - ytr[hold_i]).mean()); hist.append(hm)
        if hm < best - 0.0: best, bw, bad = hm, m.get_weights(), 0
        else:
            bad += 1
            if bad >= 5: break
    m.set_weights(bw)
    ph = np.abs(m.predict(Lva, verbose=0) - yva).mean(0)
    nt = json.load(open(a.naive_table)); ns = json.load(open(a.naive_skill))
    hz = [6, 12, 18, 24, 30, 36]
    pn = {"intercept_train_median": [ns["per_horizon"][str(h)]["intercept_train_median_MAE"] for h in hz],
          "train_mean": [nt["per_naive"]["train_mean"][str(h)]["MAE"] for h in hz],
          "persistence_last_value": [nt["per_naive"]["persistence_last_value"][str(h)]["MAE"] for h in hz]}
    mm = float(ph.mean()); res = {"seed": a.seed, "latent_dim_total": int(Ltr.shape[1]), "groups": sorted(groups), "epochs_run": len(hist),
        "best_holdout_mae": best, "val_mae_per_horizon": dict(zip(hz, map(float, ph))), "val_mae_mean": mm,
        "naive_mean": {k: float(np.mean(v)) for k, v in pn.items()}, "skill_vs": {k: 1 - mm / float(np.mean(v)) for k, v in pn.items()},
        "skill_per_horizon_vs_intercept": dict(zip(hz, [1 - float(p) / q for p, q in zip(ph, pn["intercept_train_median"])]))}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))

if __name__ == "__main__":
    p = argparse.ArgumentParser(); s = p.add_subparsers(dest="c", required=True)
    g = s.add_parser("groups"); g.add_argument("--table"); g.add_argument("--out"); g.set_defaults(f=cmd_groups)
    e = s.add_parser("ae"); e.add_argument("--groups"); e.add_argument("--group"); e.add_argument("--data_dir"); e.add_argument("--out_dir")
    e.add_argument("--adapter_dir"); e.add_argument("--seed", type=int, default=2021); e.add_argument("--latent_dim", type=int, default=8)
    e.add_argument("--filters", type=int, default=32); e.add_argument("--epochs", type=int, default=20); e.set_defaults(f=cmd_ae)
    f = s.add_parser("forecast"); f.add_argument("--groups"); f.add_argument("--data_dir"); f.add_argument("--ae_root"); f.add_argument("--out")
    f.add_argument("--naive_table"); f.add_argument("--naive_skill"); f.add_argument("--seed", type=int, default=2021); f.set_defaults(f=cmd_forecast)
    a = p.parse_args(); a.f(a)
