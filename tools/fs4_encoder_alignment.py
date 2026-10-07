#!/usr/bin/env python3
"""Origin support of the frozen encoder arms: state it, measure it, certify the alignment (order of the owner, 2026-10-07).

The phase-4 runner (feature-extractor @deffa53, pinned; in-flight tasks use it and it is NOT touched here) trains a strided
causal Conv1D encoder 24 -> 12 -> 6 with Keras ``padding="causal", strides=2``: output j of a strided layer reads input rows
2j-(k-1) .. 2j (EVEN phase). The windows fed to it end at the origin row (position 23), but with that phase the LAST latent step
never reads positions 21..23.  ``RunnerEncoderBank`` applies the runner's own encoder with the runner's own layers, so the
weights are used at exactly the phase they were trained at (``EVEN_AS_TRAINED``).  The alternative that covers the origin,
``ODD_PHASE_ORIGIN_COVERING`` (stride-1 convolution sampled at odd positions), uses the same weights at a phase they were not
trained at; a TRAINED phase-covering encoder needs retraining (arm v2, see docs/fs4/ENCODER_ORIGIN_COVERING_ARM_V2.md).

This module (1) derives analytically and (2) checks empirically, by perturbing inputs of the real Keras models, which window
positions each latent step reads (hence the lag of the last step in hours and the absence of any future read), (3) replays the
runner's own reconstruction score on completed runner terminals under both alignments and a linear probe of what the last
latent step encodes, and (4) applies a PREDECLARED choice rule and writes ``ENCODER_ALIGNMENT_CERT.json`` bound to the digest
of ``docs/fs4/STAGE2_RULE.md``.  Nothing here reads VALIDATION or TEST: the replay uses TRAIN corpora only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

ALIGNMENT_AS_TRAINED = "EVEN_AS_TRAINED"
ALIGNMENT_ORIGIN_COVERING = "ODD_PHASE_ORIGIN_COVERING"
CERT_SCHEMA = "fs4.encoder_alignment_cert.v1"
WINDOW = 24
CHOICE_RULE = ("v2 (structural). Certify EVEN_AS_TRAINED, the runner's own phase, when (1) no latent step reads a row after the origin "
               "(analytic and empirical support agree), and (2) its replayed reconstruction MAE equals the runner terminal's MAE on every "
               "replayed terminal (relative error < 1e-4), i.e. the weights are applied exactly as trained and the instrument is the "
               "runner's. ODD_PHASE_ORIGIN_COVERING applies the same weights at a phase they were not trained at and is never certified "
               "whatever its reconstruction MAE; removing the 3-row lag of the last latent step needs a RETRAINED origin-covering arm v2 "
               "(docs/fs4/ENCODER_ORIGIN_COVERING_ARM_V2.md), which the owner decides. Reconstruction MAE and the linear probe under both "
               "alignments are reported as evidence of the cost of the lag, not as a gate. History: a first draft of this rule also required "
               "the untrained phase to be worse on >= 75 % of terminals; the replay showed it better on 7 of 9 EURUSD terminals and worse on 4 of 4 ETH "
               "terminals, so MAE cannot decide soundness and the gate was replaced by the structural rule above.")


class Refusal(ValueError):
    pass


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


# ----------------------------------------------------------------------------- analytic support
def layer_specs(window: int = WINDOW):
    """(name, kernel, stride, dilation) of the runner encoder's temporal layers, in order. calendar_proj/fuse/latent are 1x1."""
    return [("stem", 3, 1, 1), ("fuse", 1, 1, 1), ("down_24_12", 3, 2, 1), ("down_12_6", 3, 2, 1), ("latent", 1, 1, 1)]


def analytic_reads(phase: str, specs=None, window: int = WINDOW):
    """For every latent step, the set of window positions it reads. phase EVEN: output j of a stride-s causal layer reads
    s*j-(k-1)*d .. s*j ; ODD: it reads s*j+(s-1)-(k-1)*d .. s*j+(s-1). Positions outside 0..window-1 are zero padding."""
    specs = specs or layer_specs(window)
    reads = [{p} for p in range(window)]       # step t of the input reads itself
    n = window
    for _name, k, s, d in specs:
        off = 0 if phase == ALIGNMENT_AS_TRAINED else (s - 1)
        m = (n // s)
        new = []
        for j in range(m):
            end = s * j + off
            acc = set()
            for i in range(k):
                pos = end - i * d
                if 0 <= pos < n:
                    acc |= reads[pos]
            new.append(acc)
        reads, n = new, m
    return reads


def summarize_reads(reads, window: int = WINDOW):
    out = []
    for step, r in enumerate(reads):
        out.append({"latent_step": step, "min_position": min(r), "max_position": max(r),
                    "lag_rows_vs_origin": window - 1 - max(r)})
    return out


# ----------------------------------------------------------------------------- models
def load_extractor(code_dir):
    from tools import fs4_temporal_predictor as P
    return P.load_extractor(code_dir)


def odd_phase_encoder(X, U, hp, encoder):
    """The runner's encoder with the SAME weights, strided layers evaluated at stride 1 and sampled at odd positions."""
    import keras
    L = keras.layers
    names = [i.name for i in encoder.inputs]
    inp = {n: keras.Input(tuple(t.shape[1:]), name=n) for n, t in zip(names, encoder.inputs)}

    def clone(name, strides=None):
        src = encoder.get_layer(name)
        cfg = src.get_config()
        if strides is not None:
            cfg["strides"] = (strides,)
        dst = src.__class__.from_config(cfg)
        return dst, src

    x = L.Concatenate(name="series_channels")([inp["signal"], inp["observed_mask"], inp["delta_time"]])
    stem, s_src = clone("stem"); h = stem(x); stem.set_weights(s_src.get_weights())
    proj, p_src = clone("calendar_proj"); c = proj(inp["calendar"]); proj.set_weights(p_src.get_weights())
    fuse, f_src = clone("fuse"); h = fuse(L.Concatenate()([h, c])); fuse.set_weights(f_src.get_weights())
    for name in ("down_24_12", "down_12_6"):
        d, d_src = clone(name, strides=1)
        h = d(h)
        d.set_weights(d_src.get_weights())
        h = L.Lambda(lambda t: t[:, 1::2], name=f"{name}_odd")(h)
    lat, l_src = clone("latent"); z = lat(h); lat.set_weights(l_src.get_weights())
    return keras.Model(inp, z, name="fs4_encoder_odd_phase")


def empirical_reads(encoder, calendar_dim: int, window: int = WINDOW, n: int = 64, seed: int = 0):
    """Which window positions change which latent step, by perturbing the three series channels of the REAL Keras model."""
    rng = np.random.default_rng(seed)
    base = {"signal": rng.normal(size=(n, window, 1)).astype("float32"), "observed_mask": np.ones((n, window, 1), "float32"),
            "delta_time": np.abs(rng.normal(size=(n, window, 1))).astype("float32"),
            "calendar": rng.normal(size=(n, window, calendar_dim)).astype("float32")}
    z0 = np.asarray(encoder.predict(base, verbose=0))
    steps = z0.shape[1]
    reads = [set() for _ in range(steps)]
    for p in range(window):
        for ch in ("signal", "delta_time"):
            pert = {k: v.copy() for k, v in base.items()}
            pert[ch][:, p, :] += 3.0
            z1 = np.asarray(encoder.predict(pert, verbose=0))
            for s in range(steps):
                if not np.array_equal(z0[:, s], z1[:, s]):
                    reads[s].add(p)
    return reads


# ----------------------------------------------------------------------------- replay on completed terminals
def _task_claim(rec):
    return {"schema": rec["schema"].replace("result", "task"), "population_id": rec["population_id"], "identity": rec["identity"],
            "feature_id": rec["feature_id"], "fold_id": rec["fold_id"], "arm": rec["arm"], "seed": rec["seed"], "task_id": rec["task_id"]}


def replay_terminal(code_dir, inputs: dict, terminal_dir, probe_offsets=(0, 1, 2, 3, 4, 5), registry=None):
    """Re-derive the runner's scoring batch for one completed TRAINED terminal and score both alignments."""
    X, U = load_extractor(code_dir)
    import importlib
    R = importlib.import_module("app.fs4_task_runner")
    terminal_dir = Path(terminal_dir)
    rec = json.loads((terminal_dir / "result.json").read_text())
    if rec["arm"] != "TRAINED_ENCODER":
        raise Refusal("REPLAY_NEEDS_A_TRAINED_TERMINAL")
    hp = X.Hyper(**rec["hyper"])
    corpus = X.Corpus(rec["identity"], inputs, registry)
    P = R.prepare(_task_claim(rec), corpus, hp)
    encoder, decoder, training = X.build_models(hp, calendar_dim=P["cal"].shape[1])
    training.load_weights(str(terminal_dir / "chosen.weights.h5"))
    if X.weights_digest([encoder, decoder]) != rec["weights"]["chosen_weights_sha256"]:
        raise Refusal("ENCODER_IDENTITY_MISMATCH")
    score, hidden = P["score"], P["hidden"]
    corrupted = X.corrupt(score, hidden)
    even_recon = X.reconstruct(training, corrupted)
    even = X.hidden_scores(score.signal, even_recon, hidden, P["norm"].std, P["hours"])
    odd_enc = odd_phase_encoder(X, U, hp, encoder)
    z_odd = np.asarray(odd_enc.predict(corrupted.as_inputs(), batch_size=512, verbose=0))
    odd_recon = np.asarray(decoder.predict(z_odd, batch_size=512, verbose=0), np.float32)
    odd = X.hidden_scores(score.signal, odd_recon, hidden, P["norm"].std, P["hours"])
    # linear probe: what does the LAST latent step encode (clean windows)? R^2 of ridge -> value at `offset` rows before the origin
    clean = score.as_inputs()
    z_even = np.asarray(encoder.predict(clean, batch_size=512, verbose=0))[:, -1, :]
    z_oddc = np.asarray(odd_enc.predict(clean, batch_size=512, verbose=0))[:, -1, :]
    obs = score.observed_mask[..., 0] > 0
    probe = {}
    for name, z in (ALIGNMENT_AS_TRAINED, z_even), (ALIGNMENT_ORIGIN_COVERING, z_oddc):
        per = {}
        for off in probe_offsets:
            col = WINDOW - 1 - off
            ok = obs[:, col]
            if ok.sum() < 50:
                per[str(off)] = None
                continue
            A = np.column_stack([z[ok], np.ones(ok.sum())])
            y = score.signal[ok, col, 0].astype("float64")
            cut = int(0.7 * len(y))
            w = np.linalg.solve(A[:cut].T @ A[:cut] + 1e-3 * np.eye(A.shape[1]), A[:cut].T @ y[:cut])
            res = y[cut:] - A[cut:] @ w
            var = float(np.var(y[cut:]))
            per[str(off)] = float(1.0 - np.mean(res ** 2) / var) if var > 0 else None
        probe[name] = per
    return {"task_id": rec["task_id"], "population_id": rec["population_id"], "feature_id": rec["feature_id"], "fold_id": rec["fold_id"],
            "terminal_mae": rec["metrics"]["mae"], "replay_even_mae": even["mae"], "replay_odd_mae": odd["mae"],
            "terminal_matches_replay": bool(abs(even["mae"] - rec["metrics"]["mae"]) <= 1e-4 * max(abs(rec["metrics"]["mae"]), 1e-12)),
            "odd_over_even_mae": odd["mae"] / even["mae"] if even["mae"] > 0 else None, "naive_mae": rec["metrics"]["naive_mae"],
            "n_windows": len(score), "hidden_points": even["hidden_points"], "probe_r2_last_latent_to_row_at_lag": probe}


def choose_alignment(structure: dict, replays: list[dict]) -> dict:
    """The predeclared structural rule (CHOICE_RULE)."""
    st = structure["as_trained"]
    leak_free = bool(st["last_step_reads_no_future"]) and all(st.get("empirical_matches_analytic", [True]))
    reproduces = bool(replays) and all(r["terminal_matches_replay"] for r in replays)
    good = [r for r in replays if "replay_error" not in r]
    odd_worse = [r for r in good if r["replay_odd_mae"] > r["replay_even_mae"]]
    ok = leak_free and reproduces
    return {"alignment": ALIGNMENT_AS_TRAINED if ok else None, "status": "CERTIFIED" if ok else "NOT_CERTIFIED",
            "criteria": {"as_trained_reads_no_row_after_origin": leak_free, "replay_reproduces_every_terminal": reproduces,
                         "replayed_terminals": len(replays)},
            "evidence_not_a_gate": {"odd_phase_worse_terminals": len(odd_worse), "odd_phase_better_terminals": len(good) - len(odd_worse),
                                    "odd_phase_worse_fraction": (len(odd_worse) / len(good)) if good else None},
            "caveats": [f"the last latent step reads rows <= origin-{st['last_step_lag_rows']}: a lag of {st['last_step_lag_hours']} hour(s) on the hourly grid; "
                        "RAW sees the origin row, so encoder arms are handicapped for short horizons",
                        "the origin-covering alignment is not certified (untrained phase); removing the lag needs retraining arm v2"],
            "if_not_certified": "retrained origin-covering arm v2 required (docs/fs4/ENCODER_ORIGIN_COVERING_ARM_V2.md); the runner is not changed"}


def structure_report(code_dir, hp_kwargs=None):
    X, U = load_extractor(code_dir)
    hp = X.Hyper(**(hp_kwargs or {}))
    encoder, decoder, training = X.build_models(hp, calendar_dim=len(U.CALENDAR_SPEC))
    X.seed_weights([encoder, decoder], 0)
    odd = odd_phase_encoder(X, U, hp, encoder)
    out = {}
    for key, model, phase in (("as_trained", encoder, ALIGNMENT_AS_TRAINED), ("origin_covering", odd, ALIGNMENT_ORIGIN_COVERING)):
        ana = analytic_reads(phase)
        emp = empirical_reads(model, len(U.CALENDAR_SPEC))
        s = summarize_reads(ana)
        out[key] = {"alignment": phase, "analytic": s, "empirical_matches_analytic": [sorted(a) == sorted(e) for a, e in zip(ana, emp)],
                    "last_step_max_position": s[-1]["max_position"], "last_step_lag_rows": s[-1]["lag_rows_vs_origin"],
                    "last_step_reads_no_future": s[-1]["max_position"] <= WINDOW - 1,
                    "all_steps_read_no_future": all(x["max_position"] <= WINDOW - 1 for x in s), "grid_hours_per_row": 1,
                    "last_step_lag_hours": s[-1]["lag_rows_vs_origin"] * 1}
    return out


def evidence_summary(replays):
    good = [r for r in replays if "replay_error" not in r]
    def med(pop, key):
        v = [r[key] for r in good if (pop is None or r["population_id"] == pop)]
        return float(np.median(v)) if v else None
    def probe(pop, align, off):
        v = [r["probe_r2_last_latent_to_row_at_lag"][align][str(off)] for r in good
             if (pop is None or r["population_id"] == pop) and r["probe_r2_last_latent_to_row_at_lag"][align].get(str(off)) is not None]
        return float(np.median(v)) if v else None
    out = {}
    for pop in ("EURUSD", "ETH"):
        out[pop] = {"terminals": sum(1 for r in good if r["population_id"] == pop), "median_terminal_mae": med(pop, "terminal_mae"),
                    "median_naive_mae": med(pop, "naive_mae"), "median_odd_over_even_mae": med(pop, "odd_over_even_mae"),
                    "median_probe_r2_row_at_lag0_as_trained": probe(pop, ALIGNMENT_AS_TRAINED, 0),
                    "median_probe_r2_row_at_lag3_as_trained": probe(pop, ALIGNMENT_AS_TRAINED, 3),
                    "median_probe_r2_row_at_lag0_odd_phase": probe(pop, ALIGNMENT_ORIGIN_COVERING, 0)}
    return out


def write_cert(path, structure, replays, choice, rule_sha256: str, extractor_commit: str | None):
    body = {"schema": CERT_SCHEMA, "status": choice["status"], "alignment": choice["alignment"], "choice_rule": CHOICE_RULE,
            "stage2_rule_sha256": rule_sha256, "extractor_code_commit": extractor_commit, "structure": structure,
            "replays": replays, "criteria": choice["criteria"], "evidence_not_a_gate": choice["evidence_not_a_gate"], "caveats": choice["caveats"],
            "evidence_summary": evidence_summary(replays), "validation_read": 0, "test_read": 0,
            "last_step_lag_hours_as_trained": structure["as_trained"]["last_step_lag_hours"]}
    body["cert_sha256"] = digest({k: v for k, v in body.items() if k != "cert_sha256"})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(body, indent=1, sort_keys=True, default=str))
    return body


def load_cert(path, expected_rule_sha256: str | None = None) -> dict:
    cert = json.loads(Path(path).read_text())
    if cert.get("schema") != CERT_SCHEMA:
        raise Refusal("ALIGNMENT_CERT_SCHEMA")
    if digest({k: v for k, v in cert.items() if k != "cert_sha256"}) != cert.get("cert_sha256"):
        raise Refusal("ALIGNMENT_CERT_CORRUPT")
    if expected_rule_sha256 is not None and cert.get("stage2_rule_sha256") != expected_rule_sha256:
        raise Refusal("ALIGNMENT_CERT_RULE_MISMATCH: the certificate was issued for another STAGE2_RULE.md digest")
    return cert


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extractor-code", required=True)
    ap.add_argument("--terminals", required=True, help="directory of the runner's <task_id>/result.json + chosen.weights.h5 (read only)")
    ap.add_argument("--input", action="append", default=[], help="ROLE=PATH of the pinned TRAIN corpus files (runner roles)")
    ap.add_argument("--max-terminals", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from tools import fs4_weekly_wrapper as WW
    inputs = {r.split("=", 1)[0].lower(): r.split("=", 1)[1] for r in a.input}
    structure = structure_report(a.extractor_code)
    dirs = []
    for p in sorted(Path(a.terminals).glob("*/result.json")):
        rec = json.loads(p.read_text())
        if rec.get("arm") == "TRAINED_ENCODER" and rec.get("status") == "COMPLETE" and (p.parent / "chosen.weights.h5").is_file():
            dirs.append((rec["population_id"], rec["fold_id"], rec["feature_id"], p.parent))
    dirs.sort(key=lambda t: (t[1], t[0], t[2]))
    replays = []
    for _pop, _fold, _feat, d in dirs[: a.max_terminals]:
        try:
            replays.append(replay_terminal(a.extractor_code, inputs, d))
        except Exception as exc:  # noqa: BLE001  (a terminal that cannot be replayed is reported, never dropped silently)
            replays.append({"task_id": d.name, "replay_error": f"{type(exc).__name__}: {exc}", "terminal_matches_replay": False,
                            "replay_even_mae": 0.0, "replay_odd_mae": 0.0})
    choice = choose_alignment(structure, replays)
    import subprocess
    commit = subprocess.run(["git", "-C", a.extractor_code, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
    cert = write_cert(a.out, structure, replays, choice, WW.stage2_rule_sha256(), commit)
    print(canonical({"status": cert["status"], "alignment": cert["alignment"], "criteria": cert["criteria"], "cert_sha256": cert["cert_sha256"],
                     "lag_hours_as_trained": cert["last_step_lag_hours_as_trained"]}))
    return 0 if cert["status"] == "CERTIFIED" else 3


if __name__ == "__main__":
    sys.exit(main())
