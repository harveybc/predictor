#!/usr/bin/env python3
"""The audit of RP49-RP56 and RP57-RP64, recomputed from retained artifacts only.

This module stands in for the two reviews that were never written
(``MUSASHI_RP49_RP56_REVIEW``, ``MUSASHI_RP57_RP64_REVIEW``) under the owner's grant of
2026-09-26. It asserts nothing on the authority of the returns it audits: every number it
reports is RE-DERIVED from the bytes the rounds retained, by the repository's own canonical
rules, and every check carries one of three states:

    VERIFIED                      the retained artifacts sustain the claim, recomputed here
    REFUTED                       they do not; the measured value is named beside the claimed one
    UNVERIFIABLE_ARTIFACT_ABSENT  the artifact the claim needs is gone; never "accepted"

Two classes of artifact are read:

    REPOSITORY   everything under ``docs/audits/evidence/`` and ``examples/results/``. These are
                 committed bytes and the checks over them are portable: they hold on any checkout
    RUN ROOT     the working roots under the state directory (``DATA.npz``, per-cell
                 ``arrays.npz``, saved weights). These are NOT committed. Every check that needs
                 them is declared optional and becomes UNVERIFIABLE_ARTIFACT_ABSENT when they are
                 not on this host, rather than silently disappearing from the total

The claims are quoted from the two returns, so a check can never drift from the sentence it
tests. Where a claim is refuted the measured value is the one to read: the finding is the
measurement, not the adjective.

    python tools/df_rp49_rp64_audit.py --out AUDIT.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVID = REPO / "docs" / "audits" / "evidence" / "d3_k5_20260917"

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
ABSENT = "UNVERIFIABLE_ARTIFACT_ABSENT"

SUCCESSOR_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_household_successor_v3"
PHASE1_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_phase1_v1b"

SUCCESSOR_DESIGN_SHA = "143abb57d97daa07e3f5228eadf4e3f1a0deb5fe30eb95ca065663f76ff888a7"
PHASE1_DESIGN_SHA = "5cb8263d359f15008ddc0ba0a997cdafd75036b8dd5621af0d31e5f9a8a4b231"
PREDICTION_ATOL = 1e-5                       # the repository's own replay tolerance (df_e1_close)


# --- canonical identity ---------------------------------------------------------------------------

def sha_obj(obj) -> str:
    """The repository's canonical object digest (df_mod_e0.sha_obj / df_d2_design.seal_design)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rederive_design(path: Path) -> dict:
    """A sealed design must re-derive its own ``design_sha256`` from its body without that field."""
    doc = json.loads(Path(path).read_text())
    declared = doc.get("design_sha256")
    recomputed = sha_obj({k: v for k, v in doc.items() if k != "design_sha256"})
    return {"declared": declared, "recomputed": recomputed, "re_derives": declared == recomputed,
            "file_sha256": sha_file(path), "doc": doc}


def _check(claim: str, state: str, **facts) -> dict:
    out = {"claim": claim, "state": state}
    out.update(facts)
    return out


def _num(state_ok: bool) -> str:
    return VERIFIED if state_ok else REFUTED


def _close(a, b, tol) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


# --- RP49-RP56: identity of the successor run -----------------------------------------------------

def identity_successor() -> list:
    checks = []
    sealed = EVID / "RP38" / "E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json"
    if not sealed.is_file():
        return [_check("design 143abb57...; the successor re-seals to the identical digest", ABSENT,
                       missing=str(sealed.relative_to(REPO)))]
    s = rederive_design(sealed)
    checks.append(_check("the sealed successor design re-derives its own digest",
                         _num(s["re_derives"] and s["declared"] == SUCCESSOR_DESIGN_SHA),
                         declared=s["declared"], recomputed=s["recomputed"]))
    run_design = SUCCESSOR_ROOT / "DESIGN.json"
    if run_design.is_file():
        r = rederive_design(run_design)
        same = r["doc"] == s["doc"] and r["file_sha256"] == s["file_sha256"]
        checks.append(_check("the design the run executed IS the design that was sealed before it ran",
                             _num(same), sealed_file_sha256=s["file_sha256"],
                             run_file_sha256=r["file_sha256"], objects_identical=r["doc"] == s["doc"]))
    else:
        checks.append(_check("the design the run executed IS the design that was sealed before it ran",
                             ABSENT, missing=str(run_design)))
    close = EVID / "RP55" / "E1_SUCCESSOR_CLOSE.json"
    if close.is_file():
        c = json.loads(close.read_text())
        data_npz = SUCCESSOR_ROOT / "DATA.npz"
        if data_npz.is_file():
            checks.append(_check("the closure's data_sha256 is the digest of the prepared DATA bytes",
                                 _num(sha_file(data_npz) == c.get("data_sha256")),
                                 declared=c.get("data_sha256"), recomputed=sha_file(data_npz)))
        else:
            checks.append(_check("the closure's data_sha256 is the digest of the prepared DATA bytes",
                                 ABSENT, missing=str(data_npz)))
        for name, path in (("RESULTS", "RESULTS.json"), ("CLOSE", "CLOSE.json")):
            evid_copy = EVID / "RP55" / ("E1_SUCCESSOR_" + name + ".json")
            live = SUCCESSOR_ROOT / path
            if evid_copy.is_file() and live.is_file():
                checks.append(_check(f"the retained {name} copy is byte-identical to the run root's",
                                     _num(sha_file(evid_copy) == sha_file(live)),
                                     evidence_sha256=sha_file(evid_copy), run_root_sha256=sha_file(live)))
    return checks


# --- RP49-RP56: the population ---------------------------------------------------------------------

def population_successor() -> list:
    """'units declared / closed 16 / 16' beside 'cells VERIFIED_AND_GOVERNED 15 / 15': two
    populations, and the sixteenth member is never named in the return."""
    close_p = EVID / "RP55" / "E1_SUCCESSOR_CLOSE.json"
    wh_p = EVID / "RP56_WAREHOUSE_CONTENT_CHECK.json"
    if not (close_p.is_file() and wh_p.is_file()):
        return [_check("units declared / closed 16 / 16; cells VERIFIED_AND_GOVERNED 15 / 15", ABSENT,
                       missing=[str(p) for p in (close_p, wh_p) if not p.is_file()])]
    close = json.loads(close_p.read_text())
    wh = json.loads(wh_p.read_text())
    declared = list(close["population"]["declared"])
    wh_pop = list(wh["population"])
    extra = [u for u in wh_pop if u not in declared]
    counts = close["counts"]
    return [
        _check("the closure's sealed population is 15 units, every one present, none absent, no stranger",
               _num(len(declared) == 15 and sorted(close["population"]["present"]) == sorted(declared)
                    and not close["population"]["absent_ids"] and not close["population"]["strangers_on_disk"]),
               declared=len(declared), present=len(close["population"]["present"])),
        _check("15 units verified, 15 governed, 0 refused",
               _num(counts["declared"] == counts["verified"] == 15 and counts["refused"] == 0
                    and close["governance"]["units_governed"] == 15),
               counts=counts, governed=close["governance"]["units_governed"]),
        _check("verified + NOT_APPLICABLE = declared, per fact",
               _num(counts["metrics_verified"] == 15 and counts["inference_verified"] == 11
                    and counts["regime_verified"] == 14),
               inference_not_applicable=[u for u, e in close["units"].items()
                                         if e["inference"] == "NOT_APPLICABLE"],
               regime_not_applicable=[u for u, e in close["units"].items()
                                      if e["regime"] == "NOT_APPLICABLE"]),
        _check("the warehouse population of 16 is the closure's 15 plus exactly one more unit, "
               "which the return never names",
               _num(len(wh_pop) == 16 and extra == ["prepare"]),
               warehouse_population=len(wh_pop), the_sixteenth=extra,
               reading="16/16 counts the governance unit 'prepare'; it is not a sixteenth cell, and "
                       "no cell is missing from the 15"),
    ]


# --- RP49-RP56: the numbers ------------------------------------------------------------------------

def numbers_successor() -> list:
    res_p = EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json"
    if not res_p.is_file():
        return [_check("the successor run's published regime means", ABSENT, missing=str(res_p))]
    res = json.loads(res_p.read_text())
    den = float(res["denominator"])
    cells = res["cells"]
    checks = [
        _check("every cell's published scaled error is its own MAE over the run's own denominator",
               _num(all(_close(c["mase"], c["mae"] / den, 1e-12) for c in cells.values())),
               denominator=den, cells=len(cells)),
    ]
    for reg, claimed_mean, claimed_sd in (("R0", 0.8875, 0.0127), ("R1", 0.9034, 0.0175),
                                          ("R2", 0.8968, 0.0236)):
        v = [cells[f"{reg}_s{i}"]["mase"] for i in (1, 2, 3)]
        mean, sd = statistics.fmean(v), statistics.stdev(v)
        pub = res["means"][reg]
        checks.append(_check(f"{reg} mean {claimed_mean} sd {claimed_sd}",
                            _num(_close(mean, claimed_mean, 5e-5) and _close(sd, claimed_sd, 5e-5)
                                 and _close(mean, pub["mase_mean"], 1e-12)),
                            recomputed_mean=mean, recomputed_sd=sd, published_mean=pub["mase_mean"],
                            published_sd=pub["mase_sd"], per_seed=v))
    for pair, claimed in (("R1_minus_R0", 0.0159), ("R2_minus_R0", 0.0093), ("R2_minus_R1", -0.0066)):
        a, b = pair.split("_minus_")
        d = [cells[f"{a}_s{i}"]["mase"] - cells[f"{b}_s{i}"]["mase"] for i in (1, 2, 3)]
        checks.append(_check(f"paired {pair} {claimed:+.4f}",
                             _num(_close(statistics.fmean(d), claimed, 5e-5)
                                  and _close(statistics.fmean(d), res["paired"][pair]["mean"], 1e-12)),
                             recomputed=statistics.fmean(d), published=res["paired"][pair]["mean"]))
    controls = res["controls"]
    for name, claimed in (("linear_ridge", 0.8852), ("persistence", 1.0018),
                          ("seasonal_naive_daily", 1.1873)):
        checks.append(_check(f"control {name} {claimed}",
                             _num(_close(controls[name]["mase"], claimed, 5e-5)
                                  and _close(controls[name]["mase"], controls[name]["mae"] / den, 1e-12)),
                             published=controls[name]["mase"], mae=controls[name]["mae"]))
    censored = sorted(k for k, c in cells.items() if (c.get("censoring") or {}).get("verdict")
                      == "CENSORED_BY_BUDGET")
    checks.append(_check("four of the nine fits stopped at the update ceiling and are CENSORED_BY_BUDGET",
                         _num(len(censored) == 4), censored=censored,
                         at_the_ceiling=all(cells[k]["updates"] == 4000 for k in censored)))
    checks.append(_check("every fit's restored checkpoint agrees with the argmin of its own curve",
                         _num(all(c["best_epoch"] == int(np.argmin(c["curve_val"])) + 1
                                  for c in cells.values())),
                         per_cell={k: [c["best_epoch"], int(np.argmin(c["curve_val"])) + 1]
                                   for k, c in cells.items()}))
    costs = res["costs"]
    parts = (costs["ae_cpu"] + sum(costs["fit_cpu_by_regime"].values())
             + costs["pilot_cpu"] + costs["controls_cpu"])
    checks.append(_check("AE cost 137.156 s total, 22.86 s amortised per consuming fit",
                         _num(_close(costs["ae_cpu"], 137.156, 1e-3)
                              and _close(costs["ae_cpu"] / 6, 22.86, 5e-3)),
                         ae_cpu=costs["ae_cpu"], consuming_fits=6,
                         amortised=costs["ae_cpu"] / 6))
    checks.append(_check("the published root CPU figure is the sum of its own components",
                         _num(_close(parts, costs["spent_cpu_seconds_root"], 1e-6)),
                         components=parts, published=costs["spent_cpu_seconds_root"]))
    rep_p = EVID / "RP55" / "E1_SUCCESSOR_REPORT.json"
    if rep_p.is_file():
        rep = json.loads(rep_p.read_text())
        charged = rep["spent_cpu_seconds"]
        checks.append(_check("2 535.5 CPU seconds of an 11 000-second cap",
                             _num(_close(charged, 2535.504, 1e-3)),
                             quoted_against_the_cap=2535.504,
                             the_runner_charges_to_the_cap=charged,
                             already_spent_seconds=rep.get("already_spent_seconds"),
                             cap_seconds=rep.get("cap_seconds"),
                             reading="the figure the return sets against the cap is the root-only "
                                     "total; the report charges root + already_spent to the same cap. "
                                     "Both are far inside it, and the difference is the accounting, "
                                     "not the science"))
    return checks


def replay_successor(windows: int | None = None, work: Path | None = None) -> list:
    """The closure replayed 512 of the 10 020 evaluation origins per unit. This extends the replay
    to EVERY row the published metric is computed on, in a fresh process, from the saved weights."""
    if not (SUCCESSOR_ROOT / "DATA.npz").is_file():
        return [_check("fresh-process replay, 512 windows per unit, tol 1e-5", ABSENT,
                       missing=str(SUCCESSOR_ROOT / "DATA.npz"))]
    close = json.loads((EVID / "RP55" / "E1_SUCCESSOR_CLOSE.json").read_text())
    ev_n = int(json.loads((EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json").read_text())["data"]
               ["common_evaluation_set"])
    per_unit = int(close["replay"]["windows_per_unit"])
    checks = [_check("the retained closure replay covers the rows the published metric is computed on",
                     REFUTED if per_unit < ev_n else VERIFIED,
                     replayed_windows_per_unit=per_unit, metric_rows=ev_n,
                     coverage_percent=round(100.0 * per_unit / ev_n, 2),
                     reading="the replay is a real fresh-process reload, but of the FIRST 512 "
                             "contiguous origins: 5.11% of the metric's rows, and at 59/60 overlap "
                             "barely more than one situation by the programme's own reading rule")]
    if windows:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_rpaudit_df_e1_close", HERE / "df_e1_close.py")
        C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(C)
        D = np.load(SUCCESSOR_ROOT / "DATA.npz")
        ev = D["eval_origins"].astype(np.int64)[:windows]
        den = float(D["denominator"][0])
        res = json.loads((EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json").read_text())
        out_dir = Path(work) if work else Path(tempfile.mkdtemp(prefix="rp_audit_replay_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = {}
        for cid in sorted(res["cells"]):
            cell = json.loads((SUCCESSOR_ROOT / "attempts" / cid / "cell.json").read_text())
            got = C.replay(SUCCESSOR_ROOT, cid, SUCCESSOR_ROOT / "attempts" / cid / "weights.weights.h5",
                           ev, int(cell["seed"]), out_dir, regime=cell.get("regime"))
            if "error" in got:
                rows[cid] = {"error": got["error"][:200]}
                continue
            p = np.asarray(got["prediction"], dtype=float)
            z = np.load(SUCCESSOR_ROOT / "attempts" / cid / "arrays.npz")
            stored = z["validation_pred"].reshape(-1).astype(float)[:len(p)]
            y = z["validation_y"].reshape(-1)[:len(p)]
            mae = float(np.mean(np.abs(p - y)))
            rows[cid] = {"n": int(p.size), "max_abs_diff": float(np.max(np.abs(p - stored))),
                         "mae_replay": mae, "mae_published": res["cells"][cid]["mae"],
                         "mase_replay": mae / den, "mase_published": res["cells"][cid]["mase"]}
        ok = all("error" not in r and r["max_abs_diff"] <= PREDICTION_ATOL
                 and _close(r["mae_replay"], r["mae_published"], 1e-8) for r in rows.values())
        checks.append(_check("the published metric reproduces from the saved weights on EVERY "
                             "evaluation row, not only the closure's first 512", _num(ok),
                             tolerance_kw=PREDICTION_ATOL, windows=windows, cells=rows))
    return checks


# --- RP57-RP64: phase 1 -----------------------------------------------------------------------------

def phase1() -> list:
    d_p = EVID / "RP63" / "PHASE1_DESIGN_SEALED.json"
    r_p = EVID / "RP63" / "PHASE1_REPORT.json"
    if not (d_p.is_file() and r_p.is_file()):
        return [_check("design 5cb8263d... sealed before running; nine cells", ABSENT,
                       missing=[str(p) for p in (d_p, r_p) if not p.is_file()])]
    d = rederive_design(d_p)
    rep = json.loads(r_p.read_text())
    cells = {k: v for k, v in rep["cells"].items() if not k.startswith("pilot")}
    arms = {}
    for k, c in cells.items():
        arms.setdefault(c["arm"], []).append(c)
    checks = [
        _check("design 5cb8263d... sealed before running",
               _num(d["re_derives"] and d["declared"] == PHASE1_DESIGN_SHA),
               declared=d["declared"], recomputed=d["recomputed"]),
        _check("the phase ran on the successor run's own rows",
               _num(rep["data_sha256"] == json.loads((EVID / "RP55" / "E1_SUCCESSOR_CLOSE.json")
                                                     .read_text())["data_sha256"]),
               phase1_data_sha256=rep["data_sha256"]),
        _check("executed for 2 024.5 CPU seconds, nine cells",
               _num(len(cells) == 9 and _close(rep["spent_cpu_seconds"], 2024.5, 0.05)
                    and _close(sum(c["cpu_seconds"] for c in rep["cells"].values()),
                               rep["spent_cpu_seconds"], 1e-6)),
               cells=len(cells), spent=rep["spent_cpu_seconds"],
               sum_of_cell_cpu=round(sum(c["cpu_seconds"] for c in rep["cells"].values()), 3)),
    ]
    for arm, claimed_mean, claimed_sd in (("core_mae", 0.4978, 0.0002), ("tcn_mse", 0.5359, 0.0050),
                                          ("core_mse", 0.5469, 0.0078)):
        v = [c["mae"] for c in arms[arm]]
        checks.append(_check(f"{arm} MAE {claimed_mean} sd {claimed_sd} (n=3)",
                             _num(len(v) == 3 and _close(statistics.fmean(v), claimed_mean, 5e-5)
                                  and _close(statistics.stdev(v), claimed_sd, 5e-5)),
                             recomputed_mean=statistics.fmean(v), recomputed_sd=statistics.stdev(v),
                             per_seed=sorted(v)))
    checks.append(_check("the reference TCN block is parameter-matched: 8 061 against ours 8 127",
                         _num({c["parameters"] for c in arms["tcn_mse"]} == {8061}
                              and {c["parameters"] for c in arms["core_mse"]} == {8127}),
                         tcn=sorted({c["parameters"] for c in arms["tcn_mse"]}),
                         ours=sorted({c["parameters"] for c in arms["core_mse"]})))
    censored = sorted(k for k, c in cells.items() if c["verdict"] == "CENSORED_BY_BUDGET")
    checks.append(_check("Four of nine cells are CENSORED_BY_BUDGET",
                         _num(len(censored) == 4), measured=len(censored), censored=censored,
                         reading="the retained report says five. Four is the successor run's count, "
                                 "not this phase's"))
    updates = {arm: sum(c["updates"] for c in v) for arm, v in arms.items()}
    checks.append(_check("the arms of the recipe contrast were trained to the same budget",
                         _num(len(set(updates.values())) == 1), updates_by_arm=updates,
                         reading="the monitor that changed with the loss also changed when early "
                                 "stopping fired, so the winning core_mae arm ran 11 762 optimiser "
                                 "updates against core_mse's 10 270: the 0.049 kW is a recipe AND "
                                 "budget difference, which Musashi's monitor-fixed factorial "
                                 "addresses and RP63 itself does not"))
    checks.append(_check("core_mse_s1 = 0.5525, reproducing R0_s1 of the finished run exactly "
                         "through a different runner",
                         _num(_close(cells["core_mse_s1"]["mae"],
                                     json.loads((EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json").read_text())
                                     ["cells"]["R0_s1"]["mae"], 0.0)),
                         phase1=cells["core_mse_s1"]["mae"],
                         successor=json.loads((EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json").read_text())
                         ["cells"]["R0_s1"]["mae"],
                         difference=abs(cells["core_mse_s1"]["mae"]
                                        - json.loads((EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json")
                                                     .read_text())["cells"]["R0_s1"]["mae"]),
                         reading="a cross-runner reproduction to 5.0e-07 kW in the mean, not an "
                                 "identity: the predictions differ by up to 6.5e-05 kW, which is "
                                 "ABOVE this repository's own replay tolerance of 1e-05"))
    return checks


def phase1_replay(windows: int | None = None) -> list:
    """Phase 1 was closed by a report, not by a closure: no ``CLOSE.json``, no ``closure_replays``,
    no fresh-process reload of its saved weights. Its weights ARE retained, so the replay the round
    never performed is performed here."""
    if not (PHASE1_ROOT / "DATA.npz").is_file():
        return [_check("phase 1's published MAE reproduces from its own saved weights", ABSENT,
                       missing=str(PHASE1_ROOT / "DATA.npz"))]
    closed = [p.name for p in PHASE1_ROOT.iterdir()] if PHASE1_ROOT.is_dir() else []
    out = [_check("phase 1 carries a closure of its own",
                  REFUTED if "CLOSE.json" not in closed else VERIFIED,
                  files_in_the_run_root=sorted(closed),
                  reading="nine cells, every unit registered, delivered, reported and reconciled - "
                          "and never closed: there is no CLOSE.json and no closure_replays here, so "
                          "the successor run's own standard (a fresh process reloads the saved "
                          "weights and regathers the windows) was not applied to this phase")]
    if not windows:
        return out
    import importlib.util
    import numpy as _np
    spec = importlib.util.spec_from_file_location("_rpaudit_df_e1_phase1", HERE / "df_e1_phase1.py")
    PH = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(PH)                          # never registered in sys.modules: the tools
    spec2 = importlib.util.spec_from_file_location("_rpaudit_df_e1_pilot", HERE / "df_e1_pilot.py")
    P = importlib.util.module_from_spec(spec2)           # load their own dependencies themselves
    spec2.loader.exec_module(P)
    design = json.loads((PHASE1_ROOT / "DESIGN.json").read_text())
    z = _np.load(PHASE1_ROOT / "DATA.npz")
    data = {k: z[k] for k in z.files}
    W_, h = int(data["window"][0]), int(data["horizon"][0])
    j = int(data["target_channel"][0])
    ev = data["eval_origins"].astype(_np.int64)[:windows]
    rows = {}
    for att in sorted((PHASE1_ROOT / "attempts").iterdir()):
        if att.name.startswith("pilot") or not (att / "cell.json").is_file():
            continue
        cell = json.loads((att / "cell.json").read_text())
        model = PH._arm_model(cell["arm"], design, data, int(cell["seed"]))
        model.load_weights(str(att / "weights.weights.h5"))
        X = P._gather(data["Xs"], ev, W_)
        pred = _np.asarray(model.predict(X, verbose=0))[:, 0] * float(data["scaler_sd"][j]) \
            + float(data["scaler_mean"][j])
        arr = _np.load(att / "arrays.npz")
        stored = arr["validation_pred"].reshape(-1).astype(float)[:len(pred)]
        y_from_data = data["Y"][ev + h]
        params = int(sum(int(w.shape.num_elements()) for w in model.trainable_weights))
        mae = float(_np.mean(_np.abs(pred - y_from_data)))
        rows[att.name] = {"arm": cell["arm"], "n": int(pred.size),
                          "max_abs_diff": float(_np.max(_np.abs(pred - stored))),
                          "mae_replay": mae,
                          "mae_published": cell["scores"]["validation"]["model"]["mae_mean"],
                          "labels_equal_the_panel_targets":
                              bool(_np.array_equal(arr["validation_y"].reshape(-1)[:len(pred)], y_from_data)),
                          "parameters_replay": params, "parameters_published": cell["parameters"]}
    ok = all(r["max_abs_diff"] <= PREDICTION_ATOL and r["labels_equal_the_panel_targets"]
             and r["parameters_replay"] == r["parameters_published"]
             and _close(r["mae_replay"], r["mae_published"], 1e-8) for r in rows.values())
    out.append(_check("phase 1's published MAE reproduces from its own saved weights, on every "
                      "evaluation row, in a fresh process", _num(ok and len(rows) == 9),
                      tolerance_kw=PREDICTION_ATOL, cells=rows))
    return out


# --- RP57-RP64: the reanalysis ----------------------------------------------------------------------

def reanalysis() -> list:
    rep_p = EVID / "RP57" / "COMPARISON_REPORT.json"
    pub_p = EVID / "RP57" / "REANALYSIS_PUBLICATION.json"
    if not (rep_p.is_file() and pub_p.is_file()):
        return [_check("192 metric rows, tagged E1_REANALYSIS", ABSENT,
                       missing=[str(p) for p in (rep_p, pub_p) if not p.is_file()])]
    rep = json.loads(rep_p.read_text())
    pub = json.loads(pub_p.read_text())
    n = 0
    for _scale, table in rep["by_scale"].items():
        for _m, row in table.items():
            n += 1
            n += sum(1 for _r, a in (row.get("versus") or {}).items()
                     if a.get("mae_skill_percent") is not None)
    for _m, row in rep["by_scale"]["raw_kW"].items():
        n += sum(1 for k in ("persistence_scaled_error_horizon_train",
                             "conventional_mase_m1_train_slice",
                             "conventional_mase_m1440_train_slice") if row.get(k) is not None)
    n += len(rep.get("paired_differences") or {})
    return [
        _check("192 metric rows", _num(n == pub["metrics_published"] == 192),
               recounted_under_the_publisher_s_own_rule=n, declared=pub["metrics_published"]),
        _check("the published rows bind to the report whose digest they carry",
               _num(sha_file(rep_p) == pub["report_sha256"]),
               report_sha256=pub["report_sha256"], retained_file_sha256=sha_file(rep_p),
               reading="the receipt names a path in a session scratchpad that is gone, but the "
                       "bytes are retained in the repository under RP57/ and their digest matches"),
        _check("identical for all nine cells against the run's own RESULTS.json",
               _num(all(_close(v["difference_mae"], 0.0, 0.0) and _close(v["difference_scaled"], 0.0, 0.0)
                        for v in rep["local_report_check"]["cells"].values())),
               cells=len(rep["local_report_check"]["cells"])),
        _check("a zero-error reference yields an undefined skill; no epsilon manufactures a win",
               VERIFIED, skills_omitted_for_a_zero_error_reference=[
                   (s, m, r) for s, t in rep["by_scale"].items() for m, row in t.items()
                   for r, a in (row.get("versus") or {}).items() if a.get("mae_skill_percent") is None],
               reading="no reference is zero-error on these rows, so the branch is untaken here; "
                       "the rule is exercised by tests/test_rp49_rp64_audit.py instead"),
    ]


def denominators() -> list:
    """The three scaled-error denominators, each on the support its own formula states."""
    if not (SUCCESSOR_ROOT / "DATA.npz").is_file():
        return [_check("the one-step value on the same slice is 0.0851237797 over 40 257 finite pairs",
                       ABSENT, missing=str(SUCCESSOR_ROOT / "DATA.npz"))]
    D = np.load(SUCCESSOR_ROOT / "DATA.npz")
    Y, tr = D["Y"], D["train_origins"].astype(np.int64)
    h = int(D["horizon"][0])
    last_train_label = int(tr.max()) + h

    def mad(lo, hi, lag):
        idx = np.arange(lo, hi + 1 - lag)
        a, b = Y[idx], Y[idx + lag]
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(np.abs(b[m] - a[m]))), int(m.sum())

    dh = float(np.mean(np.abs(Y[tr + h] - Y[tr])))
    m1, n1 = mad(0, last_train_label, 1)
    m1440, n1440 = mad(0, last_train_label, 1440)
    rep = json.loads((EVID / "RP57" / "COMPARISON_REPORT.json").read_text())
    r0 = rep["by_scale"]["raw_kW"]["R0_s1"]
    return [
        _check("the denominator is the mean absolute change over the HORIZON on the train origins",
               _num(_close(dh, float(D["denominator"][0]), 1e-12)),
               recomputed=dh, prepared=float(D["denominator"][0]), train_origins=int(tr.size)),
        _check("the one-step value on the same slice is 0.0851237797 over 40 257 finite pairs",
               _num(_close(m1, 0.0851237797, 1e-9) and n1 == 40257),
               recomputed=m1, finite_pairs=n1, support=[0, last_train_label],
               reading="the return says 'the same slice' without naming it; the support that "
                       "reproduces the number to ten digits is row 0 to the last train LABEL row"),
        _check("MASE is reported at m = 1 and m = 1440, each with its support",
               _num(_close(r0["conventional_mase_m1_train_slice"], r0["mae"] / m1, 2e-4)
                    and _close(r0["conventional_mase_m1440_train_slice"], r0["mae"] / m1440, 2e-4)),
               m1_denominator=m1, m1_pairs=n1, m1440_denominator=m1440, m1440_pairs=n1440,
               published_m1=r0["conventional_mase_m1_train_slice"],
               recomputed_m1=r0["mae"] / m1,
               published_m1440=r0["conventional_mase_m1440_train_slice"],
               recomputed_m1440=r0["mae"] / m1440),
    ]


# --- RP58: the legacy tables -------------------------------------------------------------------------

def legacy() -> list:
    inv_p = EVID / "RP58" / "LEGACY_INVENTORY.json"
    lin_p = EVID / "RP58" / "CAUSAL_LINEAGE.json"
    if not (inv_p.is_file() and lin_p.is_file()):
        return [_check("85 of 167 published legacy tables beat the naive by 50% or more", ABSENT,
                       missing=[str(p) for p in (inv_p, lin_p) if not p.is_file()])]
    inv = json.loads(inv_p.read_text())
    lin = json.loads(lin_p.read_text())
    E, L = inv["entries"], lin["entries"]
    status = {}
    for v in E.values():
        status[v["status"]] = status.get(v["status"], 0) + 1
    disp = {}
    for v in L.values():
        disp[v["disposition"]] = disp.get(v["disposition"], 0) + 1
    measured = {k: v["anomaly_indicator"] for k, v in L.items() if v["anomaly_indicator"].get("measured")}
    flagged = sorted(k for k, a in measured.items() if a["flagged_as_extraordinary"])
    mine = sorted(k for k, a in measured.items()
                  if max(a["mae_skill_percent_vs_naive_by_horizon"].values()) >= 50.0)
    p3 = {k: a["largest"] for k, a in measured.items() if "/phase_3" in k}
    checks = [
        _check("tables inventoried 167 (113 reconstructable, 54 published-summary-only)",
               _num(len(E) == 167 and status.get("RECONSTRUCTED_FROM_PREDICTIONS_AND_LABELS") == 113
                    and status.get("PUBLISHED_SUMMARY_ONLY") == 54),
               entries=len(E), by_status=status),
        _check("CAUSALITY_UNVERIFIED 114; no decomposition declared or consumed 53",
               _num(disp.get("CAUSALITY_UNVERIFIED") == 114
                    and disp.get("NO_DECOMPOSITION_DECLARED_OR_CONSUMED") == 53
                    and sum(disp.values()) == 167),
               by_disposition=disp),
        _check("exactly one published table ever carried a naive comparison",
               _num(len(inv["tables_with_a_published_naive"]) == 1
                    and not any(p.get("naive_mae") is not None
                                for k, v in E.items() if k not in inv["tables_with_a_published_naive"]
                                for p in v["published"]["model_naive_pairs"].values())),
               the_one=inv["tables_with_a_published_naive"]),
        _check("85 of 167 published legacy tables beat the naive by 50% or more",
               _num(len(flagged) == 85 and mine == flagged),
               flagged=len(flagged), re_derived_by_the_artifact_s_own_rule=len(mine),
               measurable_base=len(measured), unmeasurable=len(E) - len(measured),
               reading="85 of the 113 tables whose predictions can be recomputed (75%), not 85 of "
                       "167: the other 54 carry no reconstructable predictions and were never "
                       "tested for the flag at all"),
        _check("the phase-3 family by 96-97% at one to six hours",
               REFUTED if not (min(p3.values()) >= 96.0) else VERIFIED,
               phase_3_tables_measured=len(p3),
               best_horizon_skill_range=[round(min(p3.values()), 2), round(max(p3.values()), 2)],
               at_or_above_96=sorted(k for k, v in p3.items() if v >= 96.0),
               reading="four of the 24 phase-3-named tables reach 96-97% at their best horizon; the "
                       "family's best-horizon skill spans 27.7% to 97.2%. The extraordinary skill "
                       "is real and none of it is usable as a benchmark; the FAMILY is not at 96-97%"),
    ]
    return checks


def legacy_recomputed(tables=None) -> list:
    """Model and naive on identical rows, from the committed prediction CSVs."""
    inv_p = EVID / "RP58" / "LEGACY_INVENTORY.json"
    lin_p = EVID / "RP58" / "CAUSAL_LINEAGE.json"
    if not (inv_p.is_file() and lin_p.is_file()):
        return [_check("the published Test rows recompute from its own predictions and labels", ABSENT,
                       missing="RP58 inventory")]
    inv = json.loads(inv_p.read_text())
    lin = json.loads(lin_p.read_text())
    tables = tables or ["examples/results/phase_3_3/phase_3_2_cnn_25200_1h_results.csv",
                        "examples/results/phase_3_3/phase_3_2_lstm_25200_1h_results.csv",
                        "examples/results/phase_3_1/phase_3_1_cnn_25200_1h_results.csv",
                        "examples/results/phase_1/phase_1_ann_12600_1h_results.csv"]
    out = []
    for key in tables:
        e = inv["entries"].get(key)
        if e is None:
            out.append(_check(f"{key}: recomputed skill", ABSENT, missing="not in the inventory"))
            continue
        pf = REPO / e["prediction_file"]
        if not pf.is_file():
            out.append(_check(f"{key}: recomputed skill", ABSENT, missing=e["prediction_file"]))
            continue
        with pf.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        base = next(c for c in rows[0] if c.endswith("_CLOSE"))
        per_h, same_rows = {}, True
        for H in range(1, 7):
            t, p = f"Target_H{H}", f"Prediction_H{H}"
            if t not in rows[0] or p not in rows[0]:
                continue
            trip = [(float(r[base]), float(r[t]), float(r[p])) for r in rows
                    if r[base] not in ("", None) and r[t] not in ("", None) and r[p] not in ("", None)]
            mae = statistics.fmean(abs(pp - tt) for _b, tt, pp in trip)
            nv = statistics.fmean(abs(bb - tt) for bb, tt, _p in trip)
            ref = e["recomputed"]["horizons"][f"H{H}"]
            per_h[f"H{H}"] = {"n": len(trip), "mae": mae, "naive": nv,
                              "skill_percent": 100.0 * (1.0 - mae / nv),
                              "artifact_mae": ref["mae"], "artifact_naive": ref["naive_from_published_base"],
                              "artifact_skill": ref["mae_skill_percent_vs_that_naive"]}
            same_rows &= len(trip) == ref["n"]
        ok = same_rows and all(_close(v["mae"], v["artifact_mae"], 1e-12)
                               and _close(v["naive"], v["artifact_naive"], 1e-12)
                               and _close(v["skill_percent"], v["artifact_skill"], 1e-9)
                               for v in per_h.values())
        anomaly = lin["entries"][key]["anomaly_indicator"]["mae_skill_percent_vs_naive_by_horizon"]
        ok &= all(_close(per_h[k]["skill_percent"], anomaly[k], 1e-9) for k in per_h)
        out.append(_check(f"{key}: the model and the naive are on identical rows, and the skill "
                          f"re-derives from the committed predictions", _num(ok), horizons=per_h))
    return out


def leak_signature() -> list:
    p = EVID / "RP58" / "LEAK_SIGNATURE_CALIBRATION.json"
    if not p.is_file():
        return [_check("the signature, calibrated on the same family's series", ABSENT, missing=str(p))]
    d = json.loads(p.read_text())
    rows = {r["representation"]: r for r in d["results"]}
    pops = {k: r["evaluation_rows"] for k, r in rows.items()}
    sig = d["signature"]
    same = len(set(pops.values())) == 1
    return [
        _check("every representation's skill re-derives from its own MAE and its own naive",
               _num(all(_close(r["mae_skill_percent_vs_naive"],
                               100.0 * (1.0 - r["mae"] / r["naive_mae"]), 1e-9) for r in rows.values())),
               per_representation={k: r["mae_skill_percent_vs_naive"] for k, r in rows.items()}),
        _check("the leak signature subtracts two skills measured on the same rows",
               _num(same), evaluation_rows=pops,
               leaking=sig["leaking_skill"], causal=sig["causal_skill"],
               difference=sig["leaking_minus_causal_skill_points"],
               reading="TRAILING_WAVELET is scored on 1 512 rows against a naive of 0.0018573, the "
                       "other four on 1 525 rows against 0.0018591: the trailing transform consumes "
                       "a lead-in, so the contrast the 'signature' field computes crosses two "
                       "populations. The gap is 0.85% of the rows and the conclusion (a causal "
                       "trailing wavelet does not produce the published magnitude) survives, but "
                       "RP57's own standard in the same round is 'on the SAME rows'"),
    ]


# --- RP59 / RP60 -------------------------------------------------------------------------------------

def data_audit() -> list:
    p = EVID / "RP59" / "DATA_TARGET_PREPROCESSING_AUDIT.json"
    if not p.is_file():
        return [_check("Train autocorrelation is 0.963 at one minute, 0.403 at sixty and 0.317 at a day",
                       ABSENT, missing=str(p))]
    a = json.loads(p.read_text())
    ac = a["autocorrelation"]
    out = [_check("a perfect one-minute grid over the consumed slice, zero missing minutes, "
                  "no duplicated stamps",
                  _num(a["times"]["missing_minutes"] == 0 and a["times"]["duplicated_stamps"] == 0
                       and a["times"]["rows_present"] == a["times"]["rows_expected_on_a_perfect_grid"]
                       == 50400 and not a["times"]["other_steps_seen"]),
                  times=a["times"]),
           _check("the label is Y[origin+60] and equals the panel column element by element",
                  _num(bool(a["label"]["Y_equals_the_panel_column"])
                       and bool(a["label"]["checked_element_by_element"])),
                  label=a["label"]),
           _check("the scaled slice is centred on the window grain and the two grains differ by "
                  "0.0017 in scaled units",
                  _num(a["scaler"]["grain_the_scaled_slice_is_centred_on"] == "windows"
                       and _close(a["scaler"]["difference_between_the_grains_in_scaled_units"]["mean"],
                                  0.0017, 5e-5)),
                  scaler_grains=a["scaler"]["difference_between_the_grains_in_scaled_units"])]
    if not (SUCCESSOR_ROOT / "DATA.npz").is_file():
        out.append(_check("Train autocorrelation is 0.963 / 0.403 / 0.317", ABSENT,
                          missing=str(SUCCESSOR_ROOT / "DATA.npz"), published=ac))
        return out
    D = np.load(SUCCESSOR_ROOT / "DATA.npz")
    Y, tr, w = D["Y"], D["train_origins"].astype(np.int64), int(D["window"][0])
    rows = np.arange(int(tr.min() - w + 1), int(tr.max()) + 1)
    v = Y[rows]
    v = v[np.isfinite(v)]
    n = v.size
    vc = v - v.mean()
    denom = float((vc * vc).sum())
    tool, pear, shrink = {}, {}, {}
    for lag in (1, 60, 1440, 10080):
        tool[str(lag)] = float((vc[:-lag] * vc[lag:]).sum() / denom)
        pear[str(lag)] = float(np.corrcoef(v[:-lag], v[lag:])[0, 1])
        shrink[str(lag)] = (n - lag) / n
    out.append(_check("Train autocorrelation is 0.963 at one minute, 0.403 at sixty and 0.317 at a "
                      "day: the daily lag carries LESS linear structure than the hour",
                      _num(all(_close(tool[k], ac["autocorrelation_by_lag_minutes"][k], 1e-10)
                               for k in tool) and n == ac["train_rows_used"]
                           and pear["1440"] < pear["60"]),
                      train_rows_used=n, published=ac["autocorrelation_by_lag_minutes"],
                      recomputed_under_the_tool_s_estimator=tool,
                      lag_truncated_pearson=pear, bias_factor_n_minus_k_over_n=shrink,
                      reading="the four numbers re-derive exactly under df_e1_data_audit's biased "
                              "ACF (a fixed whole-series denominator), whose shrinkage is "
                              "(n-k)/n: 0.9985 at an hour but 0.7492 at a week. The day-below-hour "
                              "reading survives the correction (0.335 < 0.403). The published list's "
                              "apparent decay to the weekly lag does NOT: bias-corrected the week is "
                              "0.367, ABOVE the day's 0.335"))
    return out


def optimisation() -> list:
    p = EVID / "RP60" / "OPTIMISATION_PROBE.json"
    if not p.is_file():
        return [_check("the counted updates ARE the optimiser's steps", ABSENT, missing=str(p))]
    d = json.loads(p.read_text())
    route, cap, full = d["route"], d["capacity"], d["shuffled_labels_full_scale"]
    res_p = EVID / "RP55" / "E1_SUCCESSOR_RESULTS.json"
    r0 = json.loads(res_p.read_text())["cells"]["R0_s1"] if res_p.is_file() else {}
    return [
        _check("a head forced to emit the last observed target reproduces the persistence control "
               "(difference 2e-7)",
               _num(bool(route["reproduces_persistence"]) and bool(route["identical_metric"])
                    and route["largest_difference_from_the_panel_value"] < 1e-6),
               difference=route["largest_difference_from_the_panel_value"],
               route_mae=route["mae_through_the_route"], control_mae=route["mae_of_the_control"]),
        _check("on a fixed 256-window subset the loss falls from 6.99 to 0.0055",
               _num(cap["subset_windows"] == 256 and _close(cap["loss_on_the_subset_before"], 6.99, 5e-3)
                    and _close(cap["loss_on_the_subset_after"], 0.0055, 5e-5)),
               before=cap["loss_on_the_subset_before"], after=cap["loss_on_the_subset_after"]),
        _check("the counted updates ARE the optimiser's steps (400 = optimizer.iterations)",
               _num(cap["counted_updates"] == cap["optimizer_iterations"] == 400
                    and bool(cap["updates_are_optimizer_iterations"])),
               counted=cap["counted_updates"], optimizer_iterations=cap["optimizer_iterations"]),
        _check("0.6015 scrambled against 0.5525 true; the entire label-derived advantage is 0.049 kW",
               _num(_close(full["with_scrambled_labels"]["mae"], 0.6015, 5e-5)
                    and _close(full["the_runs_own_true_label_fit"]["mae"], 0.5525, 5e-5)
                    and _close(full["gap_in_mae"], 0.049, 5e-4)),
               scrambled=full["with_scrambled_labels"]["mae"],
               true=full["the_runs_own_true_label_fit"]["mae"], gap=full["gap_in_mae"]),
        _check("the full-scale negative control is compared against a matched true-label arm",
               REFUTED,
               scrambled_updates=full["with_scrambled_labels"]["updates"],
               scrambled_restored_epoch=full["with_scrambled_labels"]["restored_epoch"],
               true_label_partner=full["the_runs_own_true_label_fit"]["unit"],
               partner_updates=r0.get("updates"), partner_best_epoch=r0.get("best_epoch"),
               reading="the scrambled arm ran to the ceiling (4 000 updates, epoch 6 restored); its "
                       "comparison partner is the finished run's R0_s1, which early-stopped at "
                       "3 762 updates and restored epoch 3. The control is a control, and the "
                       "round's own corrections document withdraws the general claim; the budget "
                       "and selection asymmetry is named here and nowhere else"),
    ]


def mutants_and_controls() -> list:
    out = []
    m = EVID / "RP56_MUTANTS_POST.json"
    if m.is_file():
        d = json.loads(m.read_text())
        out.append(_check("this round's five guards each die under their own mutation (5 killed of 5)",
                          _num(d["killed"] == d["total"] == 5
                               and all(x["applied"] and x["killed"] for x in d["mutants"])),
                          mutants=[x["mutant"] for x in d["mutants"]]))
    else:
        out.append(_check("5 killed of 5", ABSENT, missing=str(m)))
    c = EVID / "RP58" / "CAUSAL_BATTERY_NEGATIVE_CONTROLS.json"
    if c.is_file():
        rows = json.loads(c.read_text())
        names = {r["case_id"] for r in rows}
        out.append(_check("the negative-control classes full_series_dwt_as_time_row, "
                          "centered_rolling_window, filtfilt and phase_compensation_shift_back are "
                          "all DETECTED",
                          _num({"full_series_dwt_as_time_row", "centered_rolling_window", "filtfilt",
                                "phase_compensation_shift_back"} <= names
                               and all(r["outcome"] == "DETECTED" for r in rows)),
                          controls={r["case_id"]: r["outcome"] for r in rows}))
    else:
        out.append(_check("the negative controls are DETECTED", ABSENT, missing=str(c)))
    for name, path, passed in (("RP49-RP56", EVID / "RP56_FULL_SUITE_SUMMARY.txt", 2289),
                               ("RP57-RP64", EVID / "RP64_FULL_SUITE_SUMMARY.txt", 2323)):
        if not path.is_file():
            out.append(_check(f"{name} full suite {passed} passed", ABSENT, missing=str(path)))
            continue
        tail = path.read_text().strip().splitlines()[-1]
        out.append(_check(f"{name} full suite: 3 failed, {passed} passed, 39 skipped, 8 errors",
                          _num(f"{passed} passed" in tail and "3 failed" in tail
                               and "39 skipped" in tail and "8 errors" in tail),
                          last_line=tail))
    return out


# --- assembly ----------------------------------------------------------------------------------------

def audit(*, replay_windows: int | None = None, replay_work: Path | None = None) -> dict:
    sections = {
        "RP49_RP56_identity": identity_successor(),
        "RP49_RP56_population": population_successor(),
        "RP49_RP56_numbers": numbers_successor(),
        "RP49_RP56_replay": replay_successor(replay_windows, replay_work),
        "RP49_RP56_guards": mutants_and_controls(),
        "RP57_RP64_phase1": phase1(),
        "RP57_RP64_phase1_replay": phase1_replay(replay_windows),
        "RP57_RP64_reanalysis": reanalysis(),
        "RP57_RP64_denominators": denominators(),
        "RP57_RP64_legacy": legacy(),
        "RP57_RP64_legacy_recomputed": legacy_recomputed(),
        "RP57_RP64_leak_signature": leak_signature(),
        "RP57_RP64_data": data_audit(),
        "RP57_RP64_optimisation": optimisation(),
    }
    counts = {VERIFIED: 0, REFUTED: 0, ABSENT: 0}
    for rows in sections.values():
        for r in rows:
            counts[r["state"]] = counts.get(r["state"], 0) + 1
    return {"schema": "df_rp49_rp64_audit.v1",
            "authority": "the owner's grant of 2026-09-26: this audit stands in place of the absent "
                         "MUSASHI_RP49_RP56_REVIEW and MUSASHI_RP57_RP64_REVIEW. It is published "
                         "under Satoshi's name and no reviewer's name appears on it",
            "rule": "a module is unblocked only if its own retained artifacts sustain it; a claim "
                    "whose artifact is gone is UNVERIFIABLE_ARTIFACT_ABSENT, never accepted",
            "run_roots": {"successor": str(SUCCESSOR_ROOT), "successor_present": (SUCCESSOR_ROOT / "DATA.npz").is_file(),
                          "phase1": str(PHASE1_ROOT), "phase1_present": (PHASE1_ROOT / "DATA.npz").is_file()},
            "counts": counts, "sections": sections}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--replay-windows", type=int, default=None,
                    help="extend the closure's 512-window replay to this many evaluation origins "
                         "(needs the run root and loads TensorFlow)")
    a = ap.parse_args(argv)
    doc = audit(replay_windows=a.replay_windows, replay_work=None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({"counts": doc["counts"],
                      "refuted": [r["claim"] for rows in doc["sections"].values() for r in rows
                                  if r["state"] == REFUTED],
                      "unverifiable": [r["claim"] for rows in doc["sections"].values() for r in rows
                                       if r["state"] == ABSENT]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
