"""I6-N audit addendum, lane strategy (finding I6N-AUD-02, owner point 6b).

Counterexamples for the strategy contract of the I6-N audit request, section 3.1,
executed against ONE copy of the strategy code given by file path.  Replaces the
original probe probes/a/strategy_counterexample.py (kept unchanged there).

APIs (one per copy found on this host):
  hs_policy     heuristic-strategy app/policies/prediction_entry_exit.py (pure functions)
  hs_plugin     heuristic-strategy app/plugins/plugin_long_short_predictions.py; the UNMODIFIED
                HeuristicStrategy.next() is called on a stub object that replaces only the
                backtrader runtime (clock, close line, broker, order calls); the plugin's own
                _should_early_close_* / _policy_params / compute_size methods are bound to it.
  legacy_plugin the 2025-03-14 copy installed in the conda base env site-packages; same harness.
  lts_copy      lts plugins_strategy/heuristic_strategy.py (copied logic; pure functions)

Scenario construction (all values are synthetic prices; no market data is read).
  Horizons in physical hours: short family 2,3,4,5,6,12; long family 18,24,36,48,60,72.
  LONG position:  entry 1.10000, TP 1.10100 (crossed when value >= TP), SL 1.09800 (value <= SL).
  SHORT position: entry 1.10000, TP 1.09900 (crossed when value <= TP), SL 1.10200 (value >= SL).
  The long family is flat (1.10020 for a long position, 1.09980 for a short) and never crosses.
  S1 "TP first": the 2 h short prediction is beyond TP, the 12 h one beyond SL.
  S2 "SL first": the same six short values in reverse time order (SL at 2 h, TP at 12 h).
  The owner's rule (request 3.1): order ALL available predictions of both families by horizon and
  request an early close only if SL is anticipated before TP.  Expected: S1 -> no early close,
  S2 -> early close, for both buy and sell.  Section A verifies the crossings by construction.
  Ties, exact barrier equality and missing heads are deliberately absent from S1/S2.

Output is deterministic (fixed seed); paths are printed with home placeholders.
Usage: python -I counterexample_v2.py --api API --module FILE [--import-root DIR]
       --gh-root DIR --conda-base DIR
"""
import argparse
import datetime as _dt
import hashlib
import importlib
import importlib.util
import inspect
import math
import os
import random
import sys
import types

NAN = float("nan")
SHORT_H = [2, 3, 4, 5, 6, 12]
LONG_H = [18, 24, 36, 48, 60, 72]
ENTRY = 1.10000
POS = {"long": {"tp": 1.10100, "sl": 1.09800, "flat": [1.10020] * 6,
                "S1": [1.10120, 1.10060, 1.10040, 1.10020, 1.10000, 1.09750]},
       "short": {"tp": 1.09900, "sl": 1.10200, "flat": [1.09980] * 6,
                 "S1": [1.09880, 1.09940, 1.09960, 1.09980, 1.10000, 1.10250]}}
for _d in POS.values():
    _d["S2"] = list(reversed(_d["S1"]))
VARIANTS = "ABCDEFG"
DT = _dt.datetime(2024, 6, 12, 10, 0)  # synthetic stamp for the stub clock only


def crossed(direction, value, tp, sl):
    if direction == "long":
        return ("TP" if value >= tp else None), ("SL" if value <= sl else None)
    return ("TP" if value <= tp else None), ("SL" if value >= sl else None)


def first_crossings(direction, short_vals, long_vals, tp, sl):
    """Chronological scan over both families merged and sorted by physical hours."""
    pts = sorted(zip(SHORT_H + LONG_H, list(short_vals) + list(long_vals)))
    first_tp = first_sl = None
    for h, v in pts:
        if not (isinstance(v, float) and math.isfinite(v)):
            continue
        t, s = crossed(direction, v, tp, sl)
        if t and first_tp is None:
            first_tp = h
        if s and first_sl is None:
            first_sl = h
    return first_tp, first_sl


def owner_expected_close(direction, short_vals, long_vals, tp, sl):
    ftp, fsl = first_crossings(direction, short_vals, long_vals, tp, sl)
    return fsl is not None and (ftp is None or fsl < ftp)


# ----------------------------------------------------------------------------- adapters
class _Clock:
    def __init__(self, dt):
        self._dt = dt

    def datetime(self, _i):
        return self._dt


class _Line:
    def __init__(self, v):
        self._v = v

    def __getitem__(self, _i):
        return self._v


class _Broker:
    def getvalue(self):
        return 10000.0

    def getcash(self):
        return 10000.0


class _Stub:
    def __len__(self):
        return 1


def _pred_frame(pd, short_cols, long_cols):
    row = {}
    row.update(short_cols)
    row.update(long_cols)
    return pd.DataFrame([row], index=pd.Index([DT], name="DATE_TIME"))


class HarnessAPI:
    """Calls the plugin's own HeuristicStrategy.next() on a stub that replaces backtrader."""

    def __init__(self, mod, legacy):
        import pandas as pd
        self.pd = pd
        self.mod = mod
        self.legacy = legacy
        self.Cls = mod.Plugin.HeuristicStrategy
        self.defaults = dict(mod.Plugin.plugin_params)
        self.variants = ["-"] if legacy else list(VARIANTS)
        self.captured = []

    def _stub(self, frame, price, in_pos, direction=None, tp=None, sl=None, variant="E"):
        st = _Stub()
        st.data0 = types.SimpleNamespace(datetime=_Clock(DT), close=_Line(price))
        st.broker = _Broker()
        st.position = 1 if in_pos else 0
        st.exit_variant = variant
        st.current_direction, st.current_tp, st.current_sl = direction, tp, sl
        st.pred_df = frame
        st.num_hourly_preds = len([c for c in frame.columns if c.startswith("Prediction_h_")])
        st.num_daily_preds = len([c for c in frame.columns if c.startswith("Prediction_d_")])
        st.trade_low = st.trade_high = None
        st.balance_history, st.date_history, st.trade_entry_dates = [], [], []
        st.order_entry_price = ENTRY if in_pos else None
        p = dict(self.defaults)
        p.setdefault("default_uncertainty_short_term", 0.0)
        p.setdefault("default_uncertainty_long_term", 0.0)
        st.params = st.p = types.SimpleNamespace(**p)
        st.events = []
        st.close = lambda: st.events.append(("close",))
        st.buy = lambda size=None: st.events.append(("buy", size))
        st.sell = lambda size=None: st.events.append(("sell", size))
        for name in ("_policy_params", "compute_size", "_should_early_close_long", "_should_early_close_short"):
            if hasattr(self.Cls, name):
                setattr(st, name, types.MethodType(getattr(self.Cls, name), st))
        for name in ("_should_early_close_long", "_should_early_close_short"):
            if hasattr(st, name):
                real = getattr(st, name)

                def rec(v, ph, pdl, sl_, e, _real=real):
                    self.captured.append(([round(float(x), 5) for x in ph], [round(float(x), 5) for x in pdl]))
                    return _real(v, ph, pdl, sl_, e)
                setattr(st, name, rec)
        return st

    @staticmethod
    def cols(prefix, values, numbers=None):
        numbers = numbers or list(range(1, len(values) + 1))
        return {f"{prefix}{n}": v for n, v in zip(numbers, values)}

    def early_close(self, direction, variant, short_vals, long_vals, tp, sl, price=ENTRY,
                    short_cols=None, long_cols=None):
        frame = _pred_frame(self.pd, short_cols or self.cols("Prediction_h_", short_vals),
                            long_cols or self.cols("Prediction_d_", long_vals))
        st = self._stub(frame, price, True, direction, tp, sl, variant if variant != "-" else "E")
        self.Cls.next(st)
        return ("close",) in st.events

    def entry(self, short_vals, long_vals, price=ENTRY):
        frame = _pred_frame(self.pd, self.cols("Prediction_h_", short_vals), self.cols("Prediction_d_", long_vals))
        st = self._stub(frame, price, False)
        self.Cls.next(st)
        orders = [e[0] for e in st.events if e[0] in ("buy", "sell")]
        if not orders:
            return None
        return ("long" if orders[0] == "buy" else "short", round(st.current_tp, 5), round(st.current_sl, 5))


class PolicyAPI:
    def __init__(self, mod):
        self.mod = mod
        self.variants = list(VARIANTS)
        self.params = mod.PredictionEntryExitParameters()

    def early_close(self, direction, variant, short_vals, long_vals, tp, sl, price=ENTRY, **_):
        return self.mod.should_early_close(direction=direction, variant=variant,
                                           short_horizon_predictions=short_vals,
                                           long_horizon_predictions=long_vals,
                                           stop_loss_price=sl, entry_price=ENTRY)

    def entry(self, short_vals, long_vals, price=ENTRY):
        g = self.mod.calculate_entry_geometry(current_price=price, long_horizon_predictions=long_vals,
                                              params=self.params)
        return None if g is None else (g.direction, round(g.take_profit_price, 5), round(g.stop_loss_price, 5))


class LtsAPI:
    def __init__(self, mod):
        self.mod = mod
        self.variants = list(VARIANTS)

    def early_close(self, direction, variant, short_vals, long_vals, tp, sl, price=ENTRY, **_):
        return self.mod.should_early_close(direction, variant, list(short_vals), list(long_vals), sl, ENTRY)

    def entry(self, short_vals, long_vals, price=ENTRY):
        s = self.mod.compute_signal(current_price=price, daily_predictions=list(long_vals),
                                    hourly_predictions=list(short_vals))
        if s["action"] == "hold":
            return None
        return ("long" if s["action"] == "buy" else "short", round(s["tp"], 5), round(s["sl"], 5))


# ----------------------------------------------------------------------------- helpers
def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:  # report, never hide
        return f"RAISES {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True, choices=["hs_policy", "hs_plugin", "legacy_plugin", "lts_copy"])
    ap.add_argument("--module", required=True)
    ap.add_argument("--import-root", default="")
    ap.add_argument("--gh-root", required=True)
    ap.add_argument("--conda-base", required=True)
    a = ap.parse_args()
    home = os.path.expanduser("~")

    def red(p):
        p = str(p)
        for real, tag in ((a.gh_root, "$GH"), (a.conda_base, "$CONDA_BASE"), (home, "~")):
            p = p.replace(real.rstrip("/"), tag)
        return p

    module_path = os.path.realpath(a.module)
    if a.import_root:
        sys.path.insert(0, a.import_root)
    if a.api == "hs_plugin":
        mod = importlib.import_module("app.plugins.plugin_long_short_predictions")
        assert os.path.realpath(mod.__file__) == module_path, (mod.__file__, module_path)
    else:
        spec = importlib.util.spec_from_file_location("strategy_under_test", module_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["strategy_under_test"] = mod
        spec.loader.exec_module(mod)
    api = {"hs_policy": lambda: PolicyAPI(mod), "hs_plugin": lambda: HarnessAPI(mod, legacy=False),
           "legacy_plugin": lambda: HarnessAPI(mod, legacy=True), "lts_copy": lambda: LtsAPI(mod)}[a.api]()
    src = open(module_path).read()

    print(f"api={a.api}")
    print(f"module={red(module_path)}")
    print(f"module_sha256={sha256(module_path)}")
    print(f"python={sys.version.split()[0]} executable_prefix={red(sys.prefix)}")
    if a.api == "hs_plugin":
        pol = mod.should_early_close.__code__.co_filename
        print(f"policy_resolved_by_plugin={red(pol)} sha256={sha256(pol)}")

    # A. construction check -------------------------------------------------------------
    print("\n[A] scenario construction (verified independently of the module under test)")
    for direction, d in POS.items():
        print(f"  {direction.upper()} position: entry {ENTRY:.5f} TP {d['tp']:.5f} SL {d['sl']:.5f}; "
              f"long family {d['flat'][0]:.5f} at {LONG_H} h")
        for sc in ("S1", "S2"):
            path = ", ".join(f"{h}h={v:.5f}" for h, v in zip(SHORT_H, d[sc]))
            ftp, fsl = first_crossings(direction, d[sc], d["flat"], d["tp"], d["sl"])
            exp = owner_expected_close(direction, d[sc], d["flat"], d["tp"], d["sl"])
            print(f"    {sc}: short path {path}")
            print(f"        first TP crossing at {ftp} h, first SL crossing at {fsl} h -> owner-expected early_close={exp}")
        assert first_crossings(direction, d["S1"], d["flat"], d["tp"], d["sl"]) == (2, 12)
        assert first_crossings(direction, d["S2"], d["flat"], d["tp"], d["sl"]) == (12, 2)
        assert sorted(d["S1"]) == sorted(d["S2"])
    print("  assertions passed: S1 = (TP@2h, SL@12h), S2 = (TP@12h, SL@2h), S2 is a time permutation of S1")

    # B. TP-first vs SL-first -----------------------------------------------------------
    print("\n[B] early close, TP-first (S1) vs SL-first (S2); owner-expected (S1, S2) = (False, True)")
    for direction, d in POS.items():
        for v in api.variants:
            r1 = safe(api.early_close, direction, v, d["S1"], d["flat"], d["tp"], d["sl"])
            r2 = safe(api.early_close, direction, v, d["S2"], d["flat"], d["tp"], d["sl"])
            ok = (r1 is False and r2 is True)
            print(f"  {direction:5s} variant {v}: S1 close={r1}  S2 close={r2}  identical={r1 == r2}  conforms={ok}")
    if a.api in ("hs_policy", "lts_copy"):
        fn = mod.should_early_close
        print(f"  should_early_close parameters: {tuple(inspect.signature(fn).parameters)}  (no take-profit argument)")
    else:
        print("  prediction early close compares predictions only with current_sl; current_tp is used only for the "
              f"realised-price TP check: {'predicted_min < self.current_sl' in src or 'self._should_early_close_long(v, preds_h, preds_d, sl, entry)' in src}")

    # C. permutation invariance ---------------------------------------------------------
    print("\n[C] 200 random within-family permutations of the short S1 path (seed 20261011)")
    rng = random.Random(20261011)
    perms = []
    for _ in range(200):
        p = POS["long"]["S1"][:]
        rng.shuffle(p)
        perms.append(p)
    for direction, d in POS.items():
        for v in api.variants:
            outs = set()
            for p in perms:
                vals = p if direction == "long" else [d["S1"][POS["long"]["S1"].index(x)] for x in p]
                outs.add(safe(api.early_close, direction, v, vals, d["flat"], d["tp"], d["sl"]))
            print(f"  {direction:5s} variant {v}: distinct decisions = {sorted(map(str, outs))}")

    # D. entry -------------------------------------------------------------------------------
    print("\n[D] entry (no open position), current price 1.10000")
    long_fix = [1.10030, 1.10050, 1.10080, 1.10060, 1.10040, 1.10020]
    short_a = [1.10010] * 6
    short_b = [1.10600, 1.09400, 1.10010, 1.10010, 1.10010, 1.10010]
    ea, eb = safe(api.entry, short_a, long_fix), safe(api.entry, short_b, long_fix)
    print(f"  D1 short changed, long fixed: entry(short_a)={ea}  entry(short_b)={eb}  unchanged={ea == eb}"
          "  (owner 3.1: must be unchanged)")
    if a.api == "hs_policy":
        print(f"     calculate_entry_geometry parameters: {tuple(inspect.signature(mod.calculate_entry_geometry).parameters)}")
    one_last = [NAN] * 5 + [1.10500]
    one_first = [1.10500] + [NAN] * 5
    e1, e2 = safe(api.entry, short_a, one_last), safe(api.entry, short_a, one_first)
    print(f"  D2 exactly ONE finite long (five NaN): NaN-first order -> {e1}; NaN-last order -> {e2}; "
          f"order-independent={e1 == e2}")
    if a.api == "lts_copy":
        e3 = safe(api.entry, short_a, [None] * 5 + [1.10500])
        print(f"     same with five None instead of NaN -> {e3}")
    perm_long = [long_fix[i] for i in (5, 3, 0, 4, 1, 2)]
    e4 = safe(api.entry, short_a, perm_long)
    print(f"  D3 long values permuted across horizons: {e4}  equal_to_D1={e4 == ea}"
          "  (owner 3.1 entry uses the min/max of the longs, so order-blind entry CONFORMS)")

    # E. numeric ordering of >=10 horizon keys ---------------------------------------------------
    print("\n[E] ordering of horizon keys when 12 heads are present")
    if a.api == "hs_policy":
        hours = SHORT_H + LONG_H
        for label, keys in (("hours-named keys path_<h>", [f"path_{h}" for h in hours]),
                            ("index-named keys path_<1..12>", [f"path_{i}" for i in range(1, 13)])):
            outputs = {k: float(i + 1) for i, k in enumerate(keys)}  # value = chronological rank 1..12
            ctx = types.SimpleNamespace(predictions=[types.SimpleNamespace(
                horizons={"short": types.SimpleNamespace(outputs=outputs)})])
            got = mod._path(ctx, ("short",))
            print(f"  _path() with {label}: returned chronological ranks {[int(x) for x in got]}"
                  f"  numeric_order={got == sorted(got)}")
    elif a.api == "hs_plugin":
        api.captured.clear()
        shuffled = [7, 2, 11, 1, 12, 5, 3, 10, 4, 9, 6, 8]
        cols = {f"Prediction_h_{n}": 1.10010 + 0.00001 * n for n in shuffled}
        api.early_close("long", "A", None, [1.10020] * 6, 1.10100, 1.09800, short_cols=cols)
        ph = api.captured[-1][0]
        ranks = [int(round((x - 1.10010) / 0.00001)) for x in ph]
        print(f"  12 short columns stored in shuffled order {shuffled}: next() reads index {ranks}"
              f"  numeric_order={ranks == sorted(ranks)}")
        print("  the integer is a column position, not physical hours (evaluate_candidate renames columns "
              f"positionally): rename_positional={'Prediction_h_{i+1}' in src}")
    elif a.api == "legacy_plugin":
        print("  reads Prediction_h_{i} for i in range(1, n+1), integer loop: "
              f"{'for i in range(1, self.num_hourly_preds + 1)' in src}; positional rename: {'Prediction_h_{i+1}' in src}")
    else:
        print("  takes plain lists; no keys, horizon identity is whatever list order the caller passes")

    # F. missing head ----------------------------------------------------------------------
    print("\n[F] missing head handling (long position, SL 1.09800, TP 1.10100, current price 1.09850)")
    price = 1.09850
    short_true = [1.10100] * 6
    long_vals = [1.09600, 1.10000, 1.10000, 1.10000, 1.10000, 1.10000]
    if a.api in ("hs_plugin", "legacy_plugin"):
        absent = {f"Prediction_h_{n}": 1.10100 for n in (1, 3, 4, 5, 6)}           # head 2 absent
        print("  columns present in the ABSENT case: Prediction_h_1, _3, _4, _5, _6 (count 5) -> next() reads _1.._5")
        nanrow = {f"Prediction_h_{n}": (NAN if n == 2 else 1.10100) for n in range(1, 7)}
        nan_first = {f"Prediction_h_{n}": (NAN if n == 1 else 1.10100) for n in range(1, 7)}
        none_msg = "inline min/max, vector not exposed" if a.api == "legacy_plugin" else "not read (variant G skips prediction exit)"
        for v in api.variants:
            api.captured.clear()
            r_abs = safe(api.early_close, "long", v, None, long_vals, 1.10100, 1.09800, price=price, short_cols=absent)
            vec_abs = api.captured[-1][0] if api.captured else none_msg
            api.captured.clear()
            r_nan = safe(api.early_close, "long", v, None, long_vals, 1.10100, 1.09800, price=price, short_cols=nanrow)
            vec_nan = api.captured[-1][0] if api.captured else none_msg
            r_nan1 = safe(api.early_close, "long", v, None, long_vals, 1.10100, 1.09800, price=price, short_cols=nan_first)
            print(f"  variant {v}: column Prediction_h_2 ABSENT -> short vector {vec_abs} close={r_abs}; "
                  f"present but NaN -> {vec_nan} close={r_nan}; NaN in Prediction_h_1 instead -> close={r_nan1}")
        fill_src = "Prediction_h_{i}', current_price)" in src
        print(f"  source reads absent columns with default=current price: {fill_src}")
    else:
        filled = [1.10100, price, 1.10100, 1.10100, 1.10100, 1.10100]
        masked = [1.10100, NAN, 1.10100, 1.10100, 1.10100, 1.10100]
        for v in api.variants:
            rf = safe(api.early_close, "long", v, filled, long_vals, 1.10100, 1.09800, price=price)
            rm = safe(api.early_close, "long", v, masked, long_vals, 1.10100, 1.09800, price=price)
            print(f"  variant {v}: head 2 filled with current price -> close={rf}; head 2 as NaN -> close={rm}; differ={rf != rm}")
        if a.api == "lts_copy":
            nan_first = [NAN] + [1.09700] * 5
            nan_last = [1.09700] * 5 + [NAN]
            for v in api.variants:
                r1 = safe(api.early_close, "long", v, nan_first, POS["long"]["flat"], 1.10100, 1.09800)
                r2 = safe(api.early_close, "long", v, nan_last, POS["long"]["flat"], 1.10100, 1.09800)
                rn = safe(api.early_close, "long", v, [None] + [1.09700] * 5, POS["long"]["flat"], 1.10100, 1.09800)
                print(f"  variant {v}: NaN masked head first -> close={r1}; NaN last -> close={r2}; "
                      f"order-dependent={r1 != r2}; None masked head -> {rn}")

    # G. identity / mask vocabulary in the source ----------------------------------------------
    print("\n[G] source occurrences of the contract vocabulary (request 3.1 / market-week contract)")
    for word in ("horizon_hours", "target_timestamp", "eligible_until", "available", "mask", "isfinite", "isna"):
        print(f"  {word}: {src.count(word)}")


if __name__ == "__main__":
    main()
