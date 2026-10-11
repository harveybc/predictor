"""Execute the effective heuristic-strategy early-close and entry functions on
the owner's mandatory counterexamples. Imports app/policies/prediction_entry_exit.py
(blob 0940589) by path; only trading_contracts is stubbed (types not used here)."""
import sys, importlib.util
sys.path.insert(0, sys.argv[1])  # stub dir
spec = importlib.util.spec_from_file_location("pee", sys.argv[2])
pee = importlib.util.module_from_spec(spec); sys.modules["pee"] = pee; spec.loader.exec_module(pee)
nan = float("nan")
# Long position: entry 1.1000, TP 1.1010, SL 1.0980. Short horizons [2,3,4,5,6,12] h.
flat = [1.1002] * 6
tp_first = [1.1012, 1.1006, 1.1004, 1.1002, 1.1000, 1.0975]   # TP at 2h, SL at 12h
sl_first = [1.0975, 1.1000, 1.1002, 1.1004, 1.1006, 1.1012]   # SL at 2h, TP at 12h
print("LONG position, entry 1.1000 TP 1.1010 SL 1.0980")
for v in "ABCDEFG":
    a = pee.should_early_close(direction="long", variant=v, short_horizon_predictions=tp_first, long_horizon_predictions=flat, stop_loss_price=1.0980, entry_price=1.1)
    b = pee.should_early_close(direction="long", variant=v, short_horizon_predictions=sl_first, long_horizon_predictions=flat, stop_loss_price=1.0980, entry_price=1.1)
    print(f"  variant {v}: TP@2h/SL@12h -> close={a}; SL@2h/TP@12h -> close={b}; identical={a==b}")
# Short position mirror: entry 1.1000, TP 1.0990, SL 1.1020
tp_first_s = [1.0988, 1.0994, 1.0996, 1.0998, 1.1000, 1.1025]
sl_first_s = [1.1025, 1.1000, 1.0998, 1.0996, 1.0994, 1.0988]
print("SHORT position, entry 1.1000 TP 1.0990 SL 1.1020")
for v in "ABCDEFG":
    a = pee.should_early_close(direction="short", variant=v, short_horizon_predictions=tp_first_s, long_horizon_predictions=[1.0998]*6, stop_loss_price=1.1020, entry_price=1.1)
    b = pee.should_early_close(direction="short", variant=v, short_horizon_predictions=sl_first_s, long_horizon_predictions=[1.0998]*6, stop_loss_price=1.1020, entry_price=1.1)
    print(f"  variant {v}: TP@2h/SL@12h -> close={a}; SL@2h/TP@12h -> close={b}; identical={a==b}")
# Any permutation of the 12 values gives the same decision (order-blind):
import itertools, random
random.seed(0)
vals = tp_first + flat
outs = set()
for _ in range(200):
    p = vals[:]; random.shuffle(p)
    outs.add(pee.should_early_close(direction="long", variant="E", short_horizon_predictions=p[:6], long_horizon_predictions=p[6:], stop_loss_price=1.0980, entry_price=1.1) if False else
             pee.should_early_close(direction="long", variant="A", short_horizon_predictions=p[:6], long_horizon_predictions=p[6:], stop_loss_price=1.0980, entry_price=1.1))
print("variant A over 200 random horizon permutations: distinct decisions =", outs)
# Entry with a single available long (five masked as NaN): no two-long abstention
params = pee.PredictionEntryExitParameters()
one = [nan, nan, nan, nan, nan, 1.1050]
g = pee.calculate_entry_geometry(current_price=1.1000, long_horizon_predictions=one, params=params)
print("entry with ONE finite long (others NaN):", None if g is None else (g.direction, round(g.take_profit_price, 5), round(g.stop_loss_price, 5), round(g.reward_risk_ratio, 3)))
two_perm_a = [1.1050, 1.0990, nan, nan, nan, nan]; two_perm_b = [1.0990, 1.1050, nan, nan, nan, nan]
ga = pee.calculate_entry_geometry(current_price=1.1, long_horizon_predictions=two_perm_a, params=params)
gb = pee.calculate_entry_geometry(current_price=1.1, long_horizon_predictions=two_perm_b, params=params)
print("entry is order-blind across long horizons:", (ga.direction, ga.take_profit_price, ga.stop_loss_price) == (gb.direction, gb.take_profit_price, gb.stop_loss_price))
# Short family change with long fixed does not alter entry (entry ignores short):
c = pee.calculate_entry_geometry.__code__
print("calculate_entry_geometry parameters:", c.co_varnames[:c.co_argcount + c.co_kwonlyargcount])
