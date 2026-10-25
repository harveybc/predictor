# Activity-first SAC tuning

Owner directive: more than2 executed trades/week is a prerequisite, not an
optional report. The previous critic-dropout contrast completed in76min;
both final policies had0 trades in TRAIN/INNER_A/2024. Control weights replayed
exactly;24/24 reports delivered/read back, independent copy verified.
The experiment established failure of that treatment, not a usable comparator.

## New finite sequence

ACT_ALPHA001 -> ACT_ALPHA010 -> ACT_ALPHA100: fixed entropy0.01/0.1/1.0.
Actor and critic dropout0.3,wd0.01, recipeR3, seed7,100k decisions, four25k
checkpoints, reward, costs, risk controls, action mapping and input windows
unchanged. Initial tensors must match the authenticated control. No feedback,
trade bonus, forced entries, stochastic financial evaluation, early stop or TEST.

At final100k TRAIN and INNER_A must each have strictly more than2 closed trades
per scored week, retaining zero-trade weeks in the denominator, original daily
cap and validity. Rank only eligible arms by balanced mean daily log return,
then smaller gap, lower INNER drawdown, stable id. Exactly2 and0 fail.
2024 never selects. A failed grid returns ACTIVITY_NOT_MET with no winner.
Native performance and activity are reported for every arm regardless of sign.

## Evidence

Producer agent-multi7373699f;
spec853c1ef85e8673c2ea3e650c754ae9cc5a2ae3ac82cfbc280609301ef3c6cb97.
68 tests passed; GPU admission passed14 synthetic updates for each arm.
At2026-10-25T09:09:08Z first arm2250 decisions/1249 gradient updates on primary
GPU, initial tensors identical to control. Sequential worker and collection
timers enabled; first mirror/outbox pump successful,0 reports expected yet.
Provisional batch duration2h from09:08UTC, based on measured prior38min/arm.

Diagnose training command counts and invalid HOLD commands plus policy/Q probes.
Automatic entropy falling and continuous-to-discrete command mapping are
hypotheses; this study does not prove the unique cause or guarantee activity.
No other lane, shared store or business-engine changes.
Full contract: agent-multi docs/work_plan/66_ACTIVITY_FIRST_SAC_TUNING.md.
