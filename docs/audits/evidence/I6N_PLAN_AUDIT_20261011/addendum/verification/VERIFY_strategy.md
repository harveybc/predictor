# Adversarial verification: I6-N addendum, lane strategy (I6N-AUD-02, owner point 6b)

Date: 2026-10-11. Verifier ran on the coordinator, CPU only, under crispdm-run. The lane's files were not modified.

## 1. Rerun and hashes

- I copied the lane's scripts, its stub and the original `probes/a/*` into `verify_strategy/tree/` with the same relative layout. Then I ran `bash run_strategy_addendum.sh` unchanged, from 12:20:17Z to 12:20:37Z, exit 0.
- All 8 regenerated `*.out` files are **byte-identical** to the retained ones: cx_v2 x4, inspect, resolve, original_probe_clean_rerun and reconcile.
- All 15 sha256 entries in MANIFEST.json match (`sha256sum -c`).
- The input hashes match:
  - policy 16394e2f;
  - plugin 1a835fee;
  - base site-packages copy 3e1ebec1;
  - lts copy 954318e8;
  - request 2e060f7d;
  - MWC 604f9480;
  - proposal d4341e72;
  - original probe .py 15970757 and .out d9e1399f.
- The heuristic-strategy and lts files are clean. The predictor app/ files are clean against 269c2d13.
- The original audit dir is unchanged: git status shows only `addendum/` as untracked.

## 2. Claims checked

| Claim | Verifier result |
| --- | --- |
| S1 chronological early close | **Reproduced.** I read the source myself (policy lines 124-182): `should_early_close` takes no TP and uses only min/max or a weighted min/max, so it ignores order. **Overclaim in scope.** S3 itself shows that no I6-N route loads these blobs, and the plan names no pin. The lane calls them the "intended plugin" without evidence. By the lane's own S4 logic, the defect is DEMONSTRATED on blobs 0940589/3a19b3f, and for I6-N it is an OWNER_REQUIREMENT gap that waits on the pin. The plan (proposal lines 30-35) already acknowledges that min/max without time order does not certify the rule. |
| S2 checkout attribution retraction | Reproduced (HEAD 79977b2, identical blobs at master and origin/master). |
| S3 no executor loads a strategy | Top-level conclusion **holds**. Two sub-claims are **refuted** (see section 3). The resolver rows are reproduced. "Three different outcomes" is really four: legacy copy, inspected copy, ImportError, NO_ENTRY_POINT. Minor. |
| S4 legacy copy, RISK | Reproduced: D1 TP 1.10072 to 1.10540 and SL 1.09980 to 1.08800; [F] NaN-first gives close=False against True. The classification is consistent. |
| S5 no horizon map | Reproduced: positional rename at plugin:111/115, and grep finds 0 occurrences. The same scope caveat as S1 applies (not loaded by I6-N). |
| S6 lexicographic `_path` | Reproduced, ranks [6,7,1,8,2,9,3,10,4,5,11,12] and [1,10,11,12,2..]. RISK is correct. |
| S7 current-price fill | Mechanism reproduced: plugin:302-305 `.get(..., current_price)`; num_hourly_preds counts the columns (plugin:256), so the harness is faithful. **Overclaim.** The only instantiation of `Plugin.HeuristicStrategy` anywhere (heuristic-strategy, predictor, lts, agent-multi, gym-fx, prediction_provider, doin-*) is `evaluate_candidate`. That route always writes contiguous `Prediction_h_1..n` columns, so an absent column cannot happen and the "direct pred_file route" has no caller. The harness bypasses `__init__`. The fill is a latent code defect, but it is a RISK on every reachable route, not a DEMONSTRATED_DEFECT. |
| S8 masks | Reproduced (lts NaN-first close=False against NaN-last True; None raises TypeError in A, C, D, E, F). The inspected policy has no mask parameter, so the defect is on the blob, with the same scope caveat. |
| S9, S10/S11, S12 | Reproduced. The classifications are appropriate. The MWC:127 and 150-151, request:206-207 and MWC section 4 citations are correct. |
| S13 original .out reconciliation | Mechanically reproduced: retained lines 1-19 equal clean lines 1-19, retained 21-24 equal clean 17-20, and the line-20 text appears 0 times in the retained .py. **Not retained:** the causal narrative (revision 0/1/2, tee then `tail -4 \| tee -a`, timestamps 11:46:06, 11:46:12 and 11:46:20) rests on the original sub-auditor's session transcript, and revision 1's source is not kept. The scratch-copy mtimes I found are 11:46:21Z for both the .py and the .out. They are consistent with an appended final run but do not show 11:46:12. The `short_signal` explanation is consistent with the retained grep (function line 30) but cannot be executed without revision 1. The README should label it "reconstructed from an unretained transcript". |

## 3. Independent refutation (S3 sub-claims)

`app/optimizer_ga.py` "imports a nonexistent top-level module, which no environment provides, and nothing imports optimizer_ga". **Refuted.**

- The audited predictor worktree tracks its own copy, `app/heuristic_strategy.py` (sha256 836d2fd14794fea16d22176e018c06a1fff922c8956fbd246ff3d794b3af5965, last commit 9b7d611a). It has a current-price `.get` fill at lines 124-140 and `__main__` at 345.
- `optimizer_ga.py` is a standalone script (`__main__` at line 144). Running `python app/optimizer_ga.py` puts `app/` first on sys.path, so `heuristic_strategy` resolves to that copy.
- Check: `crispdm-run ... python -I spec_check.py <pred_i6n>/app` under both base and trading-stack gives origin `<pred_i6n>/app/heuristic_strategy.py`, sha256 836d2fd1.
- This **fifth copy** is missing from the lane's inventory, and the lane's grep patterns are too narrow to find it.
- It is a legacy 2014-2015 script hard-wired to `../trading-signal/output.csv` and is not an I6-N executor, so the conclusion that no I6-N executor loads a strategy survives.
- The lane also does not mention the existing weekly components: `tools/e3_weekly_strategy.WeeklyExecutionPlugin` (an execution overlay) and the unbound `strategy` delivery callback in `tools/business_weekly_score.py`.
- Minimal correction: list the copy by sha256 and state that optimizer_ga is an unused standalone legacy script.

## 4. Owner-decision overrides

None found. Nothing touches early stopping, seeds or redoing the selection, and minimal corrections are stated as corrections, not as repairs being ordered.

## 5. Privacy

- No host names, IPs, UUIDs, usernames, absolute home paths or tokens appear in the lane files (grep for the host name, `/home/`, the username, dotted IPs, UUID and GPU- patterns, and the token, password and secret strings).
- Paths are written as $GH, $CONDA_BASE and ~.
- The scripts use `$HOME/anaconda3` only as a variable.

## 6. Unverified

- The README says trading-stack is the "env of every lts/agent-multi user unit". I did not check per unit; 32 user units mention trading-stack.
- The worker roles are not inspected, as the lane itself states.
