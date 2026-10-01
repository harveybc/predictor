# Counterfactual note for the heuristic strategy: what a dossier would need to be identified -- ASSUMPTIONS ONLY, NO ESTIMATION

Lane C2, 2026-10-01, DEVELOPMENT. Nothing here was fitted or measured. Design source: `strategy-experiment-design-20260929` as
summarized in project memory (entry reads the LONG family only; TP/SL are close-only signals; the default exit variant is
internally inconsistent; swap is subtracted from reported PnL but never debited to the broker; there is no per-horizon naive in
`app/`). I did not re-read the strategy code for this note, so every statement about the strategy is UNVERIFIED here and must be
re-checked against the plugin before use.

## 1. The question the strategy needs answered

"What would this trade's outcome have been had the entry been taken at a different time, or not at all?" That is a rung-3
question about an episode (one entry), asked of a policy-generated population. It splits in two:

* **Rung 2, population:** E[Y_trade | do(entry rule R = r)] - E[Y_trade | do(R = r0)], with Y_trade the net PnL (or the barrier
  outcome Y_b) of an entry at bar t under the declared TP/SL/exit rule and costs.
* **Rung 3, episode:** y_cf = f(r0, w_e) + u_e for the entry e actually taken, with u_e abducted from the realized path.

## 2. Assumptions each rung needs, written so they can be attacked

| id | assumption | why the current setup does not meet it |
|---|---|---|
| A1 | **No unmeasured confounding of entry timing**: given the measured covariates W_t (information at bar close t), entry (E_t = 1) is independent of the potential outcomes Y_t(1), Y_t(0) | the strategy enters on a signal computed from the same features that predict returns; the latent regime (volatility state, liquidity) moves both E_t and Y_t. W_t must contain every cause of the signal that also moves the outcome, and the C2 dossiers already show the 83 features are mutual deterministic transforms of one price history (300 of 498 cells fail the support screen) |
| A2 | **Positivity**: 0 < P(E_t = 1 | W_t) < 1 for every W_t in the support of interest | a deterministic entry rule gives propensity 0 or 1: there is no overlap; identification then needs randomization at entry (a declared exploration policy) or a threshold-discontinuity design, with its own assumptions |
| A3 | **Consistency / SUTVA**: the outcome of an entry depends only on its own treatment, not on open positions or earlier entries | positions carry over, sizing depends on equity and open risk; overlapping trades share bars. Needs separated episodes or a time-varying-treatment estimator (g-formula) over the sequential DAG |
| A4 | **Timing**: W_t, the signal and the decision are all emitted at or before the bar close, and the first fill is the next open (or a declared fill model) | timestamp semantics of the ETH view are undeclared (bar-close assumed, not certified); TP/SL are close-only signals so exits are also decision-time dependent |
| A5 | **Correct outcome definition**: Y_trade is the net PnL under one versioned exit rule and one cost model, and equals what the broker would book | the design review records an exit variant that contradicts itself and swap that is subtracted from reported PnL but not debited: Y_trade is not yet a single well-defined variable |
| A6 | **Structural model for rung 3** (additive noise, invertible mechanism for PnL given path, declared mediators: path volatility, MFE/MAE) | not declared; the barrier outcome is not a continuous additive-noise variable, so abduction needs the posterior of the path, or the 5 m trajectory re-read through the barrier rule (spec section 5.2) |
| A7 | **No live future** | any counterfactual quantity abducted from a realized outcome is `RETROSPECTIVE_ONLY`; the operational variant may use only W_t and the predictive distribution of U fitted on earlier trades |
| A8 | **Calibrated inference**: block-bootstrap (or wider-HAC) interval with block length from the measured autocorrelation of the PnL series and the features, and a scrambled-label control with rejection at or below nominal (C2: 0.181 -> 0.056) | overlapping trades and persistent regimes inflate HAC rejection about 3.6x; the control must be re-run on the trade population |

## 3. What the dossier would need, field by field (`causal_dossier.v1`)

* `data_manifest.asset_appearance`: a CONTRACTED appearance of the traded price (not the model-ready-view development slot) and the
  trade log as a sealed source with its own sha and `publication_clock` OBSERVED (broker fill timestamps), not assumed.
* `treatment`: the entry indicator (BINARY) or the threshold of the signal (CONTINUOUS with `dose_support`), with
  `sequential_treatment_policy` either `SEPARATED_EPISODES_ONLY` or `TIME_VARYING_ESTIMATOR`.
* `rung2.adjustment`: a back-door set W on the declared DAG that excludes mediators (path vol after entry), colliders (later signals) and
  descendants; `support.state = SUPPORTED` needs a propensity range inside [0.05, 0.95] with no trimming, which requires an
  exploration policy or randomized entry; `placebo.state = PASSED` with the scrambled-label rate at or below nominal on the trade population.
* `rung2.assumptions_declared`: `no_unmeasured_confounding_given_W` can be `true` only with an argument for each omitted cause
  (regime, liquidity, news); today it is `false` and stays so.
* `rung3`: `scm` with order, equations, `noise` ADDITIVE or POSTERIOR_INFERRED, invertibility, `fit_digest`, `alternatives_considered`; `emittable_from` per quantity.
* `selection.cf_eligible = true` only at `COUNTERFACTUAL_SENSITIVITY` evidence level, i.e. after rung 2 is identified AND the SCM is declared AND sensitivity bounds are reported.

## 4. Bottom line

With the strategy as described, none of A1, A2, A5 holds, so the dossier stays `NOT_IDENTIFIED` at rung 2 and rung 3 for any
strategy entry. The cheapest route to a first identified population effect is not an estimator but a design change: a logged
randomized-entry (or threshold-discontinuity) arm in paper trading with fill timestamps recorded, one versioned exit rule, and the
costs actually debited. Until then the strategy's evidence is predictive (rung 1) and bounded by the naive gate: on ETH 4h TRAIN no
variant-A feature or linear/tree model beats the zero-return naive (`RETURN.md` section 1.1).
