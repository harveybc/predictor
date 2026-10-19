# Learning a Trading Policy from a Hindsight Expert (ZigZag Demonstrations)

Design note for Musashi's RL lane. Scope: DEVELOPMENT only (TRAIN 2020-2022,
INNER 2023); 2024 is a single confirmation and TEST 2025 stays closed. No ETA,
no fits are claimed here. Author alias: Satoshi. Prepared 2026-10-19.

## 1. Facts this note builds on (as given, not re-measured)

- heuristic-strategy branch `satoshi/short-horizon-strategy-20261018` (@b9b4f9c)
  built a MetaTrader-semantics ZigZag hindsight expert (Depth 3 / Deviation
  36.3 pips / Backstep 2; a better-budgeted NSGA-II rerun is in progress).
  Hindsight Sharpe 17-19 on EURUSD H1 2020-2024, about 14.7 orders/week.
- The causally CONFIRMED version (pivot timestamped at confirmation, median
  lag 6 bars) gives Sharpe -1 to -2.2. Gap about 21 Sharpe: roughly half from
  timing, half from knowing the next pivot level (used as TP).
- Demonstrations: 3,791 sealed expert trades (sha256 352ddf6a...), label
  `HINDSIGHT_NON_CAUSAL_EXPERT`.
- gym-fx parity for the expert is verified (net +2.7 %, maxDD equal), but
  gym-fx today has NO per-trade SL/TP, no hour-dependent spread, no swap, no
  forced-flat calendar, and ignores same-direction repeated decisions.
- Owner hard rules: every order carries a computed effective SL and TP;
  units = percentual_order_size x rel_volume(0.10) x balance x leverage /
  price; at least 3 orders/week and at most 4/day; early stopping on the mean
  of the running TRAIN metric and INNER (2023) performance per epoch,
  patience 10.
- Comparator: the existing SAC (SB3 2.9.0), which reaches training Sharpe
  300-400 with poor validation; anti-overfitting work is just starting.
  `sb3_contrib` is not installed.

The central fact is that the expert is **non-causal**. Its actions are not a
function of the agent's observation, so no causal student can reproduce them.
What the expert gives us for free is a **labeling oracle**: for any state in
2020-2023 it can say what the hindsight-optimal action and the remaining leg
P&L are. The design below exploits the oracle without letting its future
knowledge leak into the actor's input.

## 2. Primary sources: what each offers here and where it mismatches

All retrieved 2026-10-19.

1. **DQfD** (Hester et al., AAAI 2018; https://arxiv.org/abs/1704.03732).
   Offers: TD loss plus a supervised large-margin loss on demonstrator
   actions, prioritized replay that sets the demo/self-generated ratio
   automatically. Mismatch: discrete-action Q-learning; our actions are
   continuous (direction, size, SL/TP multipliers), so only the demo-replay
   and prioritization ideas transfer to SAC.
2. **Overcoming Exploration with Demonstrations** (Nair et al. 2018;
   https://arxiv.org/abs/1709.10089). Offers: BC auxiliary loss on the actor,
   applied only where the critic says the demo action is at least as good as
   the policy's (the Q-filter), demo replay buffer. Fits an off-policy
   actor-critic such as SAC directly. Mismatch: designed for sparse-reward
   robotics with a *reachable* demonstrator; our demonstrator is unreachable,
   so the Q-filter is essential, not optional, to stop the actor copying
   unlearnable actions.
3. **AWAC** (Nair et al. 2020; https://arxiv.org/abs/2006.09359). Offers:
   offline pretraining from prior data with advantage-weighted maximum
   likelihood actor updates, then online fine-tuning without the collapse
   that plain BC-then-RL shows. Mismatch: demos must be advantageous under
   the *causal* MDP; hindsight trades are advantageous only in hindsight, so
   advantages must be estimated by a critic trained on confirmed-timing data.
4. **SQIL** (Reddy et al. 2019; https://arxiv.org/abs/1905.11108). Offers:
   reward +1 on demo transitions, 0 elsewhere; a regularized BC that pulls
   the agent back toward demonstrated states. Mismatch: it replaces the
   trading reward, which the owner evaluates on; usable only as a sanity
   baseline, not as the objective.
5. **GAIL** (Ho & Ermon 2016; https://arxiv.org/abs/1606.03476). Offers:
   imitation without a reward via a GAN-style discriminator. Mismatch:
   adversarial training is unstable with 3,791 trades and our episodes are
   long and noisy; occupancy matching would also try to match an
   unreachable state distribution. Not recommended.
6. **DAgger** (Ross, Gordon & Bagnell, AISTATS 2011;
   https://arxiv.org/abs/1011.0686). Offers: iterative aggregation where the
   expert relabels the states the *student* visits, with a no-regret
   guarantee against the covariate shift of plain BC. Fits perfectly: our
   expert labels any state at zero cost. Mismatch: the guarantee assumes the
   expert's action is realizable from the observation; with a hindsight
   expert the residual loss is the imitability gap itself, which is what we
   want to measure.
7. **Learning by Cheating** (Chen et al., CoRL 2019;
   https://arxiv.org/abs/1912.12294). Offers: the two-stage pattern,
   privileged teacher first, then a sensor-only student supervised on its own
   trajectories. Mismatch: their privileged agent is still causal in time
   (it sees the present map, not the future); ours sees the future, so the
   student can at best learn the *confirmed* approximation.
8. **Asymmetric Actor-Critic** (Pinto et al. 2017;
   https://arxiv.org/abs/1710.06542). Offers: the critic trains on full
   state, the actor only on observations; variance reduction with no leak
   into the deployed policy. Fits: feed the critic hindsight features (next
   pivot level, remaining leg P&L) during training only. Mismatch: their full
   state is contemporaneous; ours is future information, so the critic
   becomes a value estimate conditional on information the actor can never
   have. This is acceptable as a baseline variance reducer but changes the
   meaning of advantages; treat it as an ablation, not a default.
9. **Potential-based reward shaping** (Ng, Harada & Russell, ICML 1999;
   https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).
   Offers: adding F = gamma*Phi(s') - Phi(s) leaves the optimal policy
   unchanged, and this form is necessary for the guarantee. Fits: Phi can be
   computed from hindsight (the expert's remaining leg P&L from s) because
   Phi is a function of the state index, not of the actor's input. Mismatch:
   the theorem is for MDPs; our environment is a POMDP with a non-stationary
   market, so invariance holds for the fixed historical replay only.
10. **Decision Transformer** (Chen et al. 2021;
    https://arxiv.org/abs/2106.01345). Offers: return-conditioned
    autoregressive action prediction, no value function, cheap to run on a
    fixed demo set. Mismatch: trained on hindsight returns it will learn to
    predict hindsight actions and fail causally, which makes it a useful
    *negative control* for the imitability gap rather than a candidate.
11. **The Probability of Backtest Overfitting** (Bailey, Borwein, Lopez de
    Prado & Zhu, 2015; https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).
    Offers: PBO via combinatorially symmetric cross-validation, in-sample vs
    out-of-sample Sharpe degradation as the reported figure. Fits: it is the
    reporting discipline for the TRAIN-vs-INNER gap. Mismatch: CSCV assumes
    many strategy variants over one return series; we have few variants and
    an RL training loop, so we report the degradation and seeds rather than
    a full PBO estimate.

## 3. Ranked design recommendations

**R1. Interactive imitation (DAgger), and measure imitability first.**
Plain BC on 3,791 trades learns the expert's state distribution, not the
student's. Because the expert can label any state, run DAgger: the BC student
acts in gym-fx, every visited state is relabeled by the expert, the aggregated
set is refit. Before any RL, run E0: BC accuracy and the BC policy's Sharpe on
INNER. That number is the ceiling of what hindsight labels can teach a causal
network; if it is near the confirmed version (-1 to -2.2), the labels carry
little learnable signal and the lane pivots to R2(b) only.

**R2. Two separate uses of hindsight.**
(a) Action labels come only from the CONFIRMED-timing version: the entry at
the confirmation bar and the TP at the confirmed level are learnable. (b) The
full hindsight is used only where it cannot leak into the actor: as privileged
critic input (Asymmetric AC) and as a shaping potential
Phi(s) = expert's remaining leg P&L from s in units of sigma-hat, applied as
gamma*Phi(s') - Phi(s). The optimal policy is unchanged (Ng 1999); the actor's
inputs are unchanged. Log the shaped and unshaped returns separately.

**R3. Pretrain-then-finetune with the existing SAC.**
Offline stage: AWAC-style advantage-weighted BC on the aggregated DAgger set,
with the critic trained on confirmed-timing transitions. Online stage: SAC
(SB3 2.9.0) with a BC auxiliary loss gated by the Q-filter (Nair 2018), a
separate demo buffer sampled with DQfD-style priority, and a demo weight that
decays linearly to zero over the first N updates (N is a tuned hyperparameter,
reported). Pure GAIL is excluded for the stability reason in section 2. No
`sb3_contrib` dependency is needed: the BC loss and demo buffer are a custom
`SAC` subclass over the actor loss and a second replay buffer.

**R4. Decision-Transformer baseline on demonstrations.**
A small DT trained on the 3,791 trades with confirmed-timing labels, conditioned
on a target return equal to the median expert trade. Cheap, offline, and its
INNER Sharpe is the second comparator alongside SAC.

**R5. Overfitting controls (the lane's real risk).**
Purged walk-forward inside DEVELOPMENT (gap of one max-holding period between
folds); at least 3 seeds per arm; observation noise and random episode start
times; weight decay and dropout on the actor; early stopping on the mean of the
running TRAIN metric and INNER performance per epoch, patience 10 (owner rule);
report TRAIN-vs-INNER Sharpe gap per epoch (Bailey et al. degradation). The
headline figure is the **imitability gap**: expert Sharpe minus confirmed
Sharpe minus student Sharpe, on INNER.

**R6. Business compliance inside the environment, before training.**
SL and TP are part of the action: two positive multipliers of sigma-hat
(recent H1 return volatility) clipped to a bounded range; the env computes the
effective price levels and enforces them for every agent, including the
expert and SAC comparator. Units follow the owner formula exactly. The
frequency rule (>=3/week, <=4/day) enters as a penalty with a per-episode
audit, and as a hard mask on the fourth order of the day. gym-fx checklist
required before any fit:

- [ ] per-trade SL/TP fields on orders, enforced at bar high/low;
- [ ] hour-dependent spread table;
- [ ] swap debited to balance at rollover;
- [ ] forced-flat calendar (weekend, listed holidays);
- [ ] same-direction repeated decisions handled (ignore or scale-in, declared);
- [ ] units rule implemented as the only sizing path;
- [ ] order-count counters per day and per ISO week exposed in `info`;
- [ ] expert replay through the modified env reproduces the parity figures.

## 4. Experiment ladder (DEVELOPMENT only, no ETA)

| Step | What | Acceptance |
|---|---|---|
| E0 Imitability | BC on confirmed-timing labels; DAgger 5 rounds; DT baseline | BC INNER Sharpe reported with 3 seeds; gap table expert / confirmed / BC / DT; decision: continue or pivot to R2(b) only |
| E1 BC/DAgger policy | Best E0 policy run through the compliant env with SL/TP actions | Passes frequency and units audits on every episode; INNER Sharpe >= confirmed version |
| E2 SAC + demos | AWAC pretrain, SAC finetune with BC loss + Q-filter + demo buffer; shaping from R2(b) | TRAIN-INNER gap smaller than the current SAC's at matched updates; INNER Sharpe >= E1; early stopping triggered per owner rule and logged |
| E3 Ablations | remove Q-filter; remove shaping; remove privileged critic; remove DAgger (plain BC); plain SAC | Each arm 3 seeds; the headline table lists the imitability gap and the TRAIN-INNER degradation per arm |

Only after E3 is a single 2024 confirmation of the winning arm allowed; 2025
stays closed. Every arm writes its effective config, seed and env checklist
state next to its metrics.

## 5. What this note does not claim

No run was executed for this note. The source summaries come from abstracts
and first pages retrieved on 2026-10-19; details beyond them (exact loss forms
in DQfD, Nair 2018 and AWAC) must be checked in the full papers during
implementation. The NSGA-II rerun may change the expert's parameters and
therefore the demonstration set; the ladder must be re-run from E0 if the
sealed demonstration hash changes.
