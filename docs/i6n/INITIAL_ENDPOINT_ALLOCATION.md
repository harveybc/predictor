# Initial endpoint allocation

This is bounded historical integration/search development under the owner's
standing execution instruction. It is not the final full-space experiment,
an annual result, or a substitute for independent output-grid supervision.

## Order and scope

1. Retain the single-week h2, h3 and h72 cost pilots without repeating them.
2. Search h12 on the preferred GPU and h72 independently on the secondary
   GPU when both hosts admit their explicit cost classes. With only one
   admitted host, execute them sequentially, h12 first, never duplicate them.
   Each has at most 24 distinct candidate attempts, population 8, at most
   3 generations and one model seed 6102026. No repeated fits or extra seeds.
3. Every search uses the existing four-calendar-year history and purged 28-day
   inner interval preceding the 2024-01-07 development week. Outer-week scores
   are NOT candidate fitness. TRAIN/inner endpoint support is sealed before
   proposals; neither different windows nor failed predictions can drop rows.
4. Only endpoint supervision is executable here. Trajectories are filtered
   before evaluation, not silently rewritten or counted as successful fits.
   This is not the complete approved joint search, which still requires them.
5. Window/history, exact input spacing, transformation, core family, width,
   depth, reducer filters and strides, and final adapter vary jointly inside
   the admitted small cost class. The configured per-feature branches remain
   small and unchanged. Current t is present in every source input window.

H12 is first because its multipliers allow distinct histories 24, 48 and 72
hours. For h2 all four multipliers collapse to the 24-hour minimum; it would
not be a context-interaction experiment. H72 allows 72/144/288/432 hours.
There is no claim that twice the horizon is the optimum window.

## Initial resource class

- Downsampled inputs only: spacing >=2 hours, a divisor of the horizon and
  elapsed history, at most 25 sampled points. No smoothing.
- Core entrance width <=32; all other structural constraints come from the
  canonical search/model contracts. Maximum parameters 1,000,000.
- One child at a time on the preferred GPU; no simultaneous secondary GPU
  on that host. New child ceiling 5 GiB, wall 3600 s, host reserve 3 GiB,
  controller ceiling 512 MiB. The ceiling is not a measured production need.
- The separate secondary host's initial search ceiling is 3 GiB with at
  least 6 GiB MemAvailable before admission and a 512 MiB controller. Its h3
  historical pilot sampled 1,177,051,136 bytes under 3 GiB. This supports
  trying the bounded class, not a guarantee that every architecture fits.
  A memory refusal or stop does not authorize reducing this ceiling.
- Record admission-sampled peaks separately from exact internally retained
  cgroup peaks. Never substitute RSS, trim a refused cap, or retry a failed fit.
- Each stage's total wall ceiling is 24 hours plus setup allowance. A bounded
  terminal disposition is required if resources, coverage or time prevent work.

The earlier h2 fit sampled 1.31 GB under a 6 GiB ceiling. It does not establish
the maximum for every Transformer/LSTM candidate, so resource admission and
watching remain active for every child. Secondary GPU admission is independent.

## Fitness and interpretation

Minimize the existing equally weighted mean of TRAIN and inner endpoint
absolute-error ratios against each horizon's own naive. The stopping monitor
inside a fit remains the owner's mean TRAIN/inner loss, with best weights
restored. Retain original-price sums, counts and masked populations in C2.
This early-stage fitness is not a full-year weekly business score.

Publish every completed candidate's TRAIN and inner metrics separately to
OLAP, with exact readback and physical retained artifacts. Publication retries
never invoke the evaluator. Failed candidates retain a reason, not a fabricated
fitness. Status and ETA come from scripts, not periodically prompted agents.

After this integration stage, complete trajectory supervision, full DOIN
distributed placement and the full weekly development procedure before final
selection. The inherited features remain retrospectively selected with 2024;
the nominal calendar and exogenous vintage limitations remain explicit.
TEST and live trading remain closed. Do not resurrect the paused I6-E-M queue.
