# Satoshi: Bounded Data Closure And RL Preparation

Owner-approved direction: finish real data repair, then RL; forecasting and
NEAT are deferred. Read MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md,
RL_FIRST_AFTER_DATA_REPAIR_2026_10_18.md, and financial-data's
work_plan/17_CANONICAL_INVENTORY_AND_CURATION.md before doing anything.
This order authorizes the bounded tasks below, not a scientific campaign.

## Rules And Execution

Use separate clean worktrees based on the delivered curation branches, not
the dirty canonical checkouts or a historical experiment branch. Inspect
AGENTS.md in every repo. Keep commits isolated by repository and component.
Never edit DOIN, change a running service/configuration, restore old results,
use a GPU, train, read evaluation labels for model selection, buy subscriptions,
expose credentials, or alter live lake/warehouse resources under this order.

Two independent lanes may run in parallel. Each returns once. NO POLLING,
NO SLEEPS waiting for another lane, NO periodic log monitoring, NO interactive
babysitting. Existing automation owns runtime status. A dependency not ready
is recorded as one missing object plus the next deterministic command.
Read status at most once if it is genuinely needed for the assigned task.
Do not consume tokens waiting for data/campaign completion.

## D: Source Evidence And Required Inventory, Highest Priority

Work in financial-data's curation branch. New source/clock evidence and small
scripts/tests only; do not modify the identity-bearing recovery controller,
its config, the running publisher or its units. Musashi integrates candidates.

1. Read the generated native quality, gap and CLOCK_COMPARISON reports once,
   through operator-provided private paths. Never publish those paths. The
   diagnostic compares fixedUTC+1, fixedUTC+2 and Europe/Berlin hypotheses,
   completed-minute endpoints and actual hourly bar-end availability. A best
   overlap score is NOT a verified timezone or quote side.
2. Resolve Forexite archive semantics from primary provider documentation or
   retained provider metadata: exact archive variant (trade/bid), timezone/DST
   across2020-2025 and minute start/end. Compare evidence around both DST changes
   and winter/summer. If the provider's declared clock contradicts observed
   quotes, retain both statements and a reproducible counterexample; no guessed
   time shift, fitted price offset, interpolation or canonical merge.
3. Resolve the primary native H1 source's actual session/holiday schedule.
   Separate source closure, missing file and missing quote. Saturdays/holidays
   are candidates for calendar resolution, not automatically closed labels.
   A raw archive containing no required FX pair is not proof of an HTTP outage.
4. If an independent quote source is necessary, reuse existing paid/free
   acquisition adapters. Request only unresolved real intervals with overlap,
   bounded requests/bytes/retries and provider cooldowns. A daily fixing cannot
   replace an hourly quote. Native values must remain unchanged. Acquisition
   outputs live outside Git and the admitted lake; credential access uses the
   established environment without echoing values. Do not repeat exhausted
   requests or route around rate limits.
5. Build a machine-readable REQUIRED_VARIABLE_INVENTORY candidate from the
   configured source inventory and raw dependencies, not just existing files.
   Explicitly separate all catalog variables, trading-required variables and
   model candidates. Include missing files in the denominator. Include time,
   units, quote side, revision/availability, expected cadence and monthly
   coverage. Do not hardcode366 or19 as a universally complete population.
   Never silently drop a variable to make coverage look complete.
6. Any implementation needs tests first: missing month with100% finite rows,
   missing file, timezone/DST shift, incomplete minute/hour, conflicting
   duplicate, impossible OHLC, real extreme, release after t and gap not
   compressed into a fixed tick. Bounded CPU execution only (2GiB,120s per
   test group, chunks for inventory). No deletion or admission by this lane.

Return DATA_CLOSURE_RETURN.md: exact commits, primary URLs and retrieval date,
code/command and small result summary per question, confirmed versus unresolved,
inventory denominator/coverage, next required object and executable command.
Raw provider observations are not redistributed; public source locators and
hashes suffice. Do not call data admitted, lake replaced or selection complete.

## R: RL Research And Existing-Code Gap Map, No Fits

Read agent-multi and gym-fx instructions/source without modifying their dirty
checkouts. Use codebase-memory discovery first. Prefer existing SB3/plugins,
contracts and test helpers; do not create a new RL framework.

1. Identify exact source symbols for observation assembly, action decoding,
   execution, reward, all costs, forced flatten/reopen, weekly warm-start,
   checkpoint monitoring, score aggregation, outbox and campaign successors.
   A README assertion is not proof the runtime uses it. Mark each business
   requirement IMPLEMENTED_AND_TESTED, IMPLEMENTED_UNTESTED, MISSING or UNKNOWN.
2. Primary-source research only, a bounded review (at most12 relevant papers
   plus official implementation docs). Cover SAC, suitable PPO, DQN/QR-DQN and
   one maintained contemporary continuous-control challenger (TQC/CrossQ only
   if its existing implementation is compatible). Search explicitly for
   transaction costs, nonstationarity, causal execution and online adaptation.
   State when a paper uses daily equities or an order book and does not match
   our hourly EURUSD business. Do not call benchmark superiority a trading
   guarantee or invent unavailable data/code. No exhaustive survey treadmill.
3. Produce a table: question, candidate, action type, policy representation,
   data/time requirements, refit/update semantics, costs, evaluation protocol,
   public code/data, exact difference from our business, and minimum useful
   experiment. First comparator remains existing SAC; never silently replace
   the modular representation with only an MLP.
4. Specify the smallest synthetic CPU tests for missing RL00-RL10 controls.
   Separate episodic policy fitting/checkpoint selection from scored weekly
   prospective interaction. No TEST-dependent stopping and no reusing a future
   week's rewards to improve decisions already scored in that week.
5. Interactive brainstorming with Maestro is allowed as a separate design
   conversation: regime-conditioned policy/entry-exit experts, bounded online
   updates and short-horizon heuristics. Produce hypotheses, not implementation
   or a new live campaign. Forecast/NEAT implementation stays deferred.
6. Check actual PDT applicability from FINRA Notice26-10 and broker/account
   documentation. New standards effective2026-06-04 have a transition through
  2027-10-20. Do not infer that every account or EURUSD product has the same
   rule, and do not hardcode an unverified four-trades constraint/removal.

Return RL_PREPARATION_RETURN.md with source-symbol references, source links,
the gap matrix and a finite proposed campaign manifest. Include what can reuse
accepted code and what requires new tests/data acceptance. No invented ETA,
no GPU preparation/deployment and no standing monitoring agent.

## Completion

One return per lane plus a short combined index. Report exact tests that ran,
what they prove and remaining limits. Commit/push only sanitized code/docs/tests
to the assigned new branches. A push failure is reported once, never bypassed.
No merge, live installation or service action: Musashi integrates reviewed work.
