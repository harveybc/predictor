# Satoshi order: I6-DMO short static direct multi-output

## Authority and exact outcome

Owner has authorized implementation, configuration generation, ordinary SSH
deployment to the existing fleet, starting/stopping this experiment's own
services, scoped dependency installation into isolated environments, existing
warehouse/lake integration, commits and ordinary pushes. Proceed without
repeated approval for these actions. This is not authority to bypass platform
security, reveal credentials, erase historical results, rewrite Git history,
trade real money, reboot hosts or modify unrelated services.

Read, in order:

1. `docs/tres_temas_entrevista/MASTER_WORK_PLAN_INFORMATION_TO_KNOWLEDGE_PIPELINE_v3.md`.
2. `docs/i6n/direct_multi_output_static/PLAN.md`.
3. `docs/i6n/direct_multi_output_static/EXPERIMENT.short.json`.
4. `docs/i6n/direct_multi_output_static/ACCEPTANCE.md`.
5. The updated business contract and this order.

Do not execute older dated orders or resume single-horizon optimization. The
four POO-v2 services and their successor were stopped/disabled by Musashi.
All four stops reached systemd's 120-second stop limit; no live main/control
PID remains. Preserve partial fits as interrupted, not scientific terminals.
Archive-delivery timers and warehouse transports were deliberately left intact.
Recheck once before launch; do not disable unrelated Postgres, Metabase, chats
or the user's desktop. Use role aliases from private deployment, not public IPs.

## A. Bootstrap once

From the local predictor repository, fetch the published planning branch and
create one worktree. If already created, reuse after checking its actual tip;
do not force-reset it. These are existing commands, unlike the CLI in section D
which you must implement first.

```bash
git fetch origin codex/i6n-poo-control-sentinels-20261014
git worktree add -b satoshi/i6dmo-static-20261014 /var/tmp/satoshi-i6dmo-static-20261014 origin/codex/i6n-poo-control-sentinels-20261014
cd /var/tmp/satoshi-i6dmo-static-20261014
git status --short --branch
```

Pin this planning commit, scientific source commits, the selected manifest,
calendar and resolved data hashes in the new campaign's private effective
spec. Find the existing fleet paths and EnvironmentFiles locally; do not ask
the owner to paste a token or publish their contents. Let the existing service
load its secret EnvironmentFile. Do not print secret environments.

## B. Dispatch four bounded implementation lanes in parallel

Assign separate files/worktrees; acknowledge each dispatch before reporting it.
Maximum four coding agents, one bounded CPU test scope per host. Use Hermes
for deterministic inventories/test commands if available; its absence does not
block execution. No agent repeatedly monitors a runner or tails logs.

1. **Model/training:** typed design adapter, existing hierarchical plugin bridge,
   minimal branches/core/six independent heads, masked normalized loss,
   same-weight fit/inner stopping, serialization and DMO01/04/10/18 tests.
2. **Data/business:** exact DS2 origin windows, 2020-2023 full TRAIN, 2024 full
   development support, C1/C2 masks, independent naive replay and DMO02/03/05/
   06/08/09/16. Recover the seven unresolved producers/clocks without reading
   TEST, using existing paid/public-source adapters where needed. No guessed
   publication clock or silent feature substitution.
3. **Automation/OLAP:** exclusive durable queue, CLI/deploy/timers, bounded
   preflight, status and measured ETA, interrupted/adopted identities, warehouse
   outbox/readback and physical mirrors; DMO07/11/12/13/17/19/20 tests.
4. **Strategy/readiness:** map the actual plugin imported by the executor,
   short and long named outputs/masks, chronological SL-before-TP behavior,
   fill-based forced close, 10000 initial equity/costs/metrics, DMO14/15 tests.
   Prepare the adapter, but no strategy fit or TEST run before both families
   pass the frozen eligibility/provenance gates.

The primary scope is predictor. Changes to heuristic-strategy or a warehouse
adapter belong in that sibling repository with exact pinned commits. DOIN's
working consensus, migration, chain and dashboard must not be edited for this
new fixed-model experiment. Integrate code continuously, not after all lanes
have consumed their token allocations.

## C. Decisions already made: do not improvise them

- First fit: R0 short, RAW inputs, six heads 2/3/4/5/6/12, 24 elapsed hours,
  13 DS2 observations including t. No fitting six whole models.
- Branch filters 4, kernel 3, causal stride 1. Fusion in filters; positional
  encoding immediately after fusion. Core filters 8 then 4, kernel 3, stride 1.
- Each head Flatten -> Dense8/ReLU -> Dense1/linear. No BiLSTM, Bayesian layer,
  MC sampling, Transformer, decoder or denoising in this minimal identity.
- Full four-year TRAIN, full year development, no weekly/quarterly refits.
  Eight-week chronological inner tail; purge unavailable/cross-boundary labels.
- One seed 6102026; one scientific repetition; max200 epochs/patience10/
  min_delta1e-8, restore best same-weight mean fit/inner loss.
- Report original-price MAE/naive per h. The fitness sign is negative when
  better. Training scales come only from fit, never outer validation.
- Current 19 features exclude STLFSI4 and retain the retrospective 2024
  selection disclosure. Certified compatible R2 is a later contrast, not a
  reason to silently import the old lagged or hourly donor.
- First GPU is the 5090. Never run a heavy job simultaneously on its host's
  5070 Ti. Secondary/coordinator do independent bounded preparation/software tests.
- No GA for this fixed baseline. Do not revive 8-individual/3-generation
  searches, optimize twelve horizons separately, or weaken future DOIN budgets.
- Next scientific stage is a separately sealed long vector model with heads
  18/24/36/48/60/72. Prepare it after short closure, as the plan specifies.
- TEST stays closed until the complete pair and procedure are frozen. No
  test feedback may rank models, features, cadence or trading thresholds.

## D. Implement, verify, deploy, return; no periodic monitoring

Implement all commands specified in ACCEPTANCE.md. The following is the
required invocation sequence once those commands and tests actually exist:

```bash
python -m tools.i6dmo validate-design --design docs/i6n/direct_multi_output_static/EXPERIMENT.short.json
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/i6dmo
python -m tools.i6dmo preflight --design docs/i6n/direct_multi_output_static/EXPERIMENT.short.json --deployment "$I6DMO_PRIVATE_SPEC"
python -m tools.i6dmo init --design docs/i6n/direct_multi_output_static/EXPERIMENT.short.json --deployment "$I6DMO_PRIVATE_SPEC" --root "$I6DMO_ROOT"
python -m tools.i6dmo deploy --root "$I6DMO_ROOT" --deployment "$I6DMO_PRIVATE_SPEC"
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level campaign --json
```

`I6DMO_PRIVATE_SPEC` and `I6DMO_ROOT` must be defined by your private bootstrap
script, not supplied by the owner. Do not claim the above CLI exists until you
have implemented and exercised it. Reuse an existing compatible isolated venv;
install package entry points in that venv when needed. Do not modify the
desktop environment. Record actual installed modules, not only Git checkout.

GPU framework integration tests run once on an admitted worker in a bounded
scope; distinguish them from scientific training. A full first training is
the reusable cost pilot. Admit only with live RAM/slice/disk headroom; preserve
1.25x measured peak for subsequent jobs, and do not shrink a refused cap.

After deploy, verify **once**: the effective config hash, actual GPU UUID,
live claimed child, fresh heartbeat, no other heavy child on the preferred host, and the
timer installed/enabled. Then return and let the timers finish. Do not spend
tokens waiting for epochs, checking every few minutes or announcing an ETA
that was not measured. A delivery failure retries delivery, not training.

Automation must be able to finish short validation, prepare/execute the sealed
long stage, verify pair validation and produce a freeze/readiness disposition
without chat babysitting. TEST only proceeds when all recorded gates pass;
otherwise emit a precise terminal with missing artifact and owner, while
continuing genuinely independent preparation. A permission-tool denial is
reported once with its exact command/category; do not retry by another route
or turn it into repeated generic approval requests.

## E. Required return

Push implementation and this order through normal commits. One RETURN.md:

1. Exact per-repository commits and effective config/manifest/calendar hashes.
2. DMO01-DMO20 traceability, passed/failed/not-run; independent negative controls.
3. What is actually installed/running per role/GPU; no fake full utilization.
4. Generated STATUS path/command, expected denominators and measured ETA or
   UNMEASURED. In-flight first-fit time is not a completed-fit ETA.
5. OLAP metric schema/run IDs, verified readback, local and second-host copies.
6. Concise per-substage numeric result index and updated M1-M8 progress plot.
7. Limitations: 2024 development selection, source-clock uncertainty, assumed
   calendar, incomplete long family, TEST closed, partial old fits retained.

Do not ask whether to do the next authorized step. Do not report code as an
experiment. Do not certify financial success from unit tests or tuning scores.
