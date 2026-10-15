# Run the matched short contrast (MG1 vs MG2m)

**Status: PREPARING.** The runner and the model lane's real adapters meet. Nothing in this
pass ran the contrast: no prepare, init or deploy on a real host, no service or timer
installed, no scientific fit started.

Before it claims an arm unit, the worker checks the adapters statically:

- `train.py` must register `--common-population`, `--common-population-digest` and
  `--input-dir`;
- `reporting.py` must define `run_matched_unit` and `decide_matched`.

If either check fails, the unit waits as `WAIT_ARTIFACT MATCHED_*_ADAPTER_UNAVAILABLE`, and
nothing is claimed.

The plan is `NEXT_CONTRASTS_PLAN_20261015.md`, sections 2 and 4. The graph is finite and
uses the existing queue, governed worker, terminal verifier, outbox, readback, mirrors and
automatic CLOSE:

```
mg1.fit  -> mg1.report  \
                          comparison -> publish + exact readback -> two mirrors -> CLOSE
mg2m.fit -> mg2m.report /
```

**Decision rule.** For each arm, take the mean over heads 2, 3, 4, 5, 6 and 12 of the annual
pooled relative MAE, `(MAE_h - naive_h) / naive_h`, on the frozen common DEVELOPMENT rows.
Negative is better.

- The arm with the lower mean wins.
- If the two means are within `1e-8`, MG2m is kept.
- A missing arm or head closes as `INCOMPLETE_COMPARISON`, listing the missing evidence.
- A scientific negative is a completed result.
- A delivery retry never causes another fit.
- There is no TEST unit.

## Placeholders

All of these are private and never committed:

| Name | Meaning |
|---|---|
| `$RUNTIME` | Staged predictor runtime worktree on ext4, under `/var/tmp`, never `/home` |
| `$PRIVATE` | Private directory outside git (`chmod 700`) |
| `$PRIVATE/deployment.json` | The existing private deployment spec |
| `$DATA_ROOT` | The role's data root |
| `$EXPERIMENT_STORAGE_ROOT` | The configured experiment storage root |
| `$I6DMO_ROOT` | A NEW campaign root on ext4 |

Each role in `deployment.json` must declare `experiment_storage_root`, the root that holds
the population directory named by `private_location` in the committed
`POPULATION_SEAL.json`, relative to `$EXPERIMENT_STORAGE_ROOT`. Without it, a role cannot resolve the population
pin and the worker refuses that role with `MATCHED_PIN_UNRESOLVED`. The warehouse
`environment_file` is checked **by path only**. Its contents are never opened or echoed.

Run every command from `$RUNTIME`, on CPU, under a memory cap:

```bash
cd "$RUNTIME"
export CUDA_VISIBLE_DEVICES=""
```

## 1. Verify the frozen population (data lane CLI)

```bash
SEAL=docs/i6n/direct_multi_output_static/matched_short/POPULATION_SEAL.json
DIGEST="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["population_digest"])' "$SEAL")"
POPDIR="$EXPERIMENT_STORAGE_ROOT/$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_location"].split("/", 1)[1])' "$SEAL")"
crispdm-run -m 3G -t 600s -n i6dmo-matched-verify -- python -m tools.i6dmo.common_population verify \
  --population-dir "$POPDIR" \
  --expected-digest "$DIGEST" \
  --input-dir "$DATA_ROOT/inputs/extended-no-stlfsi4"
```

The digest and the directory are both read from the committed `POPULATION_SEAL.json`; neither
is typed by hand. A reseal of the designs or a rebuilt population changes both.

## 2. Prepare: write the private, host-specific JSON

```bash
crispdm-run -m 3G -t 600s -n i6dmo-matched-prepare -- python -m tools.i6dmo matched-short prepare \
  --deployment "$PRIVATE/deployment.json" \
  --input-dir "$DATA_ROOT/inputs/extended-no-stlfsi4" \
  --out "$PRIVATE/matched-short.prepared.json"
```

The designs default to the committed `matched_short/EXPERIMENT.mg1.json` and
`EXPERIMENT.mg2m.json`. The seal defaults to the committed `POPULATION_SEAL.json`. The
population directory is the seal's `private_location` under the role's storage root. Use
`--mg1-design`, `--mg2m-design`, `--population-seal` and `--population-dir` only to override
these.

`prepare` exits 2 and prints `{"state": "REFUSED", "cause": ...}` with one of these causes:

| Cause | Trigger |
|---|---|
| `TEST_REQUEST` | `--test`, or a design that opens TEST or reaches past 2025-01-01 |
| `PRIVATE_OUTPUT_INSIDE_GIT` | `--out` is inside a git checkout |
| `UNAVAILABLE_MOUNT:*` | data root, run root or storage root missing, or population directory absent |
| `UNFROZEN_COMPARISON:*` | Any typed refusal from `load_common_population`, including `POPULATION_DIGEST_MISMATCH`; a POPULATION.json digest mismatch; or a design that is not the one the population was frozen for |
| `INCOMPATIBLE_DESIGN:*` | Not the matched identity, or no sealed runtime pin |
| `INCOMPATIBLE_DONOR:*` | `--prior-root` is not a matched contrast with identical seals and pins; or it ran other entry points than the real adapters, is an engineering/CPU campaign, or has a fit receipt that is not `scientific: true` on a GPU without overrides |
| `MISSING_ARTIFACT:*` | Deployment spec, input or design missing |
| `INADEQUATE_MEMORY` / `INADEQUATE_DISK` | Below 4 GiB available memory or 20 GiB free disk |

## 3. Validate: launches nothing

```bash
crispdm-run -m 3G -t 600s -n i6dmo-matched-validate -- python -m tools.i6dmo matched-short validate \
  --prepared "$PRIVATE/matched-short.prepared.json"
```

## 4. Preflight: read-only, per role

```bash
crispdm-run -m 1G -t 600s -n i6dmo-matched-preflight -- python -m tools.i6dmo matched-short preflight \
  --prepared "$PRIVATE/matched-short.prepared.json" --out-dir "$PRIVATE/precheck"
```

This writes `$PRIVATE/precheck/PRECHECK.matched.json`. It exits 0 on PASS and 3 on FAIL.
For each role it checks:

- disk;
- the input and population digests on that role;
- on every GPU role:
  - GPU visibility;
  - initial-cap admission (no measured 25-slot peak exists, so the initial cap is used,
    never a 13-slot peak);
  - the worker's **actual** TensorFlow/Keras from package metadata against the runtime pin in
    the sealed designs (`sealed_identities.runtime`, whatever the current seal says).

A mismatch is `RUNTIME_PIN_MISMATCH:...`. Fix it by installing the pinned runtime. It is
never a reason to reseal. Worker admission applies the same runtime check at claim time: a
mismatched role is refused, and the unit routes to the next GPU role.

### What the automation refuses or waits on

- **Reuse.** A prior root's terminals are adopted only when its fit procedure matches: the same
  entry points, fit device and engineering mode. A scientific campaign also requires every donor
  fit receipt to be `scientific: true` on a GPU without overrides. Anything else is
  `PRIOR_REJECTED` with a typed `DONOR_*` reason, and the unit is fitted fresh.
- **Science only.** A scientific campaign's report refuses a fit receipt that is not
  `scientific: true` on a GPU (`NON_SCIENTIFIC_FIT`), and so does the decision. The trainer
  stamps every CPU fit `scientific: false`. A worker whose environment, injected environment
  or role `environment` carries any `I6DMO_TEST_*` variable is refused with
  `TEST_OVERRIDE_IN_SCIENTIFIC_ENVIRONMENT`.
- **Infrastructure before any training.** These codes mean the host refused before any training
  started:
  - `POPULATION_PIN_MISMATCH`, `INPUT_PIN_MISMATCH`, `MATCHED_PIN_UNRESOLVED`;
  - `NO_GPU_VISIBLE`, `RUNTIME_PIN_MISMATCH`;
  - a population/source/calendar digest refusal.

  None of them is a terminal. The unit waits as `WAIT_RESOURCE INFRA_PREFIT:<code>@<role>`,
  does not consume the attempt budget, and is claimed again later. After 1800 s, status
  reports `INFRA_PREFIT_BLOCKED` and `INFRASTRUCTURE_BLOCKED`. A failure after training started
  stays a terminal, and a fit is never retried with an altered cap or altered data.

## 5. Init, then deploy

```bash
crispdm-run -m 3G -t 600s -n i6dmo-matched-init -- python -m tools.i6dmo matched-short init \
  --prepared "$PRIVATE/matched-short.prepared.json" --root "$I6DMO_ROOT"

# render the systemd user units only (no systemctl call at all)
python -m tools.i6dmo deploy --root "$I6DMO_ROOT" --deployment "$PRIVATE/deployment.json" \
  --render-to "$PRIVATE/units-preview"

# install the unit files and daemon-reload, enable nothing
python -m tools.i6dmo deploy --root "$I6DMO_ROOT" --deployment "$PRIVATE/deployment.json" --no-start
```

`init` re-runs `validate` first. Enabling the timers, a plain `deploy` without `--no-start`,
is **not part of this pass**. It requires the model lane's adapters and the reviewed
design/test gates.

At the automatic `CLOSE`, or at a typed technical terminal whose facts are delivered and
mirrored, `tick`/`publish` stop this campaign's own units:

- each timer gets `systemctl --user disable --now`;
- each idle service gets `systemctl --user stop`;
- the invoking oneshot is not stopped from inside; it exits by itself.

Only units listed in the campaign's own `DEPLOY.json` that carry its id are touched. The result
is recorded in `$I6DMO_ROOT/TIMERS_STOPPED.json`, which is also mirrored. A failed stop shows as
`TIMER_STOP_FAILED` in `status`, is retried on the next tick/publish, and never alters `CLOSE`.

## 6. Status: campaign, arm and unit

None of these launch or alter a fit:

```bash
python -m tools.i6dmo matched-short status --root "$I6DMO_ROOT" --level campaign
python -m tools.i6dmo matched-short status --root "$I6DMO_ROOT" --level arm
python -m tools.i6dmo matched-short status --root "$I6DMO_ROOT" --level arm --arm mg1
python -m tools.i6dmo matched-short status --root "$I6DMO_ROOT" --level unit --unit mg2m.fit
python -m tools.i6dmo status --root "$I6DMO_ROOT" --level campaign --probe-systemd --json
```

What each level reports:

| Level | Fields |
|---|---|
| Arm | State, `complete`/`denominator`, live children (role, device, start, heartbeat age, progress), last progress time, failure causes, ETA |
| Unit | Code identity (`unit_code_commit` vs `campaign_code_commit`), reuse, decision |
| Campaign | Typed `failures` and `SERVICE_FAILED` |

The arm ETA is MEASURED or in-flight upper bound only from that arm's own fit. Otherwise it
is `UNKNOWN`.

The campaign level reports `SERVICE_FAILED` for any of these:

- a crashing service;
- a timer silent for more than 1800 s;
- a deployed service that never ran;
- a failed systemd unit (with `--probe-systemd`);
- a stale attempt.

None of these appear as indefinite WAITING.

## What was verified, and how

| Check | Command or test | Result |
|---|---|---|
| Every command above has `--help` | `test_every_documented_command_has_help` (11 commands) and `test_the_documented_data_lane_verify_command_has_help` | Pass |
| Every `python -m tools.i6dmo` line in this document parses | `test_run_document_names_only_existing_commands` | Pass |
| Finite local fixture end to end through the CLI | `tests/i6dmo/test_ops_matched_runner.py::test_cli_matched_short_end_to_end_on_a_local_fixture` | Pass |

The end-to-end fixture runs, in order:

1. a REAL common population built and sealed by the data lane from the synthetic input;
2. `prepare`;
3. a refused `prepare --test`;
4. `validate`;
5. `preflight` with an injected probe;
6. `init --entrypoints` with orchestration-only fake arm children;
7. `deploy --render-to`;
8. `tick`;
9. a finite worker drive;
10. the REAL comparison hook;
11. publish to the local test sink, with exact readback;
12. two scratch mirrors;
13. automatic CLOSE;
14. `status` at every level.

Test command:

```bash
CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-repair -- \
  python -m pytest -q -p no:cacheprovider tests/i6dmo/test_ops_matched_runner.py
```

Not verified here, as typed limitations:

- `tools.i6dmo.replay_monitor` has no matched/common-population mode (review D7). A GPU replay
  of a matched fit needs its own driver, as the reviewer's `review/p7_gpu_replay.py` did.

- `prepare` against the real private population and storage root. `$EXPERIMENT_STORAGE_ROOT`
  is not configured on the implementation host.
- A real GPU fit, or a fit on the real sealed population. The real adapters ran only as a CPU
  engineering fixture on a synthetic population: `tests/i6dmo/test_ops_matched_real_chain.py`.
  That fixture covers init, a real worker claim, the real `unit_runner`, a real
  `python -m tools.i6dmo.train --common-population` for both arms (2 epochs,
  `scientific=false`), the real matched report, `decide_matched`, publish with readback,
  mirrors, automatic CLOSE and `TIMERS_STOPPED`. The CPU engineering mode is refused for any
  design sealed on a non-synthetic calendar.
- Any deploy that enables timers.
