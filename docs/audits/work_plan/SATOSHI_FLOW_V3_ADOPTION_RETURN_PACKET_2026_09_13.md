# Return packet — Flow v3 adoption (P0 integration, contracts, deployment; P1 adoption)

**Date:** 2026-09-13
**Order:** `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_FLOW_V3_ADOPTION_ORDER_2026_09_13.md`
**Inputs integrated:** data-gov `02f07d7`, financial-data `7f77e3ce6`, predictor `bd51398` (by identity, chronology kept)
**Stopping at:** `FLOW_V3_INTEGRATED_CONTRACTS_INSTALLED_THROWAWAY_PROVEN_PRODUCTION_RESTART_PENDING_OPERATOR`

CPU only. No GPU, confirmation, live, venue or promotion. Nothing here opens D3.

---

## 0. The four states, up front

| state | what it means here | holds for |
|---|---|---|
| `IMPLEMENTED` | code + tests committed and pushed | predictor consumer (Musashi's, with two fixes), data-gov `governed_exec`, wrappers of preprocessor / feature-eng / feature-extractor, dispatcher gate, coverage query, contracts of three resources |
| `DEPLOYED` | production services (`:5055`, `:5056`, `:5057`) run the integrated code | **none** — the restart of the three services was refused by my execution policy (§3.2); they still run the pre-v3 code they were started with at 02:20 |
| `GOVERNED_RUN_PROVEN` | a real run through data-gov, terminal on PostgreSQL, exact reconciliation | predictor (toy config) and preprocessor (phase_1b config), both on a **throwaway** stack with a throwaway database; not on production |
| `NON_GOVERNING_ONLY` | mechanics and tests only | feature-eng and feature-extractor wrappers (no contracted inputs / cannot complete standalone); agent-multi, DOIN and live runners (evidence only, §5) |

## 1. My own faults, first

1. **Production restart not done.** P0.3.2 orders the ordered restart with the integrated code. The action was refused by the assistant's execution policy ("interfere with workloads") when I tried to write the restart script; per that policy I stopped and left the services untouched. Everything before the restart (backups, contracts, validation) and everything after it that can be proven on disposable services was done. The operator has the exact script text in this packet's local record (`~/.local/state/crispdm-data-foundation/flow_v3_deploy_2026_09_13/`) to run, or can grant the action.
2. **Musashi's integrated code had four defects that only appear outside the Flask test client.** I found them by running the real threaded servers; all four are fixed with regression tests that fail on the previous code (§3.4).
3. The first throwaway run wasted one campaign key on a 400 (metric name with a space); the harness output committed is the clean re-run.

## 2. P0.1 — integration and verification (done)

Merges (no rewrite, chronology kept): predictor `4506d8b` (merge of `bd51398`), financial-data fast-forward to `7f77e3ce6`, data-gov `249741a` (merge of `02f07d7`). Suites at those tips: data-gov 116, financial lake 58, predictor consumer 24, OLAP lake 18 (+1 skipped), Musashi's E2E exact. Terminal evidence reviewed: `gov_terminal_dataset` binds delivery id, lake, resource, role, range, bytes, source hash, delivered hash, availability-contract hash and verification state (P0.1.2 satisfied). PostgreSQL proven on a throwaway database (terminal store/replay/conflict).

## 3. P0.2 — factual contracts (done for what the next campaign consumes)

Document: `data-gov/docs/07_RESOURCE_CONTRACTS_INSTALLED_2026_09_13.md` (commit `249df1e`).

- Resources consumed by the only authorised campaign (the toy governed micro-run): `predictor_examples` `phase_1/normalized_d{4,5,6}.csv`.
- Contract installed for the three: `{DATE_TIME, DATE_TIME, NAIVE_WALL_CLOCK, null, "4h"}`, digest `4d0ead37c2e471faced996f7b065eb2c5327be00c26a3861cf53171caaf72821`.
- Derivation from bytes, not from the column name (`predictor/docs/audits/evidence/repro_runs/flow_v3_tools/p02_contract_derivation.py` + `.out`): predictor `base_d*` are byte-identical to preprocessor `phase_1b/base_d*`; `typical_price` equals `phase_3b_downsampled.csv` at the same `DATE_TIME` (6,298/6,298); the 4h value at `t` is the **mean of the hourly rows in (t−4h, t]** (21,703/22,521 exact; 21,702/21,702 on full windows), so the row is complete by `t + 1h` under either hourly-label hypothesis. The Dukascopy 4h export of feature-eng is not in this lineage (0/6,298 timestamps).
- Use class: **historical offline, day-granular**. Cuts are `[from, to+1d)` on `DATE_TIME`; the last bar of a day (20:00) completes by 21:00 < 24:00, so a cut never contains information completing after its range end (verified on the bytes). Not live-equivalent: at bar level the label may precede completion by up to 1h and the hourly time zone has no producer statement.
- Validation against physical bytes before deployment (`p02_validate_contracts.py` + `.out`, `ok=true`): `AS_IS` delivered sha256 = source = disk; one-year cuts equal an independent pandas count, byte subset, digest of streamed bytes, completion bound; the other 105 resources of the lake are refused with `resource availability contract required`.
- Closed (factual deficits, no contract invented): 105 other `predictor_examples` files (phase_1_b/c derive from the Dukascopy export whose label convention and GMT relation have no producer statement); all 5,275 `financial_files` resources — the 198 first-batch contracts declare `timestamp_meaning`, `timezone`, `availability_rule`, `availability_delay_seconds` and every variable's `available_time_rule` as `UNKNOWN`; the financial lake stays `resource_contracts: {}` (fail-closed). No financial resource is needed until D3 opens.

## 3.1 P0.3 — backups (done, no datasets)

`~/.local/state/crispdm-data-foundation/flow_v3_deploy_2026_09_13/backups/`: `data-gov_accounting.db` (sqlite `.backup`, 27 events), the three service configs, the untracked financial-lake `local.json`, `predictor_olap_before.dump` (pg_dump custom format, 420 MB, exit 0), `SHA256SUMS.txt`. Exact row counts of all 62 cube tables before: `CUBE_COUNTS_BEFORE.txt` (sha256 `63aa9242…`).

## 3.2 P0.3 — restart (NOT done) and production state

- Services `:5057`, `:5056`, `:5055`: healthz `ok`, same pids as at 02:20, working directories are the three checkouts, `NRestarts=0` on the loader, PostgreSQL and Metabase untouched.
- Probes prove they run the **old** code: `GET :5057/api/v2/terminals` → 404, `GET :5056/api/v2/download` → 404, `POST :5055/api/v2/campaigns` → 404, `/api/v1/lakes` lists only `financial_files` and `olap_cube` (no `predictor_examples`).
- Cube counts after my work (`CUBE_COUNTS_AFTER.txt`): no `gov_*` table exists in the real cube; the only differences from before are `dim_campaign 10→11`, `dim_campaign_run 10→13`, `fact_campaign_unit 188→195`, produced by the running OLAP loader, not by anything in this packet. No throwaway database remains.
- The restart procedure is prepared as a script (Appendix A): it carries the lake token and PG password from each running process's environment without printing them, stops OLAP lake → financial lake → data-gov, relaunches each from its checkout with `--load_config examples/config/default.json` (data-gov with `--save_config` to record the effective config), waits for healthz and records pids, HEADs and config digests. Even writing it to disk was refused by the policy, so it exists only as text here; it was not executed. After the restart, the production proof is one command: `tools/governed_run.py` from a clean predictor worktree with the toy config against `:5055` (the throwaway run of §3.3 is exactly that, on disposable services).

## 3.3 P0.3 — governed runs proven on a throwaway stack

`p03_throwaway_governed_runs.py` + `.out` (predictor `5d2af40`, harness `b564839`): disposable OLAP lake on a throwaway PostgreSQL database created and dropped by the script, disposable data-gov with the same `predictor_examples` contracts, predictor from a **clean detached worktree** (`08a4c04`), CPU, `--execution_purpose ARCHIVAL_REPLAY_NON_AUTHORITATIVE`.

| run | terminal | evidence |
|---|---|---|
| 1 toy micro-run, fresh out-dir | `COMPLETED` | 6 deliveries (3 `VERIFIED_TRANSFER` + 3 `VERIFIED_CACHE`, contract `4d0ead37…`, `AS_IS`), 90 metrics, 6 artifacts, reconcile `missing=[] accounting_only=[] lake_only=[]` |
| 2 same out-dir | `REFUSED` `GOVERNED_RUN_REFUSED:governing output namespace is not fresh` | no download, stored in the cube |
| 3 bogus plugin | `FAILED` `PREDICTOR_EXIT_1` | stored with its deliveries, no metrics |
| 4 OLAP lake killed while training | pending → `COMPLETED` | exit≠0 with `terminal_pending=true`, 1 envelope in `outbox/pending`; lake restarted; `flush_governed_terminals` → `sent 1 pending 0`; second flush `sent 0`; manual replay of the accepted envelope → `200 already_stored`; counts unchanged across flush 2 and replay; reconcile exact |

Cube states present: `COMPLETED`, `FAILED`, `REFUSED` (`INCONCLUSIVE`/`QUARANTINED` are not producible by this runner). Throwaway counts at the end: `gov_terminal 4`, `gov_terminal_metric 180`, `gov_terminal_dataset 18`, `gov_terminal_artifact 13`; database dropped.

## 3.4 Defects found in the integrated code (fixed, with mutation evidence)

| # | where | defect | fix | test (fails on previous code) |
|---|---|---|---|---|
| 1 | data-gov `lake_plugins/files_lake.py` (`85f71b5`) | pandas 3.0.3 + pyarrow 25: **segfault** in `string_arrow._from_sequence` on the second `read_csv` issued from a werkzeug worker thread after a streamed delivery (second `/api/v2/download`); never with the Flask test client, plain threads or the main thread; reproduced with and without holdout | pin `mode.string_storage=python` at import | `tests/user/test_threaded_server_downloads.py` (real server in a subprocess, 4 downloads): exit −11 before |
| 2 | financial-data `lake/inventory_plugins/fs_inventory.py` (`13e6b1f47`) | same segfault on the second v2 download | same pin | `lake/tests/test_threaded_server_v2_downloads.py`: exit −11 before |
| 3 | financial-data `lake/web_plugins/default_web.py` (`13e6b1f47`) | **slot leak**: `send_file` answers in direct passthrough and the dev server closes only the file wrapper, so `Response.call_on_close` never fires; two deliveries exhausted `max_downloads` for the life of the process, every later governed download `503 download slots busy` | release tied to the delivered handle (`_ReleasingHandle`, as the v1 route's `_SlotFile`) | same test: with fix 2 alone it fails with 503 on the third request; semaphore probe returns to 2 after each delivery |
| 4 | predictor `tools/governed_run.py` (`08a4c04`) | every real terminal refused `400 invalid metric`: predictor writes `Naive MAE`, `governed_terminal.v1` keys are `[A-Za-z0-9._:-]`; the refusal was invisible because `TerminalOutbox.flush` swallowed the reason | `metric_key()` (`Naive MAE` → `Naive_MAE`); `flush` returns `failures` (file → reason), recorded in `GOVERNED_RUN.json` and printed by the flush tool | `test_metric_labels_become_terminal_keys`, updated outbox test |
| 5 | data-gov `app/client.py` (`8cd45f5`) | governed deliveries carry no `Content-Disposition`, so the content cache stored files without suffix | cache entry keeps the resource's suffix | assertion in `test_governed_exec.py` |

Observation for Musashi (not changed): a terminal permanently refused with 4xx stays pending forever and, by design, blocks every later governing run of that actor (`PRIOR_TERMINAL_PENDING`). Fail-closed is right; an operator path for a provably invalid envelope is missing.

Suites after the fixes: data-gov 125 passed; financial lake 59 passed; predictor consumer + gate + dispatcher 53 passed; OLAP lake 18 passed, 1 skipped; Musashi's E2E exact on the fixed code; profile tests 1 each in preprocessor, feature-eng, feature-extractor.

## 4. Commits per repository (all pushed)

| repo | branch | commits |
|---|---|---|
| data-gov | `master` | `249741a` (merge of Musashi `02f07d7`), `249df1e` contracts + doc 07, `85f71b5` string storage + threaded test, `bf9f8fb` `tools/governed_exec.py` + tests, `8cd45f5` cache suffix |
| financial-data | `satoshi/c122-c145-20260912` | `7f77e3ce6` (Musashi, fast-forward), `13e6b1f47` two server fixes + threaded test |
| predictor | `satoshi/c166-c184-20260913` | `4506d8b` (merge of Musashi `bd51398`), `e9eeede` P0.2 tools, `08a4c04` metric keys / outbox failures / harness, `b564839`, `5d2af40` sealed P0.3 output, `91c7806` gate + coverage + dispatcher wiring, `d3d2887` preprocessor proof |
| preprocessor | `satoshi/crispdm-census-gate-20260910` | `20db3fb` `tools/governed_run.py` + profile test |
| feature-eng | `docs/agent-onboarding-20260816` | `7118b83` `tools/governed_run.py` + profile test |
| feature-extractor | `docs/agent-onboarding-20260816` | `384ad8d` `tools/governed_run.py` + profile test |

## 5. P1 — adoption matrix

`data-gov/tools/governed_exec.py` is the one place where the protocol lives for the non-predictor repositories (strict code identity, execution-spec digest before data, one campaign/unit, prior pending refuses, fresh output namespace, governed download + confirmation, CPU command, terminal `COMPLETED|FAILED|INCONCLUSIVE|REFUSED` through the durable outbox with failure reasons, reconciliation). Each repository's `tools/governed_run.py` only declares its profile (input keys, output keys, command, metrics source, artifacts). No data-gov call occurs inside `fit/transform/step/learn/batch`: every call is before the command starts or after it ends.

| runner | state | evidence / deficit |
|---|---|---|
| predictor `tools/governed_run.py` | `IMPLEMENTED`, `GOVERNED_RUN_PROVEN` (throwaway), not `DEPLOYED` | §3.3; two fixes (§3.4 #4) |
| preprocessor `tools/governed_run.py` | `IMPLEMENTED`, `GOVERNED_RUN_PROVEN` (throwaway) | `p1_preprocessor_throwaway.py` + `.out` (`d3d2887`): clean worktree `20db3fb`, `phase_1b` config over `phase_1/normalized_d4.csv` (`VERIFIED_TRANSFER`, contract `4d0ead37…`), `COMPLETED` with 12 `rows` metrics (base_d1..d6, normalized_d1..d6), 16 artifact hashes, reconcile exact; second run into the same directory `REFUSED` without download. The preprocessor's own eligibility adapter needs the shared gate root (`CRISPDM_ELIGIBILITY_GATE`) and `execution_purpose`. |
| feature-eng `tools/governed_run.py` | `IMPLEMENTED`, `NON_GOVERNING_ONLY` | inputs (`input_file`, `high_freq_dataset`, `sp500_dataset`, `vix_dataset`, `economic_calendar`) must be contracted lake resources — none is today (OHLC/Dukascopy semantics unproven, §3); the default plugin writes fixed-name CSVs into the cwd, so the wrapper runs it with the output directory as cwd; metrics: row count of `output_file` only |
| feature-extractor `tools/governed_run.py` | `IMPLEMENTED`, `NON_GOVERNING_ONLY` | six predictor-style inputs; cannot complete from its own checkout (`stl_preprocessor` comes from predictor, per its AGENTS.md); metrics restricted to documented `save_log` keys whose exact names are unverified without a run; `loss_plot_file` gets a latent-dim suffix and is not captured as an artifact |
| agent-multi / gym-fx (trading eval runners, dispatchers) | not adopted — evidence only | decision runners (`tools/p1_difficulty_lr_factorial.py`, `tools/l1_factorial_screen.py`, `app/campaign_supervisor.py`, `app/weekly_promotion.py`, …) already carry content-addressed manifests, cell records and an OLAP layer (`promotion_*_olap`, `weekly_result_*_olap`); inputs are read once in `gym-fx/app/env.py.__init__` (never in `step/reset`); `app/main.py` overwrites outputs (no freshness refusal); no data-gov hook exists. Seam: a `governed_exec` profile per decision runner with `data.input_data_file`/`dataset_manifest_file` as inputs and `results_file` as metrics source; the repository forbids running its campaigns (GPU, "do not run") and the campaign supervisor is a long-lived process, so adoption there is a design item for Musashi, not a mechanics item |
| DOIN publishers / consumers | not adopted — evidence only | no component downloads a dataset by hash: `doin-core` hashes synthetic data and chain objects only; `doin-plugins/predictor/optimizer.py` delegates to the predictor repository; results reported over the network (`TaskCompleted`, `POST /api/shared/result`) carry a float fitness, no outcome enum, no dataset digest. Seam: the delegation into predictor is where `tools/governed_run.py` applies; the DOIN result messages need a terminal identity field before the cube can bind them |
| live runners (lts, prediction_provider, heuristic-strategy) | not adopted — evidence only (frontier) | `lts` runners ingest bars from venue clients / local bridges with a zero-network sink and an offline replay (`tools/live_sim_replay.py`); `prediction_provider/plugins_feeder/data_fetcher.py` fetches Yahoo Finance at runtime without hashing (contradicts its AGENTS.md "offline" claim); `heuristic-strategy/app/plugins/plugin_api_predictions.py:242` calls the prediction API inside the backtrader per-bar `next()` — a remote call inside the step path, against the rule; an offline `CsvPredictionSource` exists |

Work-plan gate: `predictor/tools/flow_v3_gate.py` + `df_dispatch --classification/--campaign-manifest/--non-governing-reason` — a `GOVERNING` dispatch without a valid `governed_campaign.v1` manifest (deliveries, units, terminal destination, digest) is refused before any placement (`DISPATCH_REFUSAL.json`, exit 4); every dispatch seals `DISPATCH_GATE.json` (not on `--resume`, so sealed roots are untouched). Coverage per project: `predictor/tools/flow_v3_coverage.py` (SQL over `gov_terminal` through data-gov's SELECT-only query, `--sql-only` for Metabase); not run against production because no `gov_*` table exists there yet.

## 6. Factual deficits (open)

1. Production not restarted → nothing `DEPLOYED`; the operator action is described in §3.2.
2. No `financial_files` resource has time semantics from a producer statement; 198 first-batch contracts all `UNKNOWN`. Closed until provider evidence exists.
3. `phase_1_b/phase_1_c` and the other predictor phases: Dukascopy label convention and GMT relation unproven; closed.
4. Hourly time zone of the `phase_3b` lineage unproven → `NAIVE_WALL_CLOCK`, offline use only.
5. feature-extractor metrics key names and the suffixed loss-plot artifact are unverified without a run that this checkout cannot complete alone.
6. agent-multi, DOIN and live adoption need design decisions (long-lived supervisors, GPU-only runners, message schemas without terminal identity, remote call inside `next()`).
7. Permanently refused terminal envelopes have no operator path (observation, §3.4).

## Appendix A — restart procedure (not executed)

Operator-run, on the coordinator, after the backups of §3.1 exist. Old pids as observed: OLAP lake `553570`, financial lake `553040`, data-gov `554520`; verify them with `ss -ltnp` before use.

```bash
#!/usr/bin/env bash
# Flow v3 P0.3: restart OLAP lake (5057) -> financial lake (5056) -> data-gov (5055)
# with the integrated code. Environment values (lake token, PG password) are copied
# from each running process and never printed.
set -uo pipefail
D=$HOME/.local/state/crispdm-data-foundation/flow_v3_deploy_2026_09_13
G=$HOME/Documents/GitHub
PY=$HOME/anaconda3/envs/trading-stack/bin/python
mkdir -p "$D/logs"
REC="$D/RESTART_RECORD.txt"; : > "$REC"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$REC"; }

restart() {  # name old_pid port dir extra_args...
  local name=$1 old=$2 port=$3 dir=$4; shift 4
  log "== $name: old pid $old cwd=$(readlink /proc/$old/cwd) started='$(ps -o lstart= -p $old)'"
  local envfile; envfile=$(mktemp); chmod 600 "$envfile"
  tr '\0' '\n' < /proc/$old/environ | grep -E '^(DATA_GOV_LAKE_TOKEN|PGPASSWORD|PGUSER|PGHOST|PGPORT|PGDATABASE|PGUSER_WRITE|PGPASSWORD_WRITE|PYTHONPATH)=' > "$envfile"
  log "   env keys carried: $(cut -d= -f1 "$envfile" | tr '\n' ' ')"
  kill -TERM "$old"
  for i in $(seq 1 30); do ss -ltn "sport = :$port" | grep -q LISTEN || break; sleep 0.5; done
  if ss -ltn "sport = :$port" | grep -q LISTEN; then kill -KILL "$old"; sleep 1; fi
  log "   port $port free; git HEAD $(git -C "$dir" rev-parse --short HEAD)"
  ( cd "$dir" && set -a && . "$envfile" && set +a && \
    setsid nohup "$PY" -m app.main --load_config examples/config/default.json "$@" > "$D/logs/$name.log" 2>&1 < /dev/null & echo $! > "$D/logs/$name.pid" )
  rm -f "$envfile"
  local new; new=$(cat "$D/logs/$name.pid")
  for i in $(seq 1 60); do curl -sf -m 2 "http://127.0.0.1:$port/healthz" > /dev/null && break; sleep 0.5; done
  log "   new pid $new healthz=$(curl -s -m 3 http://127.0.0.1:$port/healthz) cwd=$(readlink /proc/$new/cwd)"
}

restart olap-lake 553570 5057 "$G/predictor/olap/lake"
restart financial-lake 553040 5056 "$G/financial-data/lake"
restart data-gov 554520 5055 "$G/data-gov" --save_config "$D/data-gov_effective_config.json"
log "== checkouts: data-gov $(git -C $G/data-gov rev-parse HEAD) financial-data $(git -C $G/financial-data rev-parse HEAD) predictor $(git -C $G/predictor rev-parse HEAD)"
log "== config sha256: $(sha256sum $G/data-gov/examples/config/default.json $G/financial-data/lake/examples/config/default.json $G/predictor/olap/lake/examples/config/default.json | awk '{print $1" "$2}' | tr '\n' ';')"
log "== loader: $(systemctl --user show crispdm-olap-loader -p ActiveState -p NRestarts | tr '\n' ' ')"
```

Post-restart checks (all must hold): `GET :5057/api/v2/terminals` with the lake token → 200; `GET :5056/api/v2/download` without parameters → 4xx other than 404; `POST :5055/api/v2/campaigns` with `{}` → 400; `/api/v1/lakes` lists `predictor_examples`; the effective config's `predictor_examples` lake carries the three `resource_contracts`; loader `ActiveState=active NRestarts=0`; then the governed micro-run and `CUBE_COUNTS_AFTER` (only `gov_*` may change).
