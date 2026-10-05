# Lane H — TRADING-PAPER health and M5PHET-E2E (2026-10-03)

Authority: master plan v3 §11-§13; EXPERIMENT_EXECUTION_QUEUE lanes TRADING-PAPER (priority 12) and M5PHET-E2E
(priority 14); canonical selection-first order §3 lane H. Observed 2026-10-03 ~04:50-05:00 UTC. Every check was read-only
(`systemctl --user show/status`, heartbeats, journals, SQLite opened `mode=ro`). Nothing was started, stopped or restarted;
no broker path was touched; no order was sent.

## 1. TRADING-PAPER service health (by role)

| Role / host | Service | State | Code revision | Model | Last heartbeat / activity | Order flow |
|---|---|---|---|---|---|---|
| coordinator | `lts-alpaca-model-runner` | active (running since 2026-09-28 15:03 UTC, 0 failures) | LIVE CHECKOUT of `lts`, branch `satoshi/mt5-unknown-outcome-20260926` @ `12bce5f` (clean; HEAD unchanged since before start) | `spy-daily-linear-live-v1` (SPY 1d linear classifier) | heartbeat 2026-10-03T04:52Z, state `monitoring`, `read_only: false` | **YES**: 30 paper orders requested/accepted/filled 2026-08-03 .. 2026-10-02T13:33Z; 1 open exposure |
| coordinator | `lts-alpaca-paper-observer` (+ 5 min timer) | timer active, last run OK 04:52Z | live checkout as above | none (read-only preflight) | `orders_submitted: 0`, account ACTIVE, paper | none |
| coordinator | `lts-multi-venue-shadow` (+ 5 min timer) | timer active, last run OK 04:48Z | live checkout | none (no-order snapshot) | OK | none by design |
| coordinator | `lts-paper-execution-watchdog` (+ 5 min timer) | timer active, last run OK, 0 emission failures | live checkout | — | one active event: `mt5_bridge_stale` | — |
| coordinator | `lts-ibkr-paper-observer` | inactive, no timer, never started this boot | live checkout | — | IBKR runner heartbeat last written 2026-08-23 | none |
| worker_b | `lts-mt5-execution-bridge` | active since 2026-09-28 17:34 UTC, 0 restarts | LIVE CHECKOUT of `lts` on worker_b, branch `main` @ `b05d01d` (2026-08-25, clean) | — | — | — |
| worker_b | `lts-mt5-model-runner` | active since 2026-09-28 17:34 UTC, 0 restarts | PINNED runtime copy `.runtime/lts-explicit-close-ed3cf67` @ `1c16d2d` (drop-in override) | `ethusdt-4h-linear-live-v1` (ETHUSD 4h linear) | heartbeat 2026-10-03T04:54Z, state **`snapshot_stale`**: last account snapshot received 2026-09-08T22:28Z | last fills 2026-09-01 (42 filled/closed); 46 rejected decisions to 2026-09-08; nothing since (no snapshot, so no decision) |
| worker_b | `lts-mt5-bridge-watchdog` (+ 5 min timer) | timer active, last run OK | live checkout | — | 1 active event, emitting | — |
| worker_b | qemu guest named for MT5 paper | **running** for 4 d 06 h, 4 vCPU, 8 GiB assigned (about 5.8 GiB resident), autostart **enabled** | — | — | — | the EA inside it is not delivering snapshots since 2026-09-08 |
| gamma host role | (no trading service) | — | — | — | — | — |

Also live on the coordinator (not trading): `m5phet-chat`, `m5phet-tailnet-forward`, `data-gov-m5phet` (active).

### Naive eligibility of what trades

Neither model that holds an order path is naive-eligible. Both manifests (`prediction_provider.live_linear_manifest.v1`)
declare `live_execution_eligible: false` and `live_inference_eligible: false`, carry NO same-row naive comparison at all
(the word does not occur), and their own validation champions are negative: SPY weekly RAP -0.0065 (annualized return
-0.66 %), ETH weekly RAP -1.60 (annualized return -693 %). The plan rule is "only forecasts that beat their naive per
horizon enter; none currently does". So **the Alpaca paper route is placing orders from a non-naive-eligible model**
(paper only, `environment: paper`; no real capital). The MT5 demo route would too, but is currently mute because its
snapshots are stale.

## 2. M5PHET-E2E receipt

Chosen use case (decided here): the **causal (ATE) family**, the next row of front I's family inventory after regimes —
CPU-only, a real provider with retained fitted studies, the only one besides regimes stageable on a worker without
TF/SB3 interpreters or the Laya GPU.

- Repository M5PHET, branch `satoshi/m5phet-e2e-20261003`, tip **`14d9b05`** (pushed; base `cc4b847` of
  `satoshi/i-m5phet-20261001`). Files: `tests/test_family_e2e.py` (causal row + point-in-time test),
  `docs/E2E_CAUSAL_2026_10_03.md` (receipt), `docs/I1_FAMILY_INVENTORY_2026_10_01.md` (row updated).
- Chain: envelope `area: causal`, `state_ref = causal-ate:<digest>`, pinned `as_of`, no attachment -> `validate_task`
  -> real `causal_inference` provider (entry point, causal-inference `origin/master` `bb11d64`) loading the digest-checked
  study -> typed per-question output, `execution_authorized: false` -> workbench store -> new app reads it back.
- Asserts: ATE `effect_size`/interval/unit/population/assumptions equal the artifact read independently from disk
  (ATE 2.0294, 95 % CI [1.9313, 2.1275], synthetic development study — copied, not measured); `cate` on a constant-effect
  study REFUSED `NOT_ESTIMABLE` with no number, message `PARTIAL`; identical under a second request id; byte-identical
  after restart; a clock before the study existed refuses every question.
- Runs on worker_b, isolated Python 3.12.13 venv (chat-venv pins), `crispdm-run -m 512M -t 10m`:
  RED (no state dir, required) 2 failed by name; GREEN 2 passed, 1 skipped (unsupervised, not required), 1.42 s, peak RSS
  58,408 kB; mutation (effect_size +1e-9 in the provider copy) 1 failed, reverted, green again.
- Test file sha256 `e4a863df9c88fdfcc6f5a9888c787716a3261767c07f89822bb96c910d58c6dd`; study
  `f18fafdaa95e22a5b5cf147c80f87ad0836691c3d550c13bd695992fc0a2602b`. NO_NEW_MEASUREMENT of quality
  (`causal_accuracy` refused by name).

## 3. Tips

| Repo | Branch | Tip |
|---|---|---|
| M5PHET | `satoshi/m5phet-e2e-20261003` | `14d9b05` |
| lts (coordinator, live, read only) | `satoshi/mt5-unknown-outcome-20260926` | `12bce5f` |
| lts (worker_b, live bridge, read only) | `main` | `b05d01d` |
| lts (worker_b, pinned runner copy) | — | `1c16d2d` |

## 4. Defects (recorded, not fixed — live services)

- **H-D1 (gate)**: `lts-alpaca-model-runner` executes paper orders from `spy-daily-linear-live-v1`, whose manifest says
  `live_execution_eligible: false`, has no same-row naive comparison and a negative validation champion. The runner does
  not enforce the manifest's eligibility flag nor a naive gate. Last fill 2026-10-02T13:33Z; one exposure open. Paper only.
- **H-D2 (gate)**: same for `lts-mt5-model-runner` (`ethusdt-4h-linear-live-v1`, annualized validation return -693 %);
  presently inert only because of H-D3.
- **H-D3 (operations)**: MT5 demo snapshots stale since 2026-09-08T22:28Z (25 days); watchdog event
  `mt5_bridge_stale` active and emitting. Bridge process up; the broken link is the EA in the guest (consistent with the
  2026-09-26 observation).
- **H-D4 (plan conflict)**: plan v3 §13 says the MT5 VM is neither run nor reserved; on worker_b the MT5-paper qemu guest
  IS running (4 d 06 h, 8 GiB assigned, ~5.8 GiB resident, autostart enabled), withholding that RAM from experimentation.
  Not stopped (order: do not stop the MT5 VM).
- **H-D5 (provenance)**: the Alpaca runner and the MT5 bridge run from live working checkouts (a feature branch on the
  coordinator, `main` on worker_b), not pinned commits; only the MT5 runner uses a pinned runtime copy.
- **H-D6 (unit)**: the failed transient unit `codex-modular-r3-s2021.service` (not trading) lingers on the coordinator.
- **M5PHET**: the three retained CATE studies share one example title, so a title-based example cannot single one out;
  the forecast, policy and real-Laya rows still lack an E2E with persistence (next row: forecast, needs its TF bundles
  and interpreter staged on a worker).
