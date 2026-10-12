# I6-N swarm weekly: return

Orders: SATOSHI_I6N_SWARM_WEEKLY_2026_10_11.md (ac77366a) and
SATOSHI_I6N_EXECUTE_NOW_20261012.md (b78ec898). Plan af0c3570 (da51723a).
Integration branch `satoshi/i6n-swarm-weekly-20261011`. TEST not opened, no trading.

## Verdict

INTEGRATED AND STAGED, NOT STARTED. NO_NEW_MEASUREMENT. The production start
step was denied by the permission classifier ([Production Deploy]); it was not
bypassed. No production optimizer update has occurred, so no training start is
claimed. The running S3 cost pilot on the preferred GPU was not touched.

## What was delivered

1. **Five lanes, cross-reviewed.** A trajectories/model (A.4 0f05dd78: real
   trajectory heads, per-head masks/scales/naives, outer-week scoring outside
   stopping/fitness, direct multi-output arms, host-relative contexts). B DOIN
   swarm (ledger with leases/fencing, max 3 attempts with one ledger-linked
   recovery, no CPU fallback, campaign CLI, stage auto-advance). C weekly data
   (53 refit units; 2023Q1-Q3 under declared clock assumption after one bounded
   429 retry; plan-v6 seals all 681 cells). D external archive (PUT->GET->digest
   ->commit, four separate states, decision events, incremental ETL). E search
   (exact domain, canonical identity, plan v1.2 sealed, receipt-bound fold
   fitness). Review findings were fixed regression-first; E's round-1 C findings
   were fixed in C.3 but not re-verified by E.
2. **Integration** 2ded0406 merged Musashi's launch work (admitted-scope request
   relay, trajectory-aware verification, single v2 wiring path, S3 preflight).
   Two-real-DOIN-node smoke with labelled synthetic evaluators: 96 units, 0
   duplicate fits, failure isolation, terminal adoption, late join; it exposed and
   fixed two runtime bugs (null-fitness peer result crash, stale-genesis wait).
3. **Deployment staged.** Effective private config validated (seed 6102026,
   production mode, capabilities VALIDATION_OUTER_WEEK and DIRECT_MULTI_OUTPUT_ARM);
   caps = ceil(1.25*(measured peak + coverage adjustment)): S1 3,023,954,012 B,
   S2 3,044,231,417 B; S3 not admitted (no completed pilot receipt). Code (pins in
   ARTIFACTS.json) and data root (701 files) replicated and byte-verified on both
   workers; plan sealed with 52 DAG nodes; preflight READY.

## Results

No annual or production result exists. RESULTS.csv carries only the earlier
historical endpoint search bests (h12, h72; TRAIN and one inner interval before
2024; engineering history, not validation) and the 12 annual rows as NOT_STARTED.
Each later row will carry its own horizon naive on identical rows and relative
error (MAE_h - naive_h)/naive_h.

## Blocked on the owner

1. Run (or allow) the start command in COMMANDS.md. After it, the secondary
   starts admitted S1/S2 production, the preferred joins by timer after S3, and
   the coordinator tick advances search -> annual weekly -> multi-output.
2. Run (or allow) DEPLOY_STEPS_D (§0 rebuild the lake wheel first). Until then
   publication stays in the durable outbox; nothing is claimed on-chain.

## Limits

Calendar is nominal (FX NY 17:00 assumed), not broker-verified; 2019 span under a
declared clock assumption; 2024-informed feature selection is retrospective
development; Q1/Q2 2023 inner support is thin (h72: 51 rows); the deploy tool was
written before its tests (no recorded red run); S3 admission needs a new plan with
its receipt; no warehouse snapshot was taken.
