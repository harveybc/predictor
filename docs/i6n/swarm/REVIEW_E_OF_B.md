# Lane E review of lane B (DOIN swarm, ledger, campaign CLI)

Reviewed, read-only (lane E wrote none of this code):
- predictor `satoshi/i6n-swarm-B-doin-20261011@2b0b6e25`: `tools/i6n_swarm/*`,
  `tools/i6n_swarm_campaign.py`, `tests/test_i6n_swarm_*`;
- doin-plugins `satoshi/i6n-swarm-B-doin-20261011@9564c6f`: the wrapper
  `src/doin_plugins/predictor/optimizer.py`;
- doin-node `satoshi/i6n-swarm-B-runtime-20261011@92661d6`: `src/doin_node/unified.py`.

The review checked B against orders 2.B and 3 and plan sections 3-5. B's
RETURN was still in progress, so the code is the subject.

## Method

- **B's suites**, each run once on B's own code in scratch worktrees (CPU,
  `crispdm-run -m 2G -t 120s`):
  - predictor: 70 passed (ledger, lease service, placement, runner, agent,
    bridge), then 21 passed and 1 skipped (campaign). The skip is the test
    that requires E's module, which is absent in B's checkout.
  - doin-plugins wrapper: 8 passed, with `PYTHONPATH=src:<doin-core src @a90bca2>`.
  - doin-node runtime: 7 passed.
- **Reproducers.** E's modules from `5e400352` were overlaid on B's tree in a
  scratch copy. No fits and no data were used; all fixtures are invented.
  - Script: `evidence_E/review/review_e_of_b.py`.
  - Output: `evidence_E/review/review_e_of_b_result.json`.

## Findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| E-B1 | **FIX_REQUIRED** | The DOIN `optimization_config` rendered by `plan.render_doin_config` carries no `i6n_plan`. E's plugin therefore takes its **legacy v1 path**. That path has no Conv8->Conv4 seed in the initial population, uses the old v1 domain (any dy divisor, adapter width 128), has no size classes and no staged promotion, and uses a summed pseudo-support (`fold_set_support`). Generation 0 contained S2 and S3 genotypes. This violates plan sections 3-4 and orders 2.E ("valores minimos ... de verdad en la poblacion inicial"). | B1: `has_i6n_plan=false`, wire `i6n-shared-deap-v1`, `minimal_seed_in_initial_population=false`, generation-0 classes `[S2,S1,S1,S2,S2,S2,S3,S3]` |
| E-B2 | **FIX_REQUIRED** (integration) | The wrapper reproduces with `seed = 6102026 + generation + 1`. E's v2 path accepts only the plan seed, because its RNG is already derived per horizon and generation (`i6n-search-plan-v1:<seed>:<h>:gen<g>`). Every v2 reproduction is therefore refused. Fix in B: pass `I6N_SEED` unchanged. Restart determinism is preserved by E's derivation. | B2: `REFUSED: v2 requires the plan seed` |
| E-B3 | **FIX_REQUIRED** | The wrapper builds `Bridge(node, ledger, admit=...)` without `plan`. `evaluate_wire` then makes up its own plan `{folds: {fold_ids}}`. With E's real `score_folds` (plan v1.2.0, quarter-bound fold ids) this raises `KeyError 'quarters'`, so **every candidate would be ScoreRejected once E is merged**. Fix: pass the sealed E plan (`P.e_plan(root)`) to the Bridge, and refuse to run in production without it. | B3: `REFUSED KeyError: 'quarters'`; with the real plan, scoring works |
| E-B4 | **FIX_REQUIRED** | Production ranking does not refuse when the origins check is skipped. If one fold lacks C's origins digest, `score_candidate` silently passes `None` and the result is `origins_binding: NOT_CHECKED`. If E's module is missing, `_reference_score_folds` (aggregator `B_REFERENCE_PENDING_E_MERGE`) ranks receipts of **another genotype**. The orders require ranking only through E's `score_folds` with origins VERIFIED. Fix: in production mode refuse unless the aggregator is `E.score_folds` and `origins_binding == VERIFIED`; keep the reference aggregator for fixtures only. | B4: partial origins gave `NOT_CHECKED`; the fallback gave fitness 0.9375 on receipts of `b*64` |
| E-B5 | **FIX_REQUIRED** | `Ledger.complete` / `_verify_receipt` check only the wrapper keys (fit_id, fold, experiment, effective_config_sha256, genotype). The inner `candidate_receipt` and `shared_receipt` are not cross-checked. A token holder, including through the lease service, can commit another candidate's receipt for another fold and horizon as this fit's terminal. Collect and coordinator later re-score from these terminals. Fix: in the ledger, check `candidate_receipt.{genotype_canonical_sha256, fold, effective_config_sha256, horizon_hours, experiment, seed}` against the unit, and `shared_receipt.support_digest` against the unit's frozen support. | B5: `ACCEPTED` |
| E-B6 | medium | `collect.collect` catches `ValueError` from scoring, which includes E's typed `ReceiptRefused`. It sets fitness to None and still publishes the candidate as `COMPLETE` with no reason (`collect.py`, scoring block). Fix: a typed disposition `SCORE_REFUSED:<code>`, not COMPLETE. | code read |
| E-B7 | medium | `coordinator._candidate_fitness` does not catch a scoring refusal. One refused candidate raises out of `tick()`, which aborts activation for the whole DAG and contradicts the requirement that a localized failure does not stop independent work. Fix: give that candidate a typed record and continue. | code read (`coordinator.py`, `_candidate_fitness` / `tick`) |
| E-B8 | medium | The archive candidate document records `genotype.inactive = {}`, `operator = UNRECORDED_AT_BRIDGE` and `parents = []`. Plan section 6 and orders 2.E require active and inactive genes and parentage. E's `effective_config(g)` and the v2 search state already carry `inactive_genes`, `origin` and `parent_genotype_sha256`. This is resolved together with E-B1. | code read (`collect._candidate_doc`) |
| E-B9 | low | A class that is not admitted raises `WaitExceeded` out of `evaluate_wire` instead of a typed `ScoreRejected` or WAIT_RESOURCE. With E-B1 fixed, E only proposes open classes, but the path should still be typed. | code read (`bridge._argv` / `run_fold`) |
| E-B10 | info | `bridge.size_class` duplicates E's `class_of` and agrees with it. Calling E's function would avoid drift. | code read |
| E-B11 | info | Disconnect during reproduction is covered only by the doin-node `fail_closed` test (results kept, domain stops). No test disconnects a node mid-`reproduce_shared` and resumes from `search_state`. | test list |

## Checks that pass

- **B's fit_id matches E's** `space.fit_id` exactly (B6: true).
- **Receipt binding on the worker path:** `receipt_document` binds each fold
  receipt to the experiment, fold, config sha, genotype, h and seed before the
  ledger. This mitigates E-A1-style mixing on that path; E-B5 is the ledger
  side.
- **Seed and minimization:** enforced in the wrapper's configure step, in the
  role hook (`validate_domain_role`), in the domain metadata and in the node's
  role refusal.
  - Improvement uses an absolute tolerance of 1e-4 (`_shared_improved`).
  - Reproduction is `fail_closed`: no re-run of finished fits.
  - Peer results that do not carry the matching `candidate_identity` are
    ignored (no foreign score publication).
- **Real shared plugin:** the wrapper loads the real shared plugin through
  predictor's resolver, with no default_optimizer fallback and no TensorFlow
  import (tested).
- **Ledger:**
  - Global fencing tokens; a renewal is refused after expiry or after a newer
    token.
  - Late receipts are adopted only while no terminal exists; the first verified
    terminal wins.
  - At most 3 physical attempts.
  - One linked RECOVERY without a checkpoint; RESUME only with a complete
    checkpoint (optimizer, rng, epoch, stopping).
  - FAILED, FAILED_OOM and FAILED_TIMEOUT are terminal, with no retry at a
    lower cap.
  - On boot, the previous boot's attempts on this host are marked INTERRUPTED.
  - The outbox is durable and idempotent.
  - Covered by the disconnect-during-claim, after-fit-before-ACK and ACK-lost
    tests.
- **Runner:**
  - No SIGALRM and no interval timers.
  - The child self-fences before its lease expires.
  - No CPU fallback: a GPU needs an explicit single device; CPU is allowed only
    for declared fixtures.
  - Exit classification: OOM is terminal; admission refusal and pressure stops
    are distinguished.
- **Placement rules:**
  - 5090 preferred and exclusive.
  - 4090 only with a fresh healthcheck from the current boot.
  - 4070 keeps 8 GiB RAM and 2 GiB VRAM free.
  - 5070 Ti only with an aggregate proof from measured peaks, and drains under
    pressure.
  - Caps are 1.25 times the measured peak and never lowered.
- **CLI:**
  - `status` only reads (read-only ledger, no TensorFlow; tested).
  - `preflight` launches nothing.
  - `collect` never trains.
  - `start` is idempotent.
  - `install` writes only this campaign's units.
  - DAG successors activate automatically; WAIT_DEPENDENCY causes are typed.

## Verdict

**FIX_REQUIRED** for E-B1 to E-B5, which block the 12-horizon search under the
plan, plus E-B6 and E-B7. E-B8 follows from E-B1.

The ledger, fencing, attempt policy, runner, placement and CLI contracts are
**ACCEPTED** as reviewed. No lane B file was edited.
