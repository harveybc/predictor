# ADM-DEADCACHE-01: reviewed fix, reversible deployment, and the owner's one action

## Status

- **Review.** The owner's five required cases were added at predictor `775c5545` on branch `satoshi/m06-admission-dead-cache-20260930`. It supersedes `0928dc06`.
- **Tests.** 51 of 51 pass in `tests/test_crispdm_admission.py`. They ran on worker_b through `crispdm-run -m 1G`.
- **Deployment.** **Not deployed.** Replacing shared tooling on the workers is outside the agents' current authorization; the orchestrator's earlier remote deploy was denied by its permission classifier. The deployment is therefore the owner's **one action** below.
- **Interim.** Admissible slots keep being used as they are.

## What the review changed, and why

The owner's rule is that a subtracted clean-cache estimate is **not permission to over-allocate RAM**. `0928dc06` uncharged *all* clean file cache in the batch slice. That included the clean cache of **live** scopes. A live scope's cache is already counted in its observed bytes, so uncharging it also shrank that scope's unrealised reservation (cap minus observed). In effect, the live scope's headroom was handed out twice.

`775c5545` uncharges only **dead** clean cache, meaning clean cache outside every live leased scope:

```
charged = memory.current - max(0, clean(slice) - sum clean(live leased scopes))
clean   = file - shmem - file_dirty - file_writeback - unevictable
```

It also keeps everything charged in the uncertain cases:
- If a live scope's `memory.stat` cannot be read, **nothing** is uncharged.
- An unarmed lease (one with no scope yet) holds no pages, so its whole cap stays unrealised.
- The **host gate is unchanged**: MemAvailable minus the 3 GiB desktop reserve minus unrealised reservations. The fix only changes bookkeeping against the slice ceiling. It never promises RAM beyond what the kernel reports available.

## The five owner cases, each a test (all passing)

| Owner case | Test | What it proves |
|---|---|---|
| Live loads | `test_2026_10_01_case_live_loads_their_own_clean_cache_is_never_uncharged` | A live scope's clean cache stays charged. The aggregate = in use + (cap − observed); the next request is queued at the exact ceiling. |
| Shared cache | `test_2026_10_01_case_shared_cache_only_the_part_outside_live_scopes_is_dead` | Slice cache split between a live scope (1 GiB) and a finished one (2 GiB): only the 2 GiB stops counting. |
| (shared cache, unreadable) | `test_2026_10_01_case_unreadable_live_scope_stat_uncharges_nothing` | An unreadable live-scope stat means nothing is uncharged. |
| Partial reclaim | `test_2026_10_01_case_partial_reclaim_is_recorded_and_the_gate_reads_what_remains` | `memory.reclaim` returning EAGAIN is recorded as PARTIAL and not retried; `memory.stat` is re-read, and the gate charges what remains. |
| Reservation | `test_2026_10_01_case_reservations_are_still_held_in_full_beside_dead_cache` | A held 8 GiB reservation stays fully counted beside 3 GiB of dead cache: 8 + 6 = 14 is admitted, 8 + 6 + 1 is queued. |
| memory.max | `test_2026_10_01_case_memory_max_request_above_ceiling_refused_and_host_gate_unchanged` | A request above slice `memory.max` is still REFUSED terminally. With 7 GiB of dead cache, a 4 GiB request is still QUEUED on HOST_HEADROOM (MemAvailable 6 GiB minus 3 GiB reserve). |

The earlier tests for `0928dc06` also still pass: dead cache does not queue; shmem, dirty, writeback and unevictable stay charged; the own-scope-only guard holds; exit status is kept; a signal death is re-raised.

## Bytes

| File | Target | New sha256 (`775c5545`) | Rollback sha256 (deployed `ddadf4a9`) |
|---|---|---|---|
| launcher | `~/.local/bin/crispdm-run` | `056e207a1f36120cb24d8063932988082f50eca368bb03c6e2bb3de9df69d7d7` | `499fdc1877750337006de8aad6b30a943acfea7416aa157b7537c4b121c1dfc7` |
| admission module | `~/.local/libexec/crispdm/crispdm_admission.py` | `7882d20fe30782b5f411be01b6ba58e2de6a26aa2e4cd4ed4876cc69b86948e2` | `8dc2c03b17e498697d634666309876686d051243ea4b640d65ec92c51ab8ee35` |

## The owner's ONE action

From the coordinator, in this directory, run:

```bash
bash DEPLOY_ADM_DEADCACHE.sh <worker_a-alias> <worker_b-alias>
```

`DEPLOY_ADM_DEADCACHE.sh` does the following on each host:
1. Fetches the branch and extracts both files from `775c5545` with `git show` into `~/.local/state`. It does not use `/tmp`, which is tmpfs on those hosts and charged to the job.
2. Checks both sha256 values.
3. Keeps sha-named rollback copies of the deployed bytes under `~/.local/state/crispdm-run/rollback_ddadf4a9/` and checks them.
4. Replaces both files by atomic rename. Running launchers keep the text they opened, and no child is touched.
5. Prints the new sha values, the `state` reading (`slice_memory_current` vs `slice_charged_bytes`), and the ledger line from a 256M `true` smoke job.

It changes no limit, ceiling, cache, swap, oomd or persistence setting, and it restarts nothing.

**Rollback**, one command:

```bash
bash DEPLOY_ADM_DEADCACHE.sh --rollback <worker_a-alias> <worker_b-alias>
```

**After deploying.** A request that is already queued keeps running the old module in memory. It must be re-issued with the same honest `-m`, which does not lower any cap.

**Not part of this action.** Persistence mode, root timers, reboots and any one-time slice-level reclaim are excluded, under the orders at `ac125db9` §5.

## Where the leases live

On every host the store is `~/.local/state/crispdm/admission/`: `leases/`, `ledger.jsonl`, `queue.jsonl`, `requests/`, `retained/` and `incidents/`.
