# DR01 return: the admission now reserves, and it holds the reservation for the whole load

Satoshi, successor technical lead, 2026-09-26. Order:
[SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md](../../handoffs/SATOSHI_DAY_REVIEW_CONTINUATION_2026_09_26.md)
DR01, against finding F1 of
[MUSASHI_DAY_REVIEW_2026_09_26.md](MUSASHI_DAY_REVIEW_2026_09_26.md).
Branch `satoshi/dr01-atomic-admission-20260926`, worktree `.worktrees/predictor-dr01-20260926`.

## 1. The answer to the question

**No. Two jobs that each fit alone but not together can no longer both be admitted.** The first
admission now WRITES a reservation before anything is launched, and the second is measured against
it — in the same process, in another process, and across the ssh dispatcher.

Reproduced on a simulated host, 12 GiB available, two requests of 8 GiB
([PRE_POST.txt](../evidence/DR01_20260926/PRE_POST.txt), regenerate with
`pre_post_admission.sh`; no memory is allocated and no process is started by either side):

| | first 8 GiB | second 8 GiB |
|---|---|---|
| PRE (the rule the launcher carried until today) | ADMITTED | **ADMITTED** — 16 GiB against 12 |
| POST (`tools/crispdm_admission.py`) | ADMITTED, reservation written | **QUEUED / HOST_HEADROOM**, "8.00G requested; 1.00G free (MemAvailable 12.00G − 3.00G desktop reserve − 8.00G held by 1 live reservation)" |

And live on the coordinator, against real free memory, one request admitted and a second of the
same size refused while the first was still running
([LIVE_PAIR.txt](../evidence/DR01_20260926/LIVE_PAIR.txt)). During that check a *different* agent's
4 GiB job was itself holding a reservation taken through the same launcher, and it correctly
reduced what my own request was offered — cross-process admission working between two agents, not
only inside one.

## 2. What was done

### 2.1 The incident register (DR01.1)

[OOM_INCIDENT_REGISTER.md](../evidence/DR01_20260926/OOM_INCIDENT_REGISTER.md) and its JSON.
**Two mechanisms, not one; none of the four was exhausted physical RAM.**

| # | Time | Unit | Mechanism | Declared limit | Peak | Cost lost |
|---|---|---|---|---|---|---|
| 1 | 07:59:55 | `crispdm-q2ctx-…-4121087.scope` | kernel **cgroup** OOM (`CONSTRAINT_MEMCG`, its own scope) | 2 GiB, inferred | 2.0 G | 1m53s CPU / 3m07s wall |
| 2 | 08:14:06 | `crispdm-q2ctx-…-4121086.scope` | kernel **cgroup** OOM | 2 GiB, inferred | 2.0 G | 4m02s CPU / 17m19s wall |
| 3 | 13:10:54 | `crispdm-huntdgst-…-429546.scope` | **`systemd-oomd`**, user-slice pressure 55.11% > 50% for > 20s | **NOT RECORDED** | 2.8 G | 20.9s CPU / 10m01s wall |
| 4 | 13:34:46 | `crispdm-q2deep-…-443210.scope` | **`systemd-oomd`**, pressure 55.45% > 50% for > 20s | 9 GiB, recorded | 7.4 G | 21m32s CPU / 23m09s wall |

Total lost: 27m48s CPU over 53m37s wall, plus the Q2 pytest sharding attempt and one pilot cell.

* #1 and #2 each hit **their own** 2 GiB `MemoryMax` while the host had memory. Both scopes also
  carry the same epoch `1790427407` and both `Started` at **07:56:47, in the same second** — the
  double-admission signature, in the journal and not only in a probe.
* #3 and #4 never reached their own limit. #4 was admitted at 13:11:37 with **14.02 GiB**
  `MemAvailable` recorded in its own gate file, and killed 23 minutes later for *session* pressure.
  A gate that reads `MemAvailable` immediately before launching cannot see that, and
  `crispdm-memguard` reads `MemAvailable` too, not PSI.
* Declared limits were largely unrecoverable because no launch path wrote a receipt. #4's is
  recorded (`MEMORY_GATE.jsonl`, `9G`); #1/#2's is *inferred* from `CONSTRAINT_MEMCG` plus the
  reported peak; **#3's is NOT RECORDED and is not recoverable** — writing "4 GiB, the default"
  would be a guess. From now on the ledger records the declared cap.

Nothing was killed, closed or restarted to build this table.

### 2.2 The atomic per-host reservation (DR01.3, DR01.4)

`tools/crispdm_admission.py` (new, tracked) is the one authority. Under a single exclusive
`flock` covering the reading, the decision **and** the write:

* reclaim first, then read fresh capacity: `MemAvailable`, `MemTotal`, the slice's `MemoryMax` and
  `memory.current`, and the user slice's memory **PSI** `some/avg10`;
* subtract the reservations of every live load — `Σ max(0, reserved − observed)`, because
  `MemAvailable` already accounts for what they have taken and only the unrealised part must be
  subtracted again;
* gate on four things: the desktop reserve (3 GiB, unchanged), the **observed aggregate slice
  budget** `memory.current + Σ unrealised + cap ≤ slice MemoryMax`, the live pressure (admission
  stops at PSI 25, well below the 50%-for-20s where oomd acts), and the ceilings;
* then **ADMIT and write the lease**, or **QUEUE** and write nothing, saying which reading refused
  it. A request that no ceiling can ever hold is **REFUSED**, terminally.

The commitment outlives the read because `crispdm-run` **no longer `exec`s**: the wrapper stays
alive, arms the lease with the child pid (pinned to its `/proc` start time, so a recycled pid
cannot impersonate it) and the scope cgroup, heartbeats, samples the **cgroup** peak, and releases
only after the whole tree has finished.

Crash recovery without freeing a live child's memory: the sweep decides by the **witness**, never
by the clock. An expired lease whose witness is alive is *extended* and logged
(`LEASE_EXTENDED_CHILD_ALIVE`); a release asked for under a live child is **refused** and the lease
renewed (`RELEASE_REFUSED_CHILD_ALIVE`); a lease whose witness is dead is dropped whether it
expired or not (the orphan case).

The four specific requirements of DR01.4:

* **Tree, not main process.** The observed footprint is the scope cgroup's `memory.peak` /
  `memory.current`, sampled while the load runs. A pilot footprint offered to *size* an admission
  must declare `peak_scope` of `cgroup` or `tree`; a `main_process_rss` record is REFUSED
  (`PEAK_NOT_A_TREE_PEAK`). The live smoke run recorded a tree peak of 7 282 688 B in the ledger.
* **A pilot bound to its own evidence.** `--peak-bytes` is accepted only with `--peak-evidence`,
  the retained record it was read from; the file is hashed into the lease and the number re-read
  from it. A typed peak, or one that disagrees with the record, is REFUSED
  (`PEAK_NOT_BOUND_TO_EVIDENCE`).
* **One integer.** `cap_bytes` is the single source: `MemoryMax`, `MemoryHigh` (90%) and the
  reservation all derive from it, and the launcher passes `MemoryMax=<bytes>` literally. A caller
  that also passes a size string must pass one that parses to exactly that integer, or the request
  is REFUSED as `CONTRADICTORY_CAP`. No second, discordant integer is written anywhere.
* **An observed aggregate budget, not the child's own limit.** The slice gate is the aggregate
  above. The old rule — one request against the ceiling — is gone from every path.

### 2.3 The launcher's tracked origin, and how the deployed copy is refreshed

`$HOME/.local/bin/crispdm-run` **had no tracked origin anywhere in the fleet**: nothing in any of
the fourteen repositories contained its text, so it could drift with nothing to compare against.
It is now tracked in the repository that documents and depends on it, `predictor`:

| tracked | deployed to | sha256 (deployed = tracked) |
|---|---|---|
| `tools/crispdm-run` | `~/.local/bin/crispdm-run` | see `install_crispdm_launcher.sh --check` |
| `tools/crispdm_admission.py` | `~/.local/libexec/crispdm/crispdm_admission.py` | idem |

`tools/install_crispdm_launcher.sh` is the only sanctioned refresh. It writes each file to a
temporary name **in the destination directory** and renames it over the old one: a rename replaces
the directory entry, so a `crispdm-run` already running keeps executing the text it opened, and no
limit, cgroup or environment of any running child is touched. `--check` prints both digests and
changes nothing. Deployed today at 14:52 local with one agent's job live in the batch slice; that
job was not disturbed. **Future launches only.**

### 2.4 Tests (DR01.4): simulated clock, simulated resources, no real RAM pressured

`tests/test_crispdm_admission.py`, **30 tests, all green**. Readings come from a JSON file
(`CRISPDM_ADMISSION_RESOURCES_JSON`) and the clock from `CRISPDM_ADMISSION_NOW`; nothing allocates
memory and nothing is signalled. `tests/conftest.py` additionally gives *every* test in the
repository its own lease directory over a simulated, generous host, so no test is ever decided by
the coordinator's live memory.

Each case the order names, and where it is:

| Required case | Test |
|---|---|
| two requests that each fit alone but not together | `…two_requests_that_each_fit_alone_are_not_both_admitted` (in-process **and** across processes) |
| an orphaned child | `…an_orphan_lease_whose_witness_is_dead_is_reclaimed_not_leaked`, plus `…a_recycled_pid_does_not_keep_an_orphan_lease_alive` |
| a lease expired while its process is still alive | `…an_expired_lease_over_a_live_child_is_extended_never_freed`, plus `…a_release_asked_for_under_a_live_child_keeps_the_reservation` |
| a parent-imposed limit | `…the_parent_slice_ceiling_bounds_the_observed_aggregate_not_just_the_child`, plus `…a_request_above_a_parent_ceiling_is_refused_terminally_not_queued` |
| high pressure | `…high_memory_pressure_queues_instead_of_racing_oomd` (at the measured 55.11%) |
| an OOM | `…an_out_of_memory_kill_is_terminal_and_frees_the_reservation_once`, plus `…the_launcher_propagates_an_out_of_memory_exit_and_does_not_relaunch` |
| memory released | `…memory_released_by_a_finished_load_is_available_to_the_next`, plus `…the_pair_is_admitted_one_after_the_other_not_at_once` |
| a contradictory cap | `…a_contradictory_cap_is_refused_rather_than_silently_resolved` |
| an interior `--` preserved verbatim | `…an_interior_double_dash_reaches_the_command_verbatim` — end to end through the real launcher, asserting both what the command received and what reached `systemd-run` |

Beyond the minimum: the reservation is held for the whole load and not merely the read (a second
process is queued while the child runs); a refusal starts nothing and never reaches `systemd-run`;
the pilot-evidence and tree-peak rules; deployment by rename; a lease id that stays one filesystem
segment whatever the caller calls the job; and a source-level test that no executable line in the
admission path drops caches, touches swap, disables oomd, raises a ceiling or sends a kill other
than the operator's own TERM/INT forwarded to its own child.

### 2.5 Three defects the live verification found, and what they were

Reported because they are the difference between a reservation that works and one that looks like
it works. All three were found by running the deployed launcher against the live host, all three
are fixed, and each has a test:

* **A lease id is a path.** A caller's job name carries slashes (`df-utility-fab/v0/case/…`), and
  the lease could not be written at all — an admission that writes nothing holds nothing. The id
  is now one sanitised segment; the full name stays in the lease.
* **A teardown zombie is not a live child.** `cgroup.procs` still lists tasks that are already
  zombies while a scope tears down, and a zombie holds no memory. Reading that as "alive" made a
  launcher refuse to release its own reservation the moment its load finished (11 such refusals in
  the live ledger), so it leaked until a later sweep. Each listed task is now checked, and — more
  importantly — **the holder's own reaped child is the whole answer**: when a launcher waited for
  its child, the cgroup is not consulted at all. The cgroup witness is for the detached dispatcher
  units nobody waits for.
* **An ancestor cgroup may not witness a reservation.** Under the `PRLIMIT_AS` fallback,
  `df_isolated_runner`'s child shares its parent's scope, so the lease recorded the *enclosing*
  scope — a cgroup that outlives the task and could never let it be released. A witness cgroup must
  now be the unit's own, checked by name.

After the fix, a nested launch (a `crispdm-run` inside a `crispdm-run`), a plain launch, and
another agent's concurrent 4 GiB job all admitted, armed and released cleanly, each with its
observed cgroup tree peak in the ledger and zero refused releases.

## 3. Which launch paths now share the reservation

**Share it.**

1. `predictor/tools/crispdm-run` → `~/.local/bin/crispdm-run`. The common launcher: acquire, arm,
   hold, sample the tree peak, release.
2. `predictor/tools/crispdm_admission.py` → `~/.local/libexec/crispdm/crispdm_admission.py`. The
   authority; also usable in-process (`Reservation`) and from the shell.
3. `predictor/tools/df_dispatch.py` — the fleet dispatcher's generated remote start script. Its two
   inline reads (`MemAvailable − 3G`, request vs slice `MemoryMax`) are **removed** and replaced by
   `acquire … --detached` plus `arm --cgroup <unit>.service`: the unit outlives the ssh command, so
   its cgroup is the witness and the lease expires only after the wall limit.
4. `predictor/tools/df_isolated_runner.py` — reserves in `Task.start()` before the child exists,
   arms with the child pid and its scope cgroup, releases in `Task.poll()` with the **cgroup** peak
   (not the child's `maxrss`), and records the release in the outcome. This carries
   `tools/df_profile_run.py` and everything else that builds a `Task`.
5. `predictor/tools/df_memory_calibrate.py` — every calibration probe reserves; a refusal is
   recorded as `NOT_STARTED` with the readings, never retried larger.
6. `predictor/tools/df_structural_mutation.py` — already launches every child through
   `crispdm-run` (default prefix `crispdm-run -m 1G -t 5m -n mut-… --`); inherits unchanged.
7. `predictor/tools/df_parity_fixture.py`, `M5PHET/tools/wp18_pipeline.py` — operated under
   `crispdm-run`; inherit unchanged.
8. `tools/df_memory_gated_run.py` (commit `82c633e8`, on the q2deep branch, not on this one) —
   inherits **unchanged**, because it launches only through `$HOME/.local/bin/crispdm-run` and
   never around it. Its own `MemAvailable` poll is now a redundant pre-filter, not an authority;
   when that branch lands it should call `crispdm_admission` directly and keep only the pilot
   binding, which the module now enforces anyway.
9. Everything the operator's `PreToolUse` guard `~/.claude/hooks/require-capped-compute.sh` forces
   through `crispdm-run` — `pytest`, `python`, `tools/df_*` — inherits. (Unchanged by me.)

**Still bypass it — named, with why.**

a. The per-cell fit runners that spawn their children as plain subprocesses inside their **own**
   cgroup: `df_e1_block.py`, `df_e1_huber.py`, `df_fin_runner.py`, `df_d2_unit_worker.py`,
   `df_d2_r4_replay.py`, `df_e1_close.py`, `df_mod_e0_close.py`, `df_sota_repro.py`. These share
   their parent scope's `MemoryMax`, so they must **not** take a second reservation — that would
   count the same bytes twice and deadlock. They are covered when, and only when, the outer
   invocation went through `crispdm-run`; invoked bare they are uncapped and unreserved. Two
   partial controls exist: the operator's hook blocks a bare invocation from a shell tool, and
   `crispdm_admission.py inside-scope` lets any process prove (exit 0/1) that a live reservation
   covers its cgroup. Making that proof **mandatory** inside those eight tools is a behaviour
   change to eight entrypoints and was not done under this order; it is the first follow-on.
b. `agent-multi/tools/eth_curriculum_fleet.py`, `agent-multi/tools/project3_weekly_supervisor.py` —
   call `systemd-run --user` for their own units, in another repository. They bypass.
c. `M5PHET/src/m5phet/web/worker.py` `admit_gpu()` — its own per-request gate with a 4 GiB
   `MemAvailable` floor, on a different host and for a different resource (VRAM). It bypasses the
   host RAM reservation.
d. `~/.local/bin/crispdm-memguard` and `agent-multi/tools/memory_pressure_watchdog.py` — reactive
   guards, not launchers, and neither was disabled or retuned. Both still read `MemAvailable`
   rather than PSI. Admission now stops before memguard's 4 GiB soft stop would act, so memguard
   remains a backstop rather than the mechanism.
e. **Remote roles.** `df_dispatch` now refuses at launch when
   `~/.local/libexec/crispdm/crispdm_admission.py` is absent on the role. The module has **not**
   been deployed to `WORKER_A`/`WORKER_B` by this order — that needs a window on each host — so
   until it is, worker dispatch **refuses instead of launching unreserved**. That is the intended
   failure direction and the one operator-visible consequence of DR01. The refresh is the same
   script, run on the role from its own checkout.
f. **Loads started before the deployment hold no lease.** Their *used* bytes are still counted,
   through `MemAvailable` and the slice's `memory.current`; only their unrealised headroom is
   invisible. Conservative in one direction only, and it resolves as they finish. One such job was
   live at deployment and was not touched.

## 4. Prohibitions, each one observed

No live process was killed, no application of the owner's was closed, and no resource under a
running child was modified. `systemd-oomd` was not disabled or retuned; swap was not enlarged; no
ceiling was raised (the batch slice remains `MemoryHigh` 12 GiB / `MemoryMax` 14 GiB, unread and
unwritten by me); caches were not dropped. There is **no retry after a rejection** anywhere in the
new code: `QUEUED` is waiting before a first start, `REFUSED` is terminal, and an out-of-memory kill
is a terminal outcome that the launcher propagates (exit 137 in test) and never re-asks with a
bigger cap. Validation was done on a simulated host; real memory was never pressured. Deployment
was by rename and affects future launches only. No hostname, address or secret appears in any file
added here. Nothing was staged with `git add -A`.

No heavy fit was dispatched to the coordinator under this order. The only real launches were the
trivial smoke runs and the test suite, each inside a reserved cap of 4 GiB or less.

## 5. Boundaries of this return

* The four terminations are recorded from the journal and one gate file. Incident 3's declared cap
  is **not recoverable**, and incidents 1 and 2's is inference, not a receipt.
* Nothing here attributes the session pressure of incidents 3 and 4 to a single producer.
* PSI 25 as the admission limit is a declared constant chosen to sit below where oomd acted
  (55.11% and 55.45%). It is not a measured optimum, and no experiment here calibrates it.
* The reservation is **per host**. It does not coordinate across hosts, and it says nothing about
  VRAM.
* The eight in-scope fit runners of §3(a) are covered by their parent's reservation, not by one of
  their own. That is correct accounting, not full enforcement: a bare invocation still bypasses.
* This return changes no scientific result and measures nothing. It unblocks dispatch; it does not
  authorise any particular fit.
