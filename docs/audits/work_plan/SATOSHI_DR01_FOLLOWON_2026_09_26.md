# DR01 follow-on: worker dispatch reserves again, and the fit runners must prove they are covered

Satoshi, successor technical lead, 2026-09-26. Follow-on to
[SATOSHI_DR01_ATOMIC_ADMISSION_2026_09_26.md](SATOSHI_DR01_ATOMIC_ADMISSION_2026_09_26.md),
against the two loose ends that return named: §3(e) (the module was never deployed to the worker
roles, so `df_dispatch` refused at launch instead of launching unreserved) and §3(a) (eight
per-cell fit runners covered only when the invocation happened to go through the launcher).
Branch `satoshi/dr01-followon-deploy-20260926`, off `ee30935a`.
Evidence: [DR01_FOLLOWON_20260926/](../evidence/DR01_FOLLOWON_20260926/).

Roles only. No host name, address, account identifier or token appears in this document or in any
file of the evidence folder.

## 1. The two answers

**Remote dispatch works again, and it reserves.** The launcher and its admission module are
deployed on both worker roles, deployed bytes equal to the tracked bytes, by atomic rename through
the repository's own installer. Three dispatches ran end to end (two on one worker role, one on the
other) and one was refused at launch against three live reservations of the same size and then
launched after they were released — all from the coordinator, all through `tools/df_dispatch.py`.

**Four of the eight fit runners now refuse a bare invocation; four are deliberately deferred
because a live seal pins their bytes.** The four deferred are named in §4 with the gate that pins
each and the exact change to apply when that seal is legitimately superseded. Nothing was re-sealed
to make an edit convenient, and not one byte of the four pinned files changed
(`df_e1_block.py`, `df_e1_huber.py`, `df_fin_runner.py`, `df_d2_unit_worker.py` are identical to
`ee30935a`).

## 2. Loose end 1 — the worker roles

### 2.1 What each role actually has, read before anything was declared

Read with `tools/df_host_capacity.py` (read-only probe; it refuses if an alias would leak) and
confirmed on each role from `/proc/meminfo` and `systemctl --user show crispdm-batch.slice`.
Recorded by role in [CAPACITY_BY_ROLE.json](../evidence/DR01_FOLLOWON_20260926/CAPACITY_BY_ROLE.json).

| role | CPUs | MemTotal | MemAvailable at read | batch slice MemoryHigh / MemoryMax | slice in use | swap |
|---|---|---|---|---|---|---|
| COORDINATOR | 16 | 30.53 GiB | 12.60 GiB | 12 / 14 GiB | 5.73 GiB | — |
| WORKER_A | 32 | 30.58 GiB | 8.83 GiB | 12 / 14 GiB | 0.58 GiB | 8.00 GiB |
| WORKER_B | 32 | **14.29 GiB** | 6.99 GiB | 7 / **8 GiB** | 0.20 GiB | 4.00 GiB |

**WORKER_B has less than half the coordinator's RAM, and its batch ceiling is 8 GiB, not 14.** No
reserve was copied from the coordinator, and none could be: every per-host input the admission uses
— `MemAvailable`, `MemTotal`, the slice's `MemoryMax` and `memory.current`, the user slice's memory
PSI — is read **on the role** at each admission. The one declared constant is the desktop reserve,
3 GiB, which exists in exactly one place (`DESKTOP_RESERVE_BYTES`). Declared per role:

* COORDINATOR — 3 GiB of 30.53 GiB (9.8%). The owner's session runs here; unchanged.
* WORKER_A — 3 GiB of 30.58 GiB (9.8%). Same absolute reserve; its 14 GiB ceiling is read live.
* WORKER_B — 3 GiB of 14.29 GiB (**21.0%**). On this role the reserve is a *larger* fraction, i.e.
  conservative, never permissive; and what actually binds there is the 8 GiB slice ceiling read on
  the role, not the reserve. A number sized from the coordinator's 30 GiB would have been wrong
  here; none was used.

No ceiling, swap setting or oomd configuration was read into a change on any host. I did not touch
any of them.

### 2.2 The deployment

[DEPLOYMENT_BY_ROLE.txt](../evidence/DR01_FOLLOWON_20260926/DEPLOYMENT_BY_ROLE.txt).

Before: both worker roles carried the **pre-DR01** launcher (`ad52f2de…`, identical on both) and
**no** admission module — which is precisely why `df_dispatch` refused at launch — and no lease
directory.

The roles' own checkouts predate DR01 and do not contain the installer, so the three tracked files
(`tools/crispdm-run`, `tools/crispdm_admission.py`, `tools/install_crispdm_launcher.sh`) were staged
into a **new** directory on each role, `~/.local/src/crispdm-dr01/`, and the **tracked installer was
run there**. It writes each file to a temporary name in the destination directory and renames it
over the old one: future launches only. A `crispdm-run` already running keeps executing the text it
opened; no limit, cgroup or environment of any running child was touched.

After, on both roles, `install_crispdm_launcher.sh --check` exits 0 with deployed == tracked:

| deployed | sha256 |
|---|---|
| `~/.local/bin/crispdm-run` | `ae4115dd4851bfb75c0899fb9cecb139bd16bb37ea9a1b9bbaf021055fd38d02` |
| `~/.local/libexec/crispdm/crispdm_admission.py` | `8989afcb8ccb66eca9431d69dd7904fd52905ad6351f257ee52919ebb0bb9f9b` |

The coordinator was checked only, and already matched.

The module then answered on each role (Python 3.14.4 there, 3.12 on the coordinator) with zero live
leases and the readings in §2.1. Nothing on any host was started, stopped, restarted or reloaded;
nothing was killed. Each worker role had another lane's load live in its batch slice during the
deployment; neither was disturbed.

### 2.3 The proof, from the coordinator

[REMOTE_PAIR_PROOF.txt](../evidence/DR01_FOLLOWON_20260926/REMOTE_PAIR_PROOF.txt) and
[DISPATCH_RECEIPTS.json](../evidence/DR01_FOLLOWON_20260926/DISPATCH_RECEIPTS.json). Every load was
`sleep`, which allocates nothing: the reservations are bookkeeping and no real memory was pressured.

**Two remote requests of the same size, WORKER_A, through the role's newly deployed launcher.** The
size came from the role's own reading (60% of `host_free_for_new` 3 166 695 424 B → 1811 MiB), so one
fits and two cannot:

| | verdict |
|---|---|
| first, 1811 MiB | **ADMITTED**, lease `dr01deploy-a1-…-c31e3c` written and armed; while it ran, `live_leases 1`, `held_unrealised 1.766G`, `host_free_for_new 1.556G` |
| second, 1811 MiB, while the first ran | **exit 75** — `QUEUED HOST_HEADROOM — 1.77G requested; 1.56G free (MemAvailable 6.33G − 3.00G desktop reserve − 1.77G held by 1 live reservation(s))`, `REFUSED nothing was started` |

Afterwards `live_leases 0`: the first released its own reservation when its load ended.

**Four requests of the same size on WORKER_B, the fourth through `df_dispatch` itself.**
1 342 177 280 B is exactly the request `df_placement` derives for the proof job, so every request in
this section is the same size.

| | verdict |
|---|---|
| dispatch #1 (`DISPATCH_ALIVE`) | PLACED on WORKER_B, **LAUNCHED → COMPLETED**, exit 0, wall 25.836 s, cgroup **tree** peak 1 835 008 B |
| dispatch #2 (`DISPATCH_QUEUED`) | **LAUNCHED → COMPLETED**, wall 25.776 s, tree peak 2 121 728 B — two reservations were live and the request genuinely still fit; admitted, correctly, and reported as such rather than dressed up as a refusal |
| three launcher requests of the same size, against three live reservations | each **QUEUED HOST_HEADROOM**, e.g. `1.25G requested; 235.12M free (… − 3.74G held by 3 live reservation(s))`, nothing started |
| dispatch #3 (`DISPATCH_QUEUED2`), same size, while those three were live | attempt-1 **REFUSED_AT_LAUNCH, exit 75**, nothing started, the role's own words carried back: `"1.25G requested; 171.34M free (MemAvailable 6.91G − 3.00G desktop reserve − 3.74G held by 3 live reservation(s))", "verdict": "QUEUED"` |
| the same dispatch, after those reservations were released | attempt-2 **LAUNCHED → COMPLETED**, exit 0, wall 29.094 s, tree peak 2 289 664 B |

**And end to end on the other worker role**: job `dr01deploy-wa-2`, request 66 060 288 B,
**LAUNCHED → COMPLETED**, wall 17.321 s, tree peak 2 621 440 B.

Both roles finished at `live_leases 0`. Nothing leaked.

One honest negative: a larger first attempt on WORKER_A (request 1 006 632 960 B) was **never
launched** and correctly waited —
`RAM_REQUEST_1006632960_OVER_HEADROOM_80441344(host_80441344,…)` — because `df_placement`'s own
5 GiB host reserve left 80 MB while another lane's load held that role's memory. That is placement
waiting for capacity, not the DR01 refusal. It ended on its own wall limit without a receipt.

### 2.4 A consequence of the two reserves, worth knowing before the next proof

`df_placement` keeps a **5 GiB** host reserve while admission keeps **3 GiB**. So whenever a role's
`MemAvailable` is below ~7 GiB, *any two* placement-legal jobs also fit admission, and a pair proof
at the dispatch layer cannot be constructed from placement-sized jobs alone — which is why the
same-size pair above was built by holding reservations of exactly the dispatch request size. The two
reserves are not contradictory (placement is the stricter pre-filter), and I changed neither.

## 3. Loose end 2 — the four fit runners that now refuse a bare invocation

New, tracked: **`tools/df_admission_guard.py`**. It does not reserve. It *proves*, through
`crispdm_admission.py inside-scope` and **its exit code** (0 covered, 1 not), that a live
reservation covers the cgroup this process runs in; a lease armed with a holder **pid** and no
cgroup (df_isolated_runner's `PRLIMIT_AS` fallback, where the child shares a scope that must not be
the witness) is honoured too, by checking whether that holder is an ancestor of this process. A
missing or unreadable admission module **refuses** — the same failure direction `df_dispatch` takes,
because an unenforceable cap is not a cap. It reads only: no lease is written, nothing is started,
signalled, or changed.

The refusal names why, not merely that:

```
REFUSED: df_sota_repro will not run execute outside a reserved scope (NOT_COVERED).
  why      this runner spawns its children inside its OWN cgroup, so they share its
           MemoryMax; it must NOT take a second reservation (the same bytes would be
           counted twice and the request would queue against itself).  It is capped and
           reserved only when the invocation itself went through the launcher, so an
           invocation that cannot show a covering reservation is uncapped and invisible
           to admission -- it is refused instead.
  detail   no live reservation covers this process
  cgroup   …/crispdm-batch.slice/…
  leases   live: none
  module   ~/.local/libexec/crispdm/crispdm_admission.py
  start it as:
    $HOME/.local/bin/crispdm-run -m 12G -t 8h -n sotarepro -- …
  nothing was started, and no limit, slice or kernel setting was changed.
```

Where it is now mandatory, and on which paths:

| runner | gated | not gated |
|---|---|---|
| `df_d2_r4_replay` | `--replay` (spawns `df_snr` children) | `--compare` (reads retained files only) |
| `df_e1_close` | the `--replay` worker, and a close that will replay | `--no-replay` |
| `df_mod_e0_close` | a close that will replay | `--no-replay`, `--local-from` |
| `df_sota_repro` | `execute`, `child`, `pilot`, `regenerate`, `profile-eval`, `route-trace`, and `close` unless `--skip-replay` | `seal`, `report`, `lock`, `merge`, `ledger`, `admit`, `accept-*`, `retable`, `revalidate`, `backup`, `delete-predictions`, `retire-attempt` |

Each gate fires **immediately after argument parsing**, before the runner reads a design or creates
a directory, so a refusal costs nothing and starts nothing. No second reservation is taken anywhere:
a covered runner keeps running inside its parent's one.

### Tests — both directions, for the guard and for each of the four

`tests/test_df_admission_guard.py`, **22 tests, all green** (nothing allocates memory, starts a fit
or reads the coordinator's live capacity; the lease store and every reading come from the DR01
sandbox in `tests/conftest.py`):

| case | test |
|---|---|
| bare process is not covered | `…a_bare_process_is_not_covered_and_the_guard_says_so` |
| a reservation over this cgroup covers it | `…a_reservation_over_this_cgroup_covers_this_process` |
| a reservation held by an ancestor pid covers it (the `PRLIMIT_AS` case) | `…a_reservation_held_by_an_ancestor_pid_covers_this_process` |
| somebody else's cgroup does not | `…a_reservation_over_someone_elses_cgroup_does_not_cover_this_process` |
| an absent module refuses rather than admitting | `…an_absent_admission_module_refuses_rather_than_admitting` |
| the refusal names why and how to start it | `…the_refusal_names_why_and_how_to_start_it` |
| the guard reserves nothing of its own | `…the_guard_reserves_nothing_of_its_own` |
| it never kills, raises a ceiling or drops caches (source-level) | `…the_guard_never_kills_raises_a_ceiling_or_drops_caches` |
| **bare invocation refused, per runner** (4) | `…a_bare_fit_runner_is_refused_and_the_refusal_names_why[<tool>]` — asserts exit 75 and that the text names the shared cgroup, the `MemoryMax`, the second reservation and that nothing was started |
| **covered invocation passes, per runner** (4) | `…a_covered_fit_runner_passes_the_guard[<tool>]` — the guard lets it through and it then fails, or not, for its own reasons |
| a path that fits nothing stays ungated, per runner (4) | `…a_path_that_fits_nothing_is_not_gated[<tool>]` |
| the four seal-pinned runners are still pinned and were not edited | `…the_seal_pinned_runners_are_still_pinned_and_were_not_edited` |

That last test is the tripwire for §4: it fails the day one of the four stops being pinned, which
is exactly when the guard can be added there.

## 4. The four I did not touch, and the seal that pins each

I checked every one of the eight for a seal that must stay checkable **before** editing it. Two I
had already edited on the strength of a first, incomplete reading; when the drift gate turned up I
reverted both, and their bytes are identical to `ee30935a`. Reporting that because it is the
difference between a checked claim and a lucky one.

| runner | the gate that pins its bytes | where the pinned digest lives | live sealed roots on this host |
|---|---|---|---|
| `df_e1_block` | `code_drift(design)` → `BlockRefusal("REFUSED: scientific source changed")` for a new fit, **and** `replay_identity()`'s `replay_code_sha256 = sha_file(__file__)`, the whole file, which an accepted cached replay must match | `design["source_code"]["df_e1_block.py"]`; and `replay_code_sha256` in `RP90/REVIEW_REPRODUCED_POST.json` (two entries) | **14** |
| `df_e1_huber` | `validate()` → `ValueError("scientific source changed: df_e1_huber.py")` | `design["source_code"]["df_e1_huber.py"]` | 2 |
| `df_fin_runner` | `validate()` → `FinRefusal("REFUSED: scientific source changed: df_fin_runner.py")` | `design["source_code"]["df_fin_runner.py"]` | 1 |
| `df_d2_unit_worker` | it is a member of `df_d2_design.D2_CODE_FILES`, so its bytes enter `lab_code_sha256()`, which is **compared on resume** (`prior[-1][1]["code_sha256"] == code`) | `RUN_MANIFEST.json`'s `code_sha256_at_creation`, every D2 terminal's `code_sha256`, and `lab_code_sha256s_now.df_d2_unit_worker` in `repro_runs/d2_support_r1/r3/R3_REVIEW_RECEIPT.json` | the D2 roots |

**There is no smallest change that keeps these seals.** The gate recomputes the digest of *the very
file the guard call would have to live in*; the sealed sets are
`{df_fin_runner, df_fin_task, df_e1_block, df_mod_e0, df_e1_governed}`,
`{df_e1_huber, df_e1_phase1, df_e1_pilot, df_mod_e0, df_e1_governed}` and
`{df_e1_block, df_gru_reference, df_e1_calendar, df_e1_pilot, df_mod_e0, df_e1_governed}`, so every
file on those fit paths that could hold the call is itself pinned. Putting the proof in an unpinned
shared module these runners happen to import (`df_utility_run`, `governed_run`) would gate dozens of
read-only paths as a side effect — a bigger change, not a smaller one. So: **deliberately deferred,
with the seal named**, and nothing re-sealed.

The change to apply when a seal is legitimately superseded is three lines, identical in shape to the
four that landed — for example, in `df_e1_huber.main` right after `parse_args`:

```python
if a.cell or not (a.seal or a.close):
    P._module("df_admission_guard").require_reserved_scope(
        "df_e1_huber", "fit", mem="6G", wall="4h", name="e1huber")
```

and in `df_fin_runner.main` for `("execute", "child", "cost-pilot")`, in `df_e1_block` for the fit
and child paths, in `df_d2_unit_worker.run_units`. Each needs its sealed designs re-sealed *by the
order that supersedes them*, never by me to make an edit convenient.

Two corrections to DR01 §3(a) while I am here, both measured:

* **`df_d2_unit_worker` is not the bypass DR01 called it.** Its unit children are started through
  `df_isolated_runner.Task(...).start()`, which **does** take a reservation (DR01 §3.4). Its
  residual exposure is the parent `run_units` process itself when invoked bare, and the `df_snr`
  subprocess it spawns *inside* an already-reserved worker child. Narrower than stated.
* **The sealed digests on this host match no tracked ref.** All 17 sealed roots record digests equal
  to neither `master`, nor `ee30935a`, nor this branch: each campaign sealed the bytes of the
  worktree it ran from. So the drift gate is checkable only inside that worktree, and my edits would
  not in fact have broken *those* roots. I deferred anyway, because the mechanism is live and the
  order is explicit; but the record should say that the seals are worktree-local, not global.

## 5. What still bypasses the reservation, after today

Named in full, including what is outside my scope.

**In this repository, still bypassing.**

1. `df_e1_block`, `df_e1_huber`, `df_fin_runner`, `df_d2_unit_worker` — §4. Covered when the
   invocation went through the launcher; a bare invocation is still uncapped and unreserved. Two
   partial controls remain in force: the operator's `PreToolUse` guard
   `~/.claude/hooks/require-capped-compute.sh` blocks a bare invocation from a shell tool, and
   `df_d2_unit_worker`'s unit children reserve through `df_isolated_runner`.
2. The **ungated** paths of the four runners that were closed (`--no-replay`, `--compare`,
   `--skip-replay`, `report`, `seal`, the accounting commands). They read retained files and start
   no compute; they are deliberately not gated, and a `--no-replay` close that someone extends into
   real compute would slip through. That is a maintenance obligation, not a defect today.
3. Any *new* tool that spawns children in its own cgroup and neither calls the guard nor goes
   through `crispdm-run`. There is no repository-wide control that forces the guard on a new
   entry point; `tests/test_df_admission_guard.py` covers only the four it was added to.

**In other repositories, out of my scope.**

4. `agent-multi/tools/eth_curriculum_fleet.py` and `agent-multi/tools/project3_weekly_supervisor.py`
   — they call `systemd-run --user` for their own units, with no admission call. They bypass the
   host reservation entirely. Changes there belong in that repository.
5. `tools/df_memory_gated_run.py` (commit `82c633e8`, on the q2deep branch, not on this one) — it
   launches only through `crispdm-run`, so it inherits the reservation, but its own `MemAvailable`
   poll is a redundant pre-filter that should become a direct `crispdm_admission` call when that
   branch lands.

**The GPU path on the worker's web service, out of my scope.**

6. `M5PHET/src/m5phet/web/worker.py` `admit_gpu()` — its own per-request gate with a 4 GiB
   `MemAvailable` floor, on a worker role, for a **different resource** (VRAM). It does not consult
   the host RAM reservation and the host RAM reservation says nothing about VRAM; nothing today
   coordinates the two. A GPU job admitted there can still make a worker role's RAM admission
   wrong in the direction that matters (its host-RAM footprint is unreserved), and two hosts'
   reservations do not coordinate at all. Unchanged by me, and the largest remaining hole.

**Reactive guards, not launchers** (unchanged, neither disabled nor retuned): `crispdm-memguard` and
`agent-multi/tools/memory_pressure_watchdog.py`, both still reading `MemAvailable` rather than PSI.
Admission stops before memguard's soft stop would act, so memguard stays a backstop.

**Loads started before today's deployment hold no lease on the worker roles.** Their *used* bytes
are still counted through `MemAvailable` and the slice's `memory.current`; only their unrealised
headroom is invisible. Conservative in one direction only, and it resolves as they finish. Two such
loads were live during the deployment and neither was touched.

## 6. Prohibitions, each one observed

Install only: on both worker roles nothing was started, stopped, restarted or reloaded; no slice
ceiling, swap setting or oomd configuration was changed; nothing was killed. Deployment was by
atomic rename in the destination directory — future launches only — and the loads already running on
each role were not disturbed. The hosts appear nowhere by name: they are read from the operator's
role map outside the checkouts and referred to as COORDINATOR / WORKER_A / WORKER_B, and no host
name, address, account identifier or token is printed, logged or committed anywhere in this return.
Each role's real capacity was read before its reserve was declared, and recorded by role. Every
heavy command of mine ran under `$HOME/.local/bin/crispdm-run -m … -t … -n dr01deploy --`; when the
launcher refused a 6 GiB test run against another lane's live 4.83 GiB reservation I re-asked
**smaller** (3 GiB) and queued, never larger. The proof loads were `sleep`: no real memory was
pressured anywhere. Nothing was staged with `git add -A`.

## 7. Boundaries of this return

* It measures nothing scientific and changes no result. It restores remote dispatch and narrows a
  bypass.
* Four of the eight runners are closed. The other four are deferred, with the seal named, and that
  gap is real until an order supersedes those seals.
* The pair proof used reservations, not memory: it demonstrates the accounting, not any host's
  behaviour under genuine pressure.
* The guard proves *coverage*, not sufficiency: a runner inside a 2 GiB reservation that needs 8 GiB
  is covered and will still be killed by its own cgroup limit. Sizing remains the caller's duty.
* Nothing here coordinates two hosts, and nothing here says anything about VRAM.
