# RR02 — admission finished where it actually failed: after the gate

**Date:** 2026-09-26
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Continues:** DR01 (`ee30935a`) and its follow-on (`f66ee25d`), under RR02 of the post-reboot orders
(`0d6f6f7e`), against the four defects named in
[SATOSHI_RR01_RESTART_MANIFEST_2026_09_26.md](SATOSHI_RR01_RESTART_MANIFEST_2026_09_26.md) and
[RR01_RESTART_20260926/INTERRUPTED_ATTEMPTS.json](../evidence/RR01_RESTART_20260926/INTERRUPTED_ATTEMPTS.json).
**The atomic launcher was not implemented again.** It is deployed and it works at the gate; this
return changes what happens *after* the gate.
Branch `satoshi/rr02-admission-monitor-20260926`, off `f66ee25d`.
Evidence: [RR02_MONITOR_20260926/](../evidence/RR02_MONITOR_20260926/).
`NO_NEW_MEASUREMENT`: nothing scientific is measured here and no result changes.

Roles only. No host name, address, account identifier or token appears in this document or in any
file of the evidence folder.

---

## 1. The answer to the question that matters

**Can an admitted workload still ride rising pressure to an owner-visible kill? No — not the shape
that happened.** The pressure series that was actually recorded while the two admitted scopes ran
was replayed, sample for sample at its recorded instants, through the monitor this return adds. It
stops its own scope at **21:24:37Z**, which is **153 s before** systemd-oomd killed the owner's
browser and **192 s before** it killed the owner's editor.

Nothing in that number was chosen to make it look good: the thresholds come from the host's own oomd
policy and from the retained observations (§2), the samples are the ledger's own readings, and the
replay is in
[RETAINED_PRESSURE_SERIES.json](../evidence/RR02_MONITOR_20260926/RETAINED_PRESSURE_SERIES.json).

Four things that answer does **not** say, each of which matters:

* it does not say the owner's applications would have survived. The two killed applications were
  themselves the largest memory holders on the host, 13.5 G and 10.8 G. The monitor removes **our**
  contribution; it does not control the host.
* it does not say every shape is caught. A load can still be killed by its own cgroup limit, and a
  host can still lose memory faster than a 5 s sample period sees.
* it does not cover a load that holds no lease. Anything outside the reservation is outside this
  mechanism, and §5 names every such path.
* it is a replay, not a repetition of the incident. The order forbids tuning by exhausting the
  owner's desktop, and nothing here was validated by putting a host under pressure.

**What still bypasses the reservation** is answered in full in §5 and in
[FIT_ENTRY_CENSUS.json](../evidence/RR02_MONITOR_20260926/FIT_ENTRY_CENSUS.json). The short version:
one seal-pinned runner in this repository (`df_d2_unit_worker`), one subjob launcher in `agent-multi`,
`admit_gpu()` in the M5PHET worker, any new entry point nobody has written yet, and the fact that a
reservation is per host and says nothing about VRAM.

---

## 2. Defect 1 (RR-C) — a monitor after admission, with hysteresis

### 2.1 What was wrong

Admission was a gate at entry. `crispdm-run` ran a shell loop beside its child that renewed the lease
and read the tree peak, and **never once looked at pressure**. So while the ledger correctly refused
new work at PSI some/avg10 38.76, 60.48 and 61.20 against the 25.00 limit, the two scopes already
admitted ran to the end.

### 2.2 What it is now

`tools/crispdm_admission.py` gains `PressureMonitor` and a `monitor` subcommand; `tools/crispdm-run`
runs it **in place of** the blind sampler, so it is the heartbeat, the tree-peak sampler and the
pressure watch in one place, and that place is testable with simulated `/proc` and a simulated clock.

It samples **both** pressures every 5 s: the user slice's `memory.pressure` (the host — the same
cgroup the host's oomd watches) and **this load's own scope `memory.pressure`** (the cgroup the
reservation covers). Two rules stop a load and one rule decides that the host has recovered:

| rule | what it is | what fires it |
|---|---|---|
| `SUSTAINED_ABOVE_RESPOND` | every sample in the trailing 20 s (at least four) above 37.50 | an unambiguous crossing |
| `ELEVATED_BUDGET_EXHAUSTED` | 240 s of cumulative time above 25.00 since elevation began, with no **confirmed** recovery | the oscillating shape the host actually took |
| recovery (hysteresis) | 120 s, at least 24 samples, **not one** above 25.00 | clears both accumulators |

The oscillation rule is not decoration. With 30 s between the retained samples no 20 s window can
hold four of them, so a rule that demanded every sample be high **would have watched the kills
happen**. Both rules exist because the retained evidence has both shapes in it.

### 2.3 Every threshold, and where it comes from

Derived, never tuned. Recorded with its derivation in
[RETAINED_PRESSURE_SERIES.json](../evidence/RR02_MONITOR_20260926/RETAINED_PRESSURE_SERIES.json) and
printed by `crispdm_admission.py policy`.

| constant | value | derivation |
|---|---|---|
| host oomd limit / duration | 50.00 / 20 s | **the host's own policy**, read from the journal lines retained in `OOM_INCIDENT_REGISTER.json` ("being 55.11% > 50.00% for > 20s with reclaim activity"). Recorded so the monitor acts before it; never changed. |
| `PRESSURE_ADMIT_MAX` | 25.00 | DR01, unchanged: half the oomd limit |
| `PRESSURE_RESPOND_AT` | **37.50** | the midpoint of the only two thresholds policy already declares. Strictly above 25.00, so a load legitimately admitted at PSI 24.5 does not stop itself the instant it starts; strictly below 50.00, so the response precedes the kill. |
| `PRESSURE_RESPOND_WINDOW_SECONDS` | 20 | the duration the host's own oomd policy declares |
| `PRESSURE_ELEVATED_BUDGET_SECONDS` | **240** | half of the **529 s** the retained series shows between the first sample above 37.50 (47.85 at 21:18:21Z) and the first owner-application kill (21:27:10Z) |
| `PRESSURE_RECOVERY_WINDOW_SECONDS` | **120** | twice the **60 s** longest run of consecutive samples at or below 25.00 observed inside that window (24.20, 18.34, 4.90), which was followed **90 s later** by 47.85. 60 s of calm was *observed* to be false recovery. |
| `PRESSURE_SAMPLE_SECONDS` | 5 | the launcher's existing heartbeat period, unchanged |
| `PRESSURE_STOP_TERM_GRACE_SECONDS` | 30 | the launcher's existing `timeout --kill-after` grace |

**The crossing the order names fails by a factor of four.** 52.18 → 24.5 is one sample, 30 s; the
sample after it was 7.04 and the one after that 54.03. It confirms nothing, it gives back none of the
accumulated elevation, and the next sample above the limit cancels the candidate outright. Proved
three ways: as a unit case, in the replay of the retained series, and at the real wrapper boundary in
[POST_ADMISSION_SMOKE.txt](../evidence/RR02_MONITOR_20260926/POST_ADMISSION_SMOKE.txt) case 3.

Two honest limits on the derivation, both recorded in the evidence file: the retained series was
sampled by the queue poll at 15 s and 30 s, so it bounds nothing finer than 15 s (the 5 s period is
inherited, not derived); and it is one host, one window, two scopes — an observed path, not a
distribution.

### 2.4 What a response may do, and what it may never do

A response stops **only its own identified experiment scope**:

1. SIGTERM the scope leader the launcher itself started — the pid recorded in the lease — so the
   child's own handlers run and its partial evidence is written;
2. after the 30 s grace, SIGTERM anything still in **this unit's own cgroup** — that is how a
   detached grandchild is reached;
3. SIGKILL the same, still only this unit's own cgroup.

A pid that is neither the recorded leader nor listed in **this lease's own** `cgroup.procs` is never
signalled. pid 1 and the monitor itself are excluded unconditionally. An **inherited ancestor**
cgroup is refused outright, with the reason. Nothing else is touched: no other lease, no unrelated
user application, no service, no limit, no ceiling, no cache, no swap, no oomd setting. A scope that
**holds no memory** is not stopped at all — stopping it would cost a run and free nothing.

### 2.5 The bounded integration smoke, and the bug it found

[POST_ADMISSION_SMOKE.txt](../evidence/RR02_MONITOR_20260926/POST_ADMISSION_SMOKE.txt), from the
tracked script beside it. Real launcher, real transient scope, real cgroup, real child; **simulated
readings**, and the child is `sleep`, so nothing allocates and no host is put under pressure. A
private lease store, so the fleet's own store and ledger are untouched.

| case | result |
|---|---|
| calm host | the child ends on its own terms, exit 0, nothing stopped, its body retained |
| admitted calm, then pressure rises to 60.00 — the day's own sequence | `PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND` after 4 samples in 20 s; SIGTERM to **its own scope leader only**; `term_cgroup` and `kill_cgroup` empty because the scope had already drained; `refused` empty; live leases 0 afterwards |
| the same, with one dip to **24.5** in the middle | **still stopped**, `recoveries_confirmed: 0` — three samples at or below 25.00 during the run confirmed nothing |

The exit status is **124**, not 143: the launcher has always wrapped its command in
`timeout --kill-after=30s`, and that is the status this coreutils reports when its child is
terminated. The launcher now prints the cause beside it, so the code is not read as a scientific
failure.

**The first real run found a bug in my own sustained rule, and I am recording it rather than only the
fixed version.** The first version required the samples *inside* the trailing window to span it. The
sample period is exactly a quarter of the window, so a few milliseconds of sampling drift per tick
left the window permanently one sample short, and on a host held at PSI 60.00 the sustained rule
**never fired at all** — only the 240 s budget stopped the load, 250 s in. Coverage of the window is a
property of the series, not of the samples inside it. Fixed, and pinned by a regression test that
feeds a deliberately drifting series
(`…a_drifting_sample_period_does_not_disable_the_sustained_rule`). The unit tests fed exact 5 s steps
and passed both before and after: only the real boundary showed it.

The exit cause is made durable **before anything is signalled**, in the admission store's
`incidents/<lease_id>.json`, carrying the rule, the whole sample series with both pressures, the cap,
the unit, the tree peak so far and `partial_evidence_retained: true`. RR01's lesson was exactly this:
a record that exists only in a process's pipes does not survive the event it describes. The launcher
then says on its way out that the job was stopped under pressure, so the exit code is not misread as
a scientific failure.

---

## 3. Defect 2 (RR-A) — boot identity plus process start identity

A lease now carries `boot_id` (the kernel's random boot id), `boot_time` (`/proc/stat` btime, which
is what makes a start time in ticks an absolute instant at all) and `host_key` — an **opaque** key,
`sha256(machine-id)` truncated, never a host name or an address. Liveness now asks the two identity
questions **before** it asks about a witness:

* **another host** → the lease is **never reclaimed and never judged**. A coordinator reboot must not
  touch a worker's lease. It is also **not counted against this host's capacity**, because the bytes
  it reserves are not this host's bytes; it is reported in `state` and logged once as
  `LEASE_FOREIGN_HOST_NOT_JUDGED`.
* **an earlier boot of this host** → the local load did not survive the reboot, and `pid_starttime`
  is ticks since boot and is not comparable across one. The lease is **retired as a historical
  record** (`LEASE_RETIRED_OLD_BOOT`, body kept, §4) and its capacity is released. A post-reboot
  process that matches the dead lease's pid **and** its start-time number exactly can no longer make
  it read as live — the latent defect RR01 named, now closed and tested.
* **same boot, same host** → judged exactly as before. Nothing about pid recycling was loosened.

All of it proved on simulated `/proc` and simulated clocks first, as the order requires; no real
reboot was needed and none was performed.

---

## 4. Defect 3 (RR-B) — a reclaim keeps the body

`Store.retire()` writes the whole lease body to `retained/<lease_id>.json` with the cause, the
retirement time and the observed peak, and only then removes it from the live set. Every ending goes
through it: `LEASE_RECLAIMED_WITNESS_DEAD`, `LEASE_RECLAIMED_UNARMED_GRACE_EXPIRED`,
`LEASE_RETIRED_OLD_BOOT`, `LEASE_RELEASED`, `LEASE_RELEASED_NEVER_ARMED`. A release destroyed the
body just as a reclaim did, so both were fixed; the causes are distinct words, because a witness that
died is not a lease that was never bound to one.

The test asserts the presence of **every field RR01 had to transcribe by hand** — cap, cgroup, unit,
argv digest, wall, expiry, created/armed instants, pid and start time, host reserve, name and label —
and `crispdm_admission.py retained [LEASE_ID]` reads them back. The next incident review starts from
the store, not from a transcription made minutes before the record was deleted.

---

## 5. Defect 4 (RR-D) — a refusal answered by asking for less

The store keeps a **request register** per name. A refusal (QUEUED or REFUSED) is remembered for
3600 s — the launcher's own default bounded wait, `-W 3600`, after which the refusal is stale — and a
later request under the same name for **less** than the refused cap is **REFUSED terminally**, code
`CAP_LOWERED_AFTER_REFUSAL`, with the earlier cap, the earlier code, how long ago it was, whether the
command is byte-identical to the refused one, and the rule:

> A refusal is answered by waiting for capacity or by moving the work to a host that admits it, never
> by asking for less than the work needs. If this is genuinely different, smaller work, give it its
> own `-n` name.

Three deliberate properties: an **ADMITTED** request clears the name, because capacity was found at
the size the work needs and nothing was dodged; a re-ask at the **same or a larger** cap still queues
normally, so waiting is unaffected; and a `CAP_LOWERED` refusal never becomes the *remembered*
refusal, or the remembered cap would drift downwards with each lowered re-ask and the rule would
erode itself.

**I am the caller this rule catches.** The DR01 follow-on's §6 says, in my own words, "when the
launcher refused a 6 GiB test run against another lane's live 4.83 GiB reservation I re-asked
**smaller** (3 GiB) and queued, never larger", and the retained queue log alternates 3 GiB and 1 GiB
under one name. I reported that at the time as compliance with "never retry with a bigger cap". It
was the other prohibition, and the manifest is right to call it a defect. It is now impossible under
one name and recorded when attempted.

---

## 6. The failure mode that would undo everything: the wrapper boundary

Exercised against the real launcher, not only the store.

| case | before | now |
|---|---|---|
| the launcher **SIGKILLed** while its child runs | the lease survived (DR01) | unchanged, and now asserted at the boundary: a second request for the same bytes is still QUEUED, `held_unrealised_bytes` intact |
| a **detached descendant** still in this load's own scope after the direct child exits | the lease was **released**: the next admission got bytes a live descendant was using | **closed.** A dead pid is not the end while this load's own scope still holds a live task |
| the **arm fails** | `|| true`: the lease stayed unarmed, and 120 s later the sweep reclaimed it **while the child ran** | **closed.** Arming is mandatory (3 attempts); a job that cannot be accounted for is stopped — its own scope only — and the reservation is given back after the child pid is **checked** dead |
| the **renew** fails | covered by DR01 | re-asserted: an expired lease over a live witness is extended, never freed |
| **publication interrupted** between the child's exit and the release | covered by DR01 | re-asserted, and the body is now retained when the sweep finally frees it |

**One DR01 decision is deliberately reversed, and I am reporting it rather than quietly changing a
test.** `ee30935a` made a reaped child the whole answer, because a scope tearing down still listed
tasks winding down and the launcher was observed refusing to release its own reservation until a
later sweep. That symptom now has its own fix — `cgroup_alive` checks each listed task, and a zombie
holds no memory — and reading the pid alone left exactly the case this order names. So
`tests/test_crispdm_admission.py::…a_supervised_child_that_has_been_reaped_frees_its_reservation_at_once`
now asserts the **empty** scope, and the live descendant is asserted in the RR02 battery. The leak
DR01 fixed does not come back: the launcher waits a bounded 10 s for its own scope to drain, so the
ordinary case still releases on the first attempt.

---

## 7. Coverage: the four sealed runners, closed by versioned integration

The follow-on deferred `df_e1_block`, `df_e1_huber`, `df_fin_runner` and `df_d2_unit_worker` because
each is pinned by a gate that recomputes the digest of *the very file the guard call would have to
live in* — for `df_e1_block`, `replay_identity().replay_code_sha256 = sha_file(__file__)`, which
every accepted cached replay must match. Editing them would refuse designs already sealed and
invalidate retained replays. That judgment was right and it is not revisited.

**The integration point is `df_benchmark_contract.bind()`.** It is the last thing that happens before
a runner consumes the prepared data, on the fit path of `df_e1_block`, `df_e1_huber`, `df_fin_runner`
and `df_e1_phase1`; and `df_benchmark_contract.py` is pinned by **no** design's `source_code` set and
is **not** a member of `df_d2_design.D2_CODE_FILES`. So the proof is required there, carrying a
version (`BIND_ADMISSION_GUARD_VERSION`), and:

* **not one byte** of the four pinned runners changed — asserted against `f66ee25d` in the test, not
  claimed in prose;
* every old seal still verifies, and nothing was re-sealed, superseded or invalidated;
* the proof is a **read**, through `df_admission_guard` → `crispdm_admission inside-scope`. No second
  reservation is taken: these runners spawn children in their own cgroup under their own `MemoryMax`,
  and a second reservation would count the same bytes twice and queue the work against itself.

`df_e1_phase1` was not on the follow-on's list of eight and is closed as a consequence, not as a
claim of extra scope.

**`df_d2_unit_worker` is NOT closed, and this is the reason rather than a silence.** Its fit path
touches `df_d2_design`, `df_lab_evaluation`, `df_isolated_runner`, `df_snr`, `df_operators`,
`df_snapshot` and `df_contract`, and **every one of those is inside `D2_CODE_FILES`**, whose digest is
compared on resume against `RUN_MANIFEST.json` and every D2 terminal. There is no unpinned module on
that path, and it does not use `df_benchmark_contract` at all. Closing it needs the D2 seal superseded
by the order that supersedes it — never by me to make an edit convenient. In force meanwhile: its unit
children reserve through `df_isolated_runner`, and the operator's `PreToolUse` hook blocks a bare
invocation from a shell tool. A test pins the fact that the path is still fully sealed, so the day it
stops being sealed the gap can be closed and the test says so.

**`agent-multi`**, branch `satoshi/rr02-admission-coverage-20260926`: new `tools/host_admission.py`,
a **client** of the deployed authority and never a second authority — it holds no threshold, reads no
memory and decides nothing (asserted in its test). `eth_curriculum_fleet.start()` now reserves on the
target role **before** `systemd-run`, arms the lease to the unit's **own cgroup** after the launch,
releases it if the launch failed, reports an unarmable reservation as an error instead of ignoring it,
and **refuses outright without a declared measured cap** — there is no default, because a cap chosen
to pass a gate is not a cap.

Not closed there, deliberately: the three `systemd-run` calls in `project3_weekly_supervisor` start
**services**, and gating a service start behind memory admission would create a new way for a
production service to fail to come up. The compute those services run is launched further down, in
`project3_weekly_worker`'s subjob `Popen`, which is the right place for the gate and is **not** closed
by this return; the exact change is named in the census.

**M5PHET** `web/worker.py admit_gpu()`: read and reported, not changed. It keeps the VRAM, temperature
and exclusivity checks and then reads `/proc/meminfo` with a 4 GiB `MemAvailable` floor and launches,
holding nothing between the reading and the launch — the DR01 read-then-run defect, for host RAM, once
per inference request, on a worker role. It does not consult the host RAM reservation, and the host
RAM reservation says nothing about VRAM. Its chat and model runner are live services that the reboot
restarted; a launch-path change there belongs in that repository under its own order. The exact change
is named in the census.

---

## 8. Our own footprint, and every prohibition

**One admitted memory-heavy verification at a time, on this host, throughout.** Every command of mine
ran under `crispdm-run` with a declared cap (2–3 GiB) and a wall limit, **sequentially** — never two
at once, which is precisely the increment the host did not have at 16:27. No parallel suite, no agent
fan-out, no Hermes job. The heaviest thing I ran was pytest over the four admission batteries: **98
tests, all passing** (30 DR01 + 37 RR02 monitor + 9 RR02 fit coverage + 22 guard), and its own
scope's peak was 62 MB. `agent-multi`'s 6 tests ran the same way.

* **No new heavy compute and no combined ML suite on the coordinator.** The four sealed fit batteries
  were **not** run here; they are RR01's item 2, on an admitted worker, and they are not repeated to
  test software.
* **No host OOM was provoked.** The integration smoke uses a real launcher, a real transient scope and
  a real child, with **simulated readings** and `sleep` as the load: nothing allocates.
* **oomd was never disabled**, no cap was ever enlarged, no cache or swap was cleared, the browser and
  the IDE were never signalled, no production service was stopped, and no request was ever lowered
  below its measured need — the mechanism that would have made that possible is now the one that
  refuses it.
* A source-level test asserts the absolute prohibitions over both the module and the launcher, so a
  future edit that introduces `drop_caches`, `swapoff`, a ceiling change, a service stop or a kill of
  pid 1 trips a test.
* Nothing was staged with `git add -A`. The owner's untracked files were not touched.

---

## 9. What this return does not do

* It measures nothing scientific and changes no result. `NO_NEW_MEASUREMENT`.
* It does not prove the owner's applications would have survived (§1).
* It proves **coverage**, not sufficiency: a runner inside a 2 GiB reservation that needs 8 GiB is
  covered and will still be killed by its own cgroup limit. Sizing remains the caller's duty.
* Inside the test suite the fit guard is **satisfied** by a declared sandbox reservation, in one
  labelled place (`tests/conftest.py`), so the suite does not prove the refusal direction there. That
  direction is proved directly, both ways, in `tests/test_df_admission_guard.py` (which opts out of
  the declaration) and `tests/test_df_rr02_fit_coverage.py`.
* It coordinates nothing between two hosts and says nothing about VRAM.
* It does not re-run the two interrupted pytest selections. They stay
  `INTERRUPTED_RESULT_NOT_RETAINED` and belong on an admitted worker, in bounded sequential shards —
  RR01's next action, not this one's.

## 10. Next action, concrete and owned

1. Re-run the two interrupted selections on an admitted worker, in bounded sequential shards,
   retaining exit code, revision, selection and resources.
2. Deploy this branch's launcher and module to both worker roles with the tracked installer, by atomic
   rename, future launches only — after this return is read, and without disturbing a running load.
3. When an order supersedes the D2 seal, add the proof to `df_d2_unit_worker`; until then the gap
   stands as written.
4. Close `project3_weekly_worker`'s subjob launch and M5PHET's `admit_gpu()` under their own orders,
   with the changes named in the census.

— Satoshi III (Mujuro Utsutsu), successor technical lead
