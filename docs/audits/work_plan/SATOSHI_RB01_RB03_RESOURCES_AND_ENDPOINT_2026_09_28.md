# RB01 / RB03 — the two interrupted selections, the preferred host, and the household endpoint

**Date:** 2026-09-28 (America/Bogota)
**Author:** Satoshi III (Mujuro Utsutsu), successor technical lead
**Branch:** `satoshi/rb01-rb03-resources-and-endpoint-20260928`, cut at `0ea5bff4` (RR02)
**Answers:** RB01 and RB03 of `musashi/reconcile-dispatch-20260928`, continuing RR01 `0ee09998` and RR04 `da22d7f9`
**Creates no authority.** No new training allocation, no reserve, no M4 approval, no auditor signature, no
broker authority. One service restoration is **prepared and rehearsed, not applied**: applying it is the
owner's step.

Hosts appear only by role — `COORDINATOR`, `WORKER_A`, `WORKER_B`. No host name, alias, IP address, token
or account identifier appears in this document or in any evidence file it publishes.

---

## 1. Lead

| question | answer |
|---|---|
| **selection 1** (six files, verbatim) | **exit 139, SIGSEGV.** No summary line: the interpreter died first. **89 passed, 35 skipped, 0 failed** of 228 collected. Whole-cgroup peak **3 540 209 664 B (3.30 GiB)**; CPU 2124.11 s over 851.89 s wall |
| **selection 2** (five files, verbatim) | **exit 139, SIGSEGV, same cause.** **67 passed, 35 skipped, 0 failed**. CPU 2120.22 s over 840.71 s wall; main-process peak 2 197 450 752 B; **whole-cgroup peak lost** — see §3.7 |
| **can the preferred host admit useful work?** | **QUALIFIED NO.** The 5090 itself is healthy and answered three of four probes in ≤11 s. But the host holds only **1.47 GiB** of admissible RAM behind the owner's 3 GiB reserve — 4.6 GB of its 14.3 GiB is unreclaimable kernel slab — with 76 NVRM host-memory allocation failures since 2026-09-24, and one probe held a reservation for its entire 10-minute wall and produced nothing, unexplained. Memory-heavy work went to `WORKER_A` instead; nothing waited for a hardware recovery |
| **what is the household endpoint?** | lake **`public_panels`** at **`http://127.0.0.1:5059`**, registered live in the governance kernel at `:5055`, holdout `2006-12-16`, resource `uci_235_individual_household_power/panel.parquet`, 10 890 295 B, sha256 `b3192c0b…c8db`. Its own service unit is **disabled and inactive**, so the endpoint is **REGISTERED_AND_AUTHORITATIVE_BUT_UNSERVED**. It is **not** the SOTA lake at `:5060`, which is a different registered lake over a different corpus |

---

## 2. What was not redone

RR02 `0ea5bff4` is delivered and installed. Measured today, not assumed: the launcher
`499fdc18…1dfc7` and the admission module `8dc2c03b…8ee35` are byte-identical on all three hosts and
byte-identical to the **tracked** files of that revision. Nothing was reimplemented and nothing was
reinstalled.

The dispatch order's branch `45789070` carries **neither** file and **neither** of the admission test
modules — it predates RR02 — so the selections cannot run there. They ran at `0ea5bff4`, the revision whose
tracked bytes are the installed bytes. `0ea5bff4` is a descendant of `f66ee25d`, the revision the
interrupted attempts themselves ran at, so this is a re-run at a later revision of the same lineage and is
recorded as such, not as a recovery of the historical measurement.

Evidence: `docs/audits/evidence/RB01_RB03_20260928/HOST_CAPACITY.json`.

---

## 3. RB01 — the two interrupted selections

### 3.1 Recovery first: their exits are not recoverable

Searched, exhaustively, before running anything:

| where | result |
|---|---|
| admission store, `retained/` | no body for either lease id |
| admission ledger, 29 + 17 events per lease | `ADMISSION_ADMITTED` → `LEASE_ARMED` → `LEASE_RECLAIMED_WITNESS_DEAD`, with `observed_peak_bytes: null` |
| incidents | the store has no `incidents/` directory from that boot |
| the previous boot's user journal, both scope units | *no entries* |
| pytest caches in the producing worktree | the worktree has no `.pytest_cache` |
| any retained file naming either attempt | none on disk |

**Both keep `INTERRUPTED_RESULT_NOT_RETAINED`.** Neither is `PASSED`. The historical expenditure —
**1997.051 CPU s over 1477.71 s wall** — stands as spent and unrecovered; it is not re-budgeted, not reset,
and the two attempt records in `RR01_RESTART_20260926/INTERRUPTED_ATTEMPTS.json` are untouched.

### 3.2a How to read a whole-cgroup peak in this document

Two properties of this measurement were found the hard way today, and they qualify **every** peak
quoted below. They are stated here once, they are repeated inside each evidence file that carries a
peak (`how_to_read_a_whole_cgroup_peak`), and they are not left in a covering note somewhere else.

1. **Durability — the peak is written back only by a successful release.** If the launcher's
   bounded drain loop does not succeed, it correctly leaves the reservation held rather than handing
   its bytes to a second admission, and the sweep later reclaims it as
   `LEASE_RECLAIMED_WITNESS_DEAD` with `observed_peak_bytes: null`. **So the measurement an incident
   most needs is the one an incident most easily loses**, and a null peak means *not measured* — it
   never means small. Shard 1 and shard 2 failed identically, seconds apart in CPU terms; shard 1
   released and kept 3 540 209 664 B, shard 2 was reclaimed and kept nothing (§3.7).
2. **Resolution — for a child shorter than the sampler's interval the peak is a floor, not a
   measurement.** The same GPU smoke child reported 90 726 400 B when it lived 1.6 s and
   576 438 272 B when given a dwell, against a main-process RSS peak of 967 823 360 B both times
   (§4.3).

**Consequence, applied throughout:** a cap is sized from the **larger** of the whole-cgroup peak and
the main-process peak unless the child lived long enough to be sampled. Every cap declared in this
round was sized that way.

### 3.2 The cap, sized from a whole-cgroup peak and not from a main-process figure

The order's correction is **confirmed, and it matters more than it looked**. The repository's retained
figure for `tests/test_df_sota_repro.py` in its own process is **1.35 / 1.37 / 1.36 GB** across the RP113,
RP121 and RP131 returns — so about 1.37 GB, not 8 GiB. But every one of those three tables labels the
column **`peak RSS`**: it is a main-process figure for **one** file, and it is the wrong basis for a cap.

The right basis was already on the record and is a whole-cgroup number: the two interrupted scopes' own
`observed_scope_memory_peak`, **2.9 G and 2.8 G**, under a 3 GiB declared cap — and those are *partial*
runs, so they are floors.

Declared cap for every shard: **6 GiB**. Measured outcome, which settles it: shard 1's whole-cgroup peak
was **3 540 209 664 B (3.30 GiB)** against a main-process max RSS of 2 168 536 kB (2.17 GB), and shard 2
passed 3.83 GiB while still running. **A 3 GiB cap could not have held either selection to completion**,
and the 1.37 GB figure understates the real footprint by about 2.4×. The cap was raised from measurement
and never lowered.

### 3.3 How they were run

Sequential bounded shards on `WORKER_A`, one at a time, never on the coordinator, never two at once. Each
shard: its own atomic reservation through `crispdm-run`; `OMP/OPENBLAS/MKL=1`; `-t 60m`; the shard script
refuses to start while any reservation is live on the host; boot identity, process identity, argv digest,
whole-cgroup peak, GPU inventory and CPU accounting retained per shard.

Interpreter: the pinned 3.12.13 environment (numpy 2.5.1, pandas 3.0.3, torch 2.13.0+cu130, pytest 9.1.1),
addressed by absolute path. **Named deviation from verbatim:** the recorded commands say `python3`, and
`python3` on that host is 3.14.4, which the suite does not support. The interpreter path is therefore
absolute and the argv digest differs from the historical one by that token. Everything else — the file
list, the order, `-q`, the three thread variables — is verbatim.

A collection smoke first, inside a reservation: **228 tests collected, exit 0**. The environment is sound;
what follows is not an import error.

### 3.4 Selection 1 — exit 139

```
selection: tests/test_crispdm_admission.py tests/test_df_admission_guard.py
           tests/test_df_d2_r4_comparator_guard.py tests/test_df_e1_close.py
           tests/test_df_mod_e0_close.py tests/test_df_sota_repro.py
revision:  0ea5bff48006a78cc245777161197398d43f75dd
exit code: 139 (SIGSEGV, core dumped)
counts:    no summary line — the interpreter died first.  Counted from the retained progress
           stream: 89 passed, 35 skipped, 0 failed, 0 errored, of 228 collected
resources: CPU 1960.27 s user + 163.84 s system = 2124.11 s, over 851.89 s wall
           whole-cgroup peak 3 540 209 664 B against a declared cap of 6 442 450 944 B
           main-process max RSS 2 220 580 864 B
lease:     boot id, pid, pid starttime, argv digest and cgroup all recorded; LEASE_RELEASED
```

**Where it died, from the retained fault handler:** importing `triton` — `triton/knobs.py` line 15, a
native extension — reached from `torch.utils._triton.has_triton_package` inside `torch._dynamo.utils`,
called from `tests/test_df_sota_repro.py` line 95 through `tools/df_sota_repro.py:run_cell`. 201 extension
modules were loaded at that point.

**This is an environment defect on that host, not a repository failure, and it is reproducible in
principle rather than random.** `import triton.knobs` on its own in the same interpreter succeeds, exit 0.
It segfaults only inside a process that has already loaded the other five files' stacks.

**Which retrospectively explains a convention nobody had justified.** Every retained return that reports
this suite says *"`tests/test_df_sota_repro.py` (own process)"*. That was read as a resource choice. It is
not: **run in one process with the other five files, the selection cannot complete on this environment at
all.** Both interrupted selections, as written, combine files that must not share a process — which is why
§3.6 shards per file.

### 3.5 Selection 2 — exit 139, and a refusal that was not what I first thought

The first attempt was **refused before it started**: exit 75, `SLICE_AGGREGATE_BUDGET` — *"the observed
aggregate budget would be 15.04G against the `crispdm-batch.slice` ceiling 14.00G (in use 1.04G,
unrealised reservations 8.00G)"*. Nothing was started; no limit, slice or kernel setting was changed.

**I first read that 8 GiB as my own shard-1 lease inside its release loop. That reading is wrong and I
withdraw it.** The ledger names the holder: **a concurrent lane on the same host**, admitted at
03:13:41Z and 03:14:58Z under the label `rb02-weather-train-only-pilot` at **8 589 934 592 B each** — the
RB02 Weather work of another executor. My own shard-1 lease was already `LEASE_RELEASED`.

So this was not a defect at all. **It is admission doing exactly its job across two independent
executors**: one lane's honest 8 GiB reservation left no room for another 6 GiB, and the second lane was
told so instead of being allowed to over-commit the host. Recorded, with the inference I got wrong
alongside the evidence that corrected it.

I did not adopt, stop, or interfere with that lane, and I did not run a second memory-heavy load beside
it. **The cap was not lowered.** The shard was re-asked at the same 6 GiB with `-q -W 1800` — queuing,
which is waiting before a first start, not a retry — and was admitted at 03:17:04Z.

```
selection: tests/test_crispdm_admission.py tests/test_df_d2_r4_comparator_guard.py
           tests/test_df_e1_close.py tests/test_df_mod_e0_close.py tests/test_df_sota_repro.py
revision:  0ea5bff48006a78cc245777161197398d43f75dd
exit code: 139 (SIGSEGV, core dumped) — the same triton import, the same frame
counts:    67 passed, 35 skipped, 0 failed, 0 errored, counted from the retained progress stream
resources: CPU 1958.51 s user + 161.71 s system = 2120.22 s, over 840.71 s wall
           main-process max RSS 2 197 450 752 B
           whole-cgroup peak NOT RETAINED — see §3.7
lease:     admitted after queuing; boot id, pid, pid starttime, argv digest and cgroup recorded;
           LEASE_RECLAIMED_WITNESS_DEAD
```

**Both selections, verbatim, exit 139.** Neither can be made to pass on this environment in its verbatim
form, because its verbatim form puts `tests/test_df_sota_repro.py` in the same process as the others.

### 3.6 Per-file shards — where the real counts are

The verbatim form cannot produce counts, so the union of both selections was run again as **one shard per
file**, sequentially, same 6 GiB cap, same revision. This is what "bounded sequential shards" was for, and
it is where the numbers are.

| shard | exit | passed | skipped | errors | CPU s | wall s | whole-cgroup peak |
|---|---|---|---|---|---|---|---|
| `tests/test_crispdm_admission.py` | **0** | **30** | — | — | 6.61 | 8.91 | 62 590 976 B |
| `tests/test_df_admission_guard.py` | **0** | **22** | — | — | 7.57 | 9.50 | 111 648 768 B |
| `tests/test_df_d2_r4_comparator_guard.py` | **1** | 0 | — | **17** | 2.30 | 4.21 | 21 458 944 B |
| `tests/test_df_e1_close.py` | **0** | 0 | **35** | — | 1.14 | 3.09 | 20 410 368 B |
| `tests/test_df_mod_e0_close.py` | **0** | **19** | — | — | **2104.80** | **826.67** | **3 140 186 112 B** |
| `tests/test_df_sota_repro.py` | **1** | 0 | — | **105** | 14.81 | 15.88 | 660 680 704 B |
| **total collected** | | | | | | | 30 + 22 + 17 + 35 + 19 + 105 = **228** ✓ |

**The counts reconcile to the test, which localises both deaths precisely.** Per-file the six files hold
228 tests, matching the collection smoke exactly. In company, the 17 comparator-guard tests and the 105
sota tests pass instead of erroring (§3.8), so selection 1 should have shown 30 + 22 + 17 + 19 = 88 passed
and 35 skipped before reaching sota — it showed **89 and 35**. Selection 2, which omits the admission
guard, should have shown 30 + 17 + 19 = 66 and 35 — it showed **67 and 35**.

Both are one higher, by the same one. **So both verbatim runs completed all five other files, passed the
first test of `tests/test_df_sota_repro.py`, and died on its second.** Same file, same test, twice.

**Where the 6 GiB cap was actually needed, and it was not sota.** `test_df_mod_e0_close.py` alone is
**2104.80 CPU s over 826.67 s wall with a 3 140 186 112 B whole-cgroup peak** — that is essentially the
whole cost and the whole footprint of both selections. The repository's retained `peak RSS` figure of about
1.37 GB describes `test_df_sota_repro.py`, whose own shard peaks at 660 680 704 B and costs 14.81 CPU s.
**The file the cap has to cover was never the file the cap was being sized from.**

### 3.7 A measurement gap this round found in the launcher itself

Shard 1 and shard 2 failed the same way, in the same frame, seconds apart in CPU terms. Their leases did
**not** end the same way:

| | reclaim cause | whole-cgroup peak |
|---|---|---|
| shard 1 | `LEASE_RELEASED` | **3 540 209 664 B** |
| shard 2 | `LEASE_RECLAIMED_WITNESS_DEAD` | **`null`** |

The tree peak is written back **only by a successful release** (§3.2a, rule 1). `crispdm-run` waits a bounded ~10 s for the
scope to drain and, if a descendant outlives that, correctly leaves the reservation held rather than
handing its bytes to a second admission — the sweep freed it 14 minutes later. That is the right safety
choice. But the consequence is that **the number an incident most needs is the number an incident is most
likely to lose.** RR02 made lease bodies survive reclamation (RR-B); the measured peak inside them still
does not.

Not claimed: *why* shard 2's scope outlived the loop and shard 1's did not. A 2.2 GB core dump is the
obvious suspect and it is not evidence — `coredumpctl` lists nothing on that host.

Proposed, not done: have the monitor persist the running peak into the lease as it samples, so a reclaim
inherits the last sample instead of `null`.

### 3.8 A repository defect the shards found, which the combined run hides — and it is RR02's own

`tests/test_df_d2_r4_comparator_guard.py` in its own process is **exit 1, 17 errors, 0 passed**.
`tests/test_df_sota_repro.py` in its own process is **exit 1, 105 errors, 0 passed**. Every one of the 122
is the same setup error in the autouse fixture `_crispdm_declared_coverage` of `tests/conftest.py`:

```
tests/conftest.py:81: in _crispdm_declared_coverage
    spec.loader.exec_module(A)
tools/crispdm_admission.py:470: in <module>
    @dataclass
.../dataclasses.py:749: AttributeError: 'NoneType' object has no attribute '__dict__'
```

The fixture loads `tools/crispdm_admission.py` with
`spec_from_file_location("crispdm_admission", …)` → `module_from_spec` → `exec_module`, and **never
assigns `sys.modules["crispdm_admission"]`**. `dataclasses` resolves a `KW_ONLY` check through
`sys.modules.get(cls.__module__).__dict__`, which is therefore `None`. One missing line.

**Why nobody has seen it.** The fixture is autouse, so it runs for every unmarked test — but only the
*first* such test in a process pays. `tests/test_crispdm_admission.py` imports the module itself, so in any
selection that includes it **first** the name is already in `sys.modules` and the fixture works;
`tests/test_df_admission_guard.py` carries `pytest.mark.crispdm_uncovered` and opts out entirely. Both
verbatim selections begin with `test_crispdm_admission.py`, which is exactly why those 122 tests pass there
and error alone.

**And it is a regression introduced by RR02 itself**, which is what makes it worth the paragraph. The
fixture is RR02's own declared-reservation coverage gate. The retained RP121 and RP131 returns record
`tests/test_df_sota_repro.py` **"in its own process"** at **84 passed + 1 skipped** and **104 passed +
1 skipped**. At `0ea5bff4` the same file in its own process is **105 errors**. So the convention every
retained return follows — run that file on its own — is precisely the convention this revision broke, and
it broke silently, because every combined selection that happens to start with the admission suite still
passes.

**Not repaired here, deliberately.** Editing `tests/conftest.py` would change the revision the two ordered
selections were measured at, in the same commit that reports the measurement. The repair is one line and
belongs in its own change, with its own regression test running a covered file **on its own**:

```python
sys.modules["crispdm_admission"] = A      # before spec.loader.exec_module(A)
```

§3.9 establishes the cause by experiment rather than by reading, without touching a file.

### 3.9 Both failures, established by experiment

The conftest mechanism is already **established by arithmetic on the two verbatim runs**, before any extra
run: selection 1 recorded 89 passed, and 89 is only reachable as 30 + 22 + **17** + 19 + 1. The 17
comparator-guard tests that error alone therefore *did* pass in company, and one sota test passed too. The
same decomposition holds for selection 2 at 67 = 30 + **17** + 19 + 1.

Two minimal-reproducer runs were nevertheless issued, because a two-file reproducer is worth more to
whoever repairs this than a six-file one:

* **A —** `test_crispdm_admission.py` + `test_df_d2_r4_comparator_guard.py`: does the admission suite alone
  repair the 17?
* **B —** `test_crispdm_admission.py` + `test_df_sota_repro.py`: do **two** files suffice to reproduce the
  segfault, or does it need the other four?

**Both were first issued at the shards' 6 GiB, and both were refused after waiting the full 900 s** —
`SLICE_AGGREGATE_BUDGET`, *"the observed aggregate budget would be 15.72G against the
`crispdm-batch.slice` ceiling 14.00G (in use 3.02G, unrealised reservations 6.70G)"* — because the
concurrent lane was running one honest 8 589 934 592 B weather cell after another, and 3.5 + 8.59 + 6
does not fit under 14. Exit 75 each time. **Nothing was started, no cap was lowered to get past it, and no
limit, slice or kernel setting was changed.** They waited, and then they were told no.

**A note on the caps they were then re-declared at, so this does not read as cap-shopping.** The 6 GiB was
sized for the *six-file* selection, whose footprint is almost entirely `test_df_mod_e0_close.py` at
3.14 GiB — a file neither of these commands contains. §3.6 now gives each file's own measured whole-cgroup
peak: **62 590 976 B** for the admission suite, **21 458 944 B** for the comparator guard, **660 680 704 B**
for sota. A and B are *different work*, and a cap declared from those measurements is the honest cap for
them. The forbidden move is lowering the cap **of the same work** after a refusal; that was not done — the
6 GiB requests were left to expire first, and the re-declaration is stated here rather than quietly made.

| check | selection | declared cap | exit | result | whole-cgroup peak |
|---|---|---|---|---|---|
| **A** | `test_crispdm_admission.py` + `test_df_d2_r4_comparator_guard.py` | 512 MiB | **0** | **47 passed** in 8.06 s | 62 668 800 B |
| **B** | `test_crispdm_admission.py` + `test_df_sota_repro.py` | 2 GiB | **0** | **135 passed** in 209.04 s | 1 273 561 088 B |

**A settles the conftest defect outright.** 47 = 30 + **17**. The seventeen comparator-guard tests that
*error at setup* in their own process *pass* when one file that imports the admission module runs ahead of
them. Nothing else differs. The cause in §3.8 is established by experiment, not by reading.

**B is the more interesting result, and it corrects what I was about to say.** 135 = 30 + **105**: the whole
of `tests/test_df_sota_repro.py` passes, in 209 s, with the admission suite ahead of it and nothing else.
**There is no segfault in two files.** So:

* the historical figure is *recoverable* at this revision — the file is not broken, it simply needs
  something to register the admission module before conftest's fixture reaches for it;
* and **the segfault needs more than `admission + sota`.** It requires native stacks that only
  `test_df_d2_r4_comparator_guard.py`, `test_df_e1_close.py` and/or `test_df_mod_e0_close.py` bring in. I
  did **not** minimise it further, and I do not claim which of the three is necessary. What is established
  is that three or more files are involved, that the frame is a `triton` native import, and that neither
  of the two files nearest the crash is sufficient to cause it.

**Both caps held with room and both leases released cleanly**, which is also the check that the
re-declaration was honest rather than convenient: A peaked at 62 668 800 B under 512 MiB, B at
1 273 561 088 B under 2 GiB.

---

## 4. RB02 — the preferred host, inspected before any workload

Full record: `docs/audits/evidence/RB01_RB03_20260928/GPU_SMOKE.json`.

### 4.1 What the host's memory actually is

| reading | value |
|---|---|
| MemTotal | 14 981 196 kB |
| MemAvailable | 4.47–4.78 GiB across the session |
| `SUnreclaim` | **4 600 076 kB (4.39 GiB)** |
| largest user process, by RSS | 285 MB |
| `host_free_for_new_bytes` after the 3 GiB owner reserve | **1.25–1.57 GiB** |
| memory pressure, some/full avg10 | 0.00 / 0.00 |

The memory is **in the kernel, not in an application**: nearly a third of the host is unreclaimable slab
while the biggest user process holds 285 MB. Stopping something would not return it. No cache was cleared,
no swap was touched, oomd was not disabled, and no user application was killed.

### 4.2 Driver errors: 76 of them, and none of them a device fault

`journalctl -k` carries **76** occurrences of

> `NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal(pMemDesc) @ mem_desc.c:1359`

first at **Sep 24 16:13:53**, most recently at **Sep 28 15:06:24** — so this is a standing condition across
days, not "earlier that day". `_memdescAllocInternal` is the driver failing to get **host** memory, which
is exactly the slab condition above.

Against that: **zero `NVRM: Xid` errors** and no uncorrected ECC on either device, on any of the three
hosts. The devices are not faulted. A separate benign pattern repeats every one to two minutes
(`Enabling HDA controller` / `kbifInitLtr_GB202: LTR is disabled in the hierarchy`) and is runtime power
management, not an error.

### 4.3 The smoke, inside the reservation, sized from a measured pilot

The declared cap was not guessed. The same alloc/compute/free child was first measured on the coordinator,
which has headroom: whole-cgroup peak **576 438 272 B**, main-process peak **967 823 360 B**. The larger of
the two governs, so **1.25 GiB (1 342 177 280 B)** was declared on the preferred host — 1.39× the measured
need, and inside that host's own current headroom, so no refusal had to be evaded.

One measurement caveat, found and recorded rather than smoothed over: for a child that lives ~1.6 s the
reservation's tree sampler reported 90 726 400 B, far below the same child's 967 MB RSS. **The
whole-cgroup peak is a floor, not a measurement, for a child shorter than the sampler's interval** (§3.2a, rule 2). Every
number quoted here comes from a child given a dwell long enough to be sampled.

| # | probe | declared cap | outcome | elapsed | whole-cgroup peak |
|---|---|---|---|---|---|
| 1 | alloc/compute/free, 512 MiB, first attempt | 1.25 GiB | **wall timeout, exit 124, no output** | 600 s | 724 312 064 B |
| 2 | raw CUDA driver API: `cuInit`, name, UUID, context, 512 MiB alloc, memset, sync, free, destroy | 1.25 GiB | **every return code 0** | 0.40 s | — |
| 3 | torch, one unbuffered marker per stage, 8192×8192 matmul | 1.25 GiB | **completed** | 1.11 s | 92 835 840 B (floor; short child) |
| 4 | the smoke again, unbuffered, 10 s dwell | 1.25 GiB | **computed and freed** | 11.14 s | 686 821 376 B |

**The measured device, read from inside the child**, pinned by UUID through `CUDA_VISIBLE_DEVICES`:

* driver API: `cuDeviceGetName` → `NVIDIA GeForce RTX 5090`, `cuDeviceGetUuid` → `a9f35631d36a6cc6c23beb0b36d50fb8`
* torch: `uuid=a9f35631-d36a-6cc6-c23b-eb0b36d50fb8`, capability `12.0`, 33 668 726 784 B of VRAM
* `nvidia-smi` from inside the child, during the run: the 5090 at 640 MiB used, 38 °C
* allocation and release: 838 860 800 B allocated, 1 310 720 000 B reserved, matmul finite, checksum
  609014.5, residual after `empty_cache` **33 554 432 B** — the cuBLAS workspace

Probe 4 exited **6**, the smoke's own `PARTIAL` code. It fired because the residual was exactly 32 MiB and
the threshold was written `< 32 MiB`. The threshold is one byte too tight; the device freed everything but
its workspace. Recorded, not re-run to produce a nicer code.

**Probe 1 is not explained and is not attributed.** Its whole-cgroup peak of 724 MB is the footprint of a
loaded torch holding a live CUDA context, so it was well inside the torch path when its wall ran out; its
output was buffered and nothing survived. A kernel-cache warm-up would have been a tidy story, and it is
**refuted**: no file under the compute, torch or triton caches on that host was written in that window.

### 4.4 Verdict, and what was done instead of waiting

**The device can work. The host cannot host work.** 1.47 GiB of admissible RAM, against fits that declare
6–11 GiB, with a standing driver-level host-memory failure and one unexplained ten-minute stall. Nothing of
useful size can be admitted there, and lowering a cap to fit is forbidden.

So the RB01 selections ran on `WORKER_A` — 10.5–17.1 GiB admissible against a 14 GiB slice ceiling, a cold
idle 16 GiB device — **while** the preferred host was being measured. No driver was reloaded, no host was
rebooted, and no unrelated work waited on a hardware recovery.

---

## 5. RB03 — the household endpoint resolved

Full records: `docs/audits/evidence/RB01_RB03_20260928/ENDPOINT_RESOLUTION.json`,
`docs/audits/evidence/RB03_ENDPOINT_20260928/{PREFLIGHT,REHEARSAL}.json`.

### 5.1 What the endpoint is

| | |
|---|---|
| registered lake id | **`public_panels`** |
| base URL, as the live registry publishes it | **`http://127.0.0.1:5059`** |
| holdout | `2006-12-16` — whole-resource `AS_IS` only; every date range refused |
| household resource | `uci_235_individual_household_power/panel.parquet` |
| bytes / digest | 10 890 295 / `b3192c0bcb117b2ee120a906dbcfb9550cd907abff74fea9bc2b1aa320ebc8db` |
| principals with `discover · coverage · read · download` | three, each with `deny_from: 2006-12-16` |
| serving unit | `crispdm-data-lake-public-panels.service` — `disabled`, `inactive`, `dead` |
| port probe | connection refused; nothing is bound |
| **state** | **`REGISTERED_AND_AUTHORITATIVE_BUT_UNSERVED`** |

The registration is live and unchanged. Only the process behind it is gone.

### 5.2 It is not the SOTA lake, and the refutation is stronger than stated

The store on the other port is registered as a **different lake id**, `sota_benchmarks`, with its own
holdout, its own policies and its own root. Asking it for the household panel is not a near miss; it is a
different lake.

**One correction to the order, in the order's own favour.** It says that store holds only
`thuml_tsl_electricity`. Measured today, its inventory serves **three** resources — electricity, traffic
and weather. The corpus is larger than recorded and still contains nothing of the household panel. The
substitution path stays **REFUTED**.

### 5.3 The actual retained delivery path

* **Shape:** `<run root>/cache/public_panels/<source_sha256>.parquet` — content-addressed, per run root.
* **12 warm caches** on disk hold the household bytes, every one at 10 890 295 B with digest `b3192c0b…`.
* **119 retained governed deliveries** of the household resource in the accounting store: 19
  `VERIFIED_TRANSFER`, 100 `VERIFIED_CACHE`, all `AS_IS`, all one digest, all one availability contract
  (`93d945cf…e2e2e`), from 2026-09-20T18:23:33Z to 2026-09-21T18:42:05Z.

**A warm cache does not let a consumer skip the endpoint.** `tools/governed_run.py` →
`governed_download` issues a real `GET /api/v2/download` to the kernel on **every** unit and re-streams and
re-verifies the body; `cached` only says the content-addressed target already existed. The kernel's
`http_lake` proxies that read to the lake's own `base_url`. So while the unit is down, **every** household
consumer is refused at its first byte, warm cache or not. The endpoint is a hard prerequisite.

**And a discovery preflight still passes, which is the trap.** `http_lake.discover()` swallows
*"lake unreachable"* and returns an **empty list**; `describe()` and `storage()` degrade to placeholders.
The last event on this lake is a **2026-09-27 `discover` recorded as ALLOW** — after the unit was already
down. `coverage`, `read` and `download` do not swallow. A consumer that preflights on `discover` alone will
believe the lake is healthy and then fail on its first byte. **That is why the endpoint was resolved before
any consumer was scheduled, and no consumer was scheduled.**

### 5.4 The prepared restoration — rehearsed, reversible, the owner's to apply

`tools/df_public_panels_restore.py`, five stages: `preflight · rehearse · apply · verify · rollback`.

**`preflight` — exit 0, every check green** (`RB03_ENDPOINT_20260928/PREFLIGHT.json`): unit file, lake host
configuration and store id; the root present; the household resource registered **and** declared `untimed`;
**both panel digests matching the 2026-09-20 adoption binding**; the serving provider digest matching; the
**installed** provider digest matching; the registry publishing this lake at this port with this holdout;
the port free; and a snapshot of all five other governed store units.

**`rehearse` — exit 0** (`RB03_ENDPOINT_20260928/REHEARSAL.json`). The same interpreter, module and
configuration the deployed unit names, on a throwaway loopback port, from a copy that differs **only** in
`web_port` (asserted: `config_differs_only_in: ["web_port"]`, `backend_settings_identical: true`). Never
registered, never enabled, no consumer can reach it. It proved, today:

* the inventory lists exactly the two registered resources;
* `GET /api/v2/download` of the household panel returns **HTTP 200, 10 890 295 B, sha256 `b3192c0b…c8db`**
  — **byte-exact against the adoption binding**, with `X-Availability-Contract-SHA256: 93d945cf…e2e2e`,
  the same contract digest the 119 historical deliveries carry, so **no data contract moved**;
* a ranged request over the archive is still **refused, HTTP 422** — the holdout contract holds;
* the throwaway stopped and its port freed.

**`apply` is `systemctl --user start` of that one unit and nothing else.** Not `enable`, unless the owner
passes `--also-enable`; by default the change does not survive a reboot and `rollback` is a stop. `apply`
runs `preflight` again first and refuses on any failure. `verify` then requires the household resource
listed, the registry digest unchanged and all five other store units in exactly their prior state.

Both remaining stages were exercised as far as they can be without applying anything, because a stage that
has never run is not a rollback plan:

* **`verify` while the unit is down — exit 1**, `"the endpoint does not answer: [Errno 111] Connection
  refused"` (`VERIFY_WHILE_DOWN.json`). It fails when it should, rather than passing vacuously.
* **`rollback` on the already-inactive unit — exit 0**, and it asserted what matters:
  `port_free: true`, `registry_unchanged: true`, `other_units_unchanged: true`, one step
  (`stop`), `rc 0` (`ROLLBACK_NOOP.json`). The undo path is proven to run and to change nothing else
  **before** there is anything to undo.

**What it never touches:** the governance kernel and its registry; any data contract; the lake host
configuration; the resource contracts, holdout or availability declaration; the panel bytes; any unrelated
service.

> **The adoption receipt's own documented restore line must not be used.** It reads: copy the 2026-09-20
> registry backup over the live registry and restart the kernel. The live registry has since gained the
> SOTA benchmarks lake, additively. That line would today **delete a healthy registered lake** and restart
> the kernel every other consumer depends on. The prepared restoration touches the registry not at all,
> because the registry already carries this lake, unchanged and correct.

### 5.5 RR04 consumed, and the limit of what it governs

RR04 `da22d7f9` proved real worker delivery and warehouse reconciliation. The worker key is **not** a new
external blocker: each worker holds its own. Recorded, and the limit recorded with it: **that proof does
not retroactively govern the twelve historical Q2 fits.**

### 5.6 The September 22 failed decision units — classified, none restarted, none erased

Ten retained `FAILED` terminals carry that date. One is the campaign's own designed negative control, for
which no completion is expected. The nine real decision-unit exits:

| classification | n | retained reason, in the store's own words |
|---|---|---|
| `OPERATOR_INTERRUPT` | 4 | `cell failed: KeyboardInterrupt` |
| `OPERATOR_INTERRUPT_THERMAL` | 2 | operator interrupt at the owner's order — one at 87 °C on the coordinator, one at 83–86 °C with driver thermal slowdown on a worker |
| `HOST_OOM_KILL` | 2 | `systemd-oomd` kill on the preferred host, one at a **7.1 GB** evaluation peak, one stalled after epoch 2 |
| `STALE_CHILD_NO_RESULT` | 1 | stale child of a superseded run, killed with its scope, no result |

**Every one of the nine belongs to a unit that holds a `COMPLETED` terminal** — eight superseded by a later
completion, the last two on 2026-09-23, and the ninth a redundant attempt on a unit that had already
completed earlier the same day. **Nothing is unresolved, so nothing needs restarting**, and none of the ten
records is removed or rewritten.

Two of the nine are the preferred host's own out-of-memory kills at a 7.1 GB peak. That is the same memory
condition §4 measured on that host today — which is why §4's verdict is not new information about that
host so much as the sixth day of it.

### 5.7 The six missing W1440 cells, and the estimand named correctly

**Still missing.** `long_window_own_depth_s1..s3` and `long_window_local_support_67_s1..s3`. No governed
terminal exists for any of them, in any status. The retained reason is kept verbatim: `_s1` of
`long_window_own_depth` was *"held by the gate at cap 10G for 1 395 s of polling, launched never"*, the
others *"never reached"*, under the rule that **an out-of-memory termination was not converted into a retry
with a bigger cap.**

The lineage, so nothing is conflated:

| block | design | state |
|---|---|---|
| Q2_CONTEXT v1 | `6d1aaeca…398b1` | `BUDGET_LIMITED_BEFORE_ANY_OUTCOME`, **0** cells fitted, 5 arms × 3 seeds, ceiling 4 000 updates |
| Q2_CONTEXT_BOUNDED | `47a270ee…17e17e` | **12** fits, 4 arms; the two W1440 full-depth arms restricted **before any score**; its own verdict: does not separate context from depth, does not answer the question |
| Q2_CONTEXT_DEEP v1 | `7d0bf921…8a318` | **WITHDRAWN** for carrying a machine host name |
| Q2_CONTEXT_DEEP v2 | `27ed712d…0300d` | `SEALED_NOT_EXECUTED`; three leaf values differ from v1 and **no science changed** |

The twelve landed fits are **`NOT_BOUND_TO_A_SEAL`** — they carry the withdrawn v1 digest — custody
`UNCHECKED`, **0 verified rows**, closure `FAILED`, disposition `HISTORICAL_DEV_ONLY`. They may not
promote, select or rank anything, and they are **not** re-bound to v2: v2 was sealed after their scores
existed, and re-binding would be the back-dating a seal exists to prevent. They are not refitted to fill a
table, and no allocation is reset.

**The estimand.** The ceiling is **600 observed updates**, and 600 came from the retained cost pilots'
measured rates, declared before any cell of that block had a score. Such a design can answer exactly one
thing: **the paired difference in error between arms at a matched 600-update budget**, on 10 020 identical
evaluation rows, against persistence at the horizon on the same origins. It **cannot** answer converged
accuracy, and it cannot be compared to a published number whose own training ran to convergence. Every
cell is `CENSORED_BY_BUDGET` and the block carries `UNDERTRAINED_AT_CEILING` per arm, from its own sealed
question field. The block's own N10 comparison prices the ceiling at about **+0.015 to +0.017 MAE_z** on a
W60 arm.

So the label is **`MATCHED_BUDGET_DIFFERENCE`**, never `CONVERGED_ACCURACY`. It is not redefined here, and
no rescaling recovers a converged-accuracy claim from these rows.

**One capacity fact, offered without a recommendation.** The two W1440 arms' retained cost basis is
8 458 399 744 B and 10 279 276 544 B peak RSS. `WORKER_A` held 10.5–17.1 GiB admissible today against a
14 GiB slice ceiling, so a 10 GiB and an 11 GiB declared cap would be admissible there **for the first
time**. That is capacity, not permission: executing a `SEALED_NOT_EXECUTED` design is a scientific decision
and it is not mine.

---

## 6. Report, in the §6 shape of `M5PHET/docs/WORK_PLAN_2026_09_24.md`

```
RB01 — the two interrupted selections, re-run in bounded sequential shards
repo/branch: predictor / satoshi/rb01-rb03-resources-and-endpoint-20260928, cut at 0ea5bff4
             (tip = this branch's last commit; it is pushed)
files: tools/df_public_panels_restore.py (new, RB03)
       docs/audits/work_plan/SATOSHI_RB01_RB03_RESOURCES_AND_ENDPOINT_2026_09_28.md
       docs/audits/evidence/RB01_RB03_20260928/{HOST_CAPACITY,GPU_SMOKE,ENDPOINT_RESOLUTION,
                                               HISTORICAL_UNITS_AND_Q2,SHARD_RECEIPTS,
                                               FAILURE_LOCALISATION}.json
       docs/audits/evidence/RB03_ENDPOINT_20260928/{PREFLIGHT,REHEARSAL,VERIFY_WHILE_DOWN,
                                                   ROLLBACK_NOOP}.json
suites (all on WORKER_A, revision 0ea5bff4, 6 GiB cap per shard):
  verbatim selection 1 (6 files): exit 139 SIGSEGV - 89 passed, 35 skipped, 0 failed of 228
  verbatim selection 2 (5 files): exit 139 SIGSEGV - 67 passed, 35 skipped, 0 failed
  per file: test_crispdm_admission 30 passed / test_df_admission_guard 22 passed /
            test_df_d2_r4_comparator_guard 17 ERRORS / test_df_e1_close 35 skipped /
            test_df_mod_e0_close 19 passed / test_df_sota_repro 105 ERRORS    (228 collected)
  reproducers: admission+comparator 47 passed exit 0 / admission+sota 135 passed exit 0
acceptance: recovery of the two retained exits ATTEMPTED AND FAILED -> both stay
            INTERRUPTED_RESULT_NOT_RETAINED; historical 1997.051 CPU s / 1477.71 s wall preserved,
            not re-budgeted; every shard ran on WORKER_A, one at a time, under its own atomic
            reservation, with boot identity, process identity, argv digest, whole-cgroup peak and
            GPU inventory retained; cap raised from a measured whole-cgroup peak (3.30 GiB observed
            against 6 GiB declared) and never lowered to evade the one refusal, which was waited out
what is NOT done / refused / not measured:
  - the historical exits are NOT recovered and are NOT claimed as PASSED
  - both selections in their verbatim form exit 139.  A native triton import segfaults once enough
    of the other files' stacks share the process.  Environment defect, NOT repaired.  It needs at
    least THREE of the six files: admission+sota alone is 135 passed, exit 0.  WHICH third file is
    necessary is NOT established, not minimised further and not guessed
  - tests/conftest.py has a one-line defect (a spec-loaded module never registered in sys.modules)
    that the combined run masks and the per-file shard exposes.  Diagnosed and proven by
    experiment, NOT repaired: repairing it would change the revision being measured
  - shard 2's whole-cgroup peak is LOST.  The tree peak is written back only by a successful
    release, so an incident can lose the number it most needs.  Reported, not fixed.  That rule
    and the sampler's resolution limit are now stated in S3.2a and inside every evidence file
    that carries a peak, beside the numbers they qualify
  - a concurrent lane holds reservations on the same worker.  NOT adopted, NOT interfered with; my
    shards queued behind it at the same cap
  - no measurement of the preferred host's 10-minute stall cause; no cause claimed
  - nothing ran on the coordinator except read-only inspection and one sizing pilot

RB02 — the preferred host's GPU, carefully
repo/branch/tip: as above
files: docs/audits/evidence/RB01_RB03_20260928/GPU_SMOKE.json
acceptance: driver errors and host memory inspected BEFORE any workload (76 NVRM host-memory
            allocation failures since 2026-09-24, zero Xid, zero ECC); small allocation/compute/free
            smoke performed INSIDE a 1.25 GiB reservation sized from a measured pilot; the measured
            device recorded from INSIDE the child by UUID a9f35631-d36a-6cc6-c23b-eb0b36d50fb8, both
            through the raw driver API and through torch; 838 860 800 B allocated, computed, freed to
            a 32 MiB cuBLAS workspace
verdict: QUALIFIED NO.  The device is healthy; the host has 1.47 GiB admissible against 6-11 GiB
         fits, and one probe held a reservation for its whole wall with no output.  Eligible
         memory-heavy work was placed on WORKER_A instead, concurrently, without waiting
what is NOT done / refused / not measured:
  - no driver reload, no host reboot, no cache or swap cleared, no oomd change, no user application
    killed, no global cap raised, no OOM provoked
  - no real workload placed on the preferred host
  - the stall of probe 1 is UNEXPLAINED; the kernel-cache warm-up hypothesis is REFUTED by cache
    write times

RB03 — the household endpoint resolved and a reversible restoration prepared
repo/branch/tip: as above
files: tools/df_public_panels_restore.py
       docs/audits/evidence/RB01_RB03_20260928/{ENDPOINT_RESOLUTION,HISTORICAL_UNITS_AND_Q2}.json
       docs/audits/evidence/RB03_ENDPOINT_20260928/{PREFLIGHT,REHEARSAL}.json
acceptance: the registered household endpoint is lake public_panels at http://127.0.0.1:5059,
            holdout 2006-12-16, resource uci_235_individual_household_power/panel.parquet,
            10 890 295 B, sha256 b3192c0b...c8db -> REGISTERED_AND_AUTHORITATIVE_BUT_UNSERVED.
            Retained delivery path: <run root>/cache/public_panels/<source_sha256>.parquet, 12 warm
            caches, 119 retained deliveries (19 VERIFIED_TRANSFER / 100 VERIFIED_CACHE), one digest,
            one availability contract 93d945cf...e2e2e.  preflight exit 0, all checks green;
            rehearsal exit 0 on a throwaway port from a config differing only in web_port, serving
            the household panel byte-exact with the same contract digest and still refusing a range
            (HTTP 422); throwaway stopped
verdict: still the authoritative endpoint; restoration PREPARED AND REHEARSED, NOT APPLIED
what is NOT done / refused / not measured:
  - NOT applied.  apply/rollback are the owner's steps
  - NOT re-enabled blindly: apply starts one unit and does not enable it unless asked
  - the SOTA lake substitution is REFUTED, not merely declined: different registered lake id, and
    measured today it serves three resources, none of them the household panel
  - the adoption receipt's own restore line is REJECTED as stale: it would delete the SOTA lake
  - no consumer scheduled.  A discover-only preflight against this lake still returns ALLOW with an
    empty list, so no consumer may be scheduled on discovery alone
  - the six missing W1440 cells are STILL MISSING and were not fitted; the twelve historical cells
    were not refitted; no allocation reset; the estimand of a 600-update design is labelled
    MATCHED_BUDGET_DIFFERENCE and not CONVERGED_ACCURACY
  - the September 22 failed decision units are classified (9 decision exits + 1 designed negative
    control) and NOT restarted; nothing was erased
```

---

## 6b. The two authorized follow-ons, both off the measured revision

Both were authorized after this return's measurements existed, and both are separate commits **after**
`0ea5bff4`, so nothing in §3 changes meaning.

### 6b.1 The conftest defect — repaired and pinned

```
repo/branch: predictor / satoshi/rb01-rb03-resources-and-endpoint-20260928
files: tests/conftest.py (one line), tests/test_crispdm_declared_coverage.py (new)
does a covered file now pass on its own?  YES.
  tests/test_df_d2_r4_comparator_guard.py alone: 17 passed, exit 0  (was 17 errors, exit 1)
  tests/test_crispdm_declared_coverage.py:      3 passed, exit 0
counterexample: the SAME new test file against the unrepaired 0ea5bff4 is 3 errors, exit 1 --
  it catches the defect in its own setup, being itself a covered file run on its own
what is NOT done: the measured revision is untouched; the scratch copy used for the
  counterexample was removed and that worktree is clean
```

The repair registers the module before executing it, and pops the name again if the execution
raises so a half-executed module is never left for the next test. The regression test **runs a
covered file on its own, in a subprocess**, and that shape is the whole point: the fault reaches
only the first unmarked test in a process, so a test that exercised the fixture from inside a group
would have passed on the broken code and reproduced exactly the blindness that hid this.

### 6b.2 The governance false green — closed on the wire and in the ledger

```
repo/branch: data-gov / satoshi/lake-unreachable-is-not-empty-20260928, cut at 8a5d2f9
files: lake_plugins/http_lake.py, web_plugins/default_web.py,
       inventory_plugins/default_inventory.py,
       web_plugins/templates/{dashboard,lake}.html,
       tests/unit/test_http_lake_unreachable_is_not_empty.py (new, 10),
       tests/user/test_discover_of_an_unreachable_lake.py (new, 9)
can a dead lake still produce a green preflight?  NO.
  unreachable: HTTP 503, lake named, "inventory is UNKNOWN, not empty", no `resources` key,
               ledger decision `unreachable`
  reachable and empty: HTTP 200, `resources: []`, `reachable: true`, ledger decision `allow`
  refused: HTTP 403, ledger decision `deny`
suites: new 19 passed; full data-gov suite 243 passed, 1 skipped, 1 failed
  (the failure is a pandas str-vs-object dtype comparison in an unrelated FRED parity test,
   confirmed failing identically at the base revision 8a5d2f9 -- not mine)
what is NOT done: nothing deployed, no service restarted, enabled or disabled
```

Both halves were broken. `_get` raised a bare `RuntimeError` on a transport failure and
`discover()` turned it into `[]`; the route wrote `discover` ALLOW **before** asking the lake. Now
`_get` raises `LakeUnreachable` by name, `discover()` swallows nothing at all — a 401, a 403 or a
500 is not an empty inventory either — and the route records the outcome after the lake has
answered.

**Callers checked, and what changed for each:**

| caller | before | after |
|---|---|---|
| `/api/v1/resources` | 200 + `{"resources": []}`, ledger `allow` | **503**, lake named, ledger **`unreachable`** — refuses where it previously passed |
| dashboard (`/`) | "0 resources" for a dead lake | "unreachable — inventory unknown"; excluded from the total; would otherwise have started 500ing |
| store page (`/lakes/<id>`) | empty inventory table | opens with a banner saying the inventory is unknown; its logs and statistics are local and stay accurate |
| `default_inventory.sync_lake` | cached `[]` | **propagates**, deliberately: caching emptiness moves the false green into the catalog, where it outlives the outage |
| `describe()` / `storage()` | degraded silently, looked healthy | still answer, now carry `reachable` and a named reason so zeroes read as an outage |
| `check_startup` | — | **verified not to discover**: an unreachable store still does not block the kernel from starting |
| any consumer outside this repository | — | **none.** The only client of the route is this repo's own `data_gov.client.resources` |

---

## 7. The single next owner decision

**Apply the prepared restoration of the household endpoint, or leave it down.** It is rehearsed, reversible
and contract-neutral; until it is applied, every household consumer is refused at its first byte, so the
Q2 / household lane has no runnable prerequisite left to execute. One command, and one command to undo it:

```
python tools/df_public_panels_restore.py preflight     # read-only, expect exit 0
python tools/df_public_panels_restore.py apply         # starts one unit, nothing else
python tools/df_public_panels_restore.py rollback      # stops it; the only thing it changed
```

Everything else in this return is already done or is deliberately not mine: the W1440 execution decision
belongs to the scientific owner, and the M4 review belongs to the auditor.

— Satoshi III (Mujuro Utsutsu), successor technical lead
