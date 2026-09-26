# DR01.1 — incident register: the four terminations on the coordinator, 2026-09-26

Satoshi, successor technical lead, 2026-09-26. Source: the coordinator's journal (system and
user), read only, plus the one launch receipt that exists
(`~/.local/state/crispdm-data-foundation/e1_block_q2_context_deep_v1/MEMORY_GATE.jsonl`).
No process was signalled, started or stopped to build this table. Times are local (UTC−05:00).
The host is named `<coordinator>` throughout.

**Two mechanisms, not one.** Two of the four are kernel **cgroup** out-of-memory kills: a scope
reached *its own* `MemoryMax` while the host still had memory. Two are **`systemd-oomd` memory
pressure** kills: the scope was well under its own limit and the host was not out of memory
either — `user@1000.service` was under sustained PSI pressure above 50% for more than 20 seconds
with reclaim activity, and oomd chose the batch scope as its victim. Attributing all four to
exhausted physical RAM would be wrong, and the two mechanisms need different corrections.

## The four

| # | Time | Unit | Attempt | Kill mechanism | Declared limit | Observed peak | Cost lost | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | 07:59:55 | `crispdm-q2ctx-1790427407-4121087.scope` | shard **2** of a two-way pytest split, attempt 1 | **kernel cgroup OOM** — `oom-kill:constraint=CONSTRAINT_MEMCG`, `oom_memcg` = this scope | **2 GiB**, inferred (see *Declared limits*) | **2.0 G** scope peak | 1 min 53.003 s CPU over 3 min 7.363 s wall | `Failed with result 'oom-kill'`; pid 4121109 killed, total-vm 8450096 kB, anon-rss 954844 kB, file-rss 412684 kB |
| 2 | 08:14:06 | `crispdm-q2ctx-1790427407-4121086.scope` | shard **1** of the same split, attempt 1 | **kernel cgroup OOM** — same constraint, its own scope | **2 GiB**, inferred | **2.0 G** scope peak | 4 min 2.480 s CPU over 17 min 19.320 s wall | `Failed with result 'oom-kill'`; pid 4121118 killed, total-vm 17737364 kB, anon-rss 2028920 kB, file-rss 707340 kB |
| 3 | 13:10:54 | `crispdm-huntdgst-1790445652-429546.scope` | single attempt (a git blob digest hunt) | **`systemd-oomd`** — `user@1000.service` pressure **55.11% > 50.00% for > 20s with reclaim activity`; the scope's own limit was never reached | **NOT RECORDED** by any receipt; ≥ 2.8 GiB from the observed peak | **2.8 G** (`Current Memory Usage` at the kill, and the scope's `memory peak`) | 20.874 s CPU over 10 min 1.460 s wall | oomd `killed 3 process(es) in this unit`; `Failed with result 'oom-kill'`. Its PSI at the kill: Avg10 74.37, Avg60 61.66, Avg300 44.38 |
| 4 | 13:34:46 | `crispdm-q2deep-1790446297-443210.scope` | `pilot_long_window_own_depth`, attempt 1 | **`systemd-oomd`** — pressure **55.45% > 50.00% for > 20s with reclaim activity**; 7.4 G is far below its own 9 GiB cap | **9 GiB** = 9 663 676 416 B, **recorded** (`MEMORY_GATE.jsonl`, `cap: "9G"`) | **7.4 G** (`Current Memory Usage` at the kill, and the scope's `memory peak`) | 21 min 32.106 s CPU over 23 min 9.019 s wall | oomd `killed 3 process(es) in this unit`; the gate recorded `exit_code: -9`. Its PSI at the kill: Avg10 34.59, Avg60 29.48, Avg300 23.83 |

Total measured cost lost: **27 min 48.5 s of CPU over 53 min 37.2 s of wall clock**, plus the
Q2 pytest sharding attempt and one pilot cell, none of which produced an admissible result.

## What #1 and #2 prove about admission

Both scopes carry the epoch **1790427407** in their unit names and both were `Started` at
**07:56:47**, in the same second, from `shard.sh 1` and `shard.sh 2`. Two scopes, one capacity
reading, no reservation between the reading and either launch — this is the defect itself, visible
in the journal and not only in Musashi's probe. Their aggregate committed limit was 4 GiB against
a single reading.

Their kill mechanism, however, was **not** host exhaustion: each hit *its own* 2 GiB `MemoryMax`
(`CONSTRAINT_MEMCG`, with `oom_memcg` equal to the scope itself). The correction they call for is
a cap sized on the arm's own **tree** peak, bound to evidence — not a larger host.

## What #3 and #4 prove about admission

Neither reached its own cgroup limit. #4 was admitted at 13:11:37 with **14.02 GiB**
`MemAvailable` recorded in its own gate file, having waited 45 s for exactly that reading; 23
minutes later oomd killed it for *session* pressure. A gate that reads `MemAvailable` immediately
before a launch cannot see this, and the reactive local guard (`crispdm-memguard`) reads
`MemAvailable` too, not PSI, and recorded no intervention in either episode.

The corrections they call for are different from #1/#2: read **PSI** and stop below where oomd
acts; hold a **reservation** so a second load cannot arrive beside the first; keep the desktop
reserve real; and treat the kill as **terminal** — an oomd kill is not a licence to re-ask with a
bigger cap, since the cap was never the binding constraint.

## Declared limits: a gap this order closes

`systemd-run` does not log its `-p MemoryMax`, and until today no launch path wrote a receipt of
the cap it asked for. So:

* #4's declared cap is **recorded** — `MEMORY_GATE.jsonl` names `9G` / 9 663 676 416 B.
* #3's is **unrecoverable**: no receipt exists. Its peak (2.8 G) is a lower bound, and the absence
  of a `CONSTRAINT_MEMCG` kill says the cap was above it. Recording it as "4 GiB, the launcher
  default" would be a guess, so it is recorded as NOT RECORDED.
* #1 and #2's are **inferred, not recorded**: the kernel fired `CONSTRAINT_MEMCG` on each scope,
  which happens only at that scope's own `memory.max`, and systemd reported each peak as `2G`. The
  cap was therefore 2 GiB. It is inference from two independent readings, not a receipt.

From DR01 onward the launcher writes the declared cap, the unit, the argv digest and the observed
**tree** peak into the admission ledger (`$XDG_STATE_HOME/crispdm/admission/ledger.jsonl`), so this
column is never an inference again.

## Boundaries of this register

* Nothing here attributes all the session pressure in #3 and #4 to a single producer. The journal
  does not support that, and two experimental scopes were live at the time of Musashi's review.
* The two oomd episodes are real kernel/oomd events, not a desktop notification artefact.
* No live process was killed, no application of the owner's was closed and no resource under a
  running child was modified while building this register.
