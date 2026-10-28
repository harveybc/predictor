# Shared selection execution

Authority: owner instruction and Satoshi handoff e34f8f13. This is an operations
plan, not acceptance of the feature selection or authorization of downstream RL.

## Scope and Order

1. Preserve both running base shards and the pinned scientific code 38ca383.
2. Check code/input hashes and actual base completion, not old test markers.
3. After Satoshi merges all 84 successful base units, run contrast shard 0/2.
4. After the complete contrast ledger is merged, run rescue shard 0/2.
5. Satoshi retains shard 1/2, ledger merging, aggregation and scientific repairs.
6. Keep the old SAC queue held. Do not launch RL on this provisional selection.

The supplied HGB and temporal selection models run on CPU. Moving them to GPU
would change the supplied implementation; this deployment does not do so.
One admitted heavy process at a time on our worker, 3 GiB / 86400 s per stage.
The other local machine is used for preparation, tests and orchestration support,
not an overlapping copy of shard 0/2.

## Acceptance Tests Before Deployment

| Requirement | Test / evidence |
| --- | --- |
| Old DONE does not open the barrier | Latest base start must precede a fresh DONE; all 84 exact keys required. |
| No duplicate launch | Process argv matching excludes shells querying the process list; timer is serialized with flock. |
| Same queue and shard | Generate expected keys by calling the pinned Engine.arms, including its original global index parity. |
| No partial merge accepted | Require unique successful prerequisite keys before planning the next stage. |
| No false success from exit zero | Verify every assigned key in the ledger after the scientific command. |
| Resume without losing results | Existing successful keys remain; failed/missing keys are retried by the original engine. |
| Identity does not drift | Verify pinned code and input files before each stage/probe. |
| Reboot / resource refusal | Persistent timer; bounded fit attempts; resource admission refusal is reported separately. |
| No scientific edits | Original model, data, seeds, folds, targets, scaling, budget and command arguments unchanged. |

## Current Method State

OWN_SHARD_COMPLETE: contrast and rescue each finished on their first attempt,
with every assigned key verified and no missing/failed units. The timer disabled
itself at completion. The later encoding handoff is now acknowledged; the peer
has already taken its 0/2 fallback, so it is not duplicated on our worker.

At deployment, the operations acceptance contract above had 23 passing
focused CPU tests (including zero-exit partial failure and waiting for the peer).
The persistent 60-second timer is enabled, and its first real tick identified
the existing base worker without dispatching another process. Both worker copies
have 123 identical code/input file hashes; Python sources also match Git 38ca383.
Runtime state and exact private paths are outside Git.

Only our worker emits DONE after verifying all assigned keys. It preserves the
original engine's deterministic global shard ordering, success reuse and failed
unit retries. At most three entered scientific attempts per stage, with cooldown;
admission refusals before entry do not consume that fit-attempt count. This is
bounded recovery, not a promise that arbitrary scientific errors repair themselves.

A real supervisor defect was fixed in Satoshi's private driver: its pgrep search
matched the shell running that search. Using the bracketed first character
excludes the query shell. Only that supervisor was reloaded; neither base process
was signalled. Satoshi still owns the driver and its final aggregation/publication.
His driver log lives on the local orchestration machine, not the remote peer as
the handoff initially suggested. Our worker instead checks the actual merged
ledger and exact stage keys, so it does not depend on that log's location.

Both real merge-to-fit handoffs have now been observed. The combined first
aggregate contains 84 base, 1,524 contrast and 120 rescue units without process
errors. A later read-only support review found population mismatches in some
comparisons; completion is not scientific acceptance. See
[review and candidate repair](MUSASHI_REVIEW_OF_SATOSHI_SELECTION_UPDATE_2026_10_22.md).
Its real-data follow-up also demonstrated a wrong-file loader binding: 200
effective inputs versus 260 intended. Candidate 5aeb2e7 restores the declared
file/hash and rejects missing columns; the no-fit worker probe consumes 260 and
pairs the witnessed arms. Integrating it requires a fresh full-pool identity;
the 200-input base is not reusable as the 260-input base.

## Handoff to Satoshi

The assigned worker is installed and enabled; no further manual launch request
is needed. It consumes your existing merged out_full/LEDGER.jsonl, writes the
same LOG_full start/DONE protocol, and does not perform a merge or aggregate.
It replaces only the launch mechanism on the assigned half, not the scientific
engine. Your peer launch and aggregation remain yours.

Live operations state is <psel-root>/musashi_ops/STATUS.json. The two stage
KEYS.json files contain the original engine's full and assigned key lists,
effective store hashes and the prerequisite ledger digest. Stage RUN.json and
separate ATTEMPT_N.json files retain commands, attempts and missing/failed keys.
Successful rows are checked even when the scientific child exits zero.

The peer shell wrapper still emits DONE on zero process exit even if an engine
unit recorded an error. Our worker does not trust that as a sufficient barrier:
rescue requires every expected contrast key to be successful in the merged
ledger. Keep failed-unit repair and final completeness checks in your lane;
neither this deployment nor an empty error list before completion is an audit
certificate for the selection.

Resource observation at deployment: the assigned worker has about 6.9 GiB
available RAM and 71 GiB disk free; the orchestration machine has about 19 GiB
available RAM, 82 GiB free on home and 673 GiB on the archive volume. No extra
GPU workload, duplicate base worker, scaling change or downstream RL was launched.

The historic FEATURE_SELECTION_METHOD_STATE is not a certificate for Satoshi's
new selection. This operations deployment neither advances its scientific gates
nor declares coverage, causality, selection completeness or trading usefulness.
