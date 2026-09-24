# RP136-RP139 review and live operational follow-up

Reviewed source: `d0fb9acd`, separate repair worktree. No changes to the running
RP135 producer. Owner cooling confirmation and concurrent-work instruction apply.

## Experimental progress

At review, six L512 cells have records and terminal receipts; H336s2021 is running
on the external5090. Recorded author float32 metrics, not independently accepted
by this review:

| Horizon | Seed | MSE | MAE |
|---|---|---:|---:|
|96|2021|0.125551492|0.220453292|
|96|2022|0.125848770|0.220933795|
|96|2023|0.126375318|0.221487328|
|192|2021|0.144050777|0.238342345|
|192|2022|0.142952621|0.237667948|
|192|2023|0.143947497|0.237898931|

These are new measured cells, not a verified horizon mean or a doctoral H1 result.
Matched naive and independent replay belong to the pending campaign closure.
No naive value is invented here. Table9 H96 publishes0.126/0.220, with unresolved
per-horizon lookback, so it is not a certified exact comparator for this recipe.

## Findings and scope

1. **High, next-experiment design:** NEXT_INTERVENTION_PREPARATION section2
   reintroduces AE selection on external DEV validation inputs. RP36's actual
   `df_e1_pilot.py:526` and `test_df_e1_pretraining.py:123` use a purged inner TRAIN
   tail. Correct the successor document and prove membership before fitting.
   This is not evidence that current TimeFilter or the tested E1 runner leaked.
2. **Medium, claim:** coordinator replay is numerically exact, but the actual
   report has training UUID null/attribution UNKNOWN. State exactness on the
   observed4070, not certified physical same-device identity without an original
   execution binding. This does not change the numerical scores.
3. **Medium, orchestration:** completion of protocolB, retention-only repairs and
   a renewed owner permission are not universal prerequisites for the next
   isolated TRAIN-only cost pilot. Its scientific input/architecture/causality
   tests remain prerequisites. A matched A campaign depends on accepted A.
4. **Required before matched ECL fitting:** current E1 target code is a single
   channel at one offset. The reference evaluates horizon x321 channels. Sharing
   a dataset name does not establish a matched task. RP142 requires an actual
   full-output adapter with independent target/scaler/reduction parity.

The declared implementation transition is acceptable for the narrow comparison
of recomputed catalog fields. The delegated audit independently hashed both
historical function bodies and confirmed the one-line finiteness-guard change;
8354 single numeric mutations were refused, as were an unknown digest and an
extra foreign-revision identity field. Timestamp/independent-check notes are
excluded from estimator equality; they are not separately certified by it.
Provenance revision labels are annotations, not a proof of the entire producer.
No new numerical-bypass finding in this narrow transition. This is not an audit
approval for production deletion or all legacy estimator definitions.

## Operations completed by Musashi

Fresh read-only authentication using the existing warehouse service environment
closed the missing custody check for H192s2021. Preparation and cell terminal,
design and all artifact hashes AND byte counts match. The previous report is
preserved by digest; its recorded replay is exact. See
[LIVE_CUSTODY.json](../evidence/RP139_REVIEW_2026_09_23/LIVE_CUSTODY.json).
This separate result is not silently promoted to a fresh complete closure.
Seven fixture cases prove the audit refuses missing/foreign/changed/duplicated
accepted evidence. No token was printed, committed or copied between actors.

At admission, host available RAM was approximately8GiB, scope current1.84GiB,
other shared batch usage approximately0.4GiB. Raised the live continuation
MemoryMax6->7GiB and MemoryHigh to6GiB; parent batch slice remains8GiB. Existing
peak5800300544bytes =5.40GiB, with oom=0/oom_kill=0. No restart or recipe change.
The existing scope and heartbeat remain active; no duplicate job was launched.

## Disposition

Continue RP135. Close A by bound evidence composition; preserve device-attribution
limits. Execute [RP140-RP143](../../handoffs/MUSASHI_SOTA_RP140_RP143_2026_09_23.md)
in parallel. No owner action needed. No production prediction deletion.
