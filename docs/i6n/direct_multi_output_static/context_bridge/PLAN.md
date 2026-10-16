# Short Inputs, Long Targets: Automated R0 Context Bridge

Approved by the owner, 2026-10-16. Execute after the retained training audit,
without reopening feature selection. No Satoshi, DOIN, R2, core pretraining,
cadence changes, smoothing, decompositions, denoised inputs/targets or TEST.

## Frozen Question And Design

Two matched R0 arms, identical 4,882-parameter architecture and seed6102026:

| Arm | Role | Observations | Open-market spacing | Open-market context | Exact elapsed heads |
|---|---|---:|---:|---:|---|
| mg12c | Retained-recipe matched control | 13 | 12 h | 144 h | 18/24/36/48/60/72 h |
| mg2l | Short-input long-target treatment | 13 | 2 h | 24 h | 18/24/36/48/60/72 h |

Both include t and issue hourly, with actual UTC offsets and closure flags.
Nineteen selected RAW features, branch causal Conv1D4(k3), temporal concat,
physical-time encoding, core causal Conv1D8->4(k3), six independent
Flatten->Dense8->Dense1 heads. Only input grid/context/time inputs differ.
This compares whole input recipes, not separate causal effects of span/spacing.

TRAIN2020-2023, chronological last eight weeks INNER with mature labels;
all supported DEVELOPMENT2024 rows; no TEST unit, bar or metadata scan.
Common origin/head intersection frozen before either fit. Targets, masks,
FIT-only shared input scaler (union of both arms' FIT input source rows),
FIT-only per-horizon naive loss scales and architecture/initial weights match.
Business offsets: 12 h after open, 12 h before close; exact t+h must remain
before forced flat and split end. Entry cutoff is separate; closed hours
never count as observations or get synthesized. Calendar/source qualifications
remain declared and do not trigger another selection campaign.

AdamW lr .001, decay .0001, batch256, max200, patience10, min_delta1e-8;
same-weight end-of-epoch mean FIT/INNER and best restoration. Same 5090 GPU
class for both scientific fits, sequentially, one heavy process on its host;
no parallel 5070 Ti. Other machines handle bounded independent CPU work;
do not duplicate fits just to occupy a GPU. Initial cap 4 GiB provisional,
then 1.25 times each resource class's measured full-cgroup peak, if admissible.

Both references must be fresh if code/support/scaler identities differ from
the old long campaign. This is a necessary matched control, not a seed repeat;
old results remain unchanged. No second seed or silent retry after a completed
fit. Retries of publication never fit. Restart adopts authenticated terminals.

Primary comparison: equal-head mean of annual pooled `(MAE_h-naive_h)/naive_h`
on common support. Negative better; every MAE displays its own horizon naive.
Within1e-8 retain mg12c, the reference, without claiming superiority. Each
head retains its own eligibility; aggregate superiority does not authorize
strategy/TEST. This finite campaign stops after comparison/publication/mirrors.
No implicit promotion or new experiment is started by its scientific outcome.

## Requirements And Tests Before Execution

| ID | Structural and behavioral acceptance |
|---|---|
| CB01 | New explicit design/model families and seals; all old design bytes/identities remain unchanged; mixed pairs refused |
| CB02 | Both arms have 13 slots, the same six long horizons and 4,882 parameters; R0 initial weights agree; context alone differs |
| CB03 | Frozen common masks, exact source values/t+h, t included, closed-hour ranks, split purge and FIT-only statistics independently verified |
| CB04 | Existing model/report paths accept new families; exact paired price MAE/naive replay and zero declared capacity difference |
| CB05 | Complete negative advances to publication/mirrors/CLOSE; incomplete arm never ranks; tie retains control |
| CB06 | Fresh health, mount/disk/RAM/GPU admission; same GPU class and runtime; workers cannot consume TEST |
| CB07 | Durable existing queue/timers/outbox/mirrors reused; interrupted preparation and repeated deploy do not duplicate fits |
| CB08 | Bounded status at campaign/unit level with measured/unknown ETA; close stops only this campaign's timers |

Independent bounded actual-path review after focused tests and real-population
generation, before scientific deploy. Code/configs/hashes/small results in Git;
large arrays/models on experiment storage; SQLite/WAL and locks on ext4.
All metrics to governed OLAP with exact readback, local physical mirror and
second-host mirror. Automatic comparison and close require those delivery gates.
