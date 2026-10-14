# I6-DMO return: short static direct multi-output (order section E)

Date: 2026-10-14. The short family is implemented and deployed, and its first full fit is running.
There are no scientific results yet. TEST is closed. This document reports code and deployment, not
an experiment result.

## 1. Commits and identities

| Repository / lane | Commit |
|---|---|
| predictor, deployed code | `93078a0f` on `satoshi/i6dmo-static-20261014` (the branch tip also carries this document and the milestone update) |
| predictor, model lane | `0d60f685` |
| predictor, data lane | `0d0a3349` |
| predictor, strategy lane | `457624e0` |
| predictor, ops lane | `98711afe`, plus integration fixes up to `ab1d492a` (exit-code fix) and `93078a0f` (preflight routing) |
| heuristic-strategy (pair policy pin) | `d40fd3b0` |
| planning commit | `2a440265` |

| Identity | Value |
|---|---|
| Design (short) | `541ce40ac1becceb16b14937e34e3452eee36aad8a370686dacf62860c86d381` |
| Model-lane effective config (short) | `cd78cc6ce9c03d2202d73cc7e5a0624f60f72a66f7abe498b5b027e5c148513f` |
| Campaign seal (ops `EFFECTIVE_CONFIG.json`) | `368fb4a1059b63f7e0dac4ba4ad15de360231b580ff2866243fcb8b69524119b` |
| Feature manifest (19 features, no STLFSI4) | `2ff34fa21d5841da1befdd4d898db911128d959a5e460d04cc45b54955416470` |
| Calendar | `782603fd26e879d6f837c83e9ec92d363500cb1dd150544e10a80c1ddeb88265`. This is an ASSUMED FX calendar, not a verified broker calendar. |
| Short bundle | `bundle_sha256 3e42c3f55dfffd275aa125f144144af26dfa0ec30d80b8e27f599d1511b1c205`; BUNDLE.json `0140dfbc91d6234360bcb60790c9c36588ac56b1afde028ae38fa817cd01e03a`; sampling `DS2_ASOF_LAST_AVAILABLE`; rows fit / inner / validation = 18,144 / 739 / 4,889 |
| Long bundle | `bundle_sha256 e88f72aa93b478c6e175efc671e706ce85d1125127444b26b0017973b44277a1`; BUNDLE.json `afcef506df02ceba74fac77b6b41e5e3b22ea4e2b2b69d0e4556403b880b9a75`; sampling `DS12_ASOF_LAST_AVAILABLE`; data contract `6e354fe6f8ad1930aa75aa16bc516717b04f75b4c9ef60c84c2b59c911d85e75` |
| Retained input (INPUT.npz) | `ef22ddb54921ee9f9c774a550731dcd522c9d712f2a1882f1bf4e6a35e3942b8` |

`init` refuses the superseded exact-slot bundle `5221615d`. The private spec, the pins and the
bootstrap live outside Git.

## 2. DMO01-DMO20 traceability

Every lane wrote its tests first and recorded the failing PRE run. The POST results below are CPU
tests. On the integration tree, the full suite of 177 tests passed in 5 bounded jobs. No GPU
scientific run backs any row yet. The required independent reviewer's replay was **not run**, and
`TRACEABILITY.json` is not written; the lane return files hold the per-test detail.

| DMO | Owner | Result | Negative control included |
|---|---|---|---|
| DMO01 six heads, one backbone | model | PASS | future input does not move earlier core steps |
| DMO02 13 exact as-of slots incl. t | data | PASS | perturbing t+1 and later changes no input |
| DMO03 exact h3/h5, no thinning | data | PASS | a missing endpoint is not bridged by the next row |
| DMO04 masked loss | model | PASS | NaN cannot enter the loss |
| DMO05 C1 masks, DST, splits | data | PASS | cross-boundary labels are purged |
| DMO06 outer/future perturbation | data | PASS | positive control included |
| DMO07 one fit per family | ops | PASS | a report entry equal to the fit entry is refused |
| DMO08 MAE and naive on identical rows | data + ops | PASS | tampered target; shuffled rows and reordered heads are typed failures |
| DMO09 pooled sums and counts | data + ops | PASS | unequal weeks differ from the mean of weekly MAEs; an absent week is explicit |
| DMO10 fit/inner stopping, save/load | model | PASS | a tampered archive is refused |
| DMO11 identity adoption | ops | PASS | changed code, data or config cannot adopt; a tampered prior terminal is rejected |
| DMO12 outbox exactly-once and readback | ops | PASS | a readback mismatch never closes delivery |
| DMO13 TEST closed | ops (+ data) | PASS | an open-TEST design and a tampered seal are refused; TEST rows are refused |
| DMO14 masks and chronological SL/TP | strategy | PASS | reversed hit order; a missing head is never filled |
| DMO15 forced flat with a real fill | strategy | PASS | the end of data invents no fill |
| DMO16 four years, every origin | data | PASS | incomplete coverage is refused, not shortened |
| DMO17 one heavy lease per host | ops | PASS | the 5070 Ti is blocked; foreign GPU processes are refused |
| DMO18 TRAIN-only scales, 19 features | model + data | PASS | STLFSI4 aliases are rejected |
| DMO19 progress, measured ETA or UNMEASURED | ops | PASS | stale and failed units are never COMPLETE |
| DMO20 cadence identities, no TEST feedback | ops | PASS | a gate built from TEST reads is refused |

## 3. What is installed and running

All units are user units on the coordinator, which supervises the children on their hosts.

- **Coordinator.** It runs the tick timer and three role worker timers, all enabled. The report and
  pair units run on this machine's CPU under the memory launcher. **Publish units are NOT
  installed** (see section 5).
- **Secondary GPU (4090).** `short.fit` is running here at the default 7936 MiB cap, with a live
  claimed child and a fresh heartbeat. At the single verification it was at epoch 13 of a
  200-epoch ceiling.
- **Preferred GPU (5090).** Admission was refused with `HOST_MEMORY_BELOW_GATE`: about 10.3 GiB is
  available, which is below the 7.75 GiB cap plus the host's 3 GiB reserve. The margin was not
  trimmed. The host joins later fits once a measured peak gives a cap of 1.25x that peak and the
  cap fits.
- **5070 Ti.** It is never used, because it is not a parallel slot.

**Retired first root.** The launcher refused admission with rc 75 and started nothing. Because
systemd had already collected the transient unit, the worker read the result as rc 0 and typed the
unit TERMINAL_INVALID. The fix is in `ab1d492a`: the child writes its own `EXIT_CODE`. The first
root is kept with a `RETIRED` reason, its units were removed, and no fit ran in it.

**Installed modules.** The GPU hosts use the existing isolated `tensorflow` environment: Python
3.12.13, tensorflow 2.21.0, keras 3.13.2, numpy 2.4.3. The coordinator uses Python 3.12.7 with
numpy 1.26.4 and sqlalchemy 2.0.34. Bundles, input and code were replicated to both GPU hosts, and
all 20 files matched their sha256 on each host.

## 4. Status, denominators, ETA

`python -m tools.i6dmo status --root "$I6DMO_ROOT" --level campaign --json` (the `unit`, `stage`,
`publication` and `fleet` levels are also available).

- **Expected validation units: 5.** These are `short.fit`, `short.report`, `long.fit`,
  `long.report` and `pair.validation`.
- **Final units:** not enumerated until TEST readiness. There will be two refits on
  `[2021-01-01, 2025-01-01)` and one TEST pair unit.
- **ETA: UNMEASURED.** No fit has completed yet. The in-flight epoch rate is reported only as an
  upper bound at the 200-epoch ceiling.

## 5. OLAP and mirrors

- **Publication is not deployed.** The warehouse URL and EnvironmentFile are unresolved: two
  attempts to locate them were denied as Credential Exploration and were not retried. Owner: Satoshi
  fills `warehouse.url` and `warehouse.environment_file` in the private spec and redeploys.
- **Facts are held in the outbox.** Every fact goes into the durable outbox
  (`state/outbox.sqlite3`), keyed by report_sha256, and stays there until publication runs. No
  OLAP run id or readback exists yet.
- **Mirrors.** The local and second-host mirrors are configured. They are synced and verified by the
  publish pass, so they have no `MIRRORS.json` yet.

## 6. Result index and M1-M8

- **Numeric results: none.** The first fit is in flight.
- **Milestones.** `docs/audits/evidence/canonical_20261003/MASTER_MILESTONE_STATUS.json` M6 now
  records the I6-DMO engineering state. **No percentage was changed.** `MASTER_MILESTONE_PROGRESS.png`
  was regenerated with `render_master_milestones.py`.

## 7. Limitations

- **Feature set.** The 19-feature set was selected using 2024, so VALIDATION is development data,
  not an untouched holdout.
- **Source clocks.** Two FRED features (DGS3MO, DGS7) have recovered producers, but their
  publication clocks are UNVERIFIED (ALFRED vintages are missing). This blocks a confirmatory TEST.
  The DST-gap clock disagreement is retained as stale, not leaked.
- **Calendar.** The calendar is ASSUMED, not a verified broker calendar.
- **Long family.** The long design is sealed automatically after short closure. `long.fit` checks
  the data contract `6e354fe6` before it uses the long bundle.
- **Strategy rulings pending (owner):**
  - barrier equality;
  - costs and pip value (the legacy `pip_cost` 0.00001 gives about 0.2/0.1 real pips of spread and
    slippage and needs ratification);
  - fill model;
  - Sharpe convention;
  - minimum long heads.

  The pair unit waits in WAIT_OWNER_RULING and also waits for the strategy pins.
- **TEST is closed.** No TEST array has been materialized.
- **Old POO-v2 fits.** The partial POO-v2 fits are kept as interrupted, not as scientific
  terminals. Their services are disabled and in a failed state. The archive timers and warehouse
  tunnels were deliberately left running.
