# Independent review of the I6-DMO market-time pair

Reviewer: independent (wrote none of the reviewed code). Mandate: order 1.4 of
`docs/handoffs/MUSASHI_TO_SATOSHI_MARKET_TIME_PAIR_20261015.md`
(sha256 `50d20d018661bcd698e0f1922caf8eb6657961149efc25538407414e4d2c8e7d`),
governing specification `MARKET_TIME_PAIR_PLAN.md` sections 2-6 at
`codex/i6dmo-market-time-dispatch-20261015` @ `85e2dc88`.

Two jobs: **A. data provenance** (independent, reported here) and
**B. independent falsification of the integrated temporal and metric paths**
(reported in section B).

Constraints honored: CPU only (`CUDA_VISIBLE_DEVICES=""`), every execution one
at a time under `crispdm-run -q -m 3G -t <=900s -n i6dmo-mt-review -- ...` with
`OMP_NUM_THREADS=1`, no scientific fit, no GPU, no TEST (2025+) rows, no
service started or stopped, no credential read/printed/committed, no DOIN
change, no edit to any other lane's branch. Downloads were written to a
dedicated directory outside the repository and parsed by scripts kept
elsewhere, with `python3 -I`.

---

## A. Data provenance

### A.0 Verdict

| Object | Before | After this round |
|---|---|---|
| Per-observation point-in-time vintage for DGS3MO / DGS7, 2019-2024 | MISSING (`PROVENANCE_UNRESOLVED`) | **RECOVERED** from the official unauthenticated ALFRED endpoint; retained with digests and a replay rule |
| Value-level revision history of those two series | UNKNOWN | **MEASURED**: 1 of 1,587 observations per series differs from the late anchor vintage (one back-fill, no value revision) |
| The producer's declared clock `(D+2) 00:00 UTC` | asserted, `proven: false` | **PARTIALLY FALSIFIED**: for 339 of 1,587 observations (21.4%) the ALFRED availability date is strictly later than `D+2`. Covered only if the H.15 documented same-day 16:15 ET posting is accepted as the availability instant |
| Per-release H.15 timestamps, 2019-2024 | MISSING | **STILL MISSING** — only the current schedule page is served; dated historical release pages return HTTP 404 |
| MT5 retained live-spread rows for instrument config `f2443141` | "retained but untranscribed" | **CLAIM REFUTED** — no such rows exist on this host; nothing is transcribable, no cost field can be promoted |
| Confirmatory TEST | closed | **STAYS CLOSED** |
| Labelled 2024 development execution | permitted | **STILL PERMITTED** — incomplete provenance blocks TEST only (plan section 5; dispatch section 1.4) |

**No credential was used, requested, scavenged or created.** All FRED/H.15
evidence came from endpoints that serve the public anonymously. The
`FRED_API_KEY` EnvironmentFile named as a blocker in
`RETURN_SUPPORT_STUDY_20261015.md` section 5 is **not required** for this
artifact: the plan's own allowance ("official unauthenticated evidence, where
available, does not require inventing a credential gate") is satisfied.

### A.1 What was retrieved, and how

The missing object was ALFRED per-observation `realtime_start` for DGS3MO and
DGS7 over 2019-2024. The prior round concluded no governed path existed because
the only FRED adapter in the fleet calls `fred/series/observations` (API key,
current vintage only). That conclusion is correct **about the API**, and wrong
as a statement about availability: ALFRED's public **Download Data** form
serves vintage output anonymously, and one of its four output formats is
literally *"Observations, Initial Release Only"*, whose third column is
`realtime_start_date`.

```
POST https://alfred.stlouisfed.org/series/downloaddata?seid=DGS3MO
     form[file_type]=4      # Observations, Initial Release Only
     form[file_format]=csv
     form[obs_start_date], form[obs_end_date]
     form[selected_vintage_dates][] (repeated)
-> application/zip  {README.txt, obs.,_initial_release_only.csv}
   period_start_date,DGS3MO,realtime_start_date
   2019-01-02,2.42,2019-01-03
```

Two server-side limits shape the retrieval and are recorded because they
constrain any replay: a download may name at most **450 daily vintages**, and
the per-observation `realtime_start` is **left-censored at the first vintage
named** in the request. Censoring is what makes a naive full-range download
wrong, so the chunk rule is part of the artifact:

> chunk *k* requests the observations of one calendar quarter with the vintage
> set covering `[quarter_start - 20 days, quarter_end + 20 days]`; only rows
> inside the quarter are kept. Measured maximum publication lag is 5 calendar
> days, an order of magnitude inside the 20-day pad, so every kept row's true
> first-publication vintage lies strictly inside its chunk's vintage set.

25 quarter chunks per series plus one late-anchor full-history download per
series: **52 bounded requests**, each retained as the raw server zip with its
sha256 under `provenance/alfred_raw/`.

### A.2 Retained artifacts

| Artifact | Content |
|---|---|
| `provenance/ALFRED_PROVENANCE.json` | manifest: endpoint, authentication `NONE`, output formats, obs/vintage windows, chunk rule, per-chunk vintage window, bytes, zip sha256, inner CSV sha256, rows kept, the merged table's digest, the revision test and the lag histogram, plus the replay rule |
| `provenance/DGS3MO_realtime_start_2019_2024.csv` | 1,587 rows `observation_date,initial_value,realtime_start_date`; sha256 `bdeebfdfb0edd910f59b59379f9924ebaa450a5f36c720a2a5b47455df2f1091` |
| `provenance/DGS7_realtime_start_2019_2024.csv` | 1,587 rows, same schema; sha256 `f78d6c09398007013de614a992e6fb9b4388749852675f28563189b40b0a8dbd` |
| `provenance/alfred_raw/*.zip` | 52 raw server responses (26 per series), digests in the manifest |
| `provenance/H15_RELEASE_CLOCK.json` | the five Federal Reserve pages probed, their HTTP status and digests, the quoted documented clock, and the explicit limitation |
| `provenance/MT5_COST_EVIDENCE.json` | the MT5 transcription attempt, the fresh absence search, the digests of every retained MT5 artifact, and the cost fields that cannot move |
| `provenance/PROV01_RESULT.json` | output of the independent clock-check probe |
| `provenance/PROV02_BACKFILL.json` | output of the back-fill bisection probe |
| `review_probes/fetch_alfred_provenance.py` | the retrieval, replayable |
| `review_probes/fetch_h15_release_clock.py` | the H.15 pages and the MT5 absence search |
| `review_probes/prov01_fred_clock_check.py` | the clock controls, with two injected fail-closed counterexamples |
| `review_probes/prov02_backfill_bisect.py` | bisection for the one back-filled observation |

Replay rule: ALFRED vintages are append-only, so a replay reproduces every
retained digest unless St. Louis publishes a **new** vintage touching a
2018-12..2024-12 observation. The merged per-observation table is the stable
object; the chunk zips carry the raw bytes for byte-level comparison. Verify
with `python3 -I review_probes/prov01_fred_clock_check.py provenance`.

### A.3 What the evidence says (PROV01, executed)

Controls on 1,587 observations per series, CPU, no network:

- **C1 PASS, 1,587/1,587 both series.** No observation is in the ALFRED
  database on or before its own observation date. There is no same-day
  database availability anywhere in the window.
- **C3 the lag is exactly "next business day".** Calendar-day lag by
  observation weekday, identical for both series:

  | obs weekday | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 |
  |---|---|---|---|---|---|
  | Mon | 272 | 42 | 4 | | |
  | Tue | 304 | 14 | | | |
  | Wed | 300 | 16 | | | 1 |
  | Thu | 300 | | | 17 | |
  | Fri | | | 265 | 51 | 1 |

  Every long lag is a weekend or a holiday: Friday observations land on Monday
  (3) or Tuesday after a Monday holiday (4); Thursday lag 4 is the
  Friday-holiday pattern. No unexplained residue.
- **C2: the declared clock is not covered by this artifact for 21.4%.**
  `producer_recovery.v1.json` declares, for both FRED features, *"as-of last
  value from (D+2) 00:00 UTC, NaN if older than 168 h"*. **339 of 1,587**
  observations per series have an ALFRED `realtime_start` strictly later than
  `D+2` — they are the Friday and Thursday-before-holiday observations.
- **C4 both injected counterexamples caught.** A doctored row whose
  `realtime_start` equals its observation date flips C1 to FAIL; a doctored row
  pulled back to exactly `D+2` decrements C2's uncovered count by one. The
  controls see an injected violation, so their PASS is informative.
- **C5 revision control.** Against vintage 2025-01-10, **1 of 1,587**
  observations differs per series: `2020-08-19`, first published **empty**
  (`realtime_start` 2020-08-21) and later carrying DGS3MO 0.11 / DGS7 0.48.
  No observation's published *value* was ever revised; one missing value was
  back-filled. PROV02 bisects the vintage at which that happened.

**How to read C2 honestly.** ALFRED `realtime_start` is the date a value
entered the FRED/ALFRED database; it is an **upper bound** on public
availability, because H.15 itself is posted on the observation day. The
Federal Reserve's release page states verbatim: *"The release is posted daily
Monday through Friday at 4:15pm."* and *"It is published every business day
except holidays."* (digests in `H15_RELEASE_CLOCK.json`). Under that documented
clock a Friday observation is public at Friday 20:15 UTC, well before the
declared `(D+2) 00:00 UTC` instant, and the declared clock is conservative for
**all** 1,587 observations. So:

- if the availability instant is **H.15 publication**, the declared clock is
  safe everywhere, but it rests on a *documented recurring schedule*, not on a
  retained per-release timestamp;
- if the availability instant is **ALFRED/FRED database presence** — which is
  what a FRED-based producer would actually have been able to read — the
  declared clock is **not** established for 339 observations per series.

This reviewer does not rule between the two: that is a specification decision
for the owner/Musashi. The measurement is retained either way, and it is
reported as a *bounded refutation of coverage*, **not** as a proven look-ahead.

### A.4 Exact artifacts still missing

1. **Per-release H.15 timestamps for 2019-2024.** Probed and unavailable:
   `/releases/h15/20190102/default.htm` and `/releases/h15/20190102/h15.htm`
   both return HTTP 404; `/releases/h15/` and `/releases/h15/current/` serve
   only the current release; the RSS data feed
   (`/feeds/Data/H15_H15.XML`) carries only the current release. What exists
   publicly is the recurring schedule, quoted above, with its digest. A
   per-release timestamp series would have to come from the Fed's Data
   Download Program release metadata or an archival request — neither is a
   credential problem, and neither was within this bounded round.
2. ~~The back-fill vintage of `2020-08-19`~~ — **recovered** by PROV02
   (section A.5): vintage 2020-08-25. No longer missing.
3. **MT5 EURUSD cost evidence** — see A.6.

Nothing on this list is a secret, and none of it was treated as one.

### A.5 The one back-filled observation (PROV02, executed)

PROV02 bisects the public ALFRED vintage list, one observation wide, for the
first vintage carrying a value at `2020-08-19`. Executed, 12 requests per
series over 1,087 candidate vintages:

| Series | last vintage still empty | first vintage carrying a value | value |
|---|---|---|---|
| DGS3MO | 2020-08-24 | **2020-08-25** | 0.11 |
| DGS7 | 2020-08-24 | **2020-08-25** | 0.48 |

So the `2020-08-19` observation was absent from the database at its own first
publication (2020-08-21) and for two further business days, and appeared only
at vintage **2020-08-25 — six calendar days after the observation date**. Under
the producer's declared `(D+2) 00:00 UTC` as-of rule the feature would have
read a *stale or NaN* input for that date, never 0.11 / 0.48. A current-vintage
replay sees the opposite. This is one observation of 1,587, and it is the
concrete instance of the plan's warning that release time alone does not
establish which historical value was available. Digests and the full probe
trail are in `provenance/PROV02_BACKFILL.json`.

### A.6 MT5 cost evidence: the retained-rows claim is refuted

Instrument configuration `f2443141` carries the note *"The MT5 bridge lab
retains live spread snapshots in its lab database; transcribing them is a later
evidence task, not done here"*, and `RETURN_SUPPORT_STUDY_20261015.md` section
5 repeats it as *"MT5 lab live-spread rows are retained but untranscribed"*.
Both statements are **false on this host**, and the transcription task they
defer cannot be performed:

- **The lab database does not exist.** A fresh search of the home tree,
  `/var/tmp`, `/srv`, `/opt`, `/mnt`, `/media` and `/tmp` for
  `mt5-bridge.sqlite*` / `*mt5*bridge*.sqlite*` returns **zero hits**. The
  configured path that the retained MT5 evidence itself names
  (`<home>/.local/state/lts/mt5-bridge.sqlite`) is absent. This independently
  reproduces a record already written in the `lts` repository on 2026-09-26
  (`docs/audits/evidence/mt5_unknown_outcome_20260926/NOTE.md`: *"no
  mt5-bridge.sqlite exists anywhere on the filesystem"*), which the instrument
  config's note contradicts.
- **The retained MT5 artifacts contain no spread at all.** The only surviving
  MT5 evidence is five snapshot JSONs captured 2026-08-16T01:46Z..04:22Z (plus
  a `latest.json` copy), all digested in `MT5_COST_EVIDENCE.json`. Each holds
  account, heartbeat, session, latest-bar, inference, position, order and
  service-state rows. **Zero `symbol_snapshots` rows, no `spread` field of any
  kind, and no `EURUSD` symbol anywhere** — the only live session is `ETHUSD`
  on an MT5 **demo** environment.
- **Even a recovered snapshot could not evidence these costs.** It would be a
  2026 ETHUSD demo quote; the configuration's four assumed costs are EURUSD
  costs for a 2020-2024 development backtest.

Consequently:

| Cost field | Could move to EVIDENCED? |
|---|---|
| `costs.spread_price` | **No** — stays `ASSUMED_NOT_BROKER_VERIFIED` |
| `costs.slippage_price` | **No** — stays `ASSUMED_NOT_BROKER_VERIFIED` |
| `costs.commission_per_lot_round_trip` | **No** — stays `ASSUMED_NOT_BROKER_VERIFIED` |
| `costs.swap_per_lot_per_day` | **No** — stays `ASSUMED_NOT_BROKER_VERIFIED` |

The already-`EVIDENCED` fields (`point_size`, `pip_size`, `price_decimals`,
`lot_units`, `pip_cost_parameter`) are unaffected; note that two of them cite
the MT5 bridge-lab **schema** (source code), which does exist, and not lab
rows, so their status is untouched by this finding.

**Missing artifacts, stated exactly:** MT5 `symbol_snapshots` rows for EURUSD
over the validation window with `terminal_observed_at`, `bid`, `ask`,
`spread`, `spread_points`, `point`; and a broker statement or venue fee
schedule giving the EURUSD round-trip commission and the signed per-direction
swap tiers. No broker was queried and no account was touched to establish this.

### A.7 Provenance findings for the implementing lanes

Non-blocking for development execution, blocking for TEST, in the order a
reader should act on them:

- **P-1 (documentation defect, two files).** The "retained MT5 live-spread
  rows" claim in `tools/i6dmo/instrument_eurusd.v1.json` (`notes`) and in
  `RETURN_SUPPORT_STUDY_20261015.md` section 5 is refuted and should be
  rewritten to the measured state. It is the kind of claim that would later be
  read as "evidence exists, go fetch it".
- **P-2 (recoverable record).** `producer_recovery.v1.json`'s `missing` field
  for both FRED features asserts that no unauthenticated vintage path exists.
  It does. The record should be updated to cite this artifact, and its
  `clock.proven` should move to a typed partial state rather than plain
  `false`, with the 339/1,587 coverage measurement and the unresolved
  H.15-versus-database-presence specification question named.
- **P-3 (TEST gate input).** If and when the owner rules that H.15 publication
  is the availability instant, the two FRED clocks become verifiable from a
  documented schedule plus this vintage table; the remaining gap is then the
  per-release timestamps, not a credential. `confirmatory_gate()` must keep
  refusing until that ruling and that artifact exist; nothing here authorizes
  relaxing it.
- **P-4.** The `2020-08-19` back-fill must be represented in any TEST replay
  rule: one observation per series was empty at first publication and filled
  later, so a point-in-time replay of the `diff_5d` / `logret_5d` windows that
  touch it differs from a current-vintage replay.

### A.8 Reproduction (A)

Each command one at a time, CPU, under the governed runner, from the market_time
directory of this branch:

```bash
CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 900s -n i6dmo-mt-review -- \
  python3 -I review_probes/fetch_alfred_provenance.py <out_dir>
CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 600s -n i6dmo-mt-review -- \
  python3 -I review_probes/fetch_h15_release_clock.py <out_dir>
CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 600s -n i6dmo-mt-review -- \
  python3 -I review_probes/prov01_fred_clock_check.py provenance
CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 900s -n i6dmo-mt-review -- \
  python3 -I review_probes/prov02_backfill_bisect.py provenance
```

`prov01` is offline and deterministic against the retained CSVs. The two
fetchers hit public endpoints only.

---

## B. Independent falsification of the temporal and metric paths

**Reviewed pin:** `satoshi/i6dmo-static-20261014` @ `96d4787b` ("merge the
MG2/MG12 data and model seals; verify the bundle-model array contract"), which
contains data `00c27fed` and model `5826f124` as ancestors (both verified).
Merged into this review branch; no lane branch was edited.

Sealed identities as pinned by the coordinator and re-derived here: designs
`169fbece` (MG2) / `8c10a9ca` (MG12), seals `e7c2f3e7` / `ef562ba4`, effective
`5798baaa` / `0b1b63e3`, bundles MG2 `21400c51` (BUNDLE.json `471ea311`) and
MG12 `c3867b5c` (`e6595c0e`), encoder `DynamicPhysicalTimeEncoder` v1.0.0.
Sealed runtime pins (TensorFlow 2.18.0, Keras 3.9.0) equal the installed
runtime. Candidate suite at the pin: **`pytest tests/i6dmo` 369 passed**,
424 s, CPU, one bounded job — the coordinator's count reproduces.

### B.0 Verdict

**FIX_REQUIRED.** The temporal and metric paths themselves are sound: every
plan-section-6 control over the data path, the encoder, the context channels,
serialization and the same-row per-horizon naive **passes**, each against an
injected fail-closed counterexample of mine. But **the sealed market-time
identity cannot execute a fit**: three independent seams between the lanes
refuse it, so there is no path from the sealed designs and bundles to a
measurement. All three are typed, fail-closed refusals — nothing silently does
the wrong thing — and all three are cheap to repair.

| Area (plan section 6) | Verdict | Probe |
|---|---|---|
| current-`t` inclusion; last offset exactly 0 | PASS | RB01 A1 |
| future perturbation changes nothing | PASS | RB01 A2 |
| exact ELAPSED target timestamps | PASS | RB01 A3 |
| weekend/DST offsets vs ordinary weeks | PASS | RB01 B1 |
| unplanned gaps not compressed into closures | PASS | RB01 B2 |
| chronology | PASS | RB01 C1 |
| split and purge boundaries | PASS | RB01 C2 |
| same-row per-horizon naive, original price | PASS | RB01 C3 |
| first slot delta 0 and flag 0 | PASS | RB01 D1 |
| actual per-example time/closure reach the model (controlled weights) | PASS | RB02 E1 |
| **an ordinal-index encoding would be caught** | **FAIL** | RB02 E2 |
| two context channels exact; the 24 not fitted | PASS | RB02 F1 |
| 19 branches -> 76 + 2 = 78 core channels | PASS | RB02 G1 |
| serialization/reload of ALL inputs incl. time channels | PASS | RB02 H1 |
| DS2 unchanged; no artifact adoptable by shape alone | PASS | RB02 I1 |
| **the fit path consumes a market-time bundle** | **FAIL** | RB03 J1 |
| finite graph, successor to long, PAIR_NOT_ELIGIBLE, no TEST unit | PASS | RB03 K1 |
| **the campaign accepts the sealed market-time designs** | **FAIL** | RB03 K2 |
| identity components, not shape, govern adoption | PASS | RB03 L2 |
| **the sealed effective config builds its model** | **FAIL** | RB04 |

### B.1 Blocking items, with reproducers

**B-1. `build_dmo(effective["model"])` is refused for BOTH sealed families.**
The trainer builds the network with exactly that call (`train_unit`,
tools/i6dmo/train.py). `design.effective_config()` emits a model block carrying
two keys that `model.normalize_model_config()`'s `MARKET_KEYS` set does not
accept:

```
UNKNOWN_FIELD/schema mismatch in market-time model config:
['offsets_hours', 'time_encoder']
```

`offsets_hours` is `None` (so there is no contradictory fixed elapsed ladder —
that part of plan section 2 holds), and `time_encoder` is the sealed formula
block; the two producers simply disagree about the model block's shape. This is
inside the model lane, on both `5798baaa` and `0b1b63e3`, so **the sealed
identities as they stand cannot build a network at all**, independently of any
data bundle. The positive control in the same probe shows
`model_config(features, family="short_mg2"|"long_mg12")` builds fine with 78
core channels, so the defect is the effective-config block, not market-time
models.
Reproducer: `python3 -I review_probes/rb04_sealed_effective_builds.py`.

**B-2. The trainer refuses the market-time bundle; nothing feeds the time
inputs.** `train.load_data_bundle` requires manifest schema
`i6dmo.data_bundle.v1` and the manifest key `offsets_hours`, and requires the
split npz arrays to be *exactly* `(row_id, x, origin_price, target_price,
mask)`. The market bundle is `i6dmo.market_bundle.v1` (deliberately, per
INTERFACE_DATA) with `offsets_open_market_hours` and nine arrays. The real
trainer CLI, invoked exactly as a fit unit invokes it, on the sealed MG2 design
and a freshly built MG2 bundle:

```
python -m tools.i6dmo.train --design EXPERIMENT.mg2.json --bundle <mg2> \
    --device cpu --json      ->  exit 3, status FAILED_TYPED,
                                 BUNDLE_SCHEMA
```

More fundamentally, `train.py` is untouched by this integration: it assembles a
single `x` tensor, and no code anywhere assembles
`[observations, offsets_hours, closure_crossed]` for the 3-input model outside
tests and my probes. The model lane's INTERFACE_MODEL declares
`consumes_data_bundle` for the market variant, but no consumer implements it.
`market_bundle_contract_check.py` checks the bundle against a *described*
contract, not against the trainer, so it reports CONTRACT_OK while the trainer
still refuses — I did not rely on it as evidence.
Reproducer: `python3 -I review_probes/rb03_integration_ops.py <workdir>`,
control J1.

**B-3. `campaign.init_campaign` refuses the sealed market-time designs.**
`_design_checks` hard-requires `schema == "i6dmo.design.v1"`; the sealed designs
carry `i6dmo.market_time_design.v1`, so `_seal_family` raises `Refused: SCHEMA`
for both short and long. The ops market-time graph is therefore untested against
the designs the model lane actually produces: its own fixture
(`ops_fakes.write_market_time_designs`) fabricates an elapsed-schema design with
an improvised `input` block carrying `sampling_rule` /
`spacing_open_market_hours`, which is a shape the sealed schema explicitly
forbids. The seven MTP acceptance tests pass on that fixture, with fake
trainers, which is why this seam is invisible in the suite.
Reproducer: same probe, control K2.

**B-4. An ordinal-index encoding reaches the model unrefused.** Plan section 6
requires that "ordinal-index encoding would be caught". The guard exists only in
the bundle builder (`market_grid.time_inputs` raises
`INTERNAL_OFFSET_NOT_ZERO`), and the encoder's `build` correctly refuses a fixed
per-batch grid — but the model call itself accepts slot indices
(`arange(-12..0)`) **and a positive, physically impossible future offset**
silently. Executed:

```
model_call_with_ordinal_offsets            -> ACCEPTED_SILENTLY
model_call_with_a_positive_future_offset   -> ACCEPTED_SILENTLY
encoder_with_a_fixed_per_batch_grid        -> ValueError (refused)
bundle_time_inputs_with_nonzero_last_offset-> INTERNAL_OFFSET_NOT_ZERO (refused)
```

Detectable is not the same as caught: the committed test shows the ordinal code
*differs numerically*, which is a property of the formula, not a guard. Since the
component that will feed these arrays (B-2) does not exist yet, this is the one
place where the missing seam could later be built wrong without any refusal.
Fix: assert in `DynamicPhysicalTimeEncoder.call` (or at the trainer's input
assembly) that the last offset is 0, that every offset is <= 0, and that the
first adjacent delta and closure flag are 0.
Reproducer: same probe as RB02, control E2.

### B.2 What passed, and how hard it was pushed

All counterexamples below were written by this reviewer, not reused from the
candidate's tests.

- **RB01 A1/D1.** On 18,856 FIT rows the last slot equals the origin in every
  row, every last offset is exactly 0, every offset is <= 0, and the first
  adjacent delta and closure flag are 0 in every row.
- **RB01 A2 future perturbation.** Shifting **19,291** bars strictly after `t`
  by +1234.5 (and the price by +99) leaves one origin's inputs, actual offsets
  and closure flags bit-identical; the counterexample — perturbing the *oldest
  in-window* bar — does change the inputs, so the control is informative.
- **RB01 A3 exact targets.** 28,484 (row, head) label comparisons against the
  retained price at `t + h*3600`: 0 mismatches; the shifted-by-one-hour control
  disagrees, so the equality is not trivial.
- **RB01 B1 weekend/DST.** Ordinary-week elapsed spans are exactly {24} open-
  market hours; closure-crossing spans are {71, 72, 73} over 2,600 rows — three
  distinct spans, which is the DST evidence the plan asks for, and never a fixed
  24 h label.
- **RB01 B2 unplanned gap.** Dropping one bar inside an open session keeps that
  hour an *expected* market hour; all 12 origins whose window contains it are
  excluded `INPUT_SLOT_NOT_OBSERVED` and counted per split, and **none** of
  their closure flags turns true at that slot — the gap is not compressed into
  a closure. With the bar present the same 12 origins are not excluded.
- **RB01 C2 purge.** For FIT/INNER/VALIDATION every origin lies inside its split
  and **zero** valid labels reach the split end.
- **RB01 C3 same-row naive.** `market_grid.paired_metrics` on the VALIDATION
  split (4,924 rows) agrees with my manual original-price computation on `n`,
  MAE and the per-horizon naive to <= 1e-9 on every head, on identical rows.
  Counterexample: rescaling the retained prices by 1.0001 raises
  `NaiveReplayMismatch` naming the disagreeing heads.
- **RB02 E1 controlled weights.** With all 19 branch kernels and biases set to
  zero (fused tensor identically 0, no gradient step anywhere), swapping actual
  offsets for ordinal indices moves the encoder output by 2.04 and the
  predictions by 0.270, and zeroing the closure flags moves the predictions by
  0.0159 — the per-example inputs demonstrably reach the core and the heads.
- **RB02 F1 context channels.** The two appended channels match my independent
  recomputation to 7.9e-8, the sinusoidal code to 3.3e-6 against the rule
  recomputed on the actual offsets, and the encoder layer holds **no weights**,
  so the factor 24 cannot have been fitted.
- **RB02 H1 serialization.** Reload is bit-identical (max prediction difference
  exactly 0.0) with 78 core channels; MODEL.json records the input names and
  `slot_order_is_not_wall_clock`. Counterexample: stripping `market_time` from a
  market artifact raises `MARKET_TIME_INCOMPATIBLE`.
- **RB02 I1 DS2.** The DS2 model keeps one input, 76 core channels,
  `PhysicalOffsetEncoding`, a null time-encoder field and a bit-identical
  reload; the config digests differ. Counterexamples: a market config carrying
  `offsets_hours` is refused, and a DS2 artifact whose metadata claims
  market-time is refused `MARKET_TIME_INCOMPATIBLE` — not adoptable by shape.
- **RB03 K1/L2.** The graph is exactly the six declared units with no TEST or
  final unit, `long.fit` depends on `short.report` **only** (the terminal, never
  eligibility), so the automatic successor after a scientific short failure is
  structural; `PAIR_NOT_ELIGIBLE` is a terminal status; the worker's identity
  component keys equal the campaign's.

### B.3 Scope discipline

Numerical equivalence was scoped **exactly as already reviewed**: the DMO10
protocol, its tolerances and `MONITOR_TOLERANCE` were not reopened, not
re-measured and not weakened, and no gate was relaxed. No fit, no GPU, no TEST
row, no service, no credential. The already-closed exact-DS2 baseline and the
retained ASOF arm were read only.

### B.4 Missing coverage (honest)

- **Everything GPU.** No GPU execution was permitted, so the 5090/4090 paths,
  GPU determinism and any GPU save/load identity for the new encoder are
  unverified by me.
- **The real bundles.** MG2 `21400c51` / MG12 `c3867b5c` live in a private
  campaign root that is not readable here; I rebuilt bundles from the synthetic
  fixture instead. I checked the pinned row counts in INTERFACE_DATA against the
  support study (16,496/715/4,772 and 13,174/546/3,804 — equal, reconciliation
  MATCH) but did not recompute them from the real input.
- **Deploy and publication.** `validate-design`, `preflight`, `init`, `deploy`,
  the outbox readback and the mirrors were not exercised; B-3 means `init`
  cannot currently accept the sealed designs anyway.
- **Campaign recovery and exclusive claims** were read in the candidate's ops
  tests (fake trainers) and not independently driven by me.
- **The acceptance matrix** carries only the ops-scope market-time addendum
  (MTP01-MTP07); no DATA or MODEL acceptance rows were added, which plan
  section 2 asks for.

### B.5 Reproduction (B)

```bash
CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 900s -n i6dmo-mt-review -- \
  python -m pytest -q -p no:cacheprovider tests/i6dmo
for p in rb01_data_time rb02_model_time rb03_integration_ops; do
  CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 900s -n i6dmo-mt-review -- \
    python3 -I docs/i6n/direct_multi_output_static/market_time/review_probes/$p.py <workdir>
done
CUDA_VISIBLE_DEVICES='' crispdm-run -q -m 3G -t 600s -n i6dmo-mt-review -- \
  python3 -I docs/i6n/direct_multi_output_static/market_time/review_probes/rb04_sealed_effective_builds.py
```

Retained verdicts: `review_probes/RB0{1,2,3,4}_RESULT.json`, redacted of local
paths and host names.

No statement in this review certifies any lane's code beyond what its own
executed control shows. This reviewer wrote none of the reviewed implementation
and does not self-certify the probes' subject.
