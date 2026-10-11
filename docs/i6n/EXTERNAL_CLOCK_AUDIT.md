# Independent Source-Clock Audit

`tools/i6n_external_clock_audit.py` tests the proposed fixed-EST policy against
public Dukascopy UTC bid-minute candles. It never reads retained feature arrays,
parquet, training data, or TEST. It does not correct timestamps or launch fits.

## Frozen Plan

Eighteen dates: first Monday on/after January 15, March 15 and July 15 of each
year 2019 through 2024. No dates, tolerances or acceptance rules are adjusted
after prices are observed. Recovered annual HistData ZIP hashes must match the
input `MANIFEST.json`. HistData processing reuses `hourly_closes` and
`clock_series` from `i6n_source_recovery`; both use hourly bar-end comparison
timestamps, avoiding a bar-start/bar-end one-hour confound.

Exactly one public HTTP request per planned date, no retries or redirects,
5-second connect/15-second read timeouts and a 1,000,000-byte body cap.
LZMA decoding is limited to 34,560 bytes and a 64 MiB decoder memory budget.
Records are big-endian `>5if`: UTC seconds from midnight, open, close, low,
high, volume; integer EURUSD prices are divided by 100,000. A complete ordered
1,440-minute day and valid OHLC/volume are required. Hours use minute 59 closes.

Each case compares the same 24 reference endpoints under fixed EST and
America/New_York. PASS requires 24/24 fixed-EST closes matching within 1e-12.
Winter additionally requires 24/24 NY matches. DST additionally requires
fixed-EST mean absolute error strictly below NY's, with 24 NY endpoints too.
Missing data, transport errors, decoding errors or slight feed differences
remain INCONCLUSIVE. Tolerance is not relaxed after observing discrepancies.

## Terminal Contract

`REPORT.json` contains:

```text
status: VERIFIED only when all eighteen planned cases PASS, else INCONCLUSIVE
clock_policy: FIXED_EST_BAR_START
input_manifest_sha256: SHA-256 of the exact annual MANIFEST.json bytes
cases: ordered date/status/source_zip_sha256/dukascopy_sha256 and metrics/errors
```

There are exclusive-write raw `.bi5` files, per-case JSONs, an `ATTEMPT.json`
plan, and `REPORT.sha256` over the final report bytes. The report also binds
script/helper hashes and explicitly limits independent coverage to the sampled
days. Full annual reconciliation to retained prices establishes transformation
lineage, not independent UTC correctness for every historical minute.
Missing/partial evidence cannot satisfy the terminal gate. Existing output
directories are refused to prevent implicit retries/overwrites.

HTTP is unauthenticated public transport. Artifact hashes provide local
integrity/binding, not cryptographic provider attestation. The audit detects
agreement or contradictions against the downloaded feed; it cannot prove the
provider's identity over HTTP. Source-clock resolution must retain that limit.

## Interpretation

The HistData FAQ declares fixed EST, but that declaration alone does not settle
conflicting quote/time evidence. March 11, 2024 is in the US/Europe DST-gap
period: its reported exact fixed-EST match is evidence for that period, not
for July. Direct raw-file inspection found source closes at 15:59 on March 15
and March 29, but 16:59 on April 5 and July 5, 2024. This warrants checking
season-dependent behavior before a blanket +1-hour NY-DST correction.
The audit does not assume either clock is correct when a case fails.

## Execution

Tests were designed before implementation: initial RED was a missing module;
initial GREEN was 17 audit tests plus 46 adapter tests, 63 passed in 0.95 s
under CPU 512 MiB / 120 s. Synthetic tests cover binary bounds, transport
limits, UTC/hour ends, both seasons, mismatches, report schema and
complete-plan gating; fabricated prices are not clock evidence.

One live execution is admitted as follows:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  crispdm-run -m 2G -t 300s -n i6n-external-clock-audit -- \
  python -B -m tools.i6n_external_clock_audit \
    --annual-root /var/tmp/i6n-source-annual-verified-20261011 \
    --output /var/tmp/i6n-external-clock-audit-20261011
```

The internal request deadline is 260 seconds; requests with insufficient
remaining time are not started. Per-date results are emitted immediately.
Exit 0 means all planned cases passed; exit 2 means INCONCLUSIVE. A partial
run without a sealed terminal report never authorizes a corrected-input pilot.
Implementation and unit verification do not imply empirical acceptance; read
the actual terminal report before treating the clock as resolved.
