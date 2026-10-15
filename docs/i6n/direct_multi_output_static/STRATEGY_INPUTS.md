# Pinned RAW Strategy Inputs

Code-only preparation component, 2026-10-15. Scope: new
`tools/i6dmo/strategy_inputs.py`, its focused tests and this note only.
Independent review is required before deployment. The module never downloads,
fits, starts services, selects a vendor, reads TEST or changes retained inputs.

## API

```python
prepare_strategy_bars(archive, input_dir, out_dir, expected_sha256)
```

Returns the `i6dmo.strategy_inputs.v1` reconciliation document after successful
exclusive publication. Raises `StrategyInputError(code, details)` on a contract
failure. The CLI prints a bounded JSON summary and exits 0, or prints a failure
to stderr and exits 2.

The approved production archive is the HistData EURUSD M1 2024 ZIP, digest
`58f074ba6b835eb2fbeaeb6678502a23b26768eefdfa509eb75ff5296bb2904d`.
The API accepts an explicit digest so tiny, separately pinned synthetic archives
can exercise the same path; production callers must pass the approved digest.
No fallback source or invalidated financial-data CSV is used.

`input_dir` supplies `INPUT.npz` (`ts`, `price`) and `CALENDAR.json`. All retained
2024 rows must match an observed source endpoint and close, with absolute
tolerance `1e-12` and relative tolerance zero. Earlier input rows are counted but
not reconciled by this 2024-only source. Any retained timestamp at or after
2025-01-01 is refused before prices are loaded. Optional `PREPARATION.json`,
when present, must name exactly one matching 2024 archive pin. Optional
`CLOCK.json` is retained as an unchanged document plus its file digest in the
reconciliation evidence, not promoted to certification.

The reader consumes one bounded semicolon CSV directly from the verified ZIP.
Only source year 2024 is accepted. Minute timestamps must be aligned and ordered;
identical adjacent duplicates are explicitly removed, conflicting duplicates
are rejected. All minute OHLC must be finite, positive and coherent.
Aggregation is first observed open / maximum high / minimum low / last observed
close, grouped by source hour. Unobserved hours are not filled or smoothed.

The existing `tools.i6n_external_clock_audit_v2.eu_clock_series` maps the observed
hourly endpoints: `(wall + 5h)` localized in Europe/London, converted to UTC,
then `+1h` for bar end. Endpoints must remain ordered, unique and pre-2025.
Clock assumptions and nominal calendar uncertainty remain explicit; the receipt
does not claim broker verification or full-year historical completeness.

## Outputs And Reuse

- `STRATEGY_BARS.npz`: `close_time` int64, plus float64 `open/high/low/close`.
- `CALENDAR.json`: byte-identical copy of the retained calendar.
- `RECONCILIATION.json`: source/input/helper/code/output hashes, counts, gaps,
  missing/mismatched retained endpoints, absolute errors and inherited limits.

No existing directory is repurposed. A rerun recomputes all content and accepts
an existing directory only if its exact three files have the regenerated hashes;
it writes nothing in that case. Any extra, missing, changed or symlinked output
is a conflict, with no overwrite. New directories are private and files are
created exclusively. An interrupted partial publication is refused on a later
run rather than repaired or overwritten automatically.

## Regeneration

Set `$RETAINED_INPUT`, `$ARCHIVE` and `$OUT` to the privately retained input
directory, approved archive and preparation output directory. Actual local
paths and execution proof remain private. Run from the shared audit checkout,
with CPU affinity restricted to two allowed CPUs where required by the operator:

```bash
(
  ulimit -v 1048576
  export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  timeout 120s python -m tools.i6dmo.strategy_inputs \
    --archive "$ARCHIVE" \
    --expected-sha256 58f074ba6b835eb2fbeaeb6678502a23b26768eefdfa509eb75ff5296bb2904d \
    --input-dir "$RETAINED_INPUT" \
    --out "$OUT"
)
```

Do not regenerate after changing governing code into the same output directory:
the receipt pins its generating module and clock helper, so changed code is an
intentional conflict. Use a distinct private output directory for a successor.

## Verification

Test design preceded implementation: initial collection failed at the expected
missing-module import. The synthetic suite then passed under two CPU affinity
slots, 4 GiB virtual memory, hidden CUDA, one BLAS/OMP thread and a 60-second
timeout. `--noconftest` avoids the parent fixture's TensorFlow import:

```bash
python -m pytest --noconftest tests/i6dmo/test_strategy_inputs.py \
  tests/i6dmo/test_raw_pair.py -q
```

Observed: **60 passed**, including 35 preparation tests and 25 RAW pair tests.

| Requirement | Evidence in `test_strategy_inputs.py` |
| --- | --- |
| Verified bounded original source only | digest refused before ZIP open; member/schema/year checks |
| Observed coherent OHLC, no fabricated hours | aggregation/gap assertions; invalid OHLC cases |
| Approved clock, no future-hour leakage | winter and summer anchors; helper spy; future perturbation |
| Complete retained 2024 close reconciliation | missing endpoint, wrong price, retained ordering/TEST failures |
| Preserve uncertain provenance | calendar bytes and CLOCK uncertainty; preparation archive disagreement |
| No overwrite, deterministic rerun | hash/mtime idempotency; changed/extra/empty outputs; input drift |
| Usable finite CLI | subprocess CLI preparation with timeout; no model import |

The actual private preparation ran once after the tests, with a 1-GiB virtual
memory limit, 120-second timeout and two CPU slots. Observed duration **4.83 s**;
peak RSS **124624 KiB**. It consumed 372379 observed minutes, emitted 6243 hourly
bars, matched all **6243 retained 2024 closes exactly** (max error 0), and recorded
2517 unobserved hours between its first and last observed endpoints. There were
no missing/mismatched retained rows and no duplicate source minutes. All 29155
earlier retained rows were explicitly excluded from this 2024 reconciliation.
The original INPUT and PREPARATION digests matched the existing matched seal.

Output pins for the parent's private `raw_pair` configuration:

| Object | SHA-256 |
| --- | --- |
| `STRATEGY_BARS.npz` | `2bd68f8bc3f0e0b8a37c1370efee7047c7d2339929f88b93d728c9e602593224` |
| `CALENDAR.json` | `782603fd26e879d6f837c83e9ec92d363500cb1dd150544e10a80c1ddeb88265` |
| `RECONCILIATION.json` | `744498d6ac172b9b65ff2a65754692941f38c30da068793d3b7467e97a1e3756` |

Next action is independent actual-path review, then the parent's freeze of
`raw_pair.strategy_bars` and `raw_pair.strategy_calendar` with these file pins.
No deployment, trading performance, broker certification or TEST eligibility
is established by this preparation.
