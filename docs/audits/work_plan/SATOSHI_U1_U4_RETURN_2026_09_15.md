# Return: U1–U4

Order: `docs/handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md` (`a7072d2`).
No production service was restarted, no cube history altered, no candidate deployed, no pin
moved, no P1LR unit re-enabled, no GPU, no financial experiment. Counts are mine.

## The finding I was wrong about, first

Musashi reproduced a real defect and the reproduction was correct. In a disposable database,
changing a contract's stored `canonical_bytes` from `UNKNOWN` to `0s` while leaving the key and
the cached columns alone made my reader answer:

    {"resolution": "RESOLVED", "returned_bytes_match_digest": false, "returned_lag": "UNKNOWN"}

I reproduced it before touching anything. The cause is exactly as stated: the reader returned
the view's row, and a `LEFT JOIN` proves that a key matched — it hashes nothing. My probe
hashed the returned bytes **in the probe**, which made the round look verified while the
reader verified nothing. That is the difference between a test that passes and a property that
holds, and I had the wrong one.

## U1 — outage accounting and the teardown

**The outage touched nothing.** 11 outbox roots, 97 envelopes: **zero** written or modified in
the interval 16:13:15 → 19:11:24 (-05:00). The data-gov journal over the same window holds
exactly three entries, all `GET /healthz`, all mine. I state that as a measurement, not as
"an outbox exists so nothing was lost" — which is the claim the order forbids and which I had
made in the previous return.

**One pending envelope, predating the outage by nine hours**, whose slot was already closed in
the cube. A stale marker, not a loss. Reconciled through `governed_exec.py --flush`, unmodified:
`{"failures": {}, "pending": 0, "sent": 1}`. Pending **1 → 0**; the campaign still has exactly
**one** row with `received_at` unchanged; the cube still holds **52** terminals — the number
Musashi measured independently. Duplicate protection held.

**Eighteen envelopes are named as unverifiable, not as reconciled.** All `sent`/`adjudicated`,
none pending, all from the throwaway stacks of 2026-09-13 whose SQLite cubes no longer exist.
They cannot be checked at all, and saying so is the honest state.

**The teardown.** `--teardown STACK.json` signals the process group each service was started
in, and refuses any PID whose `/proc/<pid>/cmdline` lacks that stack's own work directory, so
stale metadata after PID reuse yields `PID_REUSED_REFUSED` and the stranger lives. A **zombie**
is now read as gone; it was not, which is why teardown waited out the full grace and reported
`STILL_RUNNING` for processes that had already exited. Seven rules, including two stacks under
the **same module names** where tearing one down leaves the other serving and holding its data,
and a source check that no `pkill`/`killall`/`pgrep` exists in the harness. The survivor is a
second disposable stack. Dogfooded on the U2 route stack; all four production services
answered 200 afterwards.

Evidence: `docs/audits/evidence/outage_recovery_20260915/`.

## U2 — the reader verifies now

`resolve_delivery_availability` checks, in order: the delivery exists; it is not recorded
against two different contracts; a contract was retained; its digest algorithm and
canonicalization are ones this store can verify; **the bytes hash to the key they are filed
under**; the bytes parse as a contract; and the cached columns agree with them. Seven distinct
refusals, and **only** `VERIFIED` carries `use_class` and `completion_lag_max`, both derived
from the verified bytes rather than read from a column.

The view no longer implies a verdict: its reference column says `STORED`/`ABSENT` and every
semantic column is `stored_*`. A raw join may expose what the database holds; it may not
suggest it checked it.

On write: an unsupported canonicalization, a `use_class` this store cannot interpret, and a lag
that is not a duration are refused. `str({"hours": 4})` is a plausible-looking string that can
never be compared again, so the shape is checked instead of coerced.

**Tests: 20 rules × 2 engines = 40 green**, SQLite and a disposable PostgreSQL. The drifted
bytes now yield `UNRESOLVED_DIGEST_MISMATCH` with no claim; the cached-column disagreement, the
unsupported format, the malformed contract, the ambiguous delivery and the agreeing repeat each
have their own rule. The whole route was re-run on a disposable PostgreSQL stack: producer
killed, configuration deleted, and only then the fresh reader answered `VERIFIED` /
`ARCHIVE_RETROSPECTIVE` / `UNKNOWN`, with an independent second hash in the receipt agreeing
with the reader's own.

**Production candidate:** `SATOSHI_S2_PRODUCTION_CANDIDATE.md` — exact revisions for the four
components, the additive migration, backup commands, the no-history-change checks (counts plus
an md5 over all terminal digests, which must not move), and rollback. `PENDING_REVIEW` stays
set and `SOURCE_SHA256` still names the **deployed** revision. Neither was moved to make a
label green. The deployment decision is Musashi's.

## U3 — the public rows

**47,233 rows is confirmed** by independent measurement from the published revision
`d9be1b368a073aa66877208f6a003d8594394bf8`, with a per-file digest, header, span and count.
But rows are not observations: the union of distinct **calendar days** across all six files is
**4,125**. Four of the six are one BTC series at three resolutions plus a derived feature table
over the same days. Two instruments in total.

**Provenance from published code, not from a heading.** `fetch_binance.py` hits
`api.binance.com/api/v3/klines` for `BTCUSDT` 4h and 1d into `data/raw/binance/…`, and
`consolidate_data.py` reads exactly that file to produce the four `btcusd_*` outputs. The two
`feature_store` dailies are **Yahoo-derived** (`yf.download` on `BTC-USD` / `ETH-USD`) and
`BTC_USD_daily.csv` starts 2014-09-17, three years before BTCUSDT existed — so they are
outside the Binance question entirely. None of the six is the governed lake resource.

**Two corrections to my own reasoning**, both as the review required:

* I used "governs your use of your Binance Account" to suggest the terms might not reach a
  keyless read. The same sentence continues to "any other Binance Services made available to
  you on or through the Binance Platform". **Withdrawn.** It does not establish applicability
  either; the incorporated documents, precedence and the separate API terms remain unread.
* I justified the PDF's identity by its URL basename. The digest **was** computed independently
  from the received bytes; the basename agreeing is consistent with content-addressed hosting
  and is not what makes it evidence. Phrasing corrected.

The unresolved question is stated with five possible remedies, of which one — establishing
which edition and which API terms applied at acquisition — is research and mine.

## Suites actually run, with their exclusions

| suite | result |
|---|---|
| predictor `tests` + `olap/store/tests` | **1395 passed, 2 skipped, 0 failed** in 6m36s |
| — excluded: `tests/unit_tests`, `tests/integration_tests` | stale per AGENTS.md; they fail at import against the current architecture |
| data-gov | 167 passed |
| data-warehouse | 28 passed |
| data-lake | 33 passed |
| financial-data `store/tests` | 22 passed, 2 skipped |
| agent-multi governed set (5 files) | 38 passed |

Exact command for the predictor line, including the environment that decides what runs:

```
U2_PG_DATABASE=<disposable> PGDATABASE=<disposable> \
ARCHIVE_PROVIDER_SRC=<financial-data worktree>/store/src/financial_data_store \
CUDA_VISIBLE_DEVICES="" crispdm-run -m 4G -t 1800 -n u-final -- python -m pytest \
  tests olap/store/tests -q --ignore=tests/unit_tests --ignore=tests/integration_tests
```

`U2_PG_DATABASE` adds the PostgreSQL half of the dimension rules; without it they run on
SQLite only. `ARCHIVE_PROVIDER_SRC` points the archive rules at the candidate lake provider;
without it they resolve to the deployed one, which predates the class, and skip with that
path named. Both facts are in the files themselves.

No claim is made about any test not in that table.

## A race the focal run hid, found by running the whole suite

Two of my own new teardown rules passed alone and failed in the full run. The cause was not
the tests. Between `fork` and `exec` a live process has no argv, and under load that window is
wide enough to observe; `cmdline_of` read the empty argv as "already gone". In production that
means **teardown could skip a service it had just started** — the same class of leftover that
led me to the `pkill` in the first place.

Liveness now comes from the kernel's own state field (missing, or `Z`, is gone). An empty argv
on a live process gets a bounded wait and then `UNVERIFIABLE_IDENTITY_REFUSED`: not knowing
whose a process is has never been a reason to signal it. Nine rules now, and the tests wait for
`exec` instead of racing the scheduler.

This is the second time this round that a focal run hid something the whole suite showed. Both
times the correction was the one already given to me: verify in the code rather than in the
probe, and measure the suite rather than the part I wrote.

## Open, with owners

| item | owner |
|---|---|
| deploying the availability-contract candidate | Musashi |
| the edition and API terms in force at acquisition | Satoshi — blocked on documents a browser session would fetch |
| what to do about the published Binance-derived rows | owner / counsel; five remedies listed |
| the 16 P1LR screen records | deferred by order; dormant disposition DONE by Musashi |
| per-delivery capability tokens | deferred |
