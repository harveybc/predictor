# The public market rows: what they are, where they came from, and the one open question

U3 of `docs/handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md`. Nothing was
deleted, moved, rewritten, re-published or changed in visibility. No third party was contacted.

## The inventory, measured from the published bytes

Repository `harveybc/financial-data`, visibility **PUBLIC**, published revision
`d9be1b368a073aa66877208f6a003d8594394bf8` (`origin/master`). Every figure below is read from
`git show <revision>:<path>` — the bytes as published, not the working tree.

| file | rows | cols | sha256 (first 12) | span |
|---|---|---|---|---|
| `…/part_II_redux/data/processed/btcusd_4h.csv` | 18,332 | 6 | `1e95bb93c526` | 2017-08-17 04:00 → 2025-12-31 00:00 |
| `…/part_II_redux/data/processed/btcusd_4h_features.csv` | 18,307 | 14 | `84e989a920f5` | 2017-08-20 08:00 → 2025-12-30 |
| `…/part_II_redux/data/processed/btcusd_daily.csv` | 3,059 | 6 | `ee840bf7d1a9` | 2017-08-17 → 2025-12-31 |
| `…/part_II_redux/data/processed/btcusd_weekly.csv` | 438 | 6 | `5e4737aff50d` | 2017-08-18 → 2026-01-02 |
| `trading_research/feature_store/BTC_USD_daily.csv` | 4,123 | 26 | `186a32d3c3cf` | 2014-09-17 → 2025-12-30 |
| `trading_research/feature_store/ETH_USD_daily.csv` | 2,974 | 26 | `10036a5438d0` | 2017-11-09 → 2025-12-30 |

Full record with headers: `docs/audits/evidence/public_rows_20260915/U3_INVENTORY.json`.

**47,233 rows is confirmed** by independent measurement. But the review is right that rows are
not observations, and the difference here is large:

* the union of distinct **calendar days** across all six files is **4,125**;
* `btcusd_4h`, `btcusd_daily` and `btcusd_weekly` are the **same series** at three
  resolutions — 3,059 / 3,059 / 437 overlapping days — and `btcusd_4h_features` is a derived
  feature table over 3,055 of those same days;
* two instruments are covered in total: BTC (2014-09-17 → 2025-12-31) and ETH
  (2017-11-09 → 2025-12-30).

So the published material is **one BTC series and one ETH series**, republished at several
resolutions and once as derived features. "47,233 rows" is arithmetically right and, used
alone, overstates how much distinct market information is public.

## Provenance, from published code rather than from a heading

The heading "A. Binance OHLCV" in `TASK_II-7.1_DATA_ACQUISITION.md` is **not** what establishes
this, and the review was right to say so — that same deliverable attributes the 4h file to a
"legacy file", which by itself proves nothing. The chain below is code that is published in
the same repository:

**The four `btcusd_*` files — Binance, established.**

```
scripts/fetch_binance.py       url = 'https://api.binance.com/api/v3/klines'
                               symbol='BTCUSDT', interval='4h' and '1d'
                               → data/raw/binance/btcusd_4h_2017_2025.csv
scripts/consolidate_data.py    # BTC from Binance
                               reads RAW_DIR/binance/btcusd_4h_2017_2025.csv
                               → process_crypto → processed 4h / daily / weekly
```

The published file's first bar, `2017-08-17 04:00:00`, matches that script's default start.
The raw parquet files themselves are **not** tracked; what is published is the processed
output of that chain.

**The two `feature_store` dailies — Yahoo Finance, established, and NOT Binance.**

```
trading_research/extend_history.py    YFINANCE_TICKERS = {"BTC/USD": "BTC-USD", …}
                                      yf.download(ticker, interval …)
trading_research/exogenous_data.py    ("BTC", "BTC-USD"), ("ETH", "ETH-USD")
                                      yf.download(…, interval="1d", auto_adjust=True)
```

This matters: `BTC_USD_daily.csv` starts **2014-09-17**, three years before Binance listed
BTCUSDT, which is consistent with the yfinance chain and inconsistent with the Binance one.
My earlier note guessed at this from the magnitude of the `Volume` column; the evidence is the
code and the start date, not the guess.

**Examples versus lake resources.** All six are files committed into a public repository as
research inputs. None of them is the governed lake resource: that resource is the ETHUSDT 4h
archive behind data-gov, and under `market_data/` the published tree tracks only `.gitkeep`,
`.json` and `.md` — zero `.csv`, zero `.parquet`. The two are different artefacts and the
earlier report was wrong to let one claim cover both.

## A correction to my own reasoning about the terms

I wrote that the general Terms govern "your use of your Binance Account", and used that to
suggest they may not reach a keyless read. **That inference was too strong and is withdrawn.**
The same first-page sentence continues past the phrase I stopped at: it governs the account
*and any other Binance Services made available to you on or through the Binance Platform*. A
document that reaches "any other Binance Services" cannot be set aside merely because no
account was used. This does not establish that it applies either — the scope, the incorporated
documents (clause 1.3), the precedence order (1.4) and the separate API terms that clause
14.1.2(b) points at all remain to be evaluated, and I have not obtained the last of those.

A second correction, smaller and mine: I wrote that the retrieved PDF's digest was "identical
to the URL's own basename, so the bytes are the ones the review named". The digest **was**
computed independently from the received bytes — `sha256` of the 1,362,587-byte file is
`bf487971…`. That is the fact. Its agreement with the basename is consistent with
content-addressed hosting but is not itself proof of anything, and I should not have phrased
it as the reason.

## The unresolved rights question, stated so it can be acted on

**Question.** Do the terms applicable at acquisition time permit publishing, in a public
repository, bar values derived from `api.binance.com/api/v3/klines` — specifically 18,332 4h
BTCUSDT bars and the daily/weekly/feature tables derived from them?

**What is established:** the endpoint is documented by the producer as market-data-only
requiring no key; the four files are Binance-derived by published code; the two
`feature_store` dailies are Yahoo-derived and are **outside** this question; the governed lake
resource is not published.

**What is not:** which edition and which entity's terms applied when that acquisition ran; the
content of the separate API terms (two locators answered HTTP 202 with an empty body); and
whether "store", "further transmit" and "derivative works", as the applicable text actually
words them, reach a public research repository.

**Possible remedies, for whoever decides — I am not choosing among them:**

1. **Establish applicability first.** Obtain the edition in force at acquisition and the API
   terms, and read them against this exact use. Cheapest, and may end the question outright.
2. **Reduce to what is needed.** If the research needs the *series* rather than the *values*,
   published digests, returns or normalised features would carry the analysis without
   republishing bars. This is a change to future work, not a deletion of the past.
3. **Change visibility of the affected paths.** Technically simple, historically messy — git
   history keeps the bytes — and it is a decision about a public repository, not an
   engineering fix.
4. **Seek permission or rely on a stated allowance**, if the applicable text provides one.
5. **Accept the position deliberately**, with the reasoning recorded.

Remedies 2–5 are the owner's call and some need counsel. Remedy 1 is research and is mine; it
is blocked only on retrieving documents that a browser session would get.

Synthetic governance work (S1, S2, U1, U2) does not depend on any of this, and none of it was
blocked on it.
