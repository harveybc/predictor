# Usage rights, second pass: the primary document, read

S3 of `docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md`. This
supersedes the unresolved parts of `SATOSHI_MARKET_DATA_RIGHTS_RESEARCH_2026_09_15.md`; that
document's earlier conclusion is **withdrawn** below, in its own section, rather than edited
out. Nothing here is a legal acceptance and no rights are granted or claimed.

## The document, identified

Retrieved 2026-09-15T20:49:41Z with no credential and no account, HTTP 200, 1,362,587 bytes,
sha256 `bf4879710c904b991848972ec4818ba2cf9e4ce314c09adae84fa2750d3477f7` — identical to the
URL's own basename, so the bytes are exactly the ones the review pointed at. 73 pages
(`pdfinfo`; `file(1)` reports 8 and is wrong — recorded because it is reproducible).
Identity, not a copy, is in `docs/audits/evidence/terms_20260915/SOURCE_IDENTITY.txt`.

Its internal title is **"ADGM Binance Global Terms of Use"**, PDF metadata dated 17 July 2026,
with **"Effective Date: 21 July 2026"** on its first page.

## What it is, and who it binds — the first thing that changes the answer

This is not a general data licence. The opening text states it is an agreement between the
reader and three named ADGM entities (NEST EXCHANGE LIMITED, NEST CLEARING AND CUSTODY
LIMITED, NEST TRADING LIMITED), each regulated by the FSRA of the Abu Dhabi Global Market, and:

> "This Agreement (as defined at Clause 1.3 below), governs your use of your Binance Account
> and any other Binance Services made available to you on or through the Binance Platform."

Our acquisition and our 2026-09-15 comparison used `GET https://api.binance.com/api/v3/klines`
with **no account and no key**. Whether a document that governs "your use of your Binance
Account" reaches a keyless public read by a party with no account is exactly the question, and
this edition does not answer it. Two further facts bear on it and neither is favourable to a
quick conclusion in either direction:

* **Its effective date is after our acquisition.** 21 July 2026 post-dates the 2026-05-01 pull.
  Whatever governed that pull is a different, earlier document that I have not obtained. The
  review's instruction not to assume the July edition governed the May acquisition is correct,
  and I am not assuming it.
* **It routes API access to a different document.** Clause 14.1.2 lists how an account may be
  accessed, including at (b): *"Binance APIs, subject to separate API terms and our approval"*.
  So even for an account holder, this text defers API use to terms it does not contain. Those
  separate API terms are the next primary source and I have not read them.

## Clause 27, verbatim

Under the heading **INTELLECTUAL PROPERTY**, clause 26 states "The Binance IP shall remain
vested in Binance." Clause 27, **LICENCE OF BINANCE IP**, reads in full:

> "We grant to you a non-exclusive licence for the duration of the Agreement, or until we
> suspend or terminate your access to the Binance Services, whichever is sooner, to use the
> Binance IP, excluding the Trade Marks, solely as necessary to allow you to receive the
> Binance Services for non-commercial personal or internal business use, in accordance with
> the Agreement."

"Binance IP" is defined in the definitions section as the Created IP and other Intellectual
Property Rights owned, acquired or licensed by Binance.

## The incorporated-document hierarchy, clauses 1.3 and 1.4

Clause 1.3 makes the Agreement these Terms **plus** documents incorporated by reference: Core
Policies (Privacy Notice, Risk Warning, Prohibited Use Policy, AI Policy, Global Community
Guidelines, Live Chat Policy, the Fee Structure page); Rulebooks (Exchange Rules, Clearing
Rules); **Product Terms** under clause 4.1; and "Any other agreement that expresses to form
part of the Terms of Use."

Clause 1.4 sets precedence from highest to lowest: Exchange or Clearing Rules; Contract
Specifications; Notices; Exchange or Clearing Procedures; **Product Terms**; **these Terms**;
Core Policies. These Terms therefore sit *below* Product Terms, which is why a product-specific
document could narrow or widen clause 27 for a particular service, and why "the general terms
say X" is not by itself an answer.

## Clause 33.11, which is about market data and cuts the other way

> "ADGM Binance Entities may also make available third party indices and/or other market data
> provided by third parties to you through the Binance Platform, in each case for information
> purposes and as an accommodation to you. You acknowledge and agree that each ADGM Binance
> Entity: (i) does not guarantee the accuracy, completeness or availability of any information
> or data provided by third parties […]"

This is a disclaimer, not a grant, and it concerns third-party data made available through the
Platform. It is recorded because it is the only clause in this document that speaks about
market data at all, and because it independently supports keeping our revision policy at
UNKNOWN: the producer disclaims that its data is up to date as at the time obtained.

## The conclusion I withdraw

The earlier report said our present use "does not touch the restrictions". That was an
inference from paraphrases, and the review is right that it is unsupported. **Withdrawn.**
Three reasons, now from the actual text:

1. Not selling something does not establish permitted use. Clause 27 is a *grant* with a scope:
   use of Binance IP "solely as necessary to allow you to receive the Binance Services". A use
   that is internal and unpaid can still fall outside a licence drawn that narrowly.
2. The phrase "internal business use" in clause 27 qualifies a licence to *receive the
   Services*. Reading it as a general permission for internal analysis of retrieved data is
   the same move I am criticising — filling a gap with the reading I prefer.
3. The applicable edition and entity for our acquisition are still unestablished, so no clause
   here has been shown to apply to it at all.

## The claim about published repositories, checked instead of repeated

The earlier report asserted that "the published repositories carry **no** market data rows".
I checked the tracked contents rather than repeating it, and **the assertion is false as
stated**. `harveybc/financial-data` is PUBLIC (`gh repo view`: visibility PUBLIC) and commits
bulk OHLC bar values:

| tracked file | rows |
|---|---|
| `trading_research/project2/part_II_redux/data/processed/btcusd_4h.csv` | 18,332 |
| `trading_research/project2/part_II_redux/data/processed/btcusd_4h_features.csv` | 18,307 |
| `trading_research/feature_store/BTC_USD_daily.csv` | 4,123 |
| `trading_research/project2/part_II_redux/data/processed/btcusd_daily.csv` | 3,059 |
| `trading_research/feature_store/ETH_USD_daily.csv` | 2,974 |
| `trading_research/project2/part_II_redux/data/processed/btcusd_weekly.csv` | 438 |

What is true, and is the narrower claim that survives: **the governed lake resource is not
published.** Under `market_data/` only `.gitkeep`, `.json` and `.md` files are tracked — 440
paths, zero `.parquet` and zero `.csv`. The ETHUSDT 4h bytes behind data-gov are not in any
public repository, and the acquisition's own raw parquet files (`data/raw/binance/`) are
untracked.

On the source of what *is* published, I report what the repository documents and no more. The
`btcusd_*` files appear in `TASK_II-7.1_DATA_ACQUISITION.md` under a heading **"A. Binance
OHLCV"**, and the note beside the 4h entry says it was "sourced from legacy file
`btcusd_4h_2017_2025.csv` (Part II historical data)" — so that file's upstream is *not*
established by the repository itself, only its neighbourhood. The two `feature_store` dailies
carry a `Volume` column in units consistent with a consolidated equity-style feed rather than
exchange base units, and no Binance origin is documented for them. I did not delete, move or
republish anything, per the order.

## Access and use, kept apart

**Access** is settled and unchanged: `/api/v3/klines` is listed by the producer among
market-data-only endpoints — *"These URLs do not require any authentication (i.e. The API key
is not necessary) and serve only public market data."* (*Market Data Only*, last modified
15 September 2026). Our reads used no key.

**Use** is not settled. A public, keyless read establishes that retrieval was permitted; it
establishes nothing about storage, derivation or redistribution afterwards.

## Our scope, restated so it can be judged against a clause

One historical REST pull into a private lake (2026-05-01) and three read-only requests on
2026-09-15 totalling 529,443 bytes; internal analysis; derived artefacts limited to digests,
counts, spacing statistics and anomaly classifications; the lake served only to our own
consumers behind data-gov authentication; no revenue of any kind depends on it; the producer
and endpoint named in every artefact describing the resource. The published-rows finding above
is stated as a fact about the repository, not folded into this scope.

## What is unresolved, and what would move it

1. **The separate API terms** referenced by clause 14.1.2(b). Unread; the next primary source.
   Two candidate locators were requested once each on 2026-09-15 —
   `/en/support/faq/binance-api-terms-of-use` and `/en/terms/api` — and both answered
   **HTTP 202 with a zero-byte body**, which is an interstitial, not a document. That is the
   measured state, not a claim that the terms are unavailable; a browser session or the
   producer's own PDF locator would settle it, as the PDF above did for the general Terms.
2. **The edition in force on 2026-05-01**, and whether any of it reached a keyless read.
3. **Which entity's terms apply** to `api.binance.com` for our jurisdiction. This document is
   the ADGM entities' text; it is not established as the applicable one, and the US entity's
   document, which is different again, was deliberately not substituted.
4. **The published Binance-adjacent rows above.** This is the one item that is genuinely a
   question for the owner rather than for me: whether 47,233 committed bar values in a public
   repository are within whatever terms applied when they were acquired. I am not qualified to
   answer it and I have not acted on it. Nothing in S1 or S2 depends on it.

No third party was contacted, and no agreement was accepted.
