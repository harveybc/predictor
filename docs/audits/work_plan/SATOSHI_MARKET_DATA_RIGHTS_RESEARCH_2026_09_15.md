# Usage rights for the ETHUSDT market data: what the primary sources say, and what they do not

§6 of `docs/handoffs/MUSASHI_WORKER_ACTIVATION_AND_R1_R6_COMPLETION_2026_09_15.md`:

    "primary-source research is Satoshi's unfinished task first. Supply the actual applicable
     source, dated text/section, scope of analysis, derivatives, redistribution and
     uncertainties. Only a demonstrated need for an agreement, entitlement or legal
     clarification goes to the owner."

This was mine to do, and it had been sitting as an "owner action" for two rounds. It should
not have been. Below is what the sources actually say, what I could not obtain, and the one
thing that would need someone with authority to resolve — which is **not** yet demonstrated.

## The endpoint we used, and what governs it

Our acquisition and the 2026-09-15 comparison both used

    GET https://api.binance.com/api/v3/klines

`/api/v3/klines` is listed by the producer among the **market-data-only** endpoints. The
producer's own page states, verbatim:

> "These URLs do not require any authentication (i.e. The API key is not necessary) and serve
> only public market data."

— *Market Data Only*, developers.binance.com/docs/binance-spot-api-docs/faqs/market_data_only,
**last modified 15 September 2026**. It lists `GET /api/v3/klines` explicitly, alongside a
dedicated host `data-api.binance.vision` for unauthenticated market-data access.

That settles the **access** question: the read is public by the producer's own description,
no account key is involved, and our comparison used no key. It does **not** settle what may be
done with the bytes afterwards.

The SPOT API documentation points the licence question elsewhere, verbatim:

> "Binance products and services are subject to the Product Terms of Use. Please read it
> carefully before proceeding."

— *SPOT Exchange Terms of Use*,
developers.binance.com/docs/binance-spot-api-docs/PROD-TERMS-OF-USE, **last modified 15
September 2026**, linking to `binance.com/en/terms`.

So the governing document is the **general Product Terms of Use**, not a separate data licence.

## What I could not obtain, stated as a limit rather than filled in

I could not retrieve the substantive clauses of `binance.com/en/terms`. Fetching it returns
the page shell — footer navigation, a licensing notice and a risk warning — with the body
rendered client-side. The same is true of the regional variant `binance.com/en-AE/terms`;
`binance.com/en/legal/list` returns HTTP 404.

A web search surfaced **paraphrases** of two restrictions attributed to those terms: a
prohibition on services that charge for or otherwise profit from Binance market data, and a
broad clause forbidding, without prior written consent, that one "modify, replicate, duplicate,
copy, download, store, further transmit, disseminate, transfer… or create their derivative
works". **I am not presenting those as quoted terms.** I do not have the section number, the
heading or the verbatim sentence from the document itself, and a paraphrase of a legal clause
is not the clause. Recording it as if it were would be the same error as inventing a
completion lag.

## Our actual scope, stated precisely so it can be judged against any clause

| dimension | what we actually do |
|---|---|
| acquisition | one historical REST pull into a private lake (2026-05-01), plus **three** read-only requests on 2026-09-15 totalling 529,443 bytes |
| internal analysis | yes — characterisation, availability semantics, comparison by bar identity |
| derived artefacts | digests, row counts, spacing statistics, anomaly classifications. No bar values are republished |
| redistribution | **none**. The bytes live in a private lake behind data-gov; they are served to our own consumers under authentication and are not transmitted to third parties |
| public evidence | contains digests, counts and verdicts. The published repositories carry **no** market data rows |
| commercial use | none today. Nothing is sold, no service charges for this data, no advertising or referral revenue depends on it |
| attribution | the producer and endpoint are named in every artefact that describes the resource |

Against the two restrictions as *paraphrased*, nothing we do today profits from the data or
disseminates it to third parties. The word that would matter is "store": a literal reading of
the paraphrase would cover keeping a copy at all, which is plainly not how a public
market-data endpoint is used in practice — and precisely why the verbatim clause, not a
paraphrase, is what counts.

## Uncertainties, named

1. **The verbatim clauses are unread.** Until the actual text and section are obtained, the
   scope of "store", "disseminate" and "derivative works" for public market data is unknown.
2. **Which entity's terms apply** to `api.binance.com` for our jurisdiction is not established;
   several regional variants exist and the US entity's terms are a different document that I
   deliberately did **not** substitute.
3. **Whether the historical archive** in our lake was acquired under the same terms as the
   2026-09-15 comparison is not documented by the producer's pages; the acquisition script
   records the endpoint, not a licence acceptance.

## What this does and does not send to the owner

Not yet an owner action. No agreement, entitlement or legal clarification is *demonstrated* as
necessary, because our present use — internal analysis, derived statistics, no redistribution,
no revenue — does not touch the restrictions as far as they are known. Nothing about
synthetic testing or infrastructure adoption depends on this question, and none of it was
blocked on it.

It becomes an owner action the moment one of these is proposed: publishing bar values rather
than digests; serving this data to anyone outside our own systems; or any product or service
that charges for, or earns from, results derived from it. At that point the requirement is a
reading of the actual clauses by someone with authority to accept them — not my judgement.

## The comparison's own limits, preserved

The 2026-09-15 comparison is finite snapshot evidence: 3,000 bars, three windows, one moment.
Its definitions stay exactly as recorded — a *material* difference is a relative deviation
above 1e-12, and the 145 differing bars were classified as **representation** differences at
2e-16, one ULP of float64, arising because the archive stores the producer's decimal strings
as float64. Zero material differences is **not** a no-revision guarantee and is not to be
generalised into one.
