# The financial resource: what blocks it, what is actually known, and the matched reference it still lacks

Status: investigation and design, 2026-09-23 (RP143). **No byte of the financial resource was read for this document, and
the reserve from 2025-01-01 was not touched.** Nothing here authorises a fit.

## 1. The exact blocker, in the service's own words

The sealed financial cost pilot was refused externally, not abandoned. The operation was a governed ranged download of
`financial_files/market_data/forex/g10/eurusd/1h.parquet` for 2023-06-26 to 2024-06-23, under run-id
`satoshi-fin-cost-pilot-20260921`. The service answered **HTTP 422, "resource availability contract required"**: the deployed
lake declares no resource contract for that resource and refuses a timed delivery without one. Bytes read: zero.

That contract is **operator metadata**, and the runner is built not to invent it. What the client expects to receive is
visible in `tools/governed_run.py`, which records per delivery an `availability_contract_sha256`, an `availability_use`, an
`availability_label`, an `availability_completion_lag_max`, a `time_column` and `timezone_evidence`, and refuses a download
whose availability contract digest is missing. A delivery that cannot carry those fields cannot bound what a later result may
claim, which is why the refusal is correct and must not be worked around.

## 2. What the contract must state, and what evidence exists for each field

| field | what it must say | evidence today |
|---|---|---|
| event time column | which column is the bar's own timestamp | the parquet's schema, readable without opening the reserve; **not yet inspected** |
| timezone | the zone those timestamps are in | **disputed**, see section 3 |
| frequency | the bar period and whether gaps are absent or merely unrecorded | 129,873 hourly bars over 2005-01-03 to 2025-12-31 are recorded in the design; gap semantics are **unestablished** |
| available time | when each bar became observable to a consumer | **absent**: no producer record states a publication time or a completion lag |
| completion lag max | the worst delay between event and availability | **absent**, and it must not be assumed to be zero |
| availability use and label | how a consumer may use the data given that lag | derivable only once the two rows above exist |

Three of the six are unestablished, and two of those three are exactly the ones a causal claim depends on. This is why the
correct next step is an operator declaration backed by producer evidence, not a contract written by us to unblock a pilot.

## 3. The producer lineage, stated as what is known and what is not

The RP89 review established a real defect in the **inspected parser**: it treats a fixed EST timestamp as UTC and drops
volume, so a source label of 00:00 becomes 00:00 UTC rather than 05:00 UTC under the
[HistData file specification](https://www.histdata.com/f-a-q/data-files-detailed-specification/). That is a demonstrated
behaviour of the parser as it stands.

What it is **not**: evidence about the historical producer of every retained file. The retained parquet may have been written
by that parser, by an earlier one, or by another path entirely. Until the producing revision of the retained file is
established, a timezone correction applied to the real data would be a guess dressed as a repair. The order of work is
therefore: establish the producer of the retained bytes, then decide whether a correction is needed, then declare the
contract. Installing metadata first, as the review said, is insufficient.

An honest fourth possibility must stay open: that the retained file's provenance cannot be established at all, in which case
the resource is re-acquired from the source under a parser whose behaviour is specified and tested, rather than patched.

## 4. The matched financial reference, which does not exist yet

The benchmark contract `fx.eurusd.1h.FIN-LOSS-OPT` is **NOT_COMPARABLE** until a reference method is re-executed under it.
Nothing in the electricity work supplies one: the electricity ranking does not choose the financial winner, and there is no
financial winner today. The design of the contrast itself is already sealed and unchanged, and this document does not touch
it: MAE and Huber against Adam and AdamW, paired seeds, two receivers never compared across, weekly walk-forward folds with
weekly retraining, horizons declared in advance at 6 and 72 hourly steps, skill against persistence on the same rows, and the
reserve never read.

What a matched reference has to satisfy here, by the same standard the electricity reference was held to:

1. **Published, with a protocol we can execute.** A method with a paper and code, evaluated on hourly FX returns or prices
   with a stated split, a stated target transform and a stated reduction. A method whose protocol we would have to guess is
   not a reference.
2. **The same task as ours.** Hourly EURUSD, the same target construction and the same evaluation rows. A daily-bar or
   tick-level result is a different task and cannot be borrowed by name.
3. **Re-executed by us under the contract**, not cited. The electricity lesson is the whole point: a published number becomes
   a comparator only after it is reproduced under the frozen protocol, and a table whose lookback is unresolved stays
   qualified even after its script is executed.
4. **Matched naive and seasonal controls on the identical rows**, reported beside it, exactly as the electricity table does.

Until that reference is re-executed, any financial result we produce is a measurement without a comparator, and must be
labelled that way rather than compared to a number from another protocol.

## 5. What can proceed now, and what cannot

Can proceed, without reading the resource: inspecting the parquet **schema** and row count to fill the first three contract
fields; establishing the producing revision of the retained file from the repository's own history, the way RP138 located a
catalog implementation by hashing function bodies across revisions; drafting the contract for the operator to declare; and
selecting candidate reference methods from the domain literature with their protocols written down before any of them is run.

Cannot proceed: any fit, any delivery, any timezone correction to real data, and any opening of the reserve. The pilot stays
sealed and unchanged, and re-runs under a fresh run-id once the operator declares the contract.
