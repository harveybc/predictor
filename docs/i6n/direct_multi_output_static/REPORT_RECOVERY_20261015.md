# Market-time report correction and completed-fit recovery

The released MG2 fit completed, but short.report failed with
`UNKNOWN_DESIGN_FIELD: ['market_time', 'sealed_identities']`: its default
loader still passed the sealed market-time design into the elapsed loader.
Both market families reproduce this failure through the real report entry.

The corrected report authenticates the market bundle and rebuilds validation
through the market calendar table. It retains exact elapsed targets, current-t
inputs, full expected weeks, calendar identity and the existing C2/manual/disk
naive comparison. No arrays-loader override or scientific refit is needed.

`init --completed-short-fit-source <source-root>` creates a new campaign for
the repair. It verifies the source terminal and every artifact digest, exact
short/long scientific contracts, data pins, seed, cadence and entrypoints.
An allowlist rejects changes to training/model/data files or JSON designs.
The allowlist permits reviewed orchestration/reporting repair modules and
tests/Markdown documentation; it is a bounded repair contract, not a generic
claim that arbitrary future changes to those modules preserve training.

The reused short fit retains its exact original unit identity and code pin.
All newly executed units carry the successor code pin. The new graph records
the import and marks the fit ADOPT_ONLY; a missing or invalid imported fit
cannot trigger training. The source campaign and its failed report survive.
The complete finite graph then scores the short model, measures the long
model, scores it and applies the declared pair/strategy gates.

Verification: before correction, the new real-entry MG2/MG12 report tests
both fail with the production error. After correction, 27 report/deployment/
identity/TEST-closure tests pass. Completed-fit recovery tests verify unchanged
fit identity, correct downstream code identity, one original short fit plus
one successor long fit, rejection of changed scientific contracts, corrupted
source terminals and training code changes. All executions are bounded CPU
jobs; synthetic regression predictions are not scientific measurements.
