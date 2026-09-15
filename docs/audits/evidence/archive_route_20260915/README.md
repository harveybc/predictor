# The whole route for a retrospective archive, on a disposable stack (R4, §4)

Executed on **gamma**, 2026-09-15, against a stack built from the **candidate** packages:
`financial-data-store` (the branch that carries `ARCHIVE_RETROSPECTIVE`), `predictor-olap-store`,
`data-lake-service`, `data-warehouse-service` and `data-gov`, all installed into gamma's own
venv. Free ports, throwaway SQLite, `PG*` stripped. The deployed catalogue was never touched
and no production service was restarted.

One fixture (`synthetic_typical_price_4h_train.csv`) was re-declared in the manifest as a
retrospective archive — `completion_lag_max: "UNKNOWN"`, `use_class: ARCHIVE_RETROSPECTIVE` —
and the other six contracts were left exactly as generated, so the regression is real.

## What the route did

| case | outcome |
|---|---|
| whole archive delivery | **200**, `VERIFIED_TRANSFER`, 1,416 bytes, terminal **201**, reconciliation empty on all three lists |
| ranged delivery over the archive | **422**, refused with the declared reason: *"this resource is a retrospective archive: its publication time was never observed, so a date range over it would assert availability that no evidence supports; request the whole resource"* |
| an already-deployed synthetic contract, whole | **200**, 12,393 bytes — the change is additive |

## UNKNOWN survived, and here is exactly how

The fixture's contract says `completion_lag_max: "UNKNOWN"`. Its canonical digest is

    62965febdc580f40d5ec21c45d44a9ae78258fcebddbbe6db985cf9d37a73f6d

and that same digest appears in **both** persisted stores:

* data-gov's accounting: `governed_deliveries.availability_contract_sha256`;
* the warehouse cube: `gov_terminal_dataset.availability_contract_sha256`.

So UNKNOWN survives **by reference**, cryptographically bound: the digest can only have come
from a contract whose lag is the string `UNKNOWN`, and altering that lag to a number would
change the digest and break the match.

Stated plainly, because it is the limit of this evidence: the cube stores the **digest**, not
the lag itself. Reading the cube alone, an operator cannot tell that this delivery was a
retrospective archive without resolving the digest against the contract. That is a real gap in
the persisted schema, not a defect of the class, and it is reported rather than papered over.

## What was not done

The real archive policy was not installed and the real financial resource was not licensed or
served. The disposable stack was torn down; nothing of it is left running on gamma.
