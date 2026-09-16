# I1 evidence held outside the repository

Identities and row contents stay out of the checkout. What is recorded here is where they are
and what they hash to, so the private files can be checked against these receipts.

| file | sha256 (first 16) | what it is |
|---|---|---|
| `~/.local/state/crispdm-duckdb/rehearsal/i1_live_evidence.duckdb` | `fc972294c5374353` | the frozen live cube, `CONSISTENT_LIVE_READ` |
| `~/.local/state/crispdm-duckdb/rehearsal/i2_rehearsal.duckdb` | `75f818d9c713d564` | the same, after the rehearsed repair: 533 metric rows |
| `~/.local/state/crispdm-duckdb/prod/I1_LIVE_ROWS.json` | `776e83c560b85d95` | the row multisets as data |
| `~/.local/state/crispdm-duckdb/prod/cube.duckdb.wal.quarantined` | `d28fa82f42b278fb` | the incident log, still never opened |

The live cube's `gov_terminal_metric` content digest, as the service itself reports it, is
`85d650b83cdb52243a15712e15b699d2` over 537 rows — identical to the source measured in
`H2_SNAPSHOT.json` at 18:19:07Z. Production was not modified by this work.
