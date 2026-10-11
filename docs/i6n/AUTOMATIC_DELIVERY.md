# Automatic result delivery

The coordinator runs `tools.i6n_search_follower` once per timer activation,
under 2 GiB / 120 s resource admission. Each activation terminates. The
timer runs again 60 seconds after the previous activation ends; an exclusive
lock prevents concurrent copies/publications. No monitoring agent is needed.

Private JSON, kept outside Git, contains:

```json
{
  "pin": "<executed-code-commit>",
  "url": "<warehouse-url>",
  "token_file": "<local-mode-0600-token-file>",
  "entries": [
    {"host": "<ssh-worker>", "root": "/absolute/retained/root", "kind": "search"},
    {"host": "<ssh-worker>", "root": "/absolute/pilot/root", "kind": "pilot", "pin": "<pilot-code-commit>"}
  ]
}
```

Remote and coordinator absolute artifact roots must agree because source
bindings contain absolute paths. Copy the pinned source arrays and base configs
to the coordinator once; the follower copies terminal directories without
deleting local receipts, excludes temporary files and lock files, and never
changes remote state. The warehouse credential stays on the coordinator.

```sh
python -m tools.i6n_search_follower --config /absolute/FOLLOWER.private.json
python -m tools.i6n_search_follower --config /absolute/FOLLOWER.private.json --status
```

Only COMPLETED terminals publish. Pilot results are rescored by the original
C2 verifier; search results additionally recheck source support and the retained
TRAIN/inner receipts. Search metrics are labelled TRAIN and inner, never outer
validation. Reports preserve the horizon-specific naive and owner skill sign.
Every report receives exact OLAP readback before the local delivery receipt is
written. A matching prior receipt skips redundant delivery. This assumes trusted
local storage and does not authenticate a coherently rewritten directory.

Transport/publication failures remain PENDING_DELIVERY and retry on the next
timer activation. They cannot retrain a model, consume a new candidate, advance
the scientific stage, or turn an incomplete result into success. Failed model
terminals are retained in the copied controller state, not converted to scores.

The pilot path completed its first live cycle with one terminal delivered and
zero errors on 2026-10-11. Search delivery is covered by projection and failure
tests; its live acceptance requires the first completed search candidate.
