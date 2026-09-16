# Production candidate: the availability-contract dimension

U2 of `docs/handoffs/MUSASHI_WAREHOUSE_RECOVERY_AND_S2_REVIEW_2026_09_15.md`:

    "Prepare one production candidate manifest binding exact host/provider/governance
     revisions, schema migration, backup procedure, no-history-change checks and rollback.
     Keep PENDING_REVIEW until that candidate is independently checked."

**Nothing here is deployed and no service was restarted to produce it.** The deployment
decision is Musashi's. This is the manifest, so that decision has something exact to act on.

## The revisions this candidate binds

| component | deployed today | candidate |
|---|---|---|
| `predictor-olap-store` provider | `6d5c9ed079b23b046167ceb8a510f7b2c36d9721` | predictor `a0b2311`, module sha256 `3833037c…` |
| `data-warehouse-service` host | `6f16565d515426e489c9bf4f8e22f5961de53176` | data-warehouse `6320897cd1c64f210edd121e4750018ec5ee6018` |
| `data-lake-service` host | current deployed revision, unchanged by this candidate | data-lake `7ac4ace1ba0a9b2b4dbb48394526cae88307d815` |
| `financial-data-store` provider | installed wheel, predates `ARCHIVE_RETROSPECTIVE` | financial-data `609bfdce17ebaa904cf9bce5d2cbf107845d88d4` |
| `data-gov` | current deployed revision | data-gov `01482f051a4bcdb37d6002b2a6094f0541efbb71` |

The four are **one candidate**, not four independent ones: the canonical bytes are produced by
the lake provider, published by the lake host, verified and retained by data-gov, and stored
and re-verified by the warehouse. Deploying the warehouse alone is safe but inert — every
delivery resolves `UNRESOLVED_REFERENCE_ABSENT`, which is correct and useless. Deploying the
lake side alone changes nothing that is stored.

## Schema migration

Additive only, applied by the provider's own idempotent `_ensure_schema` on first connection:

```sql
CREATE TABLE IF NOT EXISTS public.gov_availability_contract (
  contract_sha256 TEXT NOT NULL PRIMARY KEY,
  canonical_bytes TEXT NOT NULL,
  digest_algorithm TEXT NOT NULL,
  canonicalization TEXT NOT NULL,
  use_class TEXT NOT NULL,
  completion_lag_max TEXT NOT NULL,     -- TEXT: an archive's lag is the string UNKNOWN
  availability_label TEXT,
  timezone_evidence TEXT,
  first_seen TEXT NOT NULL);

CREATE OR REPLACE VIEW public.gov_delivery_availability AS ...;   -- new view
```

No `ALTER`, no `UPDATE`, no `DELETE`, no `DROP`, no backfill. `gov_delivery_availability` is a
new name and replaces nothing.

**Legacy rows are left unresolved.** Every delivery already in `gov_terminal_dataset` resolves
`UNRESOLVED_REFERENCE_ABSENT` after the migration, because no contract was ever retained for
it. That is the truth about those rows and inventing one would be the defect this dimension
exists to prevent.

## Backup, before anything is applied

```bash
pg_dump --format=custom --file=<state>/cube.before-availability-dimension.dump "$PGDATABASE"
psql -At -c "SELECT count(*) FROM gov_terminal"            > <state>/counts.before.txt
psql -At -c "SELECT count(*) FROM gov_terminal_metric"     >> <state>/counts.before.txt
psql -At -c "SELECT count(*) FROM gov_terminal_dataset"    >> <state>/counts.before.txt
psql -At -c "SELECT md5(string_agg(terminal_sha256, ',' ORDER BY terminal_sha256))
             FROM gov_terminal"                            > <state>/terminals.before.md5
```

The deployed provider revision and host revision are already recorded above; note the running
unit's `WorkingDirectory` and `--load_config` before changing anything, because rollback needs
them verbatim.

## No-history-change checks, after

The same four commands. All must be **identical**:

* the three counts unchanged — the migration adds no fact rows and removes none;
* `terminals.before.md5` unchanged — no terminal digest moved, which is the strongest single
  check, since a terminal's identity is the digest of its own body;
* additionally: `SELECT count(*) FROM gov_availability_contract` is **0** immediately after the
  migration. A non-zero count means something wrote contracts during the window and that is a
  question, not a success.

## Rollback

Fully reversible, in this order:

1. stop the warehouse host unit; restore the previously recorded `--load_config` and the
   previous provider revision in its venv;
2. `DROP VIEW IF EXISTS public.gov_delivery_availability;`
3. `DROP TABLE IF EXISTS public.gov_availability_contract;`
4. start the unit; `/healthz` must answer 200 and the four counts must match the before file.

The restore dump is the fallback if anything unexpected happened, but it should not be needed:
nothing in this candidate writes to an existing table. The old provider ignores the new table
entirely, so step 1 alone already restores previous behaviour and steps 2–3 are cleanup.

## What was proved, and where

* 20 rules × 2 engines (SQLite and a **disposable** PostgreSQL), 40 green:
  `olap/store/tests/test_availability_contract_dimension.py`;
* the reproduced drift Musashi found, now a rule — tampered bytes yield
  `UNRESOLVED_DIGEST_MISMATCH` and **no** availability claim;
* the whole route on a disposable PostgreSQL stack, with the producer killed and its
  configuration deleted before anything was read:
  `docs/audits/evidence/archive_contract_20260915/`;
* teardown of that stack performed by the new identity-checked mechanism, with all four
  production services answering 200 afterwards.

`PENDING_REVIEW` in `predictor_olap_store/__init__.py` stays set, and `SOURCE_SHA256` still
names the **deployed** revision. Neither is moved to make a label green.
