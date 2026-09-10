# Research data inventory

This directory contains the first executable CRISP-DM inventory for the data-centric work plan.

## Files

- `crispdm_dataset_registry.v1.json`: reviewed input registry. Unknown metadata is explicit.
- `crispdm_dataset_inventory.v1.json`: profiles derived from the registered CSV bytes.
- `financial_data_manifest_reconciliation.v1.json`: structural reconciliation of the external `financial-data/features/MANIFEST.json` declarations.

The financial reconciliation counts physical slices and declared column occurrences. It does not claim that repeated columns across symbols, sources, or timeframes are distinct conceptual variables.

## Reproduce

From the `predictor` checkout:

```bash
python tools/build_crispdm_inventory.py \
  --registry examples/research/crispdm_dataset_registry.v1.json \
  --root predictor=. \
  --inventoried-at 2026-09-10T00:00:00Z \
  --output examples/research/crispdm_dataset_inventory.v1.json

python tools/reconcile_financial_data_manifest.py \
  --financial-data-root ../financial-data \
  --inventoried-at 2026-09-10T00:00:00Z \
  --output examples/research/financial_data_manifest_reconciliation.v1.json
```

Use the project environment if the default Python does not provide pandas and SQLAlchemy.

## Load into an isolated OLAP database

```bash
python olap/init_db.py
python tools/load_crispdm_inventory.py \
  --inventory examples/research/crispdm_dataset_inventory.v1.json
```

Do not reset or migrate the populated OLAP database until an isolated database has passed schema, ingestion, count, and rollback checks.
