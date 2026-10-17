# Best Forecast Registry Operation

Fixed epoch2 checkpoints from completed studies are now retained in
DIAGNOSTICS.json and the diagnostic section of LEADERBOARD.md, with actual
checkpoint-file/prediction/weight identities and their parent training history.
They publish under I6DMO_DIAGNOSTIC_CHECKPOINT with exact readback, never as
selected champions. Their metrics are recomputed from saved arrays. This
preserves useful development observations without changing checkpoint selection,
reading TEST or retrospectively promoting a favorable epoch.

This is a CPU-only sidecar for the retained I6-DMO outputs, not a trainer or
new optimizer. Keep the scientific runtime and configs unchanged. Requirements
and test mapping are in [PLAN.md](PLAN.md); recovered reference in [RETURN.md](RETURN.md).

The current private spec is `/var/tmp/i6dmo/best-registry-20261016.private.json`.
It contains the absolute roots, source code identities, pinned reference id,
warehouse URL/token-environment name and mirror destinations. The actual token
is inherited by systemd through the existing EnvironmentFile, never read by the
installer or written into this spec. Specs are mode 0600. Do not commit them.

Installation is idempotent and affects only `i6dmo-best-registry` units:

```bash
CUDA_VISIBLE_DEVICES='' python -B -m tools.i6dmo.best_registry install \
  --spec /var/tmp/i6dmo/best-registry-20261016.private.json
```

It is already installed. The timer performs bounded passes every 120 seconds,
using one BLAS thread, at most 50% of one CPU and 768 MiB RAM, no swap, with a
600-second timeout. The private spec seals the registry source-file hash;
unexpected code changes fail closed instead of changing the indexing protocol.
No agents need to monitor it. Failed deliveries remain in the existing durable
outbox; the next tick retries delivery without training or duplicate facts.

One read, no recomputation:

```bash
python -B -m tools.i6dmo.best_registry status \
  --spec /var/tmp/i6dmo/best-registry-20261016.private.json
```

Open `/var/tmp/i6dmo/best-registry-20261016/LEADERBOARD.md` for the current
rankings and `REFERENCE.json` for the owner's pinned recipe. Verified JSON/
Markdown/event copies go to the configured local large-volume archive and a
second host. Live SQLite, WAL, locks, private specs and credentials never go to
that archive. Original model/checkpoint locations are not moved or deleted.

OLAP records use phase `I6DMO_BEST_REGISTRY`. The `gov_metric` rows retain the
event score and, for new events, the horizon-specific MAE, naive and counts.
The associated `gov_report.tags_json` includes full `profile_json` and
`event_json`, plus their content/cohort identities. Native profile metrics and
common-support event metrics carry different explicit scope labels. The
initial recovery events predate the expanded MAE fields; their full original
MAE/naive/counts remain in `profile_json`, never guessed or overwritten.

`READY` requires both metric and full-provenance readback and two verified
mirrors. `WAIT_PUBLICATION`, `PUBLICATION_MISMATCH`, `WAIT_MIRRORS` and source
quarantines cannot silently become a clean closure. A failed pass preserves
the previous champions. Each valid new profile retains its seed, exact model,
input/scaler/config/code identity, early-stop formula and data roles, patience,
threshold, epoch limit, chosen epoch and weights.

Verification, without TensorFlow or datasets:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 python -B -m pytest -q \
  tests/i6dmo/test_best_registry.py tests/i6dmo/test_short_reference_inventory.py
```
