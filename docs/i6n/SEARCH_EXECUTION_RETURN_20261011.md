# Endpoint search execution return

Observed 2026-10-11 around 18:01 UTC. Runtime STATUS files remain authoritative.

## Running work

- Preferred GPU: h12 joint endpoint search, code `6f36f16e`; two completed
  candidates, zero failures at observation, continuing its bounded queue.
- Secondary GPU: h72 joint endpoint search, same code; first historical fit
  has recorded optimizer updates. Both controllers admit one child per host.
- Each search has at most 24 distinct candidate attempts, population eight,
  three generations, seed 6102026. The preferred host's other GPU is unused
  deliberately because it shares host RAM.
- The coordinator's minute timer copies terminals and publishes TRAIN/inner
  metrics independently. At 18:00:53 UTC it reported one newly published
  terminal, three already published terminals and zero delivery errors.
  Publishing never retrains a model.

The h12 script estimated 2,644 seconds remaining after its second completion.
This is a provisional throughput estimate from two candidates, not an ETA
for the entire master plan. H72 had no measured search ETA yet.

## Corrected execution defect

The first search revision encountered a real GPU placement failure in the
inherited attention-mask construction: integer MatrixBandPart had only a CPU
kernel under the production strict-placement policy. The earlier smoke did
not enforce that policy. Three pre-fit failures were retained in each search.
Both controllers were stopped cleanly; two dispatched, incomplete genomes
were explicitly excluded from their successors, with sealed predecessor
receipts. No interrupted fit was repeated or relabeled completed.

The additive I6N attention mask now uses a range comparison. The legacy I6E
implementation remains unchanged. Before relaunch, all 15 model variants
passed six real GPU updates and checkpoint reload under strict placement on
each of the two devices; both probe services exited zero. Those synthetic
checks are regression evidence, not historical scores or production cost
bounds. Successor output directories preserve all predecessor artifacts.

## Scope and results

The three completed historical pilots and their separate naive values are
recorded in [HISTORICAL_PILOT_RETURN.md](HISTORICAL_PILOT_RETURN.md). Twelve
pilot OLAP reports have exact readback. None beat its own horizon's naive in
the one development week. This does not decide annual performance.

The running searches use four years of history and frozen, purged 28-day
inner support. Fitness uses TRAIN/inner, not the outer week. All input windows
include the current observation; targets and naive use identical eligible
rows in original price units. No smoothing or TEST access occurs.

These are bounded endpoint searches using the shared DEAP adapter, not a
full distributed DOIN swarm. Independent trajectory supervision, complete
swarm placement and the full weekly development evaluation still require
integration. Do not describe this return as completion of those tasks.
Calendar, release-vintage and retrospective feature-selection limitations
in the pilot return continue to apply.
