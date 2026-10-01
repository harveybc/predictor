# Satoshi: close source and engineered-feature coverage gaps

Addendum to the progressive-selection orders at `ac125db9`. Read-only inspection
on 2026-10-01 around 02:28 UTC; no running work, broker, feed or dataset changed.
This is not a completed inventory audit of every subscription endpoint. It
identifies why current evidence cannot support exhaustive coverage.

## Findings

1. **The denominator is not a census of everything usable.** Lane B at
   `9c8e1a02` reports 3,538 covered / 15,228 distinct dataset-column rows,
   11,468 NOT_MEASURED, and 121,824 row-family cells. Reused rows lack several
   metric families. A dataset-column appearance is not necessarily a distinct
   economic signal. Coverage of this index does not prove source completeness.
2. **The current measured batches cover five datasets**, Electricity, Weather,
   Traffic, legacy EURUSD d4 and the ETH 4h technical/statistical view. They are
   not a provider-by-entitlement census. The financial TRAIN-contract export has
   eight FXMacroData appearances, all carrying the older census availability
   string UNAVAILABLE; no entry explicitly names Alpaca or Yahoo. Names alone
   cannot establish their absence: provider provenance must be joined back to
   canonical resources, including newer availability supersessions.
3. **Engineered files exist outside the demonstrated selection coverage.** A
   filename-only census in financial-data finds 200 files EACH named
   technical.parquet, statistical.parquet, wavelet.parquet, hilbert.parquet,
   multitaper.parquet, emd.parquet and fracdiff.parquet (1,400 paths). No values
   or holdout rows were read. Lane B's column index contains no `wavelet_*`,
   `ht_*`, or `mt_psd_*` columns. `MT_001` through `MT_370` are electricity
   channel names, not proof of multitaper coverage. Resolve producers/aliases
   explicitly; do not conclude file presence equals valid model inputs.
4. **A wavelet label conceals a different method.** In financial-data
   `ef0ba661`, `_scripts/workers/stage23_signal_decomposition_worker.py:124`,
   compute_wavelet uses rolling means at 16/32/64/128 and their differences.
   Its metadata admits a db4-style approximation; it never calls wavedec.
   This is a multiscale rolling-filter proxy, not a reproduced db4 DWT. Preserve
   the proxy with an honest method ID; separately implement/reuse and test a
   genuine wavelet producer before claiming that family was evaluated.
5. **Spectral semantics and timing need explicit tests.** The same producer
   reads only close; it does not transform every input signal. Its Hilbert and
   multitaper windows depend on full input length for n<1000, so the recipe
   itself can change when a prefix is extended. Both evaluate sparse endpoints
   and forward-fill. Hilbert `ht_inst_freq` and `ht_phase_difference` are the
   identical phase difference, without division by physical sample interval.
   Multitaper uses d=1 frequencies (cycles/sample). These are source-code
   observations, not a measured leakage rate or a judgment on all historical
   artifacts. Freeze windows from TRAIN, test prefix/restart invariance and
   declare sampling, frequency units, feature age and update cadence.
6. **Market-regime coverage is not established.** The index's two retained
   regime columns are ETH vol_regime_high/low, not a demonstrated hierarchical
   market-state provider. feature-eng `d081d0f`, app/regime_detector.py:247-302,
   has fixed scaler/centroids documented as fitted on 15 years of EURUSD and a
   cluster-label mapping informed by forward returns. Do not admit it into a
   temporal fold without linking the fitting period and label mapping to that
   fold; refit on its TRAIN when necessary. This is a provenance/leakage risk,
   not proof that every retained model used this detector.

Evidence paths in feature-eng `satoshi/b-selection-ps0-ps2-20261001`:

- docs/feature_metrics/m03/inventory_v3/coverage_summary.json,
  sha256 `884b595e21b40df63bf078240682ad9ed2c69c4b160972adf9f8bcbf9f6bc240`.
- docs/feature_metrics/laneB/BATCHES.v1.csv,
  sha256 `af290af4982d81879c79ab063b1905629ac0716f7e8ef554b77d7b72b8a90e96`.
- docs/feature_metrics/laneB/FINANCIAL_TRAIN_CONTRACTS.v1.json,
  sha256 `bfaf2cf63814a0be5756f648f3f5c949b97d96b1f0d0c29f674bc0d36a371406`.

Native method documentation: [PyWavelets DWT](https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html)
and [SciPy Hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html).
Neither API name certifies that a pipeline is point-in-time safe.

## B + C: reconcile the full finite source/feature universe, incrementally

Extend the existing census and feature DAG; do not create a second registry or
silently replace its denominator. Record additions, aliases and supersessions.
The owner's declared access includes Yahoo Finance, Alpaca and paid FXMacroData.
Distinguish OWNER_REPORTED subscription, documented entitlement, functioning
connector, retained bytes, point-in-time admissibility, profiled and evaluated.
Do not claim a paid Yahoo product implies an API entitlement; check the actual
product and supported access method. Do not expose credentials or bypass terms.

Required source table: provider/product, asset universe, field families,
frequencies, coverage dates, entitlement/rate-limit evidence, connector/version,
lake resource/bytes, event/publication/receipt clocks, revision policy, TRAIN
contract, license/use scope, status and the precise missing action. Reconcile
existing sources too: macro/rates, indices/VIX, commodities, crypto/on-chain,
positioning and news where present. No arbitrary new purchases or universe-wide
download: start with our declared business assets and existing subscriptions.

Source discovery must include raw storage and derived directories, not only the
currently selected views. Resolve the eight FXMacroData availability records
against current contracts; do not perpetuate stale UNAVAILABLE or invent missing
consensus/first-release vintages. A split contract alone is not availability.

Required transform-family ledger per eligible input type:

- Raw, returns/log-returns/differences, normalization and fractional differences.
- Technical indicators: price/volume/trend/momentum/volatility families and their
  compatible parameters; an OHLC indicator is not applicable to a single macro value.
- Market states: rule-based and learned/hierarchical regimes, probabilities,
  durations and transitions, fitted on TRAIN and emitted with available history.
- Wavelet, Hilbert amplitude/phase/frequency, multitaper spectral measures;
  STL and EMD/EEMD where appropriate, with proxy/native distinction.
- Calendar/release/surprise/revision/event-overlap features with real availability;
  learned branch/fused representations with their donor and fit provenance.

For each row: applicable, implemented, materialized, temporally verified,
profiled, evaluated, selected, deferred or excluded with reason/owner/next step.
NOT_APPLICABLE needs domain justification. DEFERRED is not covered or worthless.
Define causality here as no use of unavailable future inputs; it does not prove
an economic causal effect. Keep the separate causal-study ladder and assumptions.

## Keep the feature space manageable without hiding exclusions

Maintain a large **catalogue**, but materialize a bounded **active candidate set**.

1. Use a DAG: canonical source -> transform recipe -> output channels -> group
   -> branch. Identity includes parent bytes, transform/version/parameters,
   fold/fitting state, availability and units. Cache only exact compatible identities.
   Different assets, vintages or feeds do not become duplicates because names match.
2. Profile raw admissible inputs broadly. Offer a compact predeclared parameter
   grid per compatible family, not all signals x every transform x all settings.
   Start with raw + single-transform paths; compositions need explicit support
   and cost. A cap defers candidates with a reason, never deletes them from the ledger.
3. Allocate exploration across source and transform families, including low-ranked
   inputs and interacting groups. Expand settings where measured target utility
   or a declared exploration rule warrants it. Report families still untested.
4. Deduplicate exact channels; treat correlated or nonlinear redundant channels
   as reversible groups to compare, not automatic deletion. Selection/cluster
   fitting is TRAIN-only. Keep raw controls and counterexamples with joint-only utility.
5. Budget each candidate using measured raw channels, expanded channels, branches,
   fused width/time, materialization bytes, parameters, RAM/VRAM and weekly cost.
   Fused width is sum of branch widths, not just the number of source files.
   DOIN searches valid group/transform/encoder choices under these constraints.
   Dimensional bottlenecks reduce model cost, not the cost of generating every
   conceivable feature first. No silent temporal collapse or input truncation.
6. Store recipe/state/metrics and reconstructible references; cache materialized
   batches by actual reuse and measured disk budget. Do not duplicate all
   decompositions on every worker. Apply existing retention rules, not new blanket deletion.

The catalogue is exhaustive relative to a declared finite discovery snapshot;
the search is progressive, not exhaustive over infinitely many engineered features.
No method can certify in advance that no useful combination will ever be missed.
Our obligation is visible coverage, controlled exploration and reversible decisions.

## Acceptance and handback, independent of current fits

Lane B owns source/feature reconciliation; lane C checks method semantics and
temporal tests; M01/M04 own shape/cost/search limits; M06 links coverage/metrics.
Continue current valid fits. No restart of literature reproductions with new inputs.

Design tests first: a new subscribed source or derived file creates an uncovered
row; a proxy cannot certify native-wavelet coverage; future extension and chunked
replay preserve emitted features; regime fitting cannot cross the fold boundary;
equivalent cached recipes reuse while different vintages do not; feature-budget
overflow yields a named deferred candidate, not silent truncation; family
exploration and group reintroduction survive a resource-limited batch.

Next return: source entitlement-to-use matrix, transform-family coverage matrix,
old/new denominators, measured candidate cost envelope, omissions with owners,
and newly demonstrated utilities. Separate catalogue coverage from model-input
count and from evaluated-candidate count. A claim of complete coverage is refused
while a known source/family is absent from that accounting.
