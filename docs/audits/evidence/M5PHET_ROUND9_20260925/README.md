# M5PHET round 9, 2026-09-25 — the observed clock, and the point-in-time capture that starts now

## WP28 — the first event study with an OBSERVED publication clock, and why it still says NOT_IDENTIFIED
No consensus source overlaps an observed-clock source, so the consensus was replaced by a **declared model-based
expectation** built only from each series' own vintages published before the release: `SEASONAL_NAIVE(m)` and `AR(p≤4)`
by BIC, the row's model chosen by its own expanding-window out-of-sample MAE on prior releases, with that error
travelling on every row. The word *consensus* appears for it nowhere, and a calendar declaring both is refused
`TWO_EXPECTATIONS_DECLARED`.

Run: 11,135 expectations over 133 series → **48,369 event rows over 9,815 releases and 66 types**, clock
`OBSERVED_ACTUAL_PUBLICATION`, provenance `DEVELOPMENT_OBSERVED_CLOCK_MODEL_EXPECTATION`, `MISSING_PUBLICATION_CLOCK 0`.

**`ASSUMED_PUBLICATION_CLOCK` is gone for the first time.** What replaced it, decided by the code: `NOT_IDENTIFIED`
for `EXPECTATION_IS_MODEL_BASED` (actual − model forecast differs from actual − consensus by a quantity that was itself
pre-release information: measurement error in the treatment, correlated with the conditioning set) **and**
`PLACEBO_FAILED, 652 of 660`. Superposition `ADDITIVE_HOLDS` 10/10; the projection beats the naive on 289 of 648
triples; 114 of 650 β intervals exclude zero. Nothing was registered: the placebo passes on 8 of 660 cells and the
rule is all of them.

**Three reasons a year of releases cannot answer the owner's question, each measured:**
1. **The releases the question is about produced no rows at all** — NFP, CPI, PPI, retail sales, GDP, the policy
   decisions: zero. A monthly series has 10–13 releases in a ten-month archive; the expectation needs 12 prior values
   plus 8 prior out-of-sample forecasts, and the scale needs 8 prior surprises. What survived is almost entirely daily
   yield and reference-rate postings.
2. **22.4 % of releases have a surprise of exactly zero** (a daily reference rate repeats, so a seasonal-naive
   expectation is exactly right). Eight types are ≥50 % zeros; one is 100 % and its projection was refused
   `TREATMENT_NOT_IN_THE_DESIGN`. The most-released type is 59.2 % zeros.
3. **The archive's instants are declared, not verified — and one case is provably wrong**: `USD|initial_jobless_claims`
   stamps all 60 releases on a **Saturday** at 12:30 UTC for a release published Thursday, and produced zero rows only
   because there is no FX bar at a weekend instant. `OBSERVED_ACTUAL_PUBLICATION` means the dataset declared the
   instant and nobody assumed one — not that anyone verified it.

## WP32 — the point-in-time capture is running
Append-only store `m5phet.point_in_time.v1` (receipt clock outside the key, so re-reading is a duplicate and not a
revision; a row is never rewritten; `known_at(T)` never shows a row received after T; quarantine by bytes), fed by a
collector that reads only archives already on disk, on a `systemd --user` timer every three hours. First pass: **727
rows over 154 series in 14 economies**, 45 observed, 682 awaited, 0 revisions, 0 quarantined.

**The limit the agent found and stated:** the actuals archive was last acquired 2026-05-01, so today most receipt
clocks record *when we fetched*, not when it was published. A receipt clock bounds publication from above only, captures
no consensus, and cannot reach backwards. The capture is honest either way; it is informative only to the degree the
archive beneath it is refreshed.

## A regression this round caused and fixed
The abstention work made `m5phet.interpret` call `propose_with_confidence` on **every** interpreter, breaking every
implementation outside the package — feature-eng's own test doubles first. Asking an object for a method it does not
declare is precisely the mistake this framework exists to refuse. Fixed in M5PHET `f7db7f9`: the older protocol is used
as it is and its silence about confidence is never read as one; feature-eng is green again (465 passed).

## WP29 — twenty-eight corpora, and a chooser whose answer does not depend on the state

Twenty-eight corpora over twenty-three distinct source files (the household panel, the 4-hour EUR/USD fixture, six
disjoint calendar years of the 5-minute series, nine other G10 pairs at one hour, eleven daily instruments), each with
its own content digest; six foundation sets were excluded because their panels are byte-identical to the household one.
Nothing fabricated.

**The count, not a rate:** with the measured threshold in force, the checkpoint answered **nothing** at or above it —
28 method questions, 28 distinct state digests, **28 abstentions**, 0 answers ≥ 0.8. The parameter question was
therefore never put to it. Every abstention is recorded with `chosen: null` and every link refused
`ABSTENTION_HAS_NO_OUTCOME`.

**The sharper finding is how it abstained:** on all 28 corpora the head put **the same option first**
(`agglomerative`), top probability 0.3171–0.4360 (mean 0.3730) against a chance level of 0.25 over four options, with
the whole ordering unchanged. That is not a chooser that read the state and was unsure — it is a chooser whose answer
does not depend on the state. Together with WP20 (0 of 4 roles) and WP21 (0 of 256 bars, same argmax throughout), that
is three independent measurements of the same thing.

**Two machinery corrections, both required and both tested:** `compare_stages` could not rank the regimes area at all
(no metric keys), so every regimes report was `NO_NEW_MEASUREMENT` and every link would have been refused — it now
ranks on `silhouette`, with `regime_accuracy` still refused by name and the rendering saying in words that the rank
orders an internal index and says nothing about whether the clusters mean anything. And **a rank belongs to one
contest**: `decide.outcome` now carries the row's corpus seal, and the report settles the winner inside each contest
instead of refusing 28 corpora as one ambiguous election.

Calibration remains `NO_NEW_MEASUREMENT`: 0 scorable of 30 required, over 28 contests. That is the honest state — the
labels do not exist because the checkpoint declines to choose at the threshold its own calibration justifies.
