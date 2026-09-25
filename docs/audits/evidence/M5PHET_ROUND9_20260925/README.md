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

## WP26 — the search's advantage survived rows that ranked nothing

A nested split frozen before any fit (the tool refuses to overwrite it): inner rows 0–30,239, **outer rows
30,240–40,319, 9,567 sealed origins, seal `d4ac73f0adde`, protocol `48b783f88f54`**, naive `last_value` 0.667173 kW.
Disjointness checked, not asserted. Declared costs: a harder week (naive 0.667 vs 0.599) and 25 % less training data;
named residual `OUTER_ROWS_WERE_TRAINING_ROWS_OF_THE_EARLIER_ROUND` — those rows ranked nothing and were scored by
nothing, and the WP26 re-fits never read them at all.

Four stages × five seeds, everything else identical, 20 fits, GPU checked before each dispatch:

| stage | window | outer MAE (mean of 5) | sd | mean skill |
|---|---|---|---|---|
| searched_0588f5747b62 | 21 | **0.563236** | 0.005856 | 0.155787 |
| searched_b373275495e2 (the search's winner) | 21 | 0.566487 | 0.003412 | 0.150915 |
| searched_b367cba9b237 | 34 | 0.568555 | 0.003141 | 0.147816 |
| baseline_hand | 60 | 0.600899 | 0.009188 | 0.099337 |

Paired difference of the search's winner against the hand window: **−0.034412 kW (−5.73 %)**, t-interval over seeds
[−0.047875, −0.020948] and row-level bootstrap [−0.038550, −0.030221] — both exclude zero. All 15 searched fits rank
above all 5 hand fits with no overlap.

**So the published 4.50 % was not an artefact of selecting on the seal** — on this earlier, harder week the margin is
larger. What did *not* survive is the ordering inside the top three (they sit inside each other's seed spread), so
what is confirmed is that **a short window (21–34) beats 60 on this series at this horizon**, not any particular
champion. Both intervals are declared for what they measure: the t-interval is fit-to-fit variability under one split;
the bootstrap treats a contiguous week of one household as exchangeable and is optimistic by an unestimated amount.

## WP27 — served, and three defects found on the way
The bundle exported is the representation **the search chose**, not the outer-seal leader — picking that one now would
be selection on the outer holdout a round later — at the median of its five seeds. It carries its `representation_spec`
by value and its `measured_error` with seal, protocol and report digest; `provenance.quality` stays `UNMEASURED`
because the package still scores nothing: it quotes a report with its conditions.

Three defects had to be fixed first: the winner's subset-branch graph was **unloadable by anything** (Keras serialised
the channel gather as a closure's bare name — every subset model this repository ever saved is affected); the `bundle`
slot the provider declares was **never read**, so a sentence could name an engine and be ignored; and a blank state was
refused as a bundle named `''`. Control after the fix: a re-fit reproduces MAE 0.5658380994350456 / RMSE
0.8908290889272082 / 17 epochs to every published digit.

The owner's instance now serves four bundles: `pronostica Global_active_power a 60 pasos con
searched-w21-household-outer-20260925` answers with nothing but the sentence pinning the engine, and a bare
"pronostica la potencia" is now refused naming all three candidates — the correct consequence of two point bundles on
one target. Acceptance after the merge: **examples 12/12, prose 14/14, refusals 2/2, envelope questions 15/15**.

## WP30 — the router measured, and the gate stated no wider than it is

`command` / deepseek-v4-flash, 19 sentences × 5 routings = 95, report `aac62e58…`:

| | strict | when it did propose | n |
|---|---|---|---|
| router (`orchestrate.route`) | **0.8526** (81/95) | 0.8617 | 19 sentences, 95 runs |
| interpreter (`interpret`) | 0.8333 (50/60) | 0.9434 | 12 sentences, 60 runs |

The two conditionals exclude different things (the interpreter's excludes 7 `DECLINED`, the router's 1 `REFUSED`), so
they are two populations and neither rate is evidence about the other. What IS comparable is the failure kind:

**over 95 routings the router never named a different engine and never named a target, horizon, study, policy or
metric an engine does not have.** Every miss is envelope *shape* — one question where the sentence asks for two, or
`cluster_description` where the sentence asks only for the assignment — which is an under-answer visible in the review
window before anything runs. The interpreter's misses, by contrast, are `WRONG_VALUE`. Weakest sentences: `pronostica
la potencia y dame un rango` 1/5 and `cual fue el efecto del tratamiento y en jovenes` 1/5 — both ask for two things.

`/api/catalog.abstention.paths` now says exactly which paths the rule covers: `decide` covered, `interpret` only where
the plugin reports a confidence, **`route` not covered at all** — no shipped plugin reports a confidence for a
free-text envelope, not even `openai_compatible`, because `route` calls `_ask` and `_ask` discards the logprobs.

## WP31 — every answer carries what is known about its area

A rendered forecasting answer now ends with, for example: `quality (forecasting): MAE 0.526293524060822 [kW …];
interval coverage 0.9259975570032574 at nominal 0.95; skill 0.12185848183739156 vs last_value — 9824 scored rows,
labels REALISED_OUTCOME, protocol d0ebd9a4bc…, seal 33820b552ddf…`. Classification carries macro-F1 0.3778 and ECE
0.1315 over 450 rows; unsupervised is `NOT_MEASURED` **on purpose** (the WP19 report measured a reference this instance
does not serve, and another state's measurement is not published as this one's); causal and policy are `REFUSED` with
the evaluation package's own reasons. Nothing is computed at render time: a report is accepted only because the
bundle's manifest names its sha256, and the local report path is never published.

The narration guard bit once and was not relaxed: `policy_profitability`'s own refusal text ends in a clause the guard
reads as a claim of profit, so that line names the refusal and lets the reason travel verbatim in `quality.why`.

Final acceptance on 8766: examples 12/12, prose 14/14, refusals 2/2, envelope questions 15/15, outputs 16/16 faithful
with 0 figures the answers do not carry. M5PHET master e2eddd6, suite 646 passed / 1 skipped.
