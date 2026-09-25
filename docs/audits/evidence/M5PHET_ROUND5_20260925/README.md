# M5PHET round 5, 2026-09-25 — the pipeline is complete and unfitted; the event study answers in the workbench

WP24 `fused_branches` (predictor fe6cf262): the one multi-branch core, with an `rnn` (GRU) encoder and a declared
EXTRACTOR_FAMILIES table mapping feature-extractor keys to inline families — a person's declaration in the repository,
never a runtime guess. WP25 (feature-extractor d87807e, preprocessor 137edcf, predictor 949cdb94): the `rnn` encoder
implemented, `cnn_signed` removed (a second missing module the plan had not seen), every option label one truthful
sentence. WP23 steps 1-2 (M5PHET af1dea5): outcome records bound to a closure-table row by digest; the calibration
report says 22 decisions, 0 linked. WP18 complete (M5PHET 06e6e75): the household pipeline spec with every branch
mapped, unfitted. WP22 steps 4-5 (feature-eng fe8e13c, causal 75c6b11): the model-based counterfactual path and the
three event-study question types, answered in the workbench from the registered EUR/USD study.

**A regression I caused and fixed the same night**: registering the first event study made the causal `event` slot
required (m5phet.interpret treats a declared slot as required unless it says otherwise), and every ATE sentence was
refused — examples 6/9, prose 12/14 on the acceptance run. Fixed in the provider (`required: False`, causal 75c6b11)
with the regression test; acceptance back to 9/9, 14/14, 2/2, 15/15.

The owner's instance now answers, from its own state directory:
`{"area":"causal","state":{"study":"eurusd-events-assumed-clock-v1"},"questions":{"nfp":{"type":"impulse_response",
"event":"United States | Nonfarm Payrolls","outcome":"volatility"}}}` → a response-path table over five horizons,
`superposition: ADDITIVE_HOLDS`, `identification: NOT_IDENTIFIED` (assumed publication clock; 39 placebo cells failed),
`execution_authorized: false`. Every beta is a conditional association, not an identified response.

NO_NEW_MEASUREMENT for every area. Step 7 of WP18 (the fit on the 5090) awaits the owner's admission.
