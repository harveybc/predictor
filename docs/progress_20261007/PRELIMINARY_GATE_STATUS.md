# Preliminary feature-selection gate, 2026-10-07

The Phase-3 rankings are complete, but they did **not** close a binding
preselection before the original Phase-4 queue was initialized. The
`ALL_ADMISSIBLE` control expanded that queue to 6,495 tasks: 355 EURUSD and
78 ETH trainable features, three arms, five TRAIN folds. A completed ranking is
not a selected feature set.

`PRELIMINARY_GPU_TRIAGE.json` is a deterministic, TRAIN-only first wave. It
links each population to the retained Phase-3 closure and candidate-file
digests. It unions the top K=4, 8 and 12 candidates across seven measured
methods: Spearman clustering, mRMR, JMI, causal mRMR, causal JMI, univariate
mutual information and causal-supported ranking. It excludes `RANDOM_K` and
`ALL_ADMISSIBLE` from the GPU feature union while retaining them as comparison
controls. No VALIDATION or TEST rows were read to choose this wave.

| Population | First-wave GPU features | Seasonal context | Deferred, not rejected |
|---|---:|---:|---:|
| EURUSD | 82 | 9 | 274 |
| ETH | 44 | 0 | 34 |

The first wave has 126 distinct GPU features, 420 candidate subsets and at
most 1,890 feature/fold/arm tasks before already completed tasks are credited.
This is a computational triage, **not final selection**. A feature with
`NOT_IDENTIFIED` causal evidence is not proven causally irrelevant; failed
overlap and limited power remain explicit. The deferred set is preserved by
identity for expansion after the paired weekly comparison, rather than deleted.

The deployed priority dispatcher reads the original coordinator task store,
checks the pinned manifest digest, chooses only a pending `TRAINED_ENCODER`
task in this first wave, and hands its exact task ID to the existing FS4
worker. The controller still owns leases and completion. The original plan and
warehouse receipts remain unchanged. Its global 6,495-task closure cannot be
claimed from the first wave; the successor needs a separately named partial
wave closure or a reviewed, versioned successor plan. Do not mark the existing
`EXTRACTIBILITY_COMPLETE` gate true on partial evidence.

The current FS4 runner stops on the fixed masked MSE of a purged TRAIN tail.
It does **not** yet implement the requested average of comparable fit-TRAIN
and purged-TRAIN validation MSE for patience. That runner change needs a new
code pin and replay before a first-wave result is promoted from diagnostic to
selection evidence. The outer 2024 validation remains unopened. Likewise,
reconstruction against observed values is not proof of a noise-free signal:
SNR requires a defined reference or injected noise with retained identity.
The FS4 `naive_mae` is persistence for the feature being reconstructed on the
same masked points. It is **not** the naive forecast of EURUSD or ETH targets,
and beating it is neither expected nor a standalone feature-selection rule.

Future work after the primary milestones: evaluate reconstruction error and
other methods as anomaly and market-regime-transition detectors. That is not
a prerequisite for this preliminary gate or the weekly trading comparison.

## Repeatability

Regenerate the triage from the two retained `CANDIDATES_FOR_VALIDATION.json`
files with `tools/fs3_preliminary_screen.py`; the sibling
`PHASE_3_FILTER_COMPLETE.json` files are required. Run
`pytest -q tests/test_fs3_preliminary_screen.py tests/test_fs4_preliminary_next.py`.
Regenerate the PNG with `tools/render_feature_selection_progress.py --status
<fs4-status.json> --output <progress.png>`.
