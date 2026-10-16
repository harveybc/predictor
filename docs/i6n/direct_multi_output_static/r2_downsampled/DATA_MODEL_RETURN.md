# Downsampled R2 Data/Model Return

Codex helper, 2026-10-15. Implements owner-approved `PLAN.md` RD01/RD06/RD08
and the model-side portion of RD10. No science, GPU work, services, donor
replacement, source report edits or commits were performed. Independent review
and parent factory/recovery checks remain required before execution.

## Selected Arm API

- `r2_design.source_dir(arm="mg2m")`: canonical `matched_short` for `mg2m`,
  `r2_downsampled/long_source` for `mg12m`; other arms are refused.
- `build_contract(matched_dir=None, *, arm="mg2m")`: explicit downsampled arm
  in `shared.arm`. Default resolves to the unchanged short source. Long binds
  `EXPERIMENT.mg12m.json`, `POPULATION_SEAL.json`, `CAPACITY_MATCHED.json`,
  13 slots / spacing 12 / span 144 and physical heads 18/24/36/48/60/72.
- `lead_objective(matched_dir=None, *, arm="mg2m", ruling=None)`: old short O1 ruling is
  unchanged. The explicit long ruling binds this document and the owner's
  approved protocol, not a fabricated independent-review acceptance.
- `validate_r2_design(doc, *, for_execution=False, matched_dir=None)` derives
  the canonical source from `shared.arm`; arbitrary contract-adjacent sources
  are not trusted. Explicit `matched_dir` remains available for synthetic tests.
- `seal(matched_dir, out, *, arm="mg2m")`; standalone design CLI adds
  `seal --arm mg12m`, with arm-aware default source. Existing short CLI/API
  calls keep their original results. `write_ruled(..., objective=None)` infers
  the source arm and accepts an explicitly reviewed objective.
- `r2_runner._contract(path)` validates against that arm's canonical source.
  `_bind` uses `contract.shared.arm`, population and runtime. `prepare` adds
  optional `arm` / `arm_design`, retains `mg2m_design`, and passes BOTH original
  source arms to `MR.prepare`. Long's counterpart remains the sealed
  `matched_long/EXPERIMENT.mg1l.json`; no arbitrary single-arm population exists.
  Long defaults its population seal to canonical `long_source`.
- `validate` keeps the old short response fields; long returns `mg12m_design`
  and `arm`. `init` takes the selected design from the bound contract.
- `_context`, `_report` and comparisons retain long physical horizons without
  relabelling as short. Parent owns campaign/worker/unit-runner transport and
  recovery, including original short donor provenance and strict R0 reuse.

Long source custody is the retained RAW campaign `i6dmo-0f9e6c8d5d74` and
comparison `22d9ac6ce7ce5a5ff3a78f3ce5d2b29672cdd79d7f0d35d6f2ae155963a01fa3`.
The contract labels downsampled-long choice as owner-selected, not as a claim
that this arm won that historical RAW comparison.

## Independent Release Gate

This helper return is not an independent review. Default long ruling is
`OWNER_APPROVED_PENDING_INDEPENDENT_REVIEW`; validation returns
`PREPARING_INDEPENDENT_REVIEW`, and `for_execution=True` or default long
`write_ruled` refuses with `R2_INDEPENDENT_REVIEW_REQUIRED` without writing output.
Parent must obtain actual independent release acceptance, then call:

```python
objective = r2_design.lead_objective(arm="mg12m", ruling={
    "ruled_by": actual_reviewer,
    "review_commit": reviewed_source_commit,
    "review_verdict": "ACCEPT_FOR_RELEASE",  # ACCEPT is also supported
    "review_document": "../INDEPENDENT_REVIEW.md",
})
r2_design.write_ruled(base_path, ruled_path, objective=objective)
```

The supplied document must exist; its actual bytes are hashed into the ruling
and replay-validated. Parent supplies the genuine verdict and reviewed-source
identity; this API does not perform independent review. Legacy short ruling
and default sealed identity remain exact.

## Monitor Ruling

Registered loss stays `autoencoder_reconstruction` **1.0.0**, decoder 32, MSE.
Legacy short O1 records, fit settings, INNER-only monitor, donor bytes and
qualifications are unchanged. New long orchestration uses
`i6dmo.branch_mean_fit_inner.v1` with monitor
`mean_end_of_epoch_fit_purged_inner_same_weights`.

Each completed epoch performs FIT updates only, then evaluates all FIT and
purged INNER windows by the same inference path at identical weights. The
monitor is their equally weighted mean; no online accumulated training loss
enters selection. AdamW lr 0.001, weight decay 0.0001, batch 256, maximum 100
epochs, patience 10, min_delta 1e-8, per-feature seed 6102026+index. The lowest
improving mean is restored, first tie wins. Optimizer updates are measured;
inference must not change weights. An update/time budget stop fails typed and
produces no donor, including interruption during monitoring. Budgets are not
trimmed or expanded. Reconstruction remains a diagnostic, not denoising or
R0/R2 selection evidence. Registered component versions were not modified.

## Comparison And Costs

`decide_r2` always validates original paired metadata, C2, exact six actual
rows, support, replay, own naive and cached selection for long, through
`reporting.validate_matched_report`. Long runner comparison additionally
validates the authenticated child/C2 lineage. Same runtime and recorded GPU
class, 19-donor import record and matching import-arm binding are required.
The original short direct API remains compatible; new short recovery callers
may request `strict_reports=True`, or set `effective.r2.strict_reports=True`
for runner-level strict metadata and child-lineage checks.

New pretraining costs read the executing unit's cgroup-v2 `memory.peak`, a
full-subtree high-water measurement including descendants. No RSS substitution,
counter reset or memory trimming is used. Unavailable readings remain null
with explicit `UNAVAILABLE` state. New compute facts include
`Pretrain_Cgroup_Peak_Bytes` when measured. Existing null-peak receipts are not
rewritten. An edited model source hash can require a fresh short R0 under the
existing strict reuse predicate; no equivalence waiver is introduced.

## Verification

Focused tests: `tests/i6dmo/test_r2_downsampled.py`. Initial PRE: 4 failed,
1 passed; absent selected-arm API/long decision/mean monitor. First POST:
4 passed, 1 typed-source-refusal mismatch, subsequently corrected. Bounded CPU
verification passed 96 tests (28 deselected), including exact legacy short
contract and ruled-objective replay. Final reviewed-writer checks passed
3 tests (14 deselected), covering pending refusal without output, accepted
review sealing, altered-review refusal and legacy replay. Both runs used
`CUDA_VISIBLE_DEVICES=''`, `crispdm-run -m 2G -t 180s`; no broad-suite
certification is claimed.

Test fixtures contain synthetic metadata and controlled optimizer/epoch
objects. They do not pretrain scientific donors or read market arrays. Parent
canonical long source metadata is read only for binding; no source bytes are
modified. Next: independent source-scope review,
then parent-owned factory/recovery and separately admitted execution.
