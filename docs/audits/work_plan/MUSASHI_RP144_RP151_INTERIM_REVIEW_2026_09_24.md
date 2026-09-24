# RP144-RP151 interim review: progress preserved, contrast acceptance incomplete

Reviewed published return and source at `f8e9e77f9113144563e914883364c25b148407b6`.
Scope: code inspection and two bounded probes executing exact extracted AST
statements. No training, GPU, market data, live process inspection, warehouse
reconciliation or independent rerun of the reported 150 tests in this review.
The latest report describes ongoing jobs, not completion of all eight orders.

## Experimental progress, as reported

Protocol B: twelve training cells completed; H96/H192/H336 recorded mean MSE
0.125925/0.143650/0.152696; final official reduction, paired naive, dispersion and
replay pending. Do not promote them or call A-to-B a controlled context effect.
Protocol A's reported consolidated mean remains 0.161962/0.259662 against
0.158250/0.255750 published. This review does not independently recertify A.
RP146 contrast reported running at 03:45Z; current state must be rechecked.

## Findings requiring scoped correction

### F1 High: the contrast does not execute its declared checkpoint contract

`tools/df_ecl_modular.py:340` declares validation-selected checkpoints.
`run_contrast`, lines 707-735, instead fits fixed epochs, evaluates at most ten
AE or twenty downstream validation batches and records first/last losses. It
does not supply checkpoint-selection callbacks, restore a selected checkpoint,
or save the downstream trained model/optimizer. The initial weights and detector
donor are saved, not the trained downstream models. This is insufficient for
the promised model replay or full-population comparative score.

Establish whether an external wrapper supplies additional persistence before
claiming actual artifacts lost. A monitoring subset can be legitimate only when
declared and identified; its loss is not the full reference metric. Preserve the
current run at its actual scope; no retrofit of a best-checkpoint claim.

Source: [checkpoint declaration](https://github.com/harveybc/predictor/blob/f8e9e77f/tools/df_ecl_modular.py#L340),
[fit and retained results](https://github.com/harveybc/predictor/blob/f8e9e77f/tools/df_ecl_modular.py#L707).

### F2 High: regime summaries accept an empty or incomplete population

Executed the exact assignment at `tools/df_ecl_modular.py:746` via Python AST,
with the declared three seeds. Both `cells={}` and only `R0_s2021` produced:
R1 unchanged=true, R2 changed=true, common donor=true, equal allowance=true.
This proves vacuous summary checks, not that the running population is empty.
Require the sealed AE/seed/regime population before a complete verdict; report
missing cells and partial evidence explicitly.

### F3 High: a sealed-design test cannot reject unequal digests

Executed the actual assert AST from `tests/test_df_ecl_modular.py:171` with
different digests A and B: it passed because the comparison ends in `or True`.
Remove that bypass and test relevant design mutations and replay identity.
Canonical scientific identity must distinguish changing scientific inputs from
operational timestamps without making the assertion unconditional.

### F4 Medium: prescribed steps and resource accounting are not observed steps

The run reports `steps = epochs * len(sequence)` (lines 712, 733), not optimizer
iteration deltas. Its allocation calculation subtracts wall time from the minimum
of CPU and wall budgets (684), then forces at least one epoch (687-688), even if
usable is zero. CPU is checked after a whole downstream fit (741); this function
does not enforce a wall deadline. An external service may enforce hard limits,
but it does not make that prescription CPU-measured or guarantee a durable partial
result. Inspect the actual launcher; retain distinct CPU/wall accounting, observed
updates and incremental durable receipts. No finding of actual budget overrun.

## Operational disposition

RP147-149 are not complete. Owner use of the coordinator is not itself a failed
resource admission. Do not launch without admission; also do not suspend CPU-only
implementation, fixture/integration tests, source mapping and entitled collection
while waiting for inference hardware. M5PHET contract tests are not implementation
of the newly assigned provider runtime or the other domain adapters.

The return cites orders `d4bb3353`, predating the explicit M5PHET coding assignment
`27868f5d`; incorporate that addendum rather than assuming it was already read.
No global experiment hold, no duplicated B job, no production service change,
no historical result deletion and no new compute allocation from this review.
Satoshi implements the corrections and real-entrypoint PRE/POST tests.
