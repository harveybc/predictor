# Exact Short Recipe, Twelve Heads: Closed

DEVELOPMENT 2024 only; TEST remained closed. One new R0 fit, seed 6102026,
on the secondary 4090 from `1601e943`, directly based on the actual historical
short runtime `24f5a6cc`. No scientific repeat and no learned-weight donor.
The independent 407-cell continuation on the preferred GPU was not changed.

## Identity And Business Checks

- Model builder and entire trainer file retained. Model-source AST differs
  only by admitting the additional horizon tuple. Features, branch/core/head
  parameters, 13-slot MG2 window, original time encoding and training settings
  unchanged; only six additional identical head structures, 4,882->7,480 params.
- Original shared/short initial tensors and initial short forward verified
  bitwise on the executing device. Saved/reloaded predictions verified bitwise.
- Original short cohort, input values, offsets, closure flags, target values,
  per-head masks, input statistics and six FIT naive scales reproduced exactly.
- Same business predicate for ALL outputs. Its result depends on t+h; no
  all-head intersection or deletion of short rows to satisfy long horizons.
- Independent `audit_business.py`, without calling the C1 evaluator, recomputed
  all masks from the retained calendar, 12h offsets, split boundaries and exact
  observed targets: PASS over 150,680 FIT, 6,466 INNER and 43,444 VALIDATION
  valid pairs. Also checked all real input slots, exact MG2 calendar ranks,
  current t included, original feature values, origin/target prices, no future
  input and no TEST source bars. This verifies the retained assumed calendar,
  not an unretained broker calendar.
- Tests: 46 focused model/extension tests passed in 32.48 s; a separate nine
  market-grid regressions passed in 51.09 s. Two legacy protobuf deprecations;
  the model suite also emits existing Keras input-structure warnings.

## Results

One fit, 25 epochs, best epoch 15, 128.3243 seconds, cgroup peak 2,817,531,904 B
under a 4 GiB admission. Same original FIT/INNER mean stopping, patience 10,
delta 1e-8; not the continuation's new TRAIN/VALIDATION protocol. Actual
TF 2.21.0/Keras 3.13.2; historical declared/observed runtime mismatch remains
disclosed, not repaired retrospectively.

Relative error = (MAE_h-naive_h)/naive_h; negative is better. Original USD/EUR
scale; each horizon's model and naive use exactly the same rows and mask.

| Horizon | N | MAE | Naive for horizon | Relative error |
|---|---:|---:|---:|---:|
| 2 h | 4770 | 0.000755253427 | 0.000746639413 | +1.153705% |
| 3 h | 4714 | 0.000943483084 | 0.000927952906 | +1.673595% |
| 4 h | 4663 | 0.001109596069 | 0.001079245121 | +2.812239% |
| 5 h | 4607 | 0.001268924559 | 0.001224562622 | +3.622676% |
| 6 h | 4556 | 0.001414891842 | 0.001362120281 | +3.874222% |
| 12 h | 4237 | 0.002117759814 | 0.001994559830 | +6.176801% |
| 18 h | 3923 | 0.002808531464 | 0.002542159062 | +10.478196% |
| 24 h | 3610 | 0.003370406200 | 0.002966612188 | +13.611284% |
| 36 h | 2988 | 0.004285302293 | 0.003670023427 | +16.764985% |
| 48 h | 2391 | 0.005158709042 | 0.004354366374 | +18.472094% |
| 60 h | 1795 | 0.005886270602 | 0.004715665738 | +24.823746% |
| 72 h | 1190 | 0.006223830210 | 0.005103277311 | +21.957515% |

On the same 3,920 valid origins for both 12h and 18h, the model/naive/relative
are 0.002165125172/0.002036528061/+6.314527% at 12h and
0.002810418372/0.002543647959/+10.487710% at 18h. Different full-head supports
therefore do not by themselves explain this within-model gap.

The historical short-only control had -0.049615% at 6h and -0.093170% at 12h
on exactly these heads' source rows and naive values. Their improvements did
NOT survive the joint twelve-output training. Adding objectives changes
shared gradients and the selected checkpoint; equal initial tensors do not
guarantee equal trained short predictions. Hardware class also differs by
owner request. This one-seed comparison does not isolate a unique mechanism,
prove negative transfer or establish an information-theoretic barrier.
No head is eligible for the prediction strategy; no TEST/strategy run opened.

## Automatic Closure

One report containing 145 metric rows was delivered and exactly read back;
zero pending or mismatch. Thirteen immutable artifacts were verified in each
physical mirror. Both owned timers disabled themselves on CLOSED. Result
semantic digest:
`b70b5ee3f6023b571f6fd515cb30de611ae528f1c956631453a00722b7ee29c9`.
Large artifacts and physical snapshot stay outside Git; source, reproduction
commands, audit and concise numeric results are public. No secrets or host
identifiers are needed to reproduce the protocol.
