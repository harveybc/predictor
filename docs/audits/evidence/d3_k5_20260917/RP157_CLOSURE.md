# The closure authenticates its design and binds every child

No retraining and no repeated inference beyond one re-run of the nine existing checkpoints under the repaired gates. The
scores are unchanged.

## Finding 1, the design is now authenticated instead of reconstructed

`authenticate_design` refuses **before a single child is dispatched**. It recomputes the design's digest from the design's own
canonical scientific content, requires that recomputation to equal what the design claims, and requires that in turn to equal
the digest **the run recorded when it produced these cells**. An empty factorial is refused as an empty population; a design
that does not cover the cells the run holds, or that declares cells the run does not hold, is refused by name. Comparing two
caller-supplied strings is not this.

On the real run: digest `b5d5eee1b5fce981…` recomputed and equal, nine cells covered, horizon 96.

## Finding 2, every child is bound before any aggregate exists

`validate_children` requires each returned child to name its own cell, the seed and regime that cell encodes, the horizon the
design declares, a reconciled checkpoint identity, **every** expected population with a typed positive window count consistent
across cells, and typed finite numbers in **every** declared reduction. Six mutations are regressions: a dropped population, a
foreign horizon, a NaN metric, a foreign cell name, a foreign regime and a dropped reduction. Each one now produces a problem
and suppresses the regime summary instead of averaging one regime over fewer seeds than the others.

On the real run: bound, with 2,537 windows in the complete validation population and 1,802 in the label-disjoint one, in all
nine cells.

## The result, unchanged

Status COMPLETE, nine expected and nine scored, no identity failures, no problems. The corrected table stands as published:
on the 1,802 label-disjoint windows, R0 0.371174, R1 0.374584, R2 0.368596 author float32 MAE, against a matched persistence
MAE of 0.868283. R1 is worse than R0 in three of three seeds and R2 better in three of three, the paired mean being
−0.002577 MAE and −0.694 %. It remains a development observation on the validation split, with no H1 claim.
