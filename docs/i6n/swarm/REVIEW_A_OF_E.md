# Lane A review of lane E (search space, plan, identities)

Reviewed: `satoshi/i6n-swarm-E-search-20261011@78e9c527`, the sealed plan
`I6N_SEARCH_PLAN_v1_1.json` (`plan_sha256 e7de41f0...ab788c`, verified by
`verify_plan`), and E_RETURN.md. Lane A did not write any of this code. E's
branch was not edited.

Method: E's files were overlaid on lane A's branch `05d3f96b` in a scratch
worktree, so that E's generator runs against A's real evaluator. A single
finite CPU script was run under `crispdm-run -m 2G -t 120s` and finished in
10.9 s. It made no fits, used invented fixtures only and read no TEST data.
- Script: `evidence_A/review_a_of_e.py` (sha256 `7471f0ab...`).
- Output: `evidence_A/review_a_of_e_result.json` (sha256 `b0266e84...`).

E's three suites were also run once on the combined tree: 178 passed and 1
failed. The failure was my harness's fault: the overlay was missing
`E_LEGACY_MAPPING_v1.json`. With the file present, a targeted rerun of that
one test passed.

## Checks against plan sections 3 and 4 and A's evaluator

| Check | Result |
| --- | --- |
| Plan seal recomputes; `training` equals A's PLAN_STOPPING (50/5/1e-6, seed 6102026) | OK |
| History L = max(24, m*h), m in {1,2,4,6}, deduplicated (independent recomputation, 12 horizons) | OK |
| dx = every common integer divisor of h and L, including 1 | OK |
| n = 1 + L/dx; maximum 433 is reachable (h=72, L=432, dx=1, class S3) and accepted by A's mapping, cell contract and cascade rules | OK |
| dy = h/k, k in {2,3,4} where integer; h=5 has no trajectory; k=1 aliases to endpoint with `trajectory_k=null` and the same identity | OK |
| Inactive genes canonicalized (conv blocks, adapter width, endpoint k), giving the same identity; each of 8 active genes changes the identity | OK |
| `genotype_canonical_sha256` equals A's receipt formula `search.effective_identity({k: request.config[k]})` | OK (96 gen-0 + 720 sampled + 720 mutated + n=433: 0 mismatches) |
| `effective_config_sha256` equals A's `effective_config_sha256(request.config)` | OK (same set, 0 mismatches) |
| Every `evaluator_config` is accepted by A's `candidate_config` and `cell.authenticated(trajectory=True)` (equal head weights, offsets = derived outputs, n/dx match) | OK (same set) |
| A's `freeze` then `load_request` on generation-0 configs (h12, h72; synthetic fixture; no fit) | OK (8/8 each) |
| v1 -> v2 roundtrip is injective | OK (same set) |
| Per-layer strides keep t under A's `reduction_plan`; final length equals E's `final_steps` | OK (same set) |
| Conv8->Conv4 seed is really first in generation 0, for all 12 horizons (conv w8, reducer [4], stride [1], no adapter, endpoint); small alternative second; S1 only | OK |
| Typed infeasibility, no repair, input unmutated: dx, k, dy, increasing filters, stride 3, final steps, multiple reasons at once | OK (one code per defect; all reasons reported) |
| `score_folds`: mean over the four folds; three folds raise Incomplete | OK |
| `score_folds` binding of receipts to candidate and fold | **FINDING E-A1** |

## Findings

**E-A1 (FIX_REQUIRED, small).** `tools.i6n_search_plan.score_folds` only
checks each receipt against the support passed in for its key. It accepts
four receipts that came from different candidates and whose own fold labels
do not match their keys.

Reproducer (in the script): fold receipts whose
`metadata.candidate_receipt.genotype_canonical_sha256` differ (`"b"*64` and
`"a"*64`), with `candidate_receipt.fold` rotated by one key. The result was
`score_folds_mixed_candidates_and_fold_labels: ACCEPTED`.

E_RETURN names `score_folds` as the single source B should call. A
coordinator mix-up would therefore yield a fitness averaged over different
candidates without any error.

Fix: pass the expected `genotype_canonical_sha256` (and horizon). Then
require, for every fold:
- `candidate_receipt.genotype_canonical_sha256` equals the expected value;
- `candidate_receipt.fold` equals the fold key;
- `candidate_receipt.horizon_hours` equals h;
- the receipt's `support_digest` equals the frozen fold support.

Lane A's receipt v1 already carries all of these fields (INTERFACE_A A.2).
Add a test with the reproducer above.

**E-A2 (advisory).** `MIN_FINAL_STEPS = 2` is stricter than A's model rule,
which only requires that t is retained and the sequence is nonempty. It is
inherited from v1 and documented. That is acceptable, but it is an
interpretation of "vaciar la secuencia", not the literal rule.

**E-A3 (advisory).** S1 requires dx >= 2 (INT-S1-DOWNSAMPLED). As a result,
the h=5 seed uses L=30 (n=7) instead of L=24, because no dx >= 2 divides
both 5 and 24. This is consistent with "primero downsampling" but should be
visible in the plan's per-horizon seed table (it is, in `initial_genotypes`).

**E-A4 (cosmetic).** `class_of` raises `WIDTH_NOT_IN_DOMAIN` when a genotype
fits no class. That branch is unreachable, because S3 admits every domain
genotype.

## Verdict

**FIX_REQUIRED** for E-A1 only: a small guard plus a test in `score_folds`.
Everything else in my scope is ACCEPTED: domain exactness, canonicalization,
identity agreement with A's receipts, evaluator acceptance, the seed, typed
infeasibility and stride/t rules. The finding does not block A/E contract
integration, but B must not rank candidates with `score_folds` until E-A1 is
fixed or B performs the same checks.
