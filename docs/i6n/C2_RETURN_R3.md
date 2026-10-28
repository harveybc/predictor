# I6-N C2 correction round 3 return

Status: this is a candidate component in a correction round. It is not
integrated into any executor, it is not deployed, it is not a scientific
measurement, and it is not READY_FOR_CAMPAIGN. Component acceptance is the
reviewer's decision. No DOIN, fits, GPU, TEST, brokers, services, network or
databases were used. Every run was CPU only, under
`CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s`.

This file does not edit `docs/i6n/C2_RETURN.md` or `docs/i6n/C2_RETURN_R2.md`.
Both stay as historical evidence. Where they disagree with this file, this file
governs.

Governing documents (not committed on this branch; digests verified):

- Review `docs/audits/evidence/I6N_CANDIDATE_REVIEW_20261011/REVIEW.md` (Musashi),
  sha256 `ab05e154f882ad0bf4ad15bac75f62a93a8d18f19fd574b855973f994540e069`.
- Order `docs/handoffs/SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md`,
  sha256 `243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025`.
- The independent verifier's defects on `cdc85fb5`, which this round answers:
  - MEDIUM: result-level group set is not checked against the contract.
  - LOW: untyped `KeyError` when `contract` is missing.

## R3 result-level checks

### Successor pin

| Role | Commit |
|---|---|
| Audited code (PRE) | `cdc85fb59c0bc805fddd540232c809d409551dd8` |
| Previous tip (R2 return, docs only) | `24ec1d56` |
| Successor code and tests (POST, probe replay) | `994282ad25c8959e114b4394f439dd2cd3c9a9d6` |
| This document | the commit that adds this file, directly after `994282ad` on `satoshi/i6n-c2-paired-metrics-20261011` |

File digests at `994282ad`:

| File | sha256 |
|---|---|
| `tools/i6n_contracts/paired_metrics.py` | `88d4e7e9cb7ab66bfce0b4672c988ba4d4b2b85496d6113e1e5748b3c55bed50` |
| `tests/test_i6n_paired_metrics_r3.py` (new; identical to the file run PRE) | `04b7173b070753425e2f6ef5597d0237efeaef2e968a7d7cf1a65a9dd6a414e7` |
| `tests/test_i6n_paired_metrics.py` (two lines amended, see below) | `32bf6a8f6e06812b9387ff0b334d40d1e3a86ed60e276dc3b45c47d1b22c847e` |

### What changed

- `verify_result` now requires the result-level fields `contract`, `state`,
  `olap_records`, `contract_sha256`, `input_sha256`,
  `opportunity_support_sha256` and `usage_receipts_sha256`. A missing field is
  `SCHEMA_INVALID`. The contract is re-validated with the same validator that
  `score` uses, and must already be in normalised form.
- `contract_sha256` must equal `canonical_sha256(contract)`. If it does not,
  the code is `CONTRACT_DIGEST_INVALID`. To make this hold for every honest
  result, the normalised contract now keeps the optional `description`
  verbatim. Before this change, `score` hashed the raw contract including its
  description but stored the contract without it. The shipped synthetic
  example has a description. `score` now refuses with `CONTRACT_INVALID` if
  the normalised and declared digests ever differ.
- The expected group set is
  `expected_group_keys(contract) = {S, U} x horizons_hours x (roles + [ALL_ROLES])`,
  with exactly one group per key. A key outside the set is `GROUP_UNDECLARED`.
  A key listed twice is `DUPLICATE_GROUP`. A declared key with no group is
  `GROUP_MISSING`; an empty `groups` list under a non-empty contract ends here.
  - **Derivation from the contract.** The set is derived from the contract
    fields only. A contract cannot omit a population kind: `score` always
    emits both S and U, and an empty S support or a run with no receipts gives
    groups in state `NO_ELIGIBLE_ROWS` / `NO_RECEIPTS`, not absent groups.
    `roles` and `horizons_hours` are validated non-empty and unique, with
    `ALL_ROLES` excluded from `roles`. So the expected set is never empty, and
    no legitimate contract omits U or a role.
  - **Ordering.** Group order is not part of the check. A permuted honest
    result still verifies; this is a control test.
- Each group's `unit`, `scale`, `config_id`, `protocol_id` and `calendar_id`
  must equal the contract's. Its contract and population digests must also
  equal the result's. A mismatch is `RESULT_INCONSISTENT`.
- The embedded `olap_records` must equal the rows rebuilt from the verified
  groups. Otherwise the code is `RESULT_INCONSISTENT`. This hardening belongs
  to the same family as the double-counting defect: a resealed result could
  otherwise carry doubled rows in its stored copy.
- `olap_records(result)` now verifies first, then builds rows. `score` uses the
  private unverified builder `_olap_rows`, because its result is not sealed
  yet.
- `compare_configs` already called `verify_result` on both results. It now
  inherits the checks above, so it never raises `KeyError` and never returns
  COMPARABLE for a result whose groups differ from its contract.
- The CLI is unchanged. `--status` and run-reuse already route through
  `verify_result` (`_verify_stored`), so each forgery below now reports exit 6
  `RESULT_INVALID`, and a rerun reports `OUTPUT_CORRUPT` with nothing
  overwritten.
- The pure module is still stdlib-only. Its imports are unchanged.

### Defect -> test -> PRE -> POST

Every forgery is self-consistent. Groups are edited and resealed, the OLAP
rows are regenerated from the edited groups, and the result is resealed. For
the CLI rows, the RESULT wrapper digest and the STATUS summary are also
rewritten. This means only a result-level check can refuse it.

Each forgery is parametrised across four tests in
`tests/test_i6n_paired_metrics_r3.py`:

- `test_r3_verify_result_refuses_a_result_whose_groups_or_contract_disagree`
- `test_r3_compare_configs_refuses_with_a_typed_code_never_key_error_or_comparable` (both argument orders)
- `test_r3_olap_records_refuses_with_a_typed_code_and_never_double_counts`
- `test_r3_status_never_reports_success_for_a_resealed_forgery` (also checks run-reuse refusal and byte and mtime immutability)

In the table, PRE is the code at `cdc85fb5` and POST is the code at `994282ad`.
The honest fixture is one record at h=2, with contract horizons [2, 72] and two
roles. It gives 12 groups and 12 OLAP rows.

| Defect | Forgery (parameter id) | PRE `cdc85fb5`: verify / compare / olap / --status | POST `994282ad`: all four |
|---|---|---|---|
| MEDIUM (a) success with no metrics | `no_groups`: `groups=[]` under the same contract | COMPLETE / CONTRACT_MISMATCH / 0 rows / exit 0 COMPLETE_VERIFIED | `GROUP_MISSING`; status exit 6 RESULT_INVALID |
| MEDIUM (b) double counting | `duplicate_group`: S/h2/entry listed twice | COMPLETE / **COMPARABLE** / 13 rows (12 expected) / exit 0 | `DUPLICATE_GROUP`; exit 6 |
| MEDIUM, missing group | `missing_group`: last group (U/h72/ALL_ROLES) dropped | COMPLETE / **COMPARABLE** / 11 rows / exit 0 | `GROUP_MISSING`; exit 6 |
| MEDIUM, extra group outside the contract | `extra_horizon_group` (h=24, undeclared) | COMPLETE / CONTRACT_MISMATCH / 13 rows / exit 0 | `GROUP_UNDECLARED`; exit 6 |
| MEDIUM, extra group outside the contract | `extra_role_group` (role `undeclared_role`) | COMPLETE / CONTRACT_MISMATCH / 13 rows / exit 0 | `GROUP_UNDECLARED`; exit 6 |
| MEDIUM, same count but wrong set | `undeclared_group_replacing_a_declared_one` | COMPLETE / CONTRACT_MISMATCH / 12 rows / exit 0 | `GROUP_UNDECLARED`; exit 6 |
| MEDIUM, contract body vs digest | `contract_changed_digest_kept`: `contract.unit` edited, `contract_sha256` kept | COMPLETE / CONTRACT_MISMATCH / 12 rows / exit 0 | `CONTRACT_DIGEST_INVALID`; exit 6 |
| MEDIUM, contract vs groups | `contract_changed_digests_recomputed`: `contract.unit` edited, every contract digest recomputed, groups keep the old unit | COMPLETE / CONTRACT_MISMATCH / 12 rows / exit 0 | `RESULT_INCONSISTENT`; exit 6 |
| LOW, untyped KeyError | `missing_contract`: resealed result without `contract` | COMPLETE / **KeyError** / **KeyError** / exit 0 | `SCHEMA_INVALID`; exit 6 |
| Same family (stored rows) | `doubled_embedded_olap_rows`: stored `olap_records` padded with 2 repeated rows | COMPLETE / COMPARABLE / 12 rows / exit 0 | `RESULT_INCONSISTENT`; exit 6 |

Under PRE, the `compare` column shows that `compare_configs` already refused
some forgeries with `CONTRACT_MISMATCH`, because the group sets of the two
results differed. It did not refuse the duplicate-group case, the missing-U
case or the stored-rows case. The other three PRE columns accepted every
forgery.

Further tests in the same file:

| Test | PRE | POST |
|---|---|---|
| `test_r3_duplicate_group_is_not_published_twice` (olap refuses the duplicate; honest `olap_records(result) == result["olap_records"]`) | FAIL (DID NOT RAISE) | PASS |
| `test_r3_contract_with_a_description_verifies_and_its_digest_is_the_input_digest` (example contract with description verifies; an edited description is CONTRACT_DIGEST_INVALID) | FAIL (stored contract lacked the description, so its digest differed from `contract_sha256`) | PASS |
| `test_r3_expected_group_set_is_kinds_x_horizons_x_roles_plus_all_roles` (control) | PASS | PASS |
| `test_r3_permuted_groups_of_an_honest_result_still_verify` (control) | PASS | PASS |
| `test_r3_honest_pair_still_compares` (control) | PASS | PASS |
| `test_r3_group_entry_role_helper_matches_fixture` (fixture guard) | PASS | PASS |

Amended pre-existing test, in `tests/test_i6n_paired_metrics.py`:

- `test_comparison_refuses_different_group_sets` now expects `GROUP_MISSING`
  instead of `CONTRACT_MISMATCH`, because the stricter `verify_result` refuses
  the malformed result first. On `cdc85fb5` the original assertion passed with
  `CONTRACT_MISMATCH`. The amended assertion was not run PRE.
- `_forge_result` (a helper) now builds rows with `pm._olap_rows`, because the
  public `olap_records` verifies its input and the helper's body is not sealed
  yet.

The compare-side `CONTRACT_MISMATCH "group sets differ"` branch can no longer
be reached by two verified results. It is kept as a defensive check.

### PRE (retained)

```
git worktree add --detach <scratch>/pre cdc85fb59c0bc805fddd540232c809d409551dd8
cp tests/test_i6n_paired_metrics_r3.py <scratch>/pre/tests/     # sha256 04b7173b...
cd <scratch>/pre
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider -rA tests/test_i6n_paired_metrics_r3.py
-> exit 1: "42 failed, 4 passed in 0.86s"
```

The 42 failures break down as follows:

- 23 are `DID NOT RAISE`.
- 10 are `--status` exit 0 where exit 6 was expected.
- 6 are `compare_configs` returning `CONTRACT_MISMATCH` instead of the
  result-level code.
- 2 are `KeyError: 'contract'`, one from `compare_configs` and one from
  `olap_records`.
- 1 is a digest mismatch on the description test.

The 4 passes are the controls and the fixture guard. The per-API PRE outcomes
in the table above come from a scratch observer script. It was not committed.
It runs each forgery through `verify_result`, `compare_configs`,
`olap_records` and the CLI `status()` on a temporary output, against the same
PRE checkout, and it was run once.

### POST (retained)

In the C2 worktree at the successor code, before committing `994282ad`:

```
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics.py tests/test_i6n_paired_metrics_r3.py
-> run 1: exit 1, "1 failed, 208 passed" (test_comparison_refuses_different_group_sets: GROUP_MISSING != CONTRACT_MISMATCH)
-> amended that assertion (see above), run 2: exit 0, "209 passed in 1.16s"
git diff --check   -> exit 0
```

Run 2 is the only green run. No green suite was rerun. The committed tree
`994282ad` is byte-identical to the code of run 2.

### Review probe replay on the new tip

```
git worktree add --detach <scratch>/post 994282ad25c8959e114b4394f439dd2cd3c9a9d6
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python <review-dir>/probe_candidates.py c2 --repo <scratch>/post
-> exit 1: PairedMetricsError: GROUP_INCONSISTENT: groups[0]: n_declared 5 != n 1 + n_excluded 0 + failures 0
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python docs/i6n/evidence_r2/probe_c2_per_scenario.py --repo <scratch>/post
-> exit 0, commit 994282ad, component_files_dirty false:
   phantom_population    REFUSED GROUP_INCONSISTENT (n_declared 5 != n 1 + n_excluded 0 + failures 0)
   negative_error_sums   REFUSED GROUP_INCONSISTENT (sum_abs_error is negative (-0.25))
   missing_result_status exit 6, state RESULT_INVALID, disposition RESULT_MISSING, result_state null
```

This is the same outcome as on `cdc85fb5` (see `C2_RETURN_R2.md`). The
unmodified probe has no exception handling, so its exit 1 is the typed refusal
of its first scenario, not a crash. Round 3 did not regress the R2 fixes.

### Reproduction pinned to exact commits

PRE (expected exit 1, 42 failed / 4 passed):

```bash
git -C <repo> fetch origin satoshi/i6n-c2-paired-metrics-20261011
git -C <repo> worktree add --detach <pre> cdc85fb59c0bc805fddd540232c809d409551dd8
git -C <pre> show 994282ad25c8959e114b4394f439dd2cd3c9a9d6:tests/test_i6n_paired_metrics_r3.py > <pre>/tests/test_i6n_paired_metrics_r3.py
cd <pre>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics_r3.py
```

POST (expected 209 passed, clean diff check, probe as above):

```bash
git -C <repo> worktree add --detach <post> 994282ad25c8959e114b4394f439dd2cd3c9a9d6
cd <post>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider tests/test_i6n_paired_metrics.py tests/test_i6n_paired_metrics_r3.py
git diff --check cdc85fb59c0bc805fddd540232c809d409551dd8 HEAD
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python <review-dir>/probe_candidates.py c2 --repo <post>
CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python docs/i6n/evidence_r2/probe_c2_per_scenario.py --repo <post>
```

### Interface changes for readers of C2 output

- New typed codes: `CONTRACT_DIGEST_INVALID`, `GROUP_UNDECLARED`,
  `DUPLICATE_GROUP` and `GROUP_MISSING`. `RESULT_INCONSISTENT` now also covers
  group-vs-contract identity and stored OLAP rows.
- New public helper: `expected_group_keys(contract)`.
- `olap_records` refuses an unsealed or unverifiable result. Callers that
  build rows for an unsealed body must not use it.
- A normalised contract keeps `description`. `RESULT_SCHEMA_VERSION` is
  unchanged (`v2`). An honest v2 result scored by `cdc85fb5` from a contract
  with a description fails the new digest check (`CONTRACT_DIGEST_INVALID`),
  because its stored contract dropped the description. Rescore such results;
  this is a candidate and none are in custody. Contracts without a description
  give byte-identical results.
- `compare_configs` now reports a malformed result's own code (for example
  `GROUP_MISSING`) where it used to report `CONTRACT_MISMATCH`.

### Current limits

- **Candidate only.** There is no integration with C1 or any executor, no
  warehouse or OLAP write, and no OLAP schema change. The synthetic fixtures
  are not a scientific measurement, and not end-to-end or trading acceptance.
  Not READY_FOR_CAMPAIGN.
- **Unkeyed digests.** The new checks refuse results whose groups,
  contract, digests or stored rows contradict one another. A forgery that is
  plausible and fully self-consistent is still undetectable without the rows
  and external custody: for example, a rewritten contract with every group
  regenerated to match it and every digest recomputed. This round adds no
  signatures, leases or calendar system.
- **Not checked at result level:**
  - `input_sha256`, `counts` and `design_basis` are not re-derived; they need
    the input document.
  - `opportunity_population_id` and `opportunity_sealed_at` are not
    cross-checked against the groups' `population_id`.
  - Whether the S and U populations are true is out of reach. Only their
    digests' agreement with the groups is checked.
- **Group order** is not part of the contract. A permuted result verifies, and
  its rebuilt OLAP rows follow the stored order.
- **One mutation-style family per defect.** No mutation run this round. The
  guarantees rest on the PRE-failing regressions above.
- The R2 limits that are not restated here still apply: sum relations are
  necessary conditions only, C2 is independent of C1, sealing time needs
  custody, scales are compared by label, there is no lease, `atomic_json` is
  single-writer, and the fixtures use binary-fraction prices.
