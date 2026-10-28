# I6-N C2 candidate return, round 4: result-to-identity binding

Status: CANDIDATE COMPONENT. Not integrated, not READY_FOR_CAMPAIGN. Synthetic
fixtures only; no GPU, services, DOIN, TEST or OLAP writes.

Governing documents: Musashi review I6N_CANDIDATE_REVIEW_20261011/REVIEW.md
(sha256 ab05e154...), owner order SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md
(sha256 243fc9e1...). Previous rounds: C2_RETURN.md, C2_RETURN_R2.md,
C2_RETURN_R3.md (unchanged; R3 code 994282ad was accepted by a separate
reviewer, who reported the LOW item below).

## Defect closed

LOW, pre-existing (verifier of R3): the CLI never compared the inner result's
`input_sha256`/`contract_sha256` with the wrapper identity that `run()`
recomputes from the config file. A result with every internal digest
recomputed coherently (input digest replaced, or contract narrowed to fewer
horizons with groups dropped and resealed) made `--status` exit 0
COMPLETE_VERIFIED and was reused by a rerun with the honest config.

Fix: `paired_metrics_cli._verify_stored` raises
`RESULT_IDENTITY_BINDING_INVALID` unless `result.input_sha256 ==
identity.input_sha256` and `result.contract_sha256 == identity.config_sha256`.
It is used by both `run()` (reuse path, refused as OUTPUT_CORRUPT without
writing) and `--status` (exit 6 = EXIT_DAMAGED, disposition RESULT_INVALID).

| Defect | Test | PRE on 994282ad | POST |
|---|---|---|---|
| Inner result not bound to config identity: input digest replaced | test_r4_status_and_rerun_refuse_a_result_not_bound_to_the_config_identity[input_digest_replaced] | FAIL (status did not report damage) | PASS: --status 6 RESULT_INVALID; rerun OUTPUT_CORRUPT; bytes unchanged |
| Same, contract narrowed to one horizon | ...[contract_narrowed] | FAIL | PASS |
| Precondition: both forgeries pass verify_result alone | test_r4_forgeries_verify_in_isolation_so_only_the_identity_binding_can_refuse_them | PASS | PASS |

## Reproduction

PRE (test file of this round run against the R3 code):

    git worktree add --detach <scratch> 994282ad
    cp tests/test_i6n_paired_metrics_r4.py <scratch>/tests/
    cd <scratch> && CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- \
      python -m pytest -q -p no:cacheprovider -rA tests/test_i6n_paired_metrics_r4.py

Observed (run in the C2 worktree before the fix, code identical to 994282ad):
exit 1, "2 failed, 1 passed"; retained as evidence_r4/PRE_pytest_on_994282ad.txt.

POST (this commit):

    CUDA_VISIBLE_DEVICES="" crispdm-run -m 2G -t 120s -- python -m pytest -q -p no:cacheprovider \
      tests/test_i6n_paired_metrics.py tests/test_i6n_paired_metrics_r3.py tests/test_i6n_paired_metrics_r4.py

Observed: exit 0, "212 passed"; retained as evidence_r4/POST_pytest.txt.
git diff --check clean.

Pins: tests/test_i6n_paired_metrics_r4.py sha256 0f19050a35c20f0a4ff362b057d143ff8385dd4ad849693d7183a4e63bce7346;
tools/i6n_contracts/paired_metrics_cli.py sha256 590b10a6374522df443c83503c366458f308e97469713d691976fabac106fb21.

## Current limits

- Digests are unkeyed. A forgery that changes metric values inside groups
  while keeping counts, signs, group set, contract and identity consistent is
  still undetectable without the rows and external custody; only rescoring on
  reuse would catch it. No signatures, leases or calendar were added.
- `--status` alone cannot detect a directory rewritten consistently: if the
  forged result and BOTH the RESULT.json and STATUS.json identities are set to
  the forged digests (wrapper and STATUS summary resealed), `--status` reports
  COMPLETE_VERIFIED with exit 0, because it has no config to recompute the
  identity from and the digests are unkeyed. Only a rerun with the real config
  refuses it (OUTPUT_IDENTITY_MISMATCH, nothing written). An orchestrator that
  must not trust the directory should rerun with its config instead of relying
  on `--status`.
- Independent verification of this round (separate reviewer, one finite run on
  efce4b63): forgeries (a) inner input digest replaced and (b) contract
  narrowed -> `--status` 6 RESULT_IDENTITY_BINDING_INVALID, rerun 2
  OUTPUT_CORRUPT; (c) wrapper identity forged with honest STATUS -> 6
  RESULT_IDENTITY_MISMATCH, rerun 2; (d) both identities forged -> `--status`
  0, rerun 2 OUTPUT_IDENTITY_MISMATCH; honest control 0/0; bytes unchanged in
  every case; suite 212 passed; pins matched. The reviewer required only the
  two documentation corrections applied in the commit after efce4b63.
- All R3 limits still apply (C2_RETURN_R3.md).
