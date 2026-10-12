# Lane C review of lane D (external archive -> chain -> OLAP)

Reviewer: lane C (did not write D). Reviewed pins: predictor
`satoshi/i6n-swarm-D-archive-20261011@682986cd`; doin-core `4a321cd`;
doin-node `323ff7b`, `018ca2d`; data-lake `cf672f5`, `fbb6e58`;
data-warehouse `159f7fc`, `463ad56` (branch `satoshi/i6n-external-archive-20261011`
in each). All work happened in detached scratch worktrees. No D branch was
edited, and no service, network or token was touched.

## Verdict: FIX_REQUIRED

There are three defects (F1-F3) with reproducers. F1 and F2 leave a candidate
permanently non-green and retry it forever. F3 is a deploy-document rule
violation. The core custody design, which covers order, hashes, states,
duplicates, reorg, candidates and TEST, is sound and is accepted.

## Mandatory tests (each run once, CPU, `crispdm-run -m 2G -t 120s`)

| Suite | Result | D claimed |
| --- | --- | --- |
| doin-core `tests` (full) | 304 passed | 304 |
| doin-node `tests` (full, trading-stack python, worktree srcs first on PYTHONPATH) | 444 passed | 444 |
| data-lake `tests -m "not packaging"` | 39 passed, 3 deselected | 39 |
| data-warehouse `tests` | 89 passed | 89 |
| predictor `tests/test_i6n_archive.py` + `olap/store/tests/test_doin_projection_{store,migration}.py` | 29 passed (DuckDB included) | 14 + store/migration |

DuckDB 1.5.5 and duckdb_engine 0.17.0 came from a scratch venv. It was created
with `--system-site-packages --without-pip`, and a `.pth` file points at
symlinks into the existing non-prod `store-hosts-duckdb` site-packages. Nothing
was downloaded or installed.

## Plan section 6 checklist

| Check | Result | Evidence |
| --- | --- | --- |
| PUT -> exact GET -> digest -> commit | OK | `ExternalPayloadArchive.store_body` PUTs body and manifest and refuses a digest that differs from the answer. `_fetch_verified` GETs both and runs `verify_external_bytes`. `publisher.pump` submits only after `OBJECT_DURABLE`. Lake `FileByteStore` uses an fsynced temp file, a hard link, a directory fsync and a read-back hash. Tests: `test_failed_put_never_commits`, `test_readback_mismatch_returns_no_reference`. |
| Body / manifest / block hashes distinct, non-circular | OK | The body names neither digest. The manifest names `body_digest`. The block binds the tx id through the merkle root. The store also refuses equal hashes. |
| No DOIN validation removed | OK | The `chain.py` diff only adds a check after the existing hash check. doin-node 444 passed. |
| Four separate states, missing never green | OK | Separate columns; `green = all(states)`. CONFIRMING is not finalized. A reorg clears CHAIN_FINALIZED and OLAP_READBACK_VERIFIED. |
| Idempotent keys incl. C folds and VALIDATION rows | OK, with gaps F1/F2 | The row identity has the 11 shared keys including `fold`. C.2 ids `2023Q1_DCA`, `2023Q4` and `W2024-12-29` are accepted; `W2025-01-05` is refused (repro R4). VALIDATION is an admitted split. |
| Duplicate not double-counted | OK | At most one PROJECTED projection per row key; DUPLICATE events otherwise. The store and ETL tests pass. |
| Reorg revokes canonical projection, keeps history | OK | REVOKED is appended, never deleted, and reinstatement is tested. The duplicate-commit gap is documented by D. |
| All candidates kept, failed ones without score | OK | FAILED, INFEASIBLE and INTERRUPTED require `fitness=None`. A candidate-level row exists for every candidate. |
| No TEST leakage | OK (string guard) | Split TEST, a fold naming 2025 or TEST, and non-finite values are all refused. Authoritative split/cutoff enforcement stays with A/C. |
| DEPLOY_STEPS never reads token files | **FAIL (F3)** | See below. The CLI itself reads only `os.environ`. |

## Findings

**F1 (defect): the winner flag cannot be recorded after the first projection.**
INTERFACE_D says B records a candidate at its terminal disposition. The
winner (`won`) is only known when the search closes. Re-recording the same
candidate with `won=True` creates a new body and reference. The store then
refuses every one of its row keys with `CONFLICT: an active projection of this
row key carries different metrics`. That item never becomes green, the ETL
reports the error on every run, and the canonical view keeps `won=0`.
Reproducer R1: first run 0 errors; second run 1 CONFLICT; 2 items, 1 green;
canonical `won` values `[0.0]`.
Fix: record selection as its own document and row identity (for example
`fold=SEARCH_RESULT`, keyed by plan and h), or require B to record candidates
only after search closure. Pre-check row-key conflicts at `record_candidate`
time rather than at the ETL.

**F2 (defect): two attempts of one fold are accepted but can never be projected.**
`round_payload` admits receipts that differ only in `attempt.index`, but
`ROW_IDENTITY` has no attempt. The report differs through the `attempt_index`
tag, so the second receipt's rows CONFLICT. Some rows are written first, and
the item stays non-green forever. Reproducer R2: recorded=true; ETL error
CONFLICT; green 0.
Fix: refuse at record time, with one receipt per fold (the adopted terminal
attempt, lineage kept in `candidate.attempts`). Alternatively, put the attempt
in the identity of non-comparison rows.

**F3 (defect, deploy document): token files are read.**
`DEPLOY_STEPS_D.md` says nobody reads the env files. However, §2.4 ("with the
token taken from the service env by a script that never echoes it") and §3.2
("a script that takes the token from the follower env") tell a script to read
them. Fix: run these checks as transient units, for example
`systemd-run --user --wait -p EnvironmentFile=<existing env> <python> <check>`,
where the check script reads only `os.environ`, as `tools.i6n_archive` does.
Also fix the order in §3: step 3.2 must run before the stop in 3.1, as its own
text says.

**R-1 (requirement gap, low): VALIDATION rows are accepted on inner folds.**
Reproducer R3: a `2023Q1` receipt with a VALIDATION row is ACCEPTED. Inner
folds are never scored on the outer week (A.2, C.2). Refuse VALIDATION unless
the fold is a weekly unit (`W<week>`), or check against C's plan.

**R-2 (scaling risk, measured small, extrapolated).** Every ETL run re-resolves,
rewrites and re-reads every finalized reference. In addition, `_item_for` and
`_revoke` call `outbox.items()`, which is `SELECT *` including body BLOBs, for
each reference. That makes the cost quadratic. Measured with 4-fold candidates
of 17,962-byte bodies: 100 candidates took 3.92 s on the first run and 1.82 s
on the replay, and `items()` took 7.2 ms. Extrapolating to about 1,550 items
(288 search candidates plus 1,266 weekly dispositions), the scan alone is
about 0.11 s × 1,550 ≈ 170 s per run, against the 300 s cap that DEPLOY §5
suggests. Fix: skip references whose item is already verified on the same
block, and look items up by `body_digest` without reading BLOBs.

**Interface note.** INTERFACE_D.2 consumes C.1. C.2 renames the three
declared-assumption folds `2023Q1_DCA`..`2023Q3_DCA`. D's code accepts them
already (R4); only the text needs updating.

## Reproduce

```bash
B=<scratch worktrees>; PP=$B/doin-core/src:$B/doin-node/src:$B/data-lake:$B/data-warehouse:$B/predictor
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PP crispdm-run -m 2G -t 120s -- <venv python> review_d_repro.py $B/predictor
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PP crispdm-run -m 2G -t 120s -- <venv python> review_d_scale.py $B/predictor 25,100
```

Both scripts (`evidence_C/review_D/`) import D's own test fixtures (`tests/test_i6n_archive.py`:
`_rig`, `LocalChain`, `candidate`, `receipt`) and use only temporary
directories.

## Appendix: reproducer core (R1, R2)

```python
pub, archive, etl, wh, _ = T._rig(tmp); chain = T.LocalChain()
c0 = T.candidate(0, won=False); pub.record_candidate(c0, [T.receipt(c0)])
pub.pump(archive, chain.submit); chain.mint(empty_after=6); etl.run(chain.blocks)      # green
c1 = T.candidate(0, won=True);  pub.record_candidate(c1, [T.receipt(c1)])
pub.pump(archive, chain.submit); chain.mint(empty_after=6); etl.run(chain.blocks)      # CONFLICT
# R2: r2 = copy of T.receipt(c0) with attempt.index=2; pub.record_candidate(c0, [r1, r2]) -> CONFLICT
```
