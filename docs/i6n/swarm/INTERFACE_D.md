# Lane D interface (D.4)

Machine-readable contract: `INTERFACE_D.json`. D.1 was published early at
`4bedc36c`; D.2 aligned with INTERFACE_A A.2, INTERFACE_E E.1 and INTERFACE_C C.1. D.3 applies lane C's
review (REVIEW_C_OF_D.md) and C.2.

- **For B.** Call `ArchivePublisher.record_candidate(candidate, fold_receipts)`
  whenever a candidate reaches any terminal disposition. Pass the candidate
  document (`i6n.archive.candidate.v1`) and A's per-fold `candidate_receipt`
  objects. Every candidate goes in, including failed ones with
  `fitness=null`. The call only writes the local outbox (LOCAL_COMPLETE), so
  it never blocks a fit.
- **Pump and the node.** `python -m tools.i6n_archive pump` stores the bytes
  first: PUT, then an exact GET, then a digest compare. Only after that does
  it write the reference transaction to `<spool>/<tx_id>.json`. The node calls
  `doin_node.archive.external.ingest_reference_spool(...)` with
  `consensus.record_transaction`. That hook is wired by B in the runtime.
- **ETL.** `python -m tools.i6n_archive etl` checks the node's `chain.json`.
  It revokes projections whose blocks are no longer canonical and resolves
  finalized bodies by locator and digest. It then projects through the
  warehouse routes `/api/v2/doin/projections*`, reads back exactly, and ACKs.
- **Identities.** `effective_config_sha256` and `search_candidate_sha256` use
  E's formulas, and D re-verifies both. Receipt row identity uses the default
  shared keys. `role` must be the C1 canonical MANAGEMENT|ENTRY.
- **States.** LOCAL_COMPLETE, OBJECT_DURABLE, CHAIN_FINALIZED and
  OLAP_READBACK_VERIFIED are kept as separate columns, and a missing state is
  never shown green.

## D.3 changes (lane C review)

- **F1.** `won` is no longer a field of the candidate record; a candidate that
  carries it is refused with `WON_IS_A_DECISION_EVENT`. Search outcomes go
  through `record_decision`. It writes an append-only decision event
  (WINNER/FINALIST/CONTROL/NOT_SELECTED, with an increasing sequence), which
  gets its own body, reference and projection rows. The latest decision is in
  `gov_doin_candidate_decision_current`. A row key bound to different content
  is refused at record time (`ROW_KEY_CONFLICT`).
- **F2.** Up to three physical attempts of one fold are legitimate. The lowest
  attempt index is the one canonical measurement that gets projected. The
  other attempts stay in the body and in `attempt_history_json`, so nothing is
  counted twice.
- **F3.** Deploy checks use `tools.i6n_archive.probe` under
  `systemd-run -p EnvironmentFile=`. They never read a token file.
- **R-1.** VALIDATION rows on inner folds are refused with
  `VALIDATION_ON_INNER_FOLD`.
- **R-2.** The ETL is incremental, using the `etl_done` watermark and BLOB-free
  lookups.

## D.4 changes (execute-now orders, task C)

- **Off-chain readback.** `ArchiveETL.run_offchain` gives OLAP readback for
  items that are durable in the lake, hash-verified and whose reference is
  spooled to the node. Their finality is `OFFCHAIN_VERIFIED` and
  `chain_finalized` stays false. A later FINALIZED projection supersedes them
  and never duplicates them.
- **Lifecycle status.** `tools.i6n_archive.lifecycle` reports each fold and
  candidate stage separately, with the ETA kind and its sample count.
- **Not deployed.** See D_DEPLOY_RECORD.md.
