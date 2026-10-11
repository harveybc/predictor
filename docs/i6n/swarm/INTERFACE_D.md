# Lane D interface (D.2)

Machine-readable contract: `INTERFACE_D.json`. D.1 was published early at
`4bedc36c`; D.2 aligns with INTERFACE_A A.2, INTERFACE_E E.1 and INTERFACE_C C.1.

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
