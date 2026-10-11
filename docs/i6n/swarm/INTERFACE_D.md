# Lane D interface (D.1, published early)

Machine-readable contract: `INTERFACE_D.json`.

- **For B:** when a candidate reaches any terminal disposition, call
  `ArchivePublisher.record_candidate(candidate, fold_receipts)` with the
  candidate document (`i6n.archive.candidate.v1`, all candidates, failed ones
  with `fitness=null`) and A's per-fold `candidate_receipt` objects. It only
  writes the local outbox (LOCAL_COMPLETE); it never blocks a fit. `pump` and
  the ETL run independently (durable outbox, replay after outages).
- **From A:** receipt rows are consumed unchanged; D derives mae/naive/mse/
  skill from the sums and counts, never from averaged batches.
- **Archive:** body bytes first (content-addressed PUT, exact GET, digest
  compare), then the reference transaction. Body, manifest and block hashes are
  distinct and non-circular. `full_payload` keeps the same body inside the
  transaction and projects to identical analytic rows.
- **States:** LOCAL_COMPLETE, OBJECT_DURABLE, CHAIN_FINALIZED,
  OLAP_READBACK_VERIFIED are kept separately; missing is never green.
