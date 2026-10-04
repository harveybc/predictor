# M2 readiness repair return - 2026-10-04

## Scope

This repair addresses the four findings against commit `1acbf72d`. It does not
change the causal result, issue a selection manifest, authorize PS5 training,
or start, stop, or inspect a GPU process.

## Test-first evidence

The initial focused run produced 7 failures and 13 passes. The failures
reproduced all governing defects: internally resealed row-identity mutations
were adopted, the three alternative terminals had the parent extractor hashes,
the report omitted verified family identity, and the report still used
hard-coded per-feature fields.

The final focused run is:

```text
30 passed in 3.26s
```

Mutation tests now reseal both `results.jsonl` and `run_manifest.json`, then
verify rejection of changed feature, family, seed, or fold identities. Feature
identity is exercised for every row kind; family and fold identity are
exercised for every row kind that carries those fields.

## Correct alternative terminals

The retained evidence was replaced from the completed `pass_p2c` directories
on `<worker-a>`. The configured source roots remain data, not source-code
constants. The accepted family set is
`identity`, `random`, `past_to_current_siamese`.

| Feature | SHA-256 |
|---|---|
| `tv.hilbert_amp` | `2cb5445214590c7aa68bb654c1f44828d2e8371c6bf68db6dd2bcc2259309eab` |
| `tv.kalman_dev` | `3746ee3effd0ff7c5b9737b5a7da77b0ada34843cd97a36a3fbb14a4f4436cf1` |
| `tv.stl_dev` | `8c182faa5a4c862f49b691545739773a48b37a2b7e21d398c62f1aa6a86ef0af` |

## Report and artifact guarantees

`REPORT.json` now derives verified terminal names, hashes, roles, families,
folds, seed, code revision, input digest, utility, and locator exclusively from
the verified ingestor output. The generator no longer contains the old
per-feature baseline names or hashes.

The render test executes the real emitter and requires all five declared
artifacts, including the already tracked
`source_transform_coverage/ps4_emitted_profile_status.csv`.

## Preserved scientific state

- Causal population: 279 `NOT_IDENTIFIED` and 87 `OUTSIDE_JOIN_PENDING`.
- PS5 candidate state: all 366 `NOT_READY_EVIDENCE_INCOMPLETE`.
- Selection manifest: not issued.
- PS4 complete for all 366: false.

## Requirement-to-evidence traceability

| Requirement | Evidence |
|---|---|
| Adopt real `pass_p2c` terminals | Configured expected hashes plus retained manifests and results |
| Bind result rows to terminal identity | Resealed mutation tests in `test_ps3r_manifest_ingestor.py` |
| Remove report hard-coding | `verified_terminals` generated from ingestor decisions |
| Emit every declared artifact | `test_emit_writes_every_declared_artifact` |
| Preserve causal and PS5 populations | Ledger assertions in `test_m2_readiness.py` |

