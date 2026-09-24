# RP143 Musashi PRE probes

Preserves the two lightweight, in-memory probes from the read-only review at
`07140d03a265d235584235216dad3346be569853`. Their original inputs and operations
are combined in `probe_pre.py`, with resource limits and structured output added.
`observed_pre.json` records the preserved script's one execution. It is an
observation, not an assertion that these defective behaviors should remain.

Run from the worktree root:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B docs/audits/evidence/RP143_MUSASHI_REVIEW_2026_09_24/probe_pre.py
```

The script writes JSON only to stdout, disables bytecode and GPU visibility,
limits NumPy threads to one, and enforces 10/15 CPU seconds, a 20-second wall
alarm and 512 MiB address space. It imports no TensorFlow, PyTorch or pandas;
it loads no models or benchmark datasets and contacts no services. It executes
the current local source; source hashes identify what was actually examined.

## Evidence boundaries

- Binding and pooling call actual tool functions with synthetic dictionaries.
  Pooling receives synthetic accepted-terminal/verified-closure flags; this is
  not an end-to-end `compose_evidence` or production custody check. Its catalog
  reader is mocked to prevent filesystem access outside the evidence inputs.
- The mask probe runs the actual `_TrainWindows.__getitem__` through a fake
  PyDataset base and a constant synthetic Dataset. It tests repeated indexing,
  not the TensorFlow fit loop.
- The retained composition JSON is read only to count its selected replay
  classes and average its 12 stored scores. None of its original artifacts,
  identities or receipts is independently verified. Selection counts are four
  reports and eight histories labeled `IDENTITY_BOUND`; none selects the weak
  metric-only evidence. The mean is therefore not disproved by that weak-binding
  bug alone.
- Findings 3 (independent target-check gap) and 5 (reload/update proof gap) were
  source-review findings, not runtime failures reproduced by these probes.
  No model execution, production identity claim, or news-signal audit is included.
