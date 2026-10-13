# H72 Readiness Checklist

This checklist prepares H72 while H4 runs. It does not authorize H72 before the
existing terminal, common-chain and archive/OLAP gates close.

## Network warnings

- [x] Separate logical author, physical relay and next-hop relay metadata.
- [x] Exclude the physical sender by `peer_id`, not by its IP string.
- [x] Preserve the logical message identity while changing transport metadata.
- [x] Pass focused gossip and unified-node tests (96 passed).
- [ ] Pin prepared `doin-node` commit `c767509` (or its reviewed descendant) in
  the H72 private deployment spec.
- [ ] Run a disposable four-node relay test and require zero wrong-host endpoint
  attempts before production deployment.
- [ ] Deploy only after H4 is quiescent; never replace the code of a live H4 node.

## Disk space

- [ ] At H4 terminal, record candidate count and bytes for every island.
- [ ] Compute measured bytes per completed candidate per island.
- [ ] Project at least 600 H72 candidates per island before early stopping.
- [ ] Require `projected_remaining_bytes + 20 GiB reserve <= free_bytes`.
- [ ] Move immutable, hash-verified candidate payloads to the external archive
  and verify GET/readback before removing a local duplicate.
- [ ] Keep checkpoints, active candidate inputs, terminal ledger and unresolved
  outbox items local. Never delete them to satisfy admission.
- [ ] On omega, move governed historical stores only through their own verified
  migration procedure; do not manually delete `.local/state` or worktrees.
- [ ] On gamma, do not launch H72 until the projected evidence fits after the
  H4 archive is closed or the archive root is relocated to adequate storage.

## RAM and GPU

- [ ] Run one exact full-size H72 workload pilot on omega, dragon and gamma.
- [ ] Measure peak cgroup RAM, process RSS, VRAM, wall time and artifact bytes.
- [ ] Set each production cap to at least 1.25 times its measured peak when that
  cap fits; otherwise report a hardware refusal without trimming the margin.
- [ ] Keep both gamma DOIN nodes reachable, but use one host-wide heavy-fit
  lease. The 5090 has priority and the 5070 Ti consumes only free slots.
- [ ] Concurrent 5090 and 5070 Ti fits remain disabled until a fresh concurrent
  pilot, normally after a memory upgrade, proves adequate host reserve.
- [ ] A missing GPU or CUDA failure is terminally visible; never fall back to CPU.

## H72 scientific identity

- [ ] Keep the primary H72 experiment hourly/raw with exact `t+72h` target.
- [ ] Include the current known tick `t` in every input window.
- [ ] Use the same business eligibility mask for model and 72-hour naive.
- [ ] Keep TEST inaccessible to search, early stopping and fallback selection.
- [ ] Require exact seed, commits, config digest, network and genesis on all nodes.

## Resource contingency

- [ ] Open H72-DS-IN only after a typed baseline resource refusal.
- [ ] H72-DS-IN uses unsmoothed `t, t-k, ...` inputs and the original `t+72h`
  target, support and naive under a new identity.
- [ ] Open H72-DS-INOUT only if H72-DS-IN is still insufficient.
- [ ] H72-DS-INOUT must land exactly on `t+72h` and retain original-time paired
  error/naive rows under another new identity.
- [ ] Never reinterpret a fallback as continuation of the baseline blockchain.
