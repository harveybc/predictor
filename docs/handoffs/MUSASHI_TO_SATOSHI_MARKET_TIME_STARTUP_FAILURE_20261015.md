# Satoshi: repair the deployed market-time worker startup

Observed 2026-10-15 05:41 UTC, runtime `67a3f86a`, campaign
`i6dmo-f3ae94f9adcf`. No scientific terminal exists yet: short.fit QUEUED,
five units WAIT_DEPENDENCY, no running attempt. The preferred worker exits 1
before claim. Its exact failure is:

```text
worker._artifacts -> data.data_contract_sha256(design)
body = {k: design[k] for k in DATA_CONTRACT_KEYS}
KeyError: 'input'
```

The sealed market-time design deliberately has `market_time` instead of the
elapsed design's `input`. This is another legacy contract check in the real
worker path, below the sampling gate already fixed in 9dd8bbd3.

1. Fix the contract dispatch in worker._artifacts for both short and long.
   Authenticate the market-time design and its exact frozen bundle using the
   existing model/data adapter and its declared data identity. Verify how the
   deployed data_contract_sha256 pins were derived before comparing them; do
   not skip validation, invent an input block or compare unrelated digests.
   Preserve typed mismatch rejection and existing elapsed-mode behavior.
2. Add a focused regression with the real sealed MG2/MG12 designs AND real
   bundle contract pins, not fake pins that omit data_contract_sha256. Reach
   the pre-claim worker path; demonstrate PRE failure and POST acceptance,
   with wrong-contract rejection for both families. No scientific fit needed
   to establish this interface regression.
3. Generated status must expose worker startup failures, exit code and cause
   with a timestamp. WAITING_FOR_WORKER with an indefinitely failing service
   is misleading. Read the actual service state in the operational status
   collector and keep runtime/service failures distinct from scientific loss.
4. Check the current campaign again before any new deployment. If no fit has
   started, publish a new immutable runtime identity and deploy it through the
   existing mechanism, with consistent pins across executing hosts. Preserve
   the failed startup record. Do not edit the sealed pin in place. If work
   has started independently, reconcile its identity before replacing it.
5. Verify ONE actual GPU child, its lease and first real progress record; then
   return control to the existing timers for short -> long -> pair. No extra
   owner approval, no agent polling. No TEST read or DOIN change.

Fleet observation: all three hosts reachable; 5090 visible, no model fit in
GPU memory; sufficient available RAM at this check. Omega home has 13 GB free
(99% used), root scratch 49 GB, and the NTFS output volume 711 GB. The existing
archived short terminal is visible on that volume. Retain the split between
ext4 mutable state and NTFS immutable artifacts. No hardware restart indicated.
