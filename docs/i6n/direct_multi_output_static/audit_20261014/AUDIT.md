# Independent I6-DMO audit, 2026-10-14

Audited predictor `fdaae9d2`, runtime `93078a0f`, order `2a440265`.
Verdict: **REVISE_BEFORE_CONTINUE**, not perfect or end-to-end accepted.
No scientific fit was repeated and no TEST rows were read by this audit.

## Findings, in severity order

1. **P1 / AUD-DMO-01: sampling contradicts the sealed design.** The design
   requires exact timestamps (`design.py:113`), but `data.py:430-445` chooses
   each feature's last finite older observation with no age bound. The model
   never receives the retained staleness array. The independent probe accepts
   a current-feature value 70 hours old; the actual bundle records up to 72 h.
   This is not proof of future leakage, but it is a different scientific
   treatment and can omit a current feature observation. DMO02 cannot be
   accepted as the planned exact-input property. Retain the ASOF result as a
   separately labelled diagnostic; restore exact DS2 under a new identity.

2. **P1 / AUD-DMO-02: final refit is wired to the old training design.**
   `unit_runner.py:49-53` selects the original family design without applying
   the final unit's 2021-2025 window. The adapter fixes TRAIN to 2020-2024
   (`design.py:123`) and the loader enforces its boundaries. The independent
   probe reproduces the wrong training interval. A correct final bundle cannot
   simply be supplied to this unchanged path. Implement a typed final-refit
   protocol and test its actual CLI, not only queue enumeration.

3. **P1 / AUD-DMO-03: the final TEST execution path is a placeholder.**
   `strategy_adapter.py:1070` always returns FAILED /
   FINAL_TEST_HOOK_NOT_PREPARED for final_test. The independent probe reproduces
   it without opening TEST. Also complete guarded final bundle construction
   and prediction generation. The timers cannot finish the declared campaign
   while this remains true.

4. **P2 / AUD-DMO-04: publication client violates the warehouse SQL limit.**
   The original `outbox.py:218` requested LIMIT 10000; the shared warehouse
   contract permits 5000. Authenticating did not solve it: the cost report was
   POSTED but readback returned HTTP 400. Fixed in this audit branch by using
   shared MAX_LIMIT and checking COUNT(*) before/after fetch; overflow and
   cardinality mismatch fail closed. Four independent regressions failed PRE;
   all four plus four existing outbox tests passed POST. The existing five cost
   metrics are now DELIVERED with exact readback, not merely accepted by POST.

5. **P2 / AUD-DMO-05: deployment selected a filesystem with inadequate space.**
   Live STATUS showed short.report WAIT_RESOURCE / DISK_BELOW_GATE with no
   worker running. Coordinator home had about 16 GiB free, while its root
   filesystem had about 51 GiB. Its first completed fit is not still running:
   GPU receipt shows 32 epochs, best epoch 22, 2272 updates, fit 32.218 s,
   trainer wall 36.781 s, cgroup peak 2431946752 bytes. Relocate the new corrected
   campaign using actual filesystem/admission checks, not reduced safety margins.
   Measure report costs independently; do not apply the short fit ETA to long.

6. **P2 / AUD-DMO-06: new human gates and incomplete acceptance.**
   `campaign.py:47` introduces STRATEGY_OWNER_RULINGS and the return requests
   warehouse settings from the owner despite the prior operational authority.
   The policy already declares equality/fill/minimum-head conventions. Resolve
   versioned simulation defaults and units, declare unverified broker costs,
   and seal them automatically before validation; do not manufacture a new
   approval dependency. Missing TRACEABILITY.json and GPU-backed integration
   properties remain gaps. CPU test success is not DMO01-DMO20 end-to-end
   acceptance. Source clocks for two FRED inputs still block confirmatory TEST.

## Positive evidence and limits

- Independently reran the original full CPU suite: **208 passed in 130.11 s**,
  GPU disabled, governed 3 GiB/300 s scope. This supports software fixtures,
  not the exact-slot contract that those fixtures changed.
- Inspected the shared six-head backbone, per-feature causal branches, explicit
  post-fusion position encoding, core 8->4, and same-weight fit/inner early
  stopping. The GPU receipt is real; not every acceptance property has a real
  GPU replay. Current short fit remains an ASOF diagnostic, not the exact trial.
- Reproduced three protocol defects with `probes.py`; see PROBES.json.
- Existing authentication was recovered from the already provisioned private
  follower EnvironmentFile. No token was printed/copied or requested from the
  owner. One cost report/five metrics read back exactly; 19 mirrored files
  verified locally and on the secondary host. This is retained evidence
  delivery, not full predictive or TEST closure.
- Scientific worker/tick timers are stopped and disabled under AUDIT_HOLD.
  Only the bounded publication timer remains enabled for this root. Completed
  files were not deleted/relabelled; partial old POO results remain untouched.

## DOIN scope verification

Predictor diff `2a440265..fdaae9d2` contains no optimizer_plugins or
doin-data-predictor changes. Runtime clones on all three hosts are clean and
match exactly:

| Repository | Commit |
|---|---|
| doin-node | 393049dd4591076d4012f6ff1974aa42c4a390cf |
| doin-core | 9f9d3f757c5ee69c2995352a2370021fc837534c |
| doin-plugins | 2fc0523e6ee6fa371cee4d7862ca61c087a852fc |

All four retired DOIN services and their successor are still disabled. There
is no new DOIN optimization in this fixed-model trial. This audit changed no
consensus, migration, blockchain, genome, optimizer defaults or DOIN dashboard.
It does not recertify a running consensus network: the network is stopped.

## Reproduction

From this pinned audit checkout with its compatible isolated environment:

```bash
CUDA_VISIBLE_DEVICES='' python docs/i6n/direct_multi_output_static/audit_20261014/probes.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/i6dmo/test_audit_warehouse_contract.py tests/i6dmo/test_ops_dmo12_outbox.py
git diff --check
```

The first command returns 0 when all three retained defects reproduce; that
is not a successful campaign verdict. Once corrected it must report them no
longer reproduced, alongside positive/negative regression tests on new pins.
Private endpoints, credentials, deployment paths and GPU UUIDs stay outside Git.
