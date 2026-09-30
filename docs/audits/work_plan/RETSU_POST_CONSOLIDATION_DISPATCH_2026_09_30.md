# Dispatch of Musashi's post-consolidation orders

Retsu, integration owner. 2026-09-30.
Source order: `SATOSHI_POST_CONSOLIDATION_2026_09_30.md`, reviewing `788f1cfa`,
`4d548a82`, `c89a8ff4` and heuristic-strategy `4bb763d`.
The owner authorized execution of that order while Satoshi's session is stopped.
The same message restates the gates in the order: nothing was installed, no
allocation was granted, the compute petition stays at 4,800 CPU seconds and
3,600 summed child wall seconds, conditioned on a working environment and a
current admission, and the 18 GiB ceiling stays withdrawn.

## Reading of the gates

This dispatch does not treat the owner's authorization as a grant of the two
calibrations, of `trust_remote_code`, of a broker call, or of a new scoring
pass. Those remain behind the order's own conditions. The work issued now is
the part the order says can proceed: diagnose the retained GPU failure and
price an isolated repair; read the pinned remote classification source and
price its dependency closure; implement the strategy baseline, elapsed-hour
targets, support separation and tests.

No doubt was sent to Musashi before dispatch. The order text plus the owner's
restatement of the gates was specific enough to start the safe half. If that
reading is wrong, the calibrations have not been started and can still be refused.

## Lanes

One integration owner. Three independent workers. Each was dispatched into a
new worktree so the evidence commits stay untouched. No lane was described as
active before its worker id existed.

| Lane | Worker | Worktree | Source tip | Budget | Blocker at dispatch | Acceptance |
|---|---|---|---|---|---|---|
| GPU environment | `01a0f168-9bc7-7601-890a-1451d883d8d0` | `predictor-gpu-repair-20260930` | `4d548a82` | disk only, cap 20 GiB of wheels; no GPU, no admission | diagnostic allocation not granted; smoke written, not run | retained cause or `NOT_RETAINED`; pinned wheels; measured download and installed bytes; smoke command; rollback by abandoning the new env |
| Classification | `01a0f168-9bc7-7601-890a-146a1a38ac7a` | `predictor-b77-closure-20260930` | `c89a8ff4` | disk only, cap 8 GiB; no weights, no `trust_remote_code` | Python 3.13 was not on `PATH` at dispatch | static review of revision `46ed7da5b47e4bca710b756313fafaf4110c6bd1`; full closure price; one written approval request; research run `NOT_RUN` |
| Strategy | `01a0f168-9bc7-7601-890a-147751ca9fc8` | `strategy-support-20260930` | `4bb763d` | tests on synthetic frames only | historic 3,600 CPU-s ceiling is not a fresh allowance | variant E explicit and not a recovered run; elapsed-hour targets; legacy row offsets kept; support purge; prior-access audit; B0 not started |

No new accuracy, no financial result, and no model score is in scope for this dispatch.

## What this dispatch does not do

- It does not shrink the 12 GiB device envelope.
- It does not mutate an existing interpreter, the driver, a service, the MT5 VM, or a live chat port.
- It does not rerun the 241-cell strategy sweep.
- It does not spend the 4,800 / 3,600 petition. That petition is unchanged and still conditional.
