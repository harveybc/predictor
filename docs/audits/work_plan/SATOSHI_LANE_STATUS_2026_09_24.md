# One status row per lane, as RP151 requires

Written after inspecting live state, not from a snapshot: at 06:46Z on 2026-09-24 no experiment process was running on either
worker, both GPUs were idle, and the RP135 continuation service was inactive.

| lane | state | evidence | next action |
|---|---|---|---|
| RP144, protocol A | **COMPLETE and preserved** | twelve cells pooled, four-horizon mean 0.161962 / 0.259662, device attribution qualified as UNKNOWN or INFERRED | nothing; no retraining |
| RP145, protocol B | **COMPLETE** | twelve of twelve verified with bit-exact replays, mean 0.150307 / 0.246397, Table 9 lookback ambiguity explicit | nothing; the old ten-of-twelve queue is historical |
| RP146, doctoral R0/R1/R2 | **MEASURED, REPLAYED and CLOSED** | [RESULT](../evidence/d3_k5_20260917/RP155/RESULT.md): COMPLETE gates, 35,910 observed updates per fit, nine fresh-process replays, full-population scoring with same-row persistence | the outer test stays untouched; a confirmation is a separate design |
| RP147, real Laya | **NOT STARTED** | pinned identities recorded; news-signal's own state says real weights are not measured | download the pinned checkpoint in an isolated environment, then the 600 CPU s / 900 wall s residency pilot after fresh admission |
| RP148, news collection and shadow | **NOT STARTED** | no feed call made, no entitlement tested | implement receipt clocks, dedup and revisions, persistent queues and stale-input behaviour, which need no weights |
| RP149, MT5 demo and Alpaca paper | **NOT STARTED** | no broker call made, no acknowledgement or fill exists | implement the policy and risk interfaces with their refusal and recovery tests; canaries need the confirmed accounts |
| RP150, finance | **BLOCKED on an operator declaration** | HTTP 422, resource availability contract required; three of six contract fields have no evidence | the operator declares the contract; meanwhile the producer investigation and the matched reference selection continue without reading the reserve |
| RP150, calendar and domain adapters | **NOT STARTED** | CAL01 to CAL12 specified, not implemented | inventory the governed economic dataset and its vintages, then the as-of transforms with their negative tests |
| M5PHET runtime and INT | **PARTIAL, repaired through the last review** | 139 tests; INT01, INT03, INT05, INT11, INT12 verified; the five 7385937 findings repaired | INT02, INT04 and INT06 to INT10 need the governed adapter and the DOIN round trip |
| RP151, this return | **DELIVERED** | this table | none |

Nothing above says a lane is finished because its software tests pass. The four lanes marked NOT STARTED were not started, and
the financial lane is blocked by a declaration that is not mine to make.
