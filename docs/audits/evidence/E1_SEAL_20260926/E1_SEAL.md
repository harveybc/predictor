# E1 seal — `E1_PARTIAL_SEAL`

Generated 2026-09-26T19:40:45Z by `tools/df_e1_seal.py` from retained artifacts only. Nothing was fitted, loaded, scored or replayed.

**Verdict rule.** E1_SEALED only when every condition is DISCHARGED or PROHIBITION_OBSERVED; E1_PARTIAL_SEAL otherwise. Identity, rules and numbers are sealed either way — they are facts about bytes

## Conditions

| condition | source | state | evidence |
|---|---|---|---|
| `RP17_RP24_NO_SEAL_13E_V1` | MUSASHI_RP17_RP24_REVIEW_2026_09_19.md | **PROHIBITION_OBSERVED** | 13E v1 still carries the heading 'E1 design (proposal references; to be sealed after review)' and the sentence 'the E1 campaign is NOT launched on the v1 eligible: true'; no retained artifact seals it |
| `ML_BASELINES_CAUSALITY_UNVERIFIED` | MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md | **NOT_DISCHARGEABLE_BY_ARTIFACTS** | the answering return records both lineages as UNBOUND: the producer of the decomposed inputs is not in this repository |
| `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR` | MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md | **UNMET** | calendar=measured, volume=measured, context block(s) with no outcome: ['Q2_CONTEXT'] |
| `HUBER_NO_RETURN_TO_R0_R1_R2_YET` | MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md | **PROHIBITION_OBSERVED** | no post-Huber block carries an R0/R1/R2 arm; the arms measured are calendar, daily_lag, gru_adapted_w60, gru_calendar_w60, modular_w60, randomised_calendar_control, volume_112d, volume_56d |
| `POST_HUBER_PHASE2_CORRECTED` | MUSASHI_POST_HUBER_REVIEW_2026_09_21.md | **DISCHARGED** | v1 6b47509400ef… (df_e1_phase2_design.v1, state SEALED_NOT_EXECUTED) is preserved and superseded by v2 0dcb92b9c05a… (df_e1_phase2_design.v2, state SEALED_NOT_EXECUTED); both digests re-derive |
| `POST_HUBER_EXTERNAL_ACCEPTANCE` | MUSASHI_POST_HUBER_REVIEW_2026_09_21.md | **NOT_DISCHARGEABLE_BY_ARTIFACTS** | the correction is an artifact; its acceptance is a review |
| `MOD_E1_EXTERNAL_REVIEW` | SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20.md | **DISCHARGED_BY_OWNER_GRANTED_DISPOSITION_NOT_BY_EXTERNAL_REVIEW** | the return declares it itself; the two reviews are still absent ['MUSASHI_RP49_RP56_REVIEW_2026_09_20.md', 'MUSASHI_RP57_RP64_REVIEW_2026_09_21.md']; the ruling that stands in their place is ['SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md', 'SATOSHI_RP57_RP64_DISPOSITION_2026_09_26.md'], each one's identity recomputed here from its own bytes, each one declaring the owner's grant it acts under, that nothing in it is written in the reviewer's name, and the requirement it rules on; and rp49_rp64_disposition_20260926 names exactly those documents, exactly those two absent reviews, and counts the documents print |
| `OWNER_CLOSURE_TABLE_FROM_ARTIFACTS` | owner standing order, 2026-09-21 | **DISCHARGED** | docs/audits/evidence/d3_k5_20260917/RP82/CLOSURE_TABLE_RP89.json (owner_closure_table.v2, 2026-09-21T18:45:16Z) covers 70 rows over 8 runs with problems []; 39 verified and 31 preserved with a qualified scope; every row carries all thirteen required columns and every run's design identity recomputes |
| `RP136_RP139_MATCHED_ECL_ADAPTER` | MUSASHI_RP136_RP139_REVIEW_2026_09_23.md | **UNMET** | every retained block design still names a single-target, single-offset contract: REGISTRY household_W60_h60; no full-output adapter artifact is retained |

**Gaps in the seal.** `ML_BASELINES_CAUSALITY_UNVERIFIED`, `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR`, `POST_HUBER_EXTERNAL_ACCEPTANCE`, `MOD_E1_EXTERNAL_REVIEW`, `RP136_RP139_MATCHED_ECL_ADAPTER`

**Gaps that still block dispatch.** `ML_BASELINES_CAUSALITY_UNVERIFIED`, `ML_BASELINES_SEPARATE_VOLUME_CONTEXT_CALENDAR`, `POST_HUBER_EXTERNAL_ACCEPTANCE`, `RP136_RP139_MATCHED_ECL_ADAPTER`

`gaps` is what the SEAL lacks; `dispatch_blocking_gaps` is what still holds WORK back. They differ by exactly the conditions a retained ruling discharged for dispatch under a named authority without being the reviewer's signature. Neither list is an authorization: a documentary check that passes says nothing about whether a measurement is scientifically admissible

## The ruling on the reserved external review

- **ruling** — `BOTH_RANGES_ACCEPTED_WITH_NINE_FINDINGS_MOD_E1_EXTERNAL_REVIEW_DISCHARGED`
- **scope** — this discharges the requirement as a BLOCKER OF MODULE DISPATCH, on the four modules the ruling names and on the audited commit it names — and nothing else. It is not the reviewer's signature, it does not make E1 sealed, it does not accept any measurement, and it does not reach the items the ruling itself leaves with the reviewer (MOD-CONF's sealed confirmatory design, whose owner is Musashi + Satoshi)
- **identity recomputed** — `docs/audits/work_plan/SATOSHI_RP49_RP56_DISPOSITION_2026_09_26.md` `4045fce07a201754…` (23930 bytes)
- **identity recomputed** — `docs/audits/work_plan/SATOSHI_RP57_RP64_DISPOSITION_2026_09_26.md` `83d2b437a3507ff6…` (25794 bytes)
- **audited commit** — `941eb5b3`
- **what would discharge it fully** — the reviewer's own review of the RP49-RP56 and RP57-RP64 returns. The grant replaced the wait, not the reviewer


## Sealed identities

| artifact | kind | identity |
|---|---|---|
| `docs/audits/evidence/d3_k5_20260917/RP30/E1_PILOT_DESIGN.json` | design | `807a4a30577d2ea4…` |
| `docs/audits/evidence/d3_k5_20260917/RP38/E1_PILOT_DESIGN_V2_SEALED_NOT_EXECUTED.json` | design | `143abb57d97daa07…` |
| `docs/audits/evidence/d3_k5_20260917/RP63/PHASE1_DESIGN_SEALED.json` | design | `5cb8263d359f1500…` |
| `docs/audits/evidence/d3_k5_20260917/RP65/PHASE2_DESIGN_SEALED.json` | design | `6b47509400ef5e7b…` |
| `docs/audits/evidence/d3_k5_20260917/RP66/PHASE2_DESIGN_SEALED_v2.json` | design | `0dcb92b9c05ad143…` |
| `docs/audits/evidence/d3_k5_20260917/RP66/blocks/e1_block_dev_matched_v1/DESIGN.json` | design | `6f35d165bb96cea8…` |
| `docs/audits/evidence/d3_k5_20260917/RP66/blocks/e1_block_dev_matched_v2/DESIGN.json` | design | `e43380914a276d62…` |
| `docs/audits/evidence/d3_k5_20260917/RP66/blocks/e1_block_q1_calendar_v1/DESIGN.json` | design | `1e306894ee94453a…` |
| `docs/audits/evidence/d3_k5_20260917/RP66/blocks/e1_block_q2_context_v1/DESIGN.json` | design | `6d1aaecaf27c581c…` |
| `docs/audits/evidence/d3_k5_20260917/RP66/blocks/e1_block_q3_volume_v1/DESIGN.json` | design | `fbbda9f3c75ea951…` |
| `docs/audits/evidence/d3_k5_20260917/RP74/blocks/e1_block_arch_x_calendar_v1/DESIGN.json` | design | `f4e616f68d707dd0…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_context_daily_lag_v1/DESIGN.json` | design | `31e71033d93545d1…` |
| `docs/audits/evidence/d3_k5_20260917/RP34/E1_PILOT_CLOSE_REPAIRED.json` | closure | `c3783e704042d938…` |
| `docs/audits/evidence/d3_k5_20260917/RP55/E1_SUCCESSOR_CLOSE.json` | closure | `9f5aeedc8e350410…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/CLOSURE_TABLE_RP89.json` | closure_table | `857bcd1b6a2c50b5…` |
| `docs/audits/evidence/HUBER_ADAMW_2026_09_21/REPORT.json` | report_unbound_design | `70ba33f100df0896…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_dev_matched_v2/REPORT.json` | report | `e199751293c837b2…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_q1_calendar_v1/REPORT.json` | report | `f0ebdbf72ffc5b34…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_q3_volume_v1/REPORT.json` | report | `3dc8ebbefe634270…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_arch_x_calendar_v1/REPORT.json` | report | `fa9ac315659a1c79…` |
| `docs/audits/evidence/d3_k5_20260917/RP82/blocks/e1_block_context_daily_lag_v1/REPORT.json` | report | `3f15e065eac77869…` |
| `docs/audits/work_plan/MUSASHI_RP17_RP24_REVIEW_2026_09_19.md` | document | `6a11b0fa447a8678…` |
| `docs/audits/work_plan/MUSASHI_E1_ML_BASELINES_REVIEW_2026_09_20.md` | document | `4ec0cba3391f6851…` |
| `docs/audits/work_plan/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md` | document | `1f164c2cec5ceae8…` |
| `docs/audits/work_plan/MUSASHI_POST_HUBER_REVIEW_2026_09_21.md` | document | `55f3155808ad6a31…` |
| `docs/audits/work_plan/MUSASHI_RP136_RP139_REVIEW_2026_09_23.md` | document | `bf60cefd00807724…` |
| `docs/audits/work_plan/SATOSHI_PROGRAM_RP17_RP24_RETURN_2026_09_19.md` | document | `883b9e239f5b20c8…` |
| `docs/audits/work_plan/SATOSHI_PROGRAM_RP49_RP56_RETURN_2026_09_20.md` | document | `bdbd40736b233376…` |
| `docs/audits/work_plan/SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21.md` | document | `e18ff6daf49217b2…` |
| `docs/tres_temas_entrevista/program_v3/13E_E1_TASK_SHEET_2026_09_19.md` | document | `5fff9956dc60689b…` |

## Numbers as measured

- **E1_PILOT_CLOSE_REPAIRED.json** — `ALL_VERIFIED`, counts {"declared": 15, "verified": 15, "refused": 0, "metrics_verified": 15, "inference_verified": 11, "regime_verified": 14}, scopes {"SCIENTIFICALLY_VERIFIED_HISTORICAL_UNGOVERNED": 15}
- **E1_SUCCESSOR_CLOSE.json** — `ALL_VERIFIED`, counts {"declared": 15, "verified": 15, "refused": 0, "metrics_verified": 15, "inference_verified": 11, "regime_verified": 14}, scopes {"VERIFIED_AND_GOVERNED": 15}
- **DEV_MATCHED** — verified=True, 10020 common rows; gru_adapted_w60=0.484795 kW, modular_w60=0.518351 kW
- **Q1_CALENDAR** — verified=True, 10020 common rows; calendar=0.444230 kW, randomised_calendar_control=0.504504 kW
- **Q3_VOLUME** — verified=True, 10020 common rows; volume_112d=0.508690 kW, volume_56d=0.511739 kW
- **ARCH_X_CALENDAR** — verified=True, 10020 common rows; calendar=0.449555 kW, gru_adapted_w60=0.482715 kW, gru_calendar_w60=0.437860 kW, modular_w60=0.491926 kW, randomised_calendar_control=0.496969 kW
- **CONTEXT_DAILY_LAG** — verified=True, 10020 common rows; daily_lag=0.487187 kW, modular_w60=0.494256 kW
- **Q2_CONTEXT** — `BUDGET_LIMITED_BEFORE_ANY_OUTCOME`: 0 cells with an outcome. pilots only; no cell of this block was fitted to a score, so it says nothing about its question — not 'no effect'

## The owner's closure table, bound

`docs/audits/evidence/d3_k5_20260917/RP82/CLOSURE_TABLE_RP89.json` (owner_closure_table.v2, 2026-09-21T18:45:16Z, generated by `tools/df_closure_table.py`) — 70 rows, 39 verified, 31 preserved with a qualified scope, problems [].

| run :: arm | rows | model MAE kW | naive MAE kW | skill | comparability |
|---|---:|---:|---:|---:|---|
| ARCH_X_CALENDAR_RP79::calendar | 3 | 0.449555 | 0.617372 | 0.271826 | NOT_COMPARABLE |
| ARCH_X_CALENDAR_RP79::gru_adapted_w60 | 3 | 0.482715 | 0.617372 | 0.218114 | NOT_COMPARABLE |
| ARCH_X_CALENDAR_RP79::gru_calendar_w60 | 3 | 0.437860 | 0.617372 | 0.290768 | NOT_COMPARABLE |
| ARCH_X_CALENDAR_RP79::modular_w60 | 3 | 0.491926 | 0.617372 | 0.203193 | NOT_COMPARABLE |
| ARCH_X_CALENDAR_RP79::randomised_calendar_control | 3 | 0.496969 | 0.617372 | 0.195025 | NOT_COMPARABLE |
| CONTEXT_DAILY_LAG_RP87::daily_lag | 3 | 0.487187 | 0.617372 | 0.210870 | NOT_COMPARABLE |
| CONTEXT_DAILY_LAG_RP87::modular_w60 | 3 | 0.494256 | 0.617372 | 0.199420 | NOT_COMPARABLE |
| DEV_MATCHED_RP72::gru_adapted_w60 | 3 | 0.484795 | 0.617372 | 0.214744 | NOT_COMPARABLE |
| DEV_MATCHED_RP72::modular_w60 | 3 | 0.518351 | 0.617372 | 0.160392 | NOT_COMPARABLE |
| Q1_CALENDAR_RP72::calendar | 3 | 0.444230 | 0.617372 | 0.280449 | NOT_COMPARABLE |
| Q1_CALENDAR_RP72::randomised_calendar_control | 3 | 0.504504 | 0.617372 | 0.182820 | NOT_COMPARABLE |
| Q3_VOLUME_RP72::volume_112d | 3 | 0.508690 | 0.617372 | 0.176039 | NOT_COMPARABLE |
| Q3_VOLUME_RP72::volume_56d | 3 | 0.511739 | 0.617372 | 0.171102 | NOT_COMPARABLE |
| huber_adamw_musashi::huber_adam | 3 | 0.520662 | 0.617372 | 0.156648 | NOT_COMPARABLE |
| huber_adamw_musashi::huber_adamw | 3 | 0.522479 | 0.617372 | 0.153705 | NOT_COMPARABLE |
| huber_adamw_musashi::mae_adam | 3 | 0.496382 | 0.617372 | 0.195976 | NOT_COMPARABLE |
| huber_adamw_musashi::mae_adamw | 3 | 0.495073 | 0.617372 | 0.198096 | NOT_COMPARABLE |
| phase1_RP63::core_mae | 3 | 0.497770 | 0.617372 | 0.193727 | NOT_COMPARABLE |
| phase1_RP63::core_mse | 3 | 0.546930 | 0.617372 | 0.114099 | NOT_COMPARABLE |
| phase1_RP63::tcn_mse | 3 | 0.535948 | 0.617372 | 0.131888 | NOT_COMPARABLE |
| successor_RP55::R0 | 3 | 0.546929 | 0.617372 | 0.114101 | NOT_COMPARABLE |
| successor_RP55::R1 | 3 | 0.556734 | 0.617372 | 0.098219 | NOT_COMPARABLE |
| successor_RP55::R2 | 3 | 0.552660 | 0.617372 | 0.104818 | NOT_COMPARABLE |
| successor_RP55::linear_ridge | 1 | 0.545495 | 0.617372 | 0.116425 | NOT_COMPARABLE |

Naive definition, unchanged across every row: persistence at the horizon on the identical evaluation origins.

## Disclosures

- design `be2e776e5c64a842…` has no retained artifact here: the frozen Huber/AdamW v2 design. Its digest is pinned by docs/audits/work_plan/MUSASHI_HUBER_ADAMW_RESULTS_2026_09_21.md and carried by the closure table and the Huber report, but the design document itself is not in this repository, so this seal cannot recompute it from bytes
- `e1_block_dev_matched_v2` closed under code that has since changed: {"df_e1_block.py": {"sealed": "51f7eb46ea44a830c501138bb05b3818008ca36f441af4b8c86c56294111b6b3", "now": "dbd27e0c22a6deb32ea96f00117cbd382e512187c2f8838629462780cbc7502d"}}
- `e1_block_q1_calendar_v1` closed under code that has since changed: {"df_e1_block.py": {"sealed": "798711c4d99b0939335b7624089784f6caa4a74382e261fcf0e6fa34c7a92600", "now": "dbd27e0c22a6deb32ea96f00117cbd382e512187c2f8838629462780cbc7502d"}}
- `e1_block_q3_volume_v1` closed under code that has since changed: {"df_e1_block.py": {"sealed": "09a37e7e711017609953be637d3ee176c92926d0f8d332108418b826b184389c", "now": "dbd27e0c22a6deb32ea96f00117cbd382e512187c2f8838629462780cbc7502d"}}
- `e1_block_arch_x_calendar_v1` closed under code that has since changed: {"df_e1_block.py": {"sealed": "8a9a7f9184553103f41ea5519c283e20d79bcf25fb1dfdff9c1ad84d30ffc91f", "now": "dbd27e0c22a6deb32ea96f00117cbd382e512187c2f8838629462780cbc7502d"}}

these blocks were closed under code whose digest the report records and which has since changed. The recorded numbers are the numbers that were measured; a reader who wants them reproduced under today's code must say so and pay for a replay. The seal does not claim they reproduce
