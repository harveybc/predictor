# M5PHET orchestration rounds, 2026-09-24 (night)

Round 1 (parallel agents on Opus, integrated by Satoshi): WP01, WP02, WP05, WP08, WP10, WP11, plus the MCP
engine route and the NON_MODEL_FIXTURE marker. Round 2: WP03, WP04, WP06 stages 1-2, WP13, tokened harnesses.

| repo | master / branch tip | what |
|---|---|---|
| M5PHET | master fc55058 (WP01..WP04, WP08, WP11, WP13, MCP route, harness token) | workbench, plugins, MCP, harnesses |
| news-signal | master c1ea592 | wording echo; backend named; fixture marked |
| prediction_provider | musashi/m5phet-forecast-20260924 4bdd3c4 | example unit/family |
| agent-multi | satoshi/m5phet-policy-provider-20260924 c4d7fa95 | two examples, two inputs |
| causal-inference-m5phet-provider | master d53e27d | demo-modifier-v1 study, cate answers |
| feature-eng (worktree) | codex/m5phet-hierarchical-regimes-20260924 e42cd9a | regimes by words; representation spec + design job |
| lts | satoshi/crispdm-r4-20260912 9090f49 | watchdog re-emission fix (220 alerts/day) |

Acceptance after round 2 on 127.0.0.1:8766 (tokened, deepseek-v4-flash via OpenCode Go): examples 8/8, prose 14/14,
refusals 2/2 (families_8766_after_round2.json); envelope questions 15/15 (envelopes_8766_after_round2.json);
verify_outputs 16/16 faithful with both output plugins (agent run). Same numbers under the local llama3.2:3b
interpreter (agent run under a temporary M5PHET_CONFIG).

wp06_household_design_candidates.json: the design job's four candidates over the 50,400-row household DEV slice,
with the ADF/KPSS and autocorrelation results that motivated each window and lag. No model was fitted; nothing
here measures a model. NO_NEW_MEASUREMENT for every area.
