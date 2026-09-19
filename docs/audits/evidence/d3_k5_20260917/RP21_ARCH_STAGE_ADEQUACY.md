# Adequacy and costs (RP21)

## Per cell

| cell | task | arch | host | model | persistence | linear | gap | ≤ lin+0.03 | oracle | best upd | last upd | allowance | stop | best at ceiling | reach | W/P | cpu s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DX__trend_event__s1__0__profiles | DX/trend_event/profiles | 0 | WORKER_A | FAILED | | | | | | | | | | | | | |
| DX__trend_event__s1__A__profiles | DX/trend_event/profiles | A | COORDINATOR | FAILED | | | | | | | | | | | | | |
| DX__trend_event__s1__B__profiles | DX/trend_event/profiles | B | COORDINATOR | FAILED | | | | | | | | | | | | | |
| DX__trend_event__s1__C__profiles | DX/trend_event/profiles | C | WORKER_A | FAILED | | | | | | | | | | | | | |
| H2__h0__s1__0__profiles | H2/h0/profiles | 0 | COORDINATOR | 0.6026 | 0.7874 | 0.6283 | -0.0257 | True | 0.5590 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 3 | A: 2.00, B: 2.00 | 11.0 |
| H2__h0__s1__0__random_0 | H2/h0/random_0 | 0 | COORDINATOR | 0.6024 | 0.7874 | 0.6283 | -0.0259 | True | 0.5590 | 693 | 957 | 1100 | EARLY_STOPPING | False | 3 | A: 2.00, B: 2.00 | 10.5 |
| H2__h0__s1__A__profiles | H2/h0/profiles | A | COORDINATOR | 0.6093 | 0.7874 | 0.6283 | -0.0190 | True | 0.5590 | 429 | 693 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 2.00 | 27.5 |
| H2__h0__s1__A__random_0 | H2/h0/random_0 | A | COORDINATOR | 0.6059 | 0.7874 | 0.6283 | -0.0224 | True | 0.5590 | 627 | 891 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 2.00 | 29.5 |
| H2__h0__s1__B__profiles | H2/h0/profiles | B | COORDINATOR | 0.6519 | 0.7874 | 0.6283 | +0.0236 | True | 0.5590 | 825 | 1089 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 47.8 |
| H2__h0__s1__B__random_0 | H2/h0/random_0 | B | COORDINATOR | 0.6717 | 0.7874 | 0.6283 | +0.0434 | False | 0.5590 | 825 | 1089 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 46.8 |
| H2__h0__s1__C__profiles | H2/h0/profiles | C | WORKER_B | 0.6176 | 0.7874 | 0.6283 | -0.0107 | True | 0.5590 | 462 | 726 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 21.8 |
| H2__h0__s1__C__random_0 | H2/h0/random_0 | C | WORKER_B | 0.6237 | 0.7874 | 0.6283 | -0.0046 | True | 0.5590 | 462 | 726 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 21.8 |
| H2__h0__s2__0__profiles | H2/h0/profiles | 0 | WORKER_A | 0.6215 | 0.7936 | 0.6358 | -0.0144 | True | 0.5607 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 3 | A: 2.00, B: 2.00 | 7.4 |
| H2__h0__s2__0__random_0 | H2/h0/random_0 | 0 | WORKER_A | 0.6212 | 0.7936 | 0.6358 | -0.0146 | True | 0.5607 | 957 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 2.00 | 6.2 |
| H2__h0__s2__A__profiles | H2/h0/profiles | A | WORKER_A | 0.6133 | 0.7936 | 0.6358 | -0.0225 | True | 0.5607 | 594 | 858 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 2.00 | 16.3 |
| H2__h0__s2__A__random_0 | H2/h0/random_0 | A | WORKER_A | 0.6125 | 0.7936 | 0.6358 | -0.0233 | True | 0.5607 | 495 | 759 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 2.00 | 14.1 |
| H2__h0__s2__B__profiles | H2/h0/profiles | B | COORDINATOR | 0.6496 | 0.7936 | 0.6358 | +0.0138 | True | 0.5607 | 495 | 759 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 38.6 |
| H2__h0__s2__B__random_0 | H2/h0/random_0 | B | COORDINATOR | 0.6605 | 0.7936 | 0.6358 | +0.0247 | True | 0.5607 | 594 | 858 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 41.1 |
| H2__h0__s2__C__profiles | H2/h0/profiles | C | WORKER_A | 0.6158 | 0.7936 | 0.6358 | -0.0200 | True | 0.5607 | 429 | 693 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 25.8 |
| H2__h0__s2__C__random_0 | H2/h0/random_0 | C | WORKER_A | 0.6185 | 0.7936 | 0.6358 | -0.0173 | True | 0.5607 | 396 | 660 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 2.00 | 25.2 |
| H2__h3__s1__0__profiles | H2/h3/profiles | 0 | WORKER_B | 0.4078 | 0.5060 | 0.4177 | -0.0099 | True | 0.3635 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 3 | A: 2.00, B: 0.80 | 4.9 |
| H2__h3__s1__0__random_0 | H2/h3/random_0 | 0 | WORKER_B | 0.4084 | 0.5060 | 0.4177 | -0.0093 | True | 0.3635 | 693 | 957 | 1100 | EARLY_STOPPING | False | 3 | A: 2.00, B: 0.80 | 4.7 |
| H2__h3__s1__A__profiles | H2/h3/profiles | A | WORKER_B | 0.4006 | 0.5060 | 0.4177 | -0.0170 | True | 0.3635 | 693 | 957 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 0.80 | 13.7 |
| H2__h3__s1__A__random_0 | H2/h3/random_0 | A | WORKER_B | 0.4020 | 0.5060 | 0.4177 | -0.0157 | True | 0.3635 | 594 | 858 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 0.80 | 12.4 |
| H2__h3__s1__B__profiles | H2/h3/profiles | B | WORKER_A | 0.4369 | 0.5060 | 0.4177 | +0.0192 | True | 0.3635 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 0.80 | 23.6 |
| H2__h3__s1__B__random_0 | H2/h3/random_0 | B | WORKER_A | 0.4328 | 0.5060 | 0.4177 | +0.0152 | True | 0.3635 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 0.80 | 23.8 |
| H2__h3__s1__C__profiles | H2/h3/profiles | C | WORKER_A | 0.4091 | 0.5060 | 0.4177 | -0.0086 | True | 0.3635 | 396 | 660 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 0.80 | 25.9 |
| H2__h3__s1__C__random_0 | H2/h3/random_0 | C | WORKER_A | 0.4091 | 0.5060 | 0.4177 | -0.0085 | True | 0.3635 | 429 | 693 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 0.80 | 25.2 |
| H2__h3__s2__0__profiles | H2/h3/profiles | 0 | WORKER_A | 0.4212 | 0.5085 | 0.4247 | -0.0035 | True | 0.3656 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 0.80 | 6.1 |
| H2__h3__s2__0__random_0 | H2/h3/random_0 | 0 | WORKER_A | 0.4193 | 0.5085 | 0.4247 | -0.0055 | True | 0.3656 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 0.80 | 6.1 |
| H2__h3__s2__A__profiles | H2/h3/profiles | A | WORKER_A | 0.4095 | 0.5085 | 0.4247 | -0.0152 | True | 0.3656 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 7 | A: 2.00, B: 0.80 | 16.9 |
| H2__h3__s2__A__random_0 | H2/h3/random_0 | A | WORKER_A | 0.4109 | 0.5085 | 0.4247 | -0.0139 | True | 0.3656 | 726 | 990 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 0.80 | 15.9 |
| H2__h3__s2__B__profiles | H2/h3/profiles | B | COORDINATOR | 0.4426 | 0.5085 | 0.4247 | +0.0178 | True | 0.3656 | 594 | 858 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 0.80 | 41.1 |
| H2__h3__s2__B__random_0 | H2/h3/random_0 | B | COORDINATOR | 0.4290 | 0.5085 | 0.4247 | +0.0043 | True | 0.3656 | 660 | 924 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 0.80 | 41.6 |
| H2__h3__s2__C__profiles | H2/h3/profiles | C | WORKER_A | 0.4191 | 0.5085 | 0.4247 | -0.0057 | True | 0.3656 | 363 | 627 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 0.80 | 23.8 |
| H2__h3__s2__C__random_0 | H2/h3/random_0 | C | WORKER_A | 0.4164 | 0.5085 | 0.4247 | -0.0084 | True | 0.3656 | 462 | 726 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 0.80 | 25.8 |
| H3__r0__s1__0__extractor | H3/r0/extractor | 0 | COORDINATOR | 0.6068 | 0.6907 | 0.6432 | -0.0364 | True | 0.4879 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 1.00 | 11.2 |
| H3__r0__s1__0__sequence | H3/r0/sequence | 0 | COORDINATOR | 0.6068 | 0.6907 | 0.6432 | -0.0364 | True | 0.4879 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 1.00 | 11.2 |
| H3__r0__s1__0__summary | H3/r0/summary | 0 | COORDINATOR | 0.6902 | 0.6907 | 0.6432 | +0.0470 | False | 0.4879 | 99 | 363 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 7.1 |
| H3__r0__s1__A__extractor | H3/r0/extractor | A | COORDINATOR | 0.6113 | 0.6907 | 0.6432 | -0.0319 | True | 0.4879 | 429 | 693 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 27.1 |
| H3__r0__s1__A__sequence | H3/r0/sequence | A | COORDINATOR | 0.6072 | 0.6907 | 0.6432 | -0.0360 | True | 0.4879 | 561 | 825 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 19.6 |
| H3__r0__s1__A__summary | H3/r0/summary | A | COORDINATOR | 0.6779 | 0.6907 | 0.6432 | +0.0347 | False | 0.4879 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 21.4 |
| H3__r0__s1__B__extractor | H3/r0/extractor | B | WORKER_A | 0.6560 | 0.6907 | 0.6432 | +0.0128 | True | 0.4879 | 660 | 924 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 21.3 |
| H3__r0__s1__B__sequence | H3/r0/sequence | B | WORKER_A | 0.6553 | 0.6907 | 0.6432 | +0.0121 | True | 0.4879 | 528 | 792 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 13.2 |
| H3__r0__s1__B__summary | H3/r0/summary | B | WORKER_A | 0.6845 | 0.6907 | 0.6432 | +0.0413 | False | 0.4879 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 13.3 |
| H3__r0__s1__C__extractor | H3/r0/extractor | C | WORKER_B | 0.6138 | 0.6907 | 0.6432 | -0.0295 | True | 0.4879 | 330 | 594 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 19.6 |
| H3__r0__s1__C__sequence | H3/r0/sequence | C | WORKER_B | 0.6103 | 0.6907 | 0.6432 | -0.0329 | True | 0.4879 | 264 | 528 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 13.2 |
| H3__r0__s1__C__summary | H3/r0/summary | C | WORKER_B | 0.6829 | 0.6907 | 0.6432 | +0.0397 | False | 0.4879 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 19.4 |
| H3__r0__s2__0__extractor | H3/r0/extractor | 0 | WORKER_A | 0.6166 | 0.6832 | 0.6552 | -0.0386 | True | 0.4880 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 1.00 | 5.9 |
| H3__r0__s2__0__sequence | H3/r0/sequence | 0 | WORKER_A | 0.6166 | 0.6832 | 0.6552 | -0.0386 | True | 0.4880 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 3 | A: 2.00, B: 1.00 | 6.3 |
| H3__r0__s2__0__summary | H3/r0/summary | 0 | WORKER_A | 0.6830 | 0.6832 | 0.6552 | +0.0278 | True | 0.4880 | 132 | 396 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 3.8 |
| H3__r0__s2__A__extractor | H3/r0/extractor | A | WORKER_A | 0.6176 | 0.6832 | 0.6552 | -0.0375 | True | 0.4880 | 363 | 627 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 13.3 |
| H3__r0__s2__A__sequence | H3/r0/sequence | A | WORKER_A | 0.6154 | 0.6832 | 0.6552 | -0.0398 | True | 0.4880 | 561 | 825 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 9.6 |
| H3__r0__s2__A__summary | H3/r0/summary | A | WORKER_A | 0.6756 | 0.6832 | 0.6552 | +0.0205 | True | 0.4880 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 10.6 |
| H3__r0__s2__B__extractor | H3/r0/extractor | B | COORDINATOR | 0.6558 | 0.6832 | 0.6552 | +0.0007 | True | 0.4880 | 363 | 627 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 35.6 |
| H3__r0__s2__B__sequence | H3/r0/sequence | B | COORDINATOR | 0.6517 | 0.6832 | 0.6552 | -0.0035 | True | 0.4880 | 726 | 990 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 26.1 |
| H3__r0__s2__B__summary | H3/r0/summary | B | COORDINATOR | 0.6779 | 0.6832 | 0.6552 | +0.0227 | True | 0.4880 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 27.2 |
| H3__r0__s2__C__extractor | H3/r0/extractor | C | WORKER_A | 0.6244 | 0.6832 | 0.6552 | -0.0307 | True | 0.4880 | 231 | 495 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 20.9 |
| H3__r0__s2__C__sequence | H3/r0/sequence | C | WORKER_A | 0.6235 | 0.6832 | 0.6552 | -0.0317 | True | 0.4880 | 132 | 396 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 13.6 |
| H3__r0__s2__C__summary | H3/r0/summary | C | WORKER_A | 0.6795 | 0.6832 | 0.6552 | +0.0244 | True | 0.4880 | 495 | 759 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 17.1 |
| H3__r1__s1__0__extractor | H3/r1/extractor | 0 | WORKER_B | 0.5495 | 0.6864 | 0.5546 | -0.0050 | True | 0.4890 | 759 | 1023 | 1100 | EARLY_STOPPING | False | 3 | A: 2.00, B: 1.00 | 4.9 |
| H3__r1__s1__0__sequence | H3/r1/sequence | 0 | WORKER_B | 0.5495 | 0.6864 | 0.5546 | -0.0050 | True | 0.4890 | 759 | 1023 | 1100 | EARLY_STOPPING | False | 3 | A: 2.00, B: 1.00 | 4.7 |
| H3__r1__s1__0__sequence_gap | H3/r1/sequence_gap | 0 | WORKER_B | 0.6664 | 0.6864 | 0.5546 | +0.1118 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 5.0 |
| H3__r1__s1__0__summary | H3/r1/summary | 0 | WORKER_B | 0.6816 | 0.6864 | 0.5546 | +0.1271 | False | 0.4890 | 561 | 825 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 4.2 |
| H3__r1__s1__0__summary_last | H3/r1/summary_last | 0 | WORKER_B | 0.6051 | 0.6864 | 0.5546 | +0.0505 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 1 | A: 2.00, B: 1.00 | 4.5 |
| H3__r1__s1__A__extractor | H3/r1/extractor | A | WORKER_B | 0.5346 | 0.6864 | 0.5546 | -0.0200 | True | 0.4890 | 561 | 825 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 10.9 |
| H3__r1__s1__A__extractor_summary | H3/r1/extractor_summary | A | COORDINATOR | 0.6285 | 0.6864 | 0.5546 | +0.0739 | False | 0.4890 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 30.2 |
| H3__r1__s1__A__sequence | H3/r1/sequence | A | WORKER_B | 0.5325 | 0.6864 | 0.5546 | -0.0221 | True | 0.4890 | 594 | 858 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 8.1 |
| H3__r1__s1__A__sequence__dsum | H3/r1/sequence/dsum | A | COORDINATOR | 0.5634 | 0.6864 | 0.5546 | +0.0088 | True | 0.4890 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 7 | A: 2.00, B: 1.00 | 21.5 |
| H3__r1__s1__A__sequence_gap | H3/r1/sequence_gap | A | WORKER_B | 0.6664 | 0.6864 | 0.5546 | +0.1119 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 8.9 |
| H3__r1__s1__A__summary | H3/r1/summary | A | WORKER_B | 0.6741 | 0.6864 | 0.5546 | +0.1196 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 8.7 |
| H3__r1__s1__A__summary__dsum | H3/r1/summary/dsum | A | COORDINATOR | 0.6180 | 0.6864 | 0.5546 | +0.0634 | False | 0.4890 | 165 | 429 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 15.8 |
| H3__r1__s1__A__summary_last | H3/r1/summary_last | A | WORKER_B | 0.5898 | 0.6864 | 0.5546 | +0.0352 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 5 | A: 2.00, B: 1.00 | 8.7 |
| H3__r1__s1__B__extractor | H3/r1/extractor | B | WORKER_A | 0.5824 | 0.6864 | 0.5546 | +0.0278 | True | 0.4890 | 957 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 23.6 |
| H3__r1__s1__B__extractor_summary | H3/r1/extractor_summary | B | WORKER_A | 0.6477 | 0.6864 | 0.5546 | +0.0931 | False | 0.4890 | 726 | 990 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 17.8 |
| H3__r1__s1__B__sequence | H3/r1/sequence | B | WORKER_A | 0.5821 | 0.6864 | 0.5546 | +0.0276 | True | 0.4890 | 528 | 792 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 11.7 |
| H3__r1__s1__B__sequence__dsum | H3/r1/sequence/dsum | B | WORKER_A | 0.6144 | 0.6864 | 0.5546 | +0.0599 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 13.3 |
| H3__r1__s1__B__sequence_gap | H3/r1/sequence_gap | B | WORKER_A | 0.6678 | 0.6864 | 0.5546 | +0.1133 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 13.6 |
| H3__r1__s1__B__summary | H3/r1/summary | B | WORKER_A | 0.6751 | 0.6864 | 0.5546 | +0.1205 | False | 0.4890 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 13.6 |
| H3__r1__s1__B__summary__dsum | H3/r1/summary/dsum | B | WORKER_A | 0.6537 | 0.6864 | 0.5546 | +0.0991 | False | 0.4890 | 66 | 330 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 9.0 |
| H3__r1__s1__B__summary_last | H3/r1/summary_last | B | WORKER_A | 0.6041 | 0.6864 | 0.5546 | +0.0495 | False | 0.4890 | 759 | 1023 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 13.1 |
| H3__r1__s1__C__extractor | H3/r1/extractor | C | COORDINATOR | 0.5483 | 0.6864 | 0.5546 | -0.0063 | True | 0.4890 | 429 | 693 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 55.2 |
| H3__r1__s1__C__sequence | H3/r1/sequence | C | COORDINATOR | 0.5451 | 0.6864 | 0.5546 | -0.0094 | True | 0.4890 | 330 | 594 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 35.1 |
| H3__r1__s1__C__sequence_gap | H3/r1/sequence_gap | C | COORDINATOR | 0.6668 | 0.6864 | 0.5546 | +0.1122 | False | 0.4890 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 45.7 |
| H3__r1__s1__C__summary | H3/r1/summary | C | COORDINATOR | 0.6741 | 0.6864 | 0.5546 | +0.1195 | False | 0.4890 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 44.4 |
| H3__r1__s1__C__summary_last | H3/r1/summary_last | C | COORDINATOR | 0.5988 | 0.6864 | 0.5546 | +0.0442 | False | 0.4890 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 45.1 |
| H3__r1__s2__0__extractor | H3/r1/extractor | 0 | WORKER_A | 0.5688 | 0.6987 | 0.5682 | +0.0006 | True | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 3 | A: 2.00, B: 1.00 | 5.9 |
| H3__r1__s2__0__sequence | H3/r1/sequence | 0 | WORKER_A | 0.5688 | 0.6987 | 0.5682 | +0.0006 | True | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 3 | A: 2.00, B: 1.00 | 6.2 |
| H3__r1__s2__0__sequence_gap | H3/r1/sequence_gap | 0 | WORKER_A | 0.6818 | 0.6987 | 0.5682 | +0.1136 | False | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 6.2 |
| H3__r1__s2__0__summary | H3/r1/summary | 0 | WORKER_A | 0.6963 | 0.6987 | 0.5682 | +0.1281 | False | 0.4949 | 429 | 693 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 4.6 |
| H3__r1__s2__0__summary_last | H3/r1/summary_last | 0 | WORKER_A | 0.6223 | 0.6987 | 0.5682 | +0.0541 | False | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 1 | A: 2.00, B: 1.00 | 5.7 |
| H3__r1__s2__A__extractor | H3/r1/extractor | A | WORKER_A | 0.5487 | 0.6987 | 0.5682 | -0.0195 | True | 0.4949 | 594 | 858 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 14.8 |
| H3__r1__s2__A__extractor_summary | H3/r1/extractor_summary | A | COORDINATOR | 0.6393 | 0.6987 | 0.5682 | +0.0711 | False | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 30.0 |
| H3__r1__s2__A__sequence | H3/r1/sequence | A | WORKER_A | 0.5439 | 0.6987 | 0.5682 | -0.0243 | True | 0.4949 | 594 | 858 | 1100 | EARLY_STOPPING | False | 7 | A: 2.00, B: 1.00 | 9.7 |
| H3__r1__s2__A__sequence__dsum | H3/r1/sequence/dsum | A | COORDINATOR | 0.5757 | 0.6987 | 0.5682 | +0.0075 | True | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 7 | A: 2.00, B: 1.00 | 21.1 |
| H3__r1__s2__A__sequence_gap | H3/r1/sequence_gap | A | WORKER_A | 0.6732 | 0.6987 | 0.5682 | +0.1050 | False | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 10.9 |
| H3__r1__s2__A__summary | H3/r1/summary | A | WORKER_A | 0.6828 | 0.6987 | 0.5682 | +0.1146 | False | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 10.5 |
| H3__r1__s2__A__summary__dsum | H3/r1/summary/dsum | A | COORDINATOR | 0.6354 | 0.6987 | 0.5682 | +0.0672 | False | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 20.6 |
| H3__r1__s2__A__summary_last | H3/r1/summary_last | A | WORKER_A | 0.5894 | 0.6987 | 0.5682 | +0.0212 | True | 0.4949 | 990 | 1100 | 1100 | UPDATE_BUDGET | False | 5 | A: 2.00, B: 1.00 | 10.5 |
| H3__r1__s2__B__extractor | H3/r1/extractor | B | COORDINATOR | 0.5913 | 0.6987 | 0.5682 | +0.0231 | True | 0.4949 | 528 | 792 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 39.0 |
| H3__r1__s2__B__extractor_summary | H3/r1/extractor_summary | B | WORKER_A | 0.6725 | 0.6987 | 0.5682 | +0.1043 | False | 0.4949 | 462 | 726 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 15.5 |
| H3__r1__s2__B__sequence | H3/r1/sequence | B | COORDINATOR | 0.5914 | 0.6987 | 0.5682 | +0.0232 | True | 0.4949 | 429 | 693 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 23.5 |
| H3__r1__s2__B__sequence__dsum | H3/r1/sequence/dsum | B | WORKER_A | 0.6467 | 0.6987 | 0.5682 | +0.0785 | False | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 13.0 |
| H3__r1__s2__B__sequence_gap | H3/r1/sequence_gap | B | COORDINATOR | 0.6762 | 0.6987 | 0.5682 | +0.1080 | False | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 27.9 |
| H3__r1__s2__B__summary | H3/r1/summary | B | COORDINATOR | 0.6894 | 0.6987 | 0.5682 | +0.1212 | False | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 26.8 |
| H3__r1__s2__B__summary__dsum | H3/r1/summary/dsum | B | WORKER_A | 0.6693 | 0.6987 | 0.5682 | +0.1011 | False | 0.4949 | 66 | 330 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 8.6 |
| H3__r1__s2__B__summary_last | H3/r1/summary_last | B | COORDINATOR | 0.6218 | 0.6987 | 0.5682 | +0.0536 | False | 0.4949 | 726 | 990 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 25.7 |
| H3__r1__s2__C__extractor | H3/r1/extractor | C | WORKER_A | 0.5614 | 0.6987 | 0.5682 | -0.0068 | True | 0.4949 | 462 | 726 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 25.9 |
| H3__r1__s2__C__sequence | H3/r1/sequence | C | WORKER_A | 0.5558 | 0.6987 | 0.5682 | -0.0123 | True | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 20.9 |
| H3__r1__s2__C__sequence_gap | H3/r1/sequence_gap | C | WORKER_A | 0.6815 | 0.6987 | 0.5682 | +0.1133 | False | 0.4949 | 1089 | 1100 | 1100 | UPDATE_BUDGET | False | 48 | A: 2.00, B: 1.00 | 21.2 |
| H3__r1__s2__C__summary | H3/r1/summary | C | WORKER_A | 0.6905 | 0.6987 | 0.5682 | +0.1223 | False | 0.4949 | 429 | 693 | 1100 | EARLY_STOPPING | False | 48 | A: 2.00, B: 1.00 | 16.5 |
| H3__r1__s2__C__summary_last | H3/r1/summary_last | C | WORKER_A | 0.6070 | 0.6987 | 0.5682 | +0.0389 | False | 0.4949 | 1122 | 1100 | 1100 | UPDATE_BUDGET | True | 48 | A: 2.00, B: 1.00 | 20.9 |

## Causes per task and architecture

| task | arch | n | MASE mean | sd seeds | best at ceiling | flags |
|---|---|---|---|---|---|---|
| H2/h0/profiles | 0 | 2 | 0.6120 | 0.0134 | 2 | LIMITED_OPTIMISATION (2/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 24; by construction for ARCH-0/A, and for B/C only against P_B at h3); SEED_VARIATION (sd 0.0134 across 2 seeds) |
| H2/h0/profiles | A | 2 | 0.6113 | 0.0028 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 24; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h0/profiles | B | 2 | 0.6508 | 0.0016 | 0 |  |
| H2/h0/profiles | C | 2 | 0.6167 | 0.0013 | 0 |  |
| H2/h0/random_0 | 0 | 2 | 0.6118 | 0.0133 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 24; by construction for ARCH-0/A, and for B/C only against P_B at h3); SEED_VARIATION (sd 0.0133 across 2 seeds) |
| H2/h0/random_0 | A | 2 | 0.6092 | 0.0047 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 24; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h0/random_0 | B | 2 | 0.6661 | 0.0079 | 0 |  |
| H2/h0/random_0 | C | 2 | 0.6211 | 0.0037 | 0 |  |
| H2/h3/profiles | 0 | 2 | 0.4145 | 0.0095 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/profiles | A | 2 | 0.4051 | 0.0063 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/profiles | B | 2 | 0.4397 | 0.0040 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 48 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/profiles | C | 2 | 0.4141 | 0.0071 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 48 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/random_0 | 0 | 2 | 0.4138 | 0.0077 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/random_0 | A | 2 | 0.4064 | 0.0062 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/random_0 | B | 2 | 0.4309 | 0.0027 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 48 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H2/h3/random_0 | C | 2 | 0.4128 | 0.0051 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 48 < slowest period 60; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r0/extractor | 0 | 2 | 0.6117 | 0.0069 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r0/extractor | A | 2 | 0.6145 | 0.0045 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r0/extractor | B | 2 | 0.6559 | 0.0001 | 0 |  |
| H3/r0/extractor | C | 2 | 0.6191 | 0.0075 | 0 |  |
| H3/r0/sequence | 0 | 2 | 0.6117 | 0.0069 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r0/sequence | A | 2 | 0.6113 | 0.0058 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r0/sequence | B | 2 | 0.6535 | 0.0026 | 0 |  |
| H3/r0/sequence | C | 2 | 0.6169 | 0.0093 | 0 |  |
| H3/r0/summary | 0 | 2 | 0.6866 | 0.0051 | 0 |  |
| H3/r0/summary | A | 2 | 0.6768 | 0.0016 | 0 |  |
| H3/r0/summary | B | 2 | 0.6812 | 0.0047 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance) |
| H3/r0/summary | C | 2 | 0.6812 | 0.0024 | 0 |  |
| H3/r1/extractor | 0 | 2 | 0.5592 | 0.0136 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3); SEED_VARIATION (sd 0.0136 across 2 seeds) |
| H3/r1/extractor | A | 2 | 0.5417 | 0.0100 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r1/extractor | B | 2 | 0.5868 | 0.0063 | 0 |  |
| H3/r1/extractor | C | 2 | 0.5549 | 0.0092 | 0 |  |
| H3/r1/extractor_summary | A | 2 | 0.6339 | 0.0076 | 2 | LIMITED_OPTIMISATION (2/2 best checkpoints at the allowance) |
| H3/r1/extractor_summary | B | 2 | 0.6601 | 0.0176 | 0 | SEED_VARIATION (sd 0.0176 across 2 seeds) |
| H3/r1/sequence | 0 | 2 | 0.5592 | 0.0136 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 3 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3); SEED_VARIATION (sd 0.0136 across 2 seeds) |
| H3/r1/sequence | A | 2 | 0.5382 | 0.0081 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r1/sequence | B | 2 | 0.5867 | 0.0065 | 0 |  |
| H3/r1/sequence | C | 2 | 0.5505 | 0.0076 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance) |
| H3/r1/sequence/dsum | A | 2 | 0.5696 | 0.0088 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 7 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r1/sequence/dsum | B | 2 | 0.6305 | 0.0228 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); SEED_VARIATION (sd 0.0228 across 2 seeds) |
| H3/r1/sequence_gap | 0 | 2 | 0.6741 | 0.0109 | 0 | SEED_VARIATION (sd 0.0109 across 2 seeds) |
| H3/r1/sequence_gap | A | 2 | 0.6698 | 0.0048 | 0 |  |
| H3/r1/sequence_gap | B | 2 | 0.6720 | 0.0059 | 0 |  |
| H3/r1/sequence_gap | C | 2 | 0.6741 | 0.0104 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); SEED_VARIATION (sd 0.0104 across 2 seeds) |
| H3/r1/summary | 0 | 2 | 0.6890 | 0.0104 | 0 | SEED_VARIATION (sd 0.0104 across 2 seeds) |
| H3/r1/summary | A | 2 | 0.6785 | 0.0061 | 0 |  |
| H3/r1/summary | B | 2 | 0.6822 | 0.0102 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); SEED_VARIATION (sd 0.0102 across 2 seeds) |
| H3/r1/summary | C | 2 | 0.6823 | 0.0116 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); SEED_VARIATION (sd 0.0116 across 2 seeds) |
| H3/r1/summary/dsum | A | 2 | 0.6267 | 0.0123 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); SEED_VARIATION (sd 0.0123 across 2 seeds) |
| H3/r1/summary/dsum | B | 2 | 0.6615 | 0.0110 | 0 | SEED_VARIATION (sd 0.0110 across 2 seeds) |
| H3/r1/summary_last | 0 | 2 | 0.6137 | 0.0122 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance); CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 1 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3); SEED_VARIATION (sd 0.0122 across 2 seeds) |
| H3/r1/summary_last | A | 2 | 0.5896 | 0.0003 | 0 | CONTEXT_REACH_BELOW_SLOWEST_PERIOD (reach 5 < slowest period 48; by construction for ARCH-0/A, and for B/C only against P_B at h3) |
| H3/r1/summary_last | B | 2 | 0.6129 | 0.0125 | 0 | SEED_VARIATION (sd 0.0125 across 2 seeds) |
| H3/r1/summary_last | C | 2 | 0.6029 | 0.0058 | 1 | LIMITED_OPTIMISATION (1/2 best checkpoints at the allowance) |

## Costs on the exact intersection (24 task × seed pairs; only (task, seed) pairs executed by EVERY architecture; pilots, donor-sensitivity cells and the summary-trained donors excluded; hosts differ, so per-host strata are shown apart)

| arch | cells | cpu total | cpu per cell | fit total | by host | donor cells | donor cpu | uses | donor cpu per use |
|---|---|---|---|---|---|---|---|---|---|
| A | 24 | 350 | 14.6 | 235 | {"COORDINATOR": 125, "WORKER_A": 153, "WORKER_B": 71} | 4 | 66 | 12 | 5.5 |
| B | 24 | 660 | 27.5 | 490 | {"COORDINATOR": 489, "WORKER_A": 171} | 4 | 120 | 12 | 10.0 |
| C | 24 | 630 | 26.2 | 492 | {"WORKER_B": 96, "WORKER_A": 308, "COORDINATOR": 226} | 4 | 122 | 12 | 10.1 |
| 0 | 24 | 154 | 6.4 | 86 | {"COORDINATOR": 51, "WORKER_A": 70, "WORKER_B": 33} | 4 | 28 | 12 | 2.3 |

## Learning-curve design (PREPARED_NOT_LAUNCHED)

- allowances [300, 600, 1100, 2200]; observed best updates quantiles 50/90/100: [726.0, 1122.0, 1122.0]
- cells 32; projected CPU 1504 s from the pilots' per-update cost
