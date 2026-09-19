# ARCH comparison — tables (validation; population 112 cells, verified 108, closure PARTIAL)

## Receiver adequacy (H2 profiles at the top level, r = 1)
| arch | cell | model MASE | naive | linear | oracle | beats naive |
|---|---|---|---|---|---|---|
| A | H2__h3__s1__A__profiles | 0.4006 | 0.5060 | 0.4177 | 0.3635 | True |
| A | H2__h3__s2__A__profiles | 0.4095 | 0.5085 | 0.4247 | 0.3656 | True |
| B | H2__h3__s1__B__profiles | 0.4369 | 0.5060 | 0.4177 | 0.3635 | True |
| B | H2__h3__s2__B__profiles | 0.4426 | 0.5085 | 0.4247 | 0.3656 | True |
| C | H2__h3__s1__C__profiles | 0.4091 | 0.5060 | 0.4177 | 0.3635 | True |
| C | H2__h3__s2__C__profiles | 0.4191 | 0.5085 | 0.4247 | 0.3656 | True |
| 0 | H2__h3__s1__0__profiles | 0.4078 | 0.5060 | 0.4177 | 0.3635 | True |
| 0 | H2__h3__s2__0__profiles | 0.4212 | 0.5085 | 0.4247 | 0.3656 | True |

## Effects per architecture (replicate = unit; SD with n replicates; descriptive)
| arch | interpretable | e(h) | slope | d_0 | d_1 | gamma | rho_1 (last − pooled) | donor delta | n rep |
|---|---|---|---|---|---|---|---|---|---|
| A | True | h0: +0.0022, h3: -0.0014 | -0.0012 | -0.0654 | -0.0300 | +0.0354 | -0.1102 | +0.0831 | 2 |
| B | True | h0: -0.0154, h3: +0.0088 | +0.0080 | -0.0277 | -0.0182 | +0.0095 | -0.0773 | +0.0645 | 2 |
| C | True | h0: -0.0043, h3: +0.0013 | +0.0019 | -0.0643 | -0.0303 | +0.0340 | -0.1015 | None | 2 |
| 0 | True | h0: +0.0002, h3: +0.0007 | +0.0001 | -0.0749 | -0.0347 | +0.0402 | -0.0951 | None | 2 |

## Cells (raw error and MASE, denominator, control delta, support, updates, stop, cost)
| cell | arch | arm | status | MAE val | MASE val | naive | linear | oracle | MASE test | denom mean | reach | updates | stop | cpu s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DX__trend_event__s1__0__profiles | | | FAILED | | | | | | | | | | | |
| DX__trend_event__s1__A__profiles | | | FAILED | | | | | | | | | | | |
| DX__trend_event__s1__B__profiles | | | FAILED | | | | | | | | | | | |
| DX__trend_event__s1__C__profiles | | | FAILED | | | | | | | | | | | |
| H2__h0__s1__0__profiles | 0 | profiles | VERIFIED | 0.5015 | 0.6026 | 0.7874 | 0.6283 | 0.5590 | 0.5915 | 0.8438 | 3 | 1100 | UPDATE_BUDGET | 11.0 |
| H2__h0__s1__0__random_0 | 0 | random_0 | VERIFIED | 0.5015 | 0.6024 | 0.7874 | 0.6283 | 0.5590 | 0.5931 | 0.8438 | 3 | 957 | EARLY_STOPPING | 10.5 |
| H2__h0__s1__A__profiles | A | profiles | VERIFIED | 0.5066 | 0.6093 | 0.7874 | 0.6283 | 0.5590 | 0.5905 | 0.8438 | 7 | 693 | EARLY_STOPPING | 27.5 |
| H2__h0__s1__A__random_0 | A | random_0 | VERIFIED | 0.5041 | 0.6059 | 0.7874 | 0.6283 | 0.5590 | 0.5886 | 0.8438 | 7 | 891 | EARLY_STOPPING | 29.5 |
| H2__h0__s1__B__profiles | B | profiles | VERIFIED | 0.5444 | 0.6519 | 0.7874 | 0.6283 | 0.5590 | 0.6400 | 0.8438 | 48 | 1089 | EARLY_STOPPING | 47.8 |
| H2__h0__s1__B__random_0 | B | random_0 | VERIFIED | 0.5613 | 0.6717 | 0.7874 | 0.6283 | 0.5590 | 0.6396 | 0.8438 | 48 | 1089 | EARLY_STOPPING | 46.8 |
| H2__h0__s1__C__profiles | C | profiles | VERIFIED | 0.5150 | 0.6176 | 0.7874 | 0.6283 | 0.5590 | 0.6024 | 0.8438 | 48 | 726 | EARLY_STOPPING | 21.8 |
| H2__h0__s1__C__random_0 | C | random_0 | VERIFIED | 0.5202 | 0.6237 | 0.7874 | 0.6283 | 0.5590 | 0.6035 | 0.8438 | 48 | 726 | EARLY_STOPPING | 21.8 |
| H2__h0__s2__0__profiles | 0 | profiles | VERIFIED | 0.5082 | 0.6215 | 0.7936 | 0.6358 | 0.5607 | 0.6196 | 0.8277 | 3 | 1100 | UPDATE_BUDGET | 7.4 |
| H2__h0__s2__0__random_0 | 0 | random_0 | VERIFIED | 0.5076 | 0.6212 | 0.7936 | 0.6358 | 0.5607 | 0.6171 | 0.8277 | 3 | 1100 | UPDATE_BUDGET | 6.2 |
| H2__h0__s2__A__profiles | A | profiles | VERIFIED | 0.4994 | 0.6133 | 0.7936 | 0.6358 | 0.5607 | 0.6132 | 0.8277 | 7 | 858 | EARLY_STOPPING | 16.3 |
| H2__h0__s2__A__random_0 | A | random_0 | VERIFIED | 0.4992 | 0.6125 | 0.7936 | 0.6358 | 0.5607 | 0.6172 | 0.8277 | 7 | 759 | EARLY_STOPPING | 14.1 |
| H2__h0__s2__B__profiles | B | profiles | VERIFIED | 0.5314 | 0.6496 | 0.7936 | 0.6358 | 0.5607 | 0.6621 | 0.8277 | 48 | 759 | EARLY_STOPPING | 38.6 |
| H2__h0__s2__B__random_0 | B | random_0 | VERIFIED | 0.5404 | 0.6605 | 0.7936 | 0.6358 | 0.5607 | 0.6646 | 0.8277 | 48 | 858 | EARLY_STOPPING | 41.1 |
| H2__h0__s2__C__profiles | C | profiles | VERIFIED | 0.5025 | 0.6158 | 0.7936 | 0.6358 | 0.5607 | 0.6273 | 0.8277 | 48 | 693 | EARLY_STOPPING | 25.8 |
| H2__h0__s2__C__random_0 | C | random_0 | VERIFIED | 0.5056 | 0.6185 | 0.7936 | 0.6358 | 0.5607 | 0.6283 | 0.8277 | 48 | 660 | EARLY_STOPPING | 25.2 |
| H2__h3__s1__0__profiles | 0 | profiles | VERIFIED | 0.5220 | 0.4078 | 0.5060 | 0.4177 | 0.3635 | 0.4018 | 1.2954 | 3 | 1100 | UPDATE_BUDGET | 4.9 |
| H2__h3__s1__0__random_0 | 0 | random_0 | VERIFIED | 0.5229 | 0.4084 | 0.5060 | 0.4177 | 0.3635 | 0.4012 | 1.2954 | 3 | 957 | EARLY_STOPPING | 4.7 |
| H2__h3__s1__A__profiles | A | profiles | VERIFIED | 0.5122 | 0.4006 | 0.5060 | 0.4177 | 0.3635 | 0.3968 | 1.2954 | 7 | 957 | EARLY_STOPPING | 13.7 |
| H2__h3__s1__A__random_0 | A | random_0 | VERIFIED | 0.5136 | 0.4020 | 0.5060 | 0.4177 | 0.3635 | 0.3948 | 1.2954 | 7 | 858 | EARLY_STOPPING | 12.4 |
| H2__h3__s1__B__profiles | B | profiles | VERIFIED | 0.5605 | 0.4369 | 0.5060 | 0.4177 | 0.3635 | 0.4177 | 1.2954 | 48 | 1100 | UPDATE_BUDGET | 23.6 |
| H2__h3__s1__B__random_0 | B | random_0 | VERIFIED | 0.5546 | 0.4328 | 0.5060 | 0.4177 | 0.3635 | 0.4204 | 1.2954 | 48 | 1100 | UPDATE_BUDGET | 23.8 |
| H2__h3__s1__C__profiles | C | profiles | VERIFIED | 0.5234 | 0.4091 | 0.5060 | 0.4177 | 0.3635 | 0.4005 | 1.2954 | 48 | 660 | EARLY_STOPPING | 25.9 |
| H2__h3__s1__C__random_0 | C | random_0 | VERIFIED | 0.5232 | 0.4091 | 0.5060 | 0.4177 | 0.3635 | 0.3942 | 1.2954 | 48 | 693 | EARLY_STOPPING | 25.2 |
| H2__h3__s2__0__profiles | 0 | profiles | VERIFIED | 0.5280 | 0.4212 | 0.5085 | 0.4247 | 0.3656 | 0.4271 | 1.2698 | 3 | 1100 | UPDATE_BUDGET | 6.1 |
| H2__h3__s2__0__random_0 | 0 | random_0 | VERIFIED | 0.5256 | 0.4193 | 0.5085 | 0.4247 | 0.3656 | 0.4252 | 1.2698 | 3 | 1100 | UPDATE_BUDGET | 6.1 |
| H2__h3__s2__A__profiles | A | profiles | VERIFIED | 0.5118 | 0.4095 | 0.5085 | 0.4247 | 0.3656 | 0.4164 | 1.2698 | 7 | 1100 | UPDATE_BUDGET | 16.9 |
| H2__h3__s2__A__random_0 | A | random_0 | VERIFIED | 0.5134 | 0.4109 | 0.5085 | 0.4247 | 0.3656 | 0.4203 | 1.2698 | 7 | 990 | EARLY_STOPPING | 15.9 |
| H2__h3__s2__B__profiles | B | profiles | VERIFIED | 0.5557 | 0.4426 | 0.5085 | 0.4247 | 0.3656 | 0.4462 | 1.2698 | 48 | 858 | EARLY_STOPPING | 41.1 |
| H2__h3__s2__B__random_0 | B | random_0 | VERIFIED | 0.5371 | 0.4290 | 0.5085 | 0.4247 | 0.3656 | 0.4400 | 1.2698 | 48 | 924 | EARLY_STOPPING | 41.6 |
| H2__h3__s2__C__profiles | C | profiles | VERIFIED | 0.5243 | 0.4191 | 0.5085 | 0.4247 | 0.3656 | 0.4231 | 1.2698 | 48 | 627 | EARLY_STOPPING | 23.8 |
| H2__h3__s2__C__random_0 | C | random_0 | VERIFIED | 0.5201 | 0.4164 | 0.5085 | 0.4247 | 0.3656 | 0.4239 | 1.2698 | 48 | 726 | EARLY_STOPPING | 25.8 |
| H3__r0__s1__0__extractor | 0 | extractor | VERIFIED | 0.5813 | 0.6068 | 0.6907 | 0.6432 | 0.4879 | 0.5960 | 0.9540 | 3 | 1100 | UPDATE_BUDGET | 11.2 |
| H3__r0__s1__0__sequence | 0 | sequence | VERIFIED | 0.5813 | 0.6068 | 0.6907 | 0.6432 | 0.4879 | 0.5960 | 0.9540 | 3 | 1100 | UPDATE_BUDGET | 11.2 |
| H3__r0__s1__0__summary | 0 | summary | VERIFIED | 0.6622 | 0.6902 | 0.6907 | 0.6432 | 0.4879 | 0.6682 | 0.9540 | 48 | 363 | EARLY_STOPPING | 7.1 |
| H3__r0__s1__A__extractor | A | extractor | VERIFIED | 0.5860 | 0.6113 | 0.6907 | 0.6432 | 0.4879 | 0.6005 | 0.9540 | 7 | 693 | EARLY_STOPPING | 27.1 |
| H3__r0__s1__A__sequence | A | sequence | VERIFIED | 0.5819 | 0.6072 | 0.6907 | 0.6432 | 0.4879 | 0.6026 | 0.9540 | 7 | 825 | EARLY_STOPPING | 19.6 |
| H3__r0__s1__A__summary | A | summary | VERIFIED | 0.6504 | 0.6779 | 0.6907 | 0.6432 | 0.4879 | 0.6586 | 0.9540 | 48 | 1100 | UPDATE_BUDGET | 21.4 |
| H3__r0__s1__B__extractor | B | extractor | VERIFIED | 0.6281 | 0.6560 | 0.6907 | 0.6432 | 0.4879 | 0.6393 | 0.9540 | 48 | 924 | EARLY_STOPPING | 21.3 |
| H3__r0__s1__B__sequence | B | sequence | VERIFIED | 0.6274 | 0.6553 | 0.6907 | 0.6432 | 0.4879 | 0.6407 | 0.9540 | 48 | 792 | EARLY_STOPPING | 13.2 |
| H3__r0__s1__B__summary | B | summary | VERIFIED | 0.6568 | 0.6845 | 0.6907 | 0.6432 | 0.4879 | 0.6625 | 0.9540 | 48 | 1100 | UPDATE_BUDGET | 13.3 |
| H3__r0__s1__C__extractor | C | extractor | VERIFIED | 0.5880 | 0.6138 | 0.6907 | 0.6432 | 0.4879 | 0.6034 | 0.9540 | 48 | 594 | EARLY_STOPPING | 19.6 |
| H3__r0__s1__C__sequence | C | sequence | VERIFIED | 0.5846 | 0.6103 | 0.6907 | 0.6432 | 0.4879 | 0.6027 | 0.9540 | 48 | 528 | EARLY_STOPPING | 13.2 |
| H3__r0__s1__C__summary | C | summary | VERIFIED | 0.6553 | 0.6829 | 0.6907 | 0.6432 | 0.4879 | 0.6578 | 0.9540 | 48 | 1100 | UPDATE_BUDGET | 19.4 |
| H3__r0__s2__0__extractor | 0 | extractor | VERIFIED | 0.5820 | 0.6166 | 0.6832 | 0.6552 | 0.4880 | 0.6245 | 0.9397 | 3 | 1100 | UPDATE_BUDGET | 5.9 |
| H3__r0__s2__0__sequence | 0 | sequence | VERIFIED | 0.5820 | 0.6166 | 0.6832 | 0.6552 | 0.4880 | 0.6245 | 0.9397 | 3 | 1100 | UPDATE_BUDGET | 6.3 |
| H3__r0__s2__0__summary | 0 | summary | VERIFIED | 0.6450 | 0.6830 | 0.6832 | 0.6552 | 0.4880 | 0.7004 | 0.9397 | 48 | 396 | EARLY_STOPPING | 3.8 |
| H3__r0__s2__A__extractor | A | extractor | VERIFIED | 0.5829 | 0.6176 | 0.6832 | 0.6552 | 0.4880 | 0.6317 | 0.9397 | 7 | 627 | EARLY_STOPPING | 13.3 |
| H3__r0__s2__A__sequence | A | sequence | VERIFIED | 0.5810 | 0.6154 | 0.6832 | 0.6552 | 0.4880 | 0.6317 | 0.9397 | 7 | 825 | EARLY_STOPPING | 9.6 |
| H3__r0__s2__A__summary | A | summary | VERIFIED | 0.6382 | 0.6756 | 0.6832 | 0.6552 | 0.4880 | 0.6917 | 0.9397 | 48 | 1100 | UPDATE_BUDGET | 10.6 |
| H3__r0__s2__B__extractor | B | extractor | VERIFIED | 0.6195 | 0.6558 | 0.6832 | 0.6552 | 0.4880 | 0.6681 | 0.9397 | 48 | 627 | EARLY_STOPPING | 35.6 |
| H3__r0__s2__B__sequence | B | sequence | VERIFIED | 0.6157 | 0.6517 | 0.6832 | 0.6552 | 0.4880 | 0.6650 | 0.9397 | 48 | 990 | EARLY_STOPPING | 26.1 |
| H3__r0__s2__B__summary | B | summary | VERIFIED | 0.6402 | 0.6779 | 0.6832 | 0.6552 | 0.4880 | 0.6930 | 0.9397 | 48 | 1100 | UPDATE_BUDGET | 27.2 |
| H3__r0__s2__C__extractor | C | extractor | VERIFIED | 0.5892 | 0.6244 | 0.6832 | 0.6552 | 0.4880 | 0.6293 | 0.9397 | 48 | 495 | EARLY_STOPPING | 20.9 |
| H3__r0__s2__C__sequence | C | sequence | VERIFIED | 0.5882 | 0.6235 | 0.6832 | 0.6552 | 0.4880 | 0.6300 | 0.9397 | 48 | 396 | EARLY_STOPPING | 13.6 |
| H3__r0__s2__C__summary | C | summary | VERIFIED | 0.6418 | 0.6795 | 0.6832 | 0.6552 | 0.4880 | 0.6976 | 0.9397 | 48 | 759 | EARLY_STOPPING | 17.1 |
| H3__r1__s1__0__extractor | 0 | extractor | VERIFIED | 0.5226 | 0.5495 | 0.6864 | 0.5546 | 0.4890 | 0.5388 | 0.9516 | 3 | 1023 | EARLY_STOPPING | 4.9 |
| H3__r1__s1__0__sequence | 0 | sequence | VERIFIED | 0.5226 | 0.5495 | 0.6864 | 0.5546 | 0.4890 | 0.5388 | 0.9516 | 3 | 1023 | EARLY_STOPPING | 4.7 |
| H3__r1__s1__0__sequence_gap | 0 | sequence_gap | VERIFIED | 0.6377 | 0.6664 | 0.6864 | 0.5546 | 0.4890 | 0.6399 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 5.0 |
| H3__r1__s1__0__summary | 0 | summary | VERIFIED | 0.6522 | 0.6816 | 0.6864 | 0.5546 | 0.4890 | 0.6602 | 0.9516 | 48 | 825 | EARLY_STOPPING | 4.2 |
| H3__r1__s1__0__summary_last | 0 | summary_last | VERIFIED | 0.5779 | 0.6051 | 0.6864 | 0.5546 | 0.4890 | 0.5885 | 0.9516 | 1 | 1100 | UPDATE_BUDGET | 4.5 |
| H3__r1__s1__A__extractor | A | extractor | VERIFIED | 0.5079 | 0.5346 | 0.6864 | 0.5546 | 0.4890 | 0.5241 | 0.9516 | 7 | 825 | EARLY_STOPPING | 10.9 |
| H3__r1__s1__A__extractor_summary | A | extractor_summary | VERIFIED | 0.6002 | 0.6285 | 0.6864 | 0.5546 | 0.4890 | 0.6098 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 30.2 |
| H3__r1__s1__A__sequence | A | sequence | VERIFIED | 0.5058 | 0.5325 | 0.6864 | 0.5546 | 0.4890 | 0.5231 | 0.9516 | 7 | 858 | EARLY_STOPPING | 8.1 |
| H3__r1__s1__A__sequence__dsum | A | sequence | VERIFIED | 0.5362 | 0.5634 | 0.6864 | 0.5546 | 0.4890 | 0.5433 | 0.9516 | 7 | 1100 | UPDATE_BUDGET | 21.5 |
| H3__r1__s1__A__sequence_gap | A | sequence_gap | VERIFIED | 0.6378 | 0.6664 | 0.6864 | 0.5546 | 0.4890 | 0.6355 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 8.9 |
| H3__r1__s1__A__summary | A | summary | VERIFIED | 0.6452 | 0.6741 | 0.6864 | 0.5546 | 0.4890 | 0.6403 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 8.7 |
| H3__r1__s1__A__summary__dsum | A | summary | VERIFIED | 0.5899 | 0.6180 | 0.6864 | 0.5546 | 0.4890 | 0.6025 | 0.9516 | 48 | 429 | EARLY_STOPPING | 15.8 |
| H3__r1__s1__A__summary_last | A | summary_last | VERIFIED | 0.5625 | 0.5898 | 0.6864 | 0.5546 | 0.4890 | 0.5760 | 0.9516 | 5 | 1100 | UPDATE_BUDGET | 8.7 |
| H3__r1__s1__B__extractor | B | extractor | VERIFIED | 0.5545 | 0.5824 | 0.6864 | 0.5546 | 0.4890 | 0.5605 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 23.6 |
| H3__r1__s1__B__extractor_summary | B | extractor_summary | VERIFIED | 0.6194 | 0.6477 | 0.6864 | 0.5546 | 0.4890 | 0.6201 | 0.9516 | 48 | 990 | EARLY_STOPPING | 17.8 |
| H3__r1__s1__B__sequence | B | sequence | VERIFIED | 0.5541 | 0.5821 | 0.6864 | 0.5546 | 0.4890 | 0.5587 | 0.9516 | 48 | 792 | EARLY_STOPPING | 11.7 |
| H3__r1__s1__B__sequence__dsum | B | sequence | VERIFIED | 0.5862 | 0.6144 | 0.6864 | 0.5546 | 0.4890 | 0.6004 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 13.3 |
| H3__r1__s1__B__sequence_gap | B | sequence_gap | VERIFIED | 0.6392 | 0.6678 | 0.6864 | 0.5546 | 0.4890 | 0.6407 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 13.6 |
| H3__r1__s1__B__summary | B | summary | VERIFIED | 0.6459 | 0.6751 | 0.6864 | 0.5546 | 0.4890 | 0.6496 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 13.6 |
| H3__r1__s1__B__summary__dsum | B | summary | VERIFIED | 0.6252 | 0.6537 | 0.6864 | 0.5546 | 0.4890 | 0.6224 | 0.9516 | 48 | 330 | EARLY_STOPPING | 9.0 |
| H3__r1__s1__B__summary_last | B | summary_last | VERIFIED | 0.5768 | 0.6041 | 0.6864 | 0.5546 | 0.4890 | 0.5907 | 0.9516 | 48 | 1023 | EARLY_STOPPING | 13.1 |
| H3__r1__s1__C__extractor | C | extractor | VERIFIED | 0.5211 | 0.5483 | 0.6864 | 0.5546 | 0.4890 | 0.5337 | 0.9516 | 48 | 693 | EARLY_STOPPING | 55.2 |
| H3__r1__s1__C__sequence | C | sequence | VERIFIED | 0.5180 | 0.5451 | 0.6864 | 0.5546 | 0.4890 | 0.5312 | 0.9516 | 48 | 594 | EARLY_STOPPING | 35.1 |
| H3__r1__s1__C__sequence_gap | C | sequence_gap | VERIFIED | 0.6381 | 0.6668 | 0.6864 | 0.5546 | 0.4890 | 0.6445 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 45.7 |
| H3__r1__s1__C__summary | C | summary | VERIFIED | 0.6453 | 0.6741 | 0.6864 | 0.5546 | 0.4890 | 0.6518 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 44.4 |
| H3__r1__s1__C__summary_last | C | summary_last | VERIFIED | 0.5715 | 0.5988 | 0.6864 | 0.5546 | 0.4890 | 0.5800 | 0.9516 | 48 | 1100 | UPDATE_BUDGET | 45.1 |
| H3__r1__s2__0__extractor | 0 | extractor | VERIFIED | 0.5260 | 0.5688 | 0.6987 | 0.5682 | 0.4949 | 0.5655 | 0.9255 | 3 | 1100 | UPDATE_BUDGET | 5.9 |
| H3__r1__s2__0__sequence | 0 | sequence | VERIFIED | 0.5260 | 0.5688 | 0.6987 | 0.5682 | 0.4949 | 0.5655 | 0.9255 | 3 | 1100 | UPDATE_BUDGET | 6.2 |
| H3__r1__s2__0__sequence_gap | 0 | sequence_gap | VERIFIED | 0.6339 | 0.6818 | 0.6987 | 0.5682 | 0.4949 | 0.6879 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 6.2 |
| H3__r1__s2__0__summary | 0 | summary | VERIFIED | 0.6473 | 0.6963 | 0.6987 | 0.5682 | 0.4949 | 0.7022 | 0.9255 | 48 | 693 | EARLY_STOPPING | 4.6 |
| H3__r1__s2__0__summary_last | 0 | summary_last | VERIFIED | 0.5777 | 0.6223 | 0.6987 | 0.5682 | 0.4949 | 0.6270 | 0.9255 | 1 | 1100 | UPDATE_BUDGET | 5.7 |
| H3__r1__s2__A__extractor | A | extractor | VERIFIED | 0.5068 | 0.5487 | 0.6987 | 0.5682 | 0.4949 | 0.5568 | 0.9255 | 7 | 858 | EARLY_STOPPING | 14.8 |
| H3__r1__s2__A__extractor_summary | A | extractor_summary | VERIFIED | 0.5934 | 0.6393 | 0.6987 | 0.5682 | 0.4949 | 0.6470 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 30.0 |
| H3__r1__s2__A__sequence | A | sequence | VERIFIED | 0.5021 | 0.5439 | 0.6987 | 0.5682 | 0.4949 | 0.5532 | 0.9255 | 7 | 858 | EARLY_STOPPING | 9.7 |
| H3__r1__s2__A__sequence__dsum | A | sequence | VERIFIED | 0.5326 | 0.5757 | 0.6987 | 0.5682 | 0.4949 | 0.5855 | 0.9255 | 7 | 1100 | UPDATE_BUDGET | 21.1 |
| H3__r1__s2__A__sequence_gap | A | sequence_gap | VERIFIED | 0.6254 | 0.6732 | 0.6987 | 0.5682 | 0.4949 | 0.6765 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 10.9 |
| H3__r1__s2__A__summary | A | summary | VERIFIED | 0.6349 | 0.6828 | 0.6987 | 0.5682 | 0.4949 | 0.6875 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 10.5 |
| H3__r1__s2__A__summary__dsum | A | summary | VERIFIED | 0.5896 | 0.6354 | 0.6987 | 0.5682 | 0.4949 | 0.6460 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 20.6 |
| H3__r1__s2__A__summary_last | A | summary_last | VERIFIED | 0.5455 | 0.5894 | 0.6987 | 0.5682 | 0.4949 | 0.5921 | 0.9255 | 5 | 1100 | UPDATE_BUDGET | 10.5 |
| H3__r1__s2__B__extractor | B | extractor | VERIFIED | 0.5469 | 0.5913 | 0.6987 | 0.5682 | 0.4949 | 0.6036 | 0.9255 | 48 | 792 | EARLY_STOPPING | 39.0 |
| H3__r1__s2__B__extractor_summary | B | extractor_summary | VERIFIED | 0.6251 | 0.6725 | 0.6987 | 0.5682 | 0.4949 | 0.6728 | 0.9255 | 48 | 726 | EARLY_STOPPING | 15.5 |
| H3__r1__s2__B__sequence | B | sequence | VERIFIED | 0.5466 | 0.5914 | 0.6987 | 0.5682 | 0.4949 | 0.5994 | 0.9255 | 48 | 693 | EARLY_STOPPING | 23.5 |
| H3__r1__s2__B__sequence__dsum | B | sequence | VERIFIED | 0.5998 | 0.6467 | 0.6987 | 0.5682 | 0.4949 | 0.6493 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 13.0 |
| H3__r1__s2__B__sequence_gap | B | sequence_gap | VERIFIED | 0.6285 | 0.6762 | 0.6987 | 0.5682 | 0.4949 | 0.6790 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 27.9 |
| H3__r1__s2__B__summary | B | summary | VERIFIED | 0.6407 | 0.6894 | 0.6987 | 0.5682 | 0.4949 | 0.6940 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 26.8 |
| H3__r1__s2__B__summary__dsum | B | summary | VERIFIED | 0.6221 | 0.6693 | 0.6987 | 0.5682 | 0.4949 | 0.6681 | 0.9255 | 48 | 330 | EARLY_STOPPING | 8.6 |
| H3__r1__s2__B__summary_last | B | summary_last | VERIFIED | 0.5768 | 0.6218 | 0.6987 | 0.5682 | 0.4949 | 0.6065 | 0.9255 | 48 | 990 | EARLY_STOPPING | 25.7 |
| H3__r1__s2__C__extractor | C | extractor | VERIFIED | 0.5184 | 0.5614 | 0.6987 | 0.5682 | 0.4949 | 0.5692 | 0.9255 | 48 | 726 | EARLY_STOPPING | 25.9 |
| H3__r1__s2__C__sequence | C | sequence | VERIFIED | 0.5131 | 0.5558 | 0.6987 | 0.5682 | 0.4949 | 0.5636 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 20.9 |
| H3__r1__s2__C__sequence_gap | C | sequence_gap | VERIFIED | 0.6335 | 0.6815 | 0.6987 | 0.5682 | 0.4949 | 0.6847 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 21.2 |
| H3__r1__s2__C__summary | C | summary | VERIFIED | 0.6420 | 0.6905 | 0.6987 | 0.5682 | 0.4949 | 0.6964 | 0.9255 | 48 | 693 | EARLY_STOPPING | 16.5 |
| H3__r1__s2__C__summary_last | C | summary_last | VERIFIED | 0.5625 | 0.6070 | 0.6987 | 0.5682 | 0.4949 | 0.6111 | 0.9255 | 48 | 1100 | UPDATE_BUDGET | 20.9 |

## Costs per architecture
| arch | cells | cpu total s | cpu mean s | fit mean s | updates mean | stop reasons |
|---|---|---|---|---|---|---|
| A | 30 | 489 | 16.3 | 10.9 | 940 | {'EARLY_STOPPING': 16, 'UPDATE_BUDGET': 14} |
| B | 30 | 737 | 24.6 | 18.1 | 923 | {'EARLY_STOPPING': 19, 'UPDATE_BUDGET': 11} |
| C | 24 | 630 | 26.2 | 20.5 | 779 | {'EARLY_STOPPING': 17, 'UPDATE_BUDGET': 7} |
| 0 | 24 | 154 | 6.4 | 3.6 | 993 | {'UPDATE_BUDGET': 16, 'EARLY_STOPPING': 8} |

## Diagnostic trend/event (adequacy only)
| arch | cell | model | naive | linear | oracle | beats naive | within linear + 0.03 |
|---|---|---|---|---|---|---|---|

Bootstrap: percentile intervals from resampling 2 replicates with replacement: descriptive precision, not confirmation
- A: H2_slope -0.0016..-0.0007, H3_gamma +0.0327..+0.0381, H3_d1 -0.0325..-0.0276
- B: H2_slope +0.0079..+0.0081, H3_gamma +0.0044..+0.0146, H3_d1 -0.0218..-0.0146
- C: H2_slope +0.0018..+0.0020, H3_gamma +0.0259..+0.0421, H3_d1 -0.0305..-0.0301
- 0: H2_slope -0.0003..+0.0006, H3_gamma +0.0324..+0.0480, H3_d1 -0.0354..-0.0340
