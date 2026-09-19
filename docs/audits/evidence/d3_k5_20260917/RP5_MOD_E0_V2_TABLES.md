# MOD-E0-DEV `mod-e0-dev-v2` — verification and effects (validation split; descriptive, no confirmation)

all_verified = True, parent_equal = True, live warehouse query = True, warehouse equal = True (69 units)

## Cells

| cell | arm | ARI(profiles, latent) | params (trainable) | updates | stop | MASE val | naive val | oracle val | MASE test | cpu s |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| `H2__h0__s1__profiles` | profiles | 0.16 | 9608 | 1815 | EARLY_STOPPING | 0.6519 | 0.7874 | 0.5590 | 0.6400 | 48.6 |
| `H2__h0__s1__random_0` | random_0 | 0.16 | 9608 | 1815 | EARLY_STOPPING | 0.6717 | 0.7874 | 0.5590 | 0.6396 | 48.4 |
| `H2__h0__s1__random_1` | random_1 | 0.16 | 9608 | 1881 | EARLY_STOPPING | 0.6580 | 0.7874 | 0.5590 | 0.6372 | 50.3 |
| `H2__h0__s1__random_2` | random_2 | 0.16 | 9608 | 2046 | EARLY_STOPPING | 0.6687 | 0.7874 | 0.5590 | 0.6398 | 53.1 |
| `H2__h0__s2__profiles` | profiles | 0.49 | 9608 | 1485 | EARLY_STOPPING | 0.6496 | 0.7936 | 0.5607 | 0.6621 | 42.2 |
| `H2__h0__s2__random_0` | random_0 | 0.49 | 9608 | 1584 | EARLY_STOPPING | 0.6605 | 0.7936 | 0.5607 | 0.6646 | 45.1 |
| `H2__h0__s2__random_1` | random_1 | 0.49 | 9608 | 1551 | EARLY_STOPPING | 0.6563 | 0.7936 | 0.5607 | 0.6595 | 44.4 |
| `H2__h0__s2__random_2` | random_2 | 0.49 | 9608 | 1419 | EARLY_STOPPING | 0.6601 | 0.7936 | 0.5607 | 0.6651 | 41.7 |
| `H2__h0__s3__profiles` | profiles | 0.16 | 9608 | 1815 | EARLY_STOPPING | 0.6559 | 0.7885 | 0.5586 | 0.6540 | 49.5 |
| `H2__h0__s3__random_0` | random_0 | 0.16 | 9608 | 1749 | EARLY_STOPPING | 0.6608 | 0.7885 | 0.5586 | 0.6581 | 46.7 |
| `H2__h0__s3__random_1` | random_1 | 0.16 | 9608 | 1815 | EARLY_STOPPING | 0.6529 | 0.7885 | 0.5586 | 0.6589 | 49.6 |
| `H2__h0__s3__random_2` | random_2 | 0.16 | 9608 | 1815 | EARLY_STOPPING | 0.6634 | 0.7885 | 0.5586 | 0.6622 | 49.1 |
| `H2__h1__s1__profiles` | profiles | 1.00 | 9608 | 1914 | EARLY_STOPPING | 0.5500 | 0.6378 | 0.4645 | 0.5245 | 51.3 |
| `H2__h1__s1__random_0` | random_0 | 1.00 | 9608 | 1980 | EARLY_STOPPING | 0.5440 | 0.6378 | 0.4645 | 0.5337 | 53.6 |
| `H2__h1__s1__random_1` | random_1 | 1.00 | 9608 | 2013 | EARLY_STOPPING | 0.5430 | 0.6378 | 0.4645 | 0.5309 | 54.9 |
| `H2__h1__s1__random_2` | random_2 | 1.00 | 9608 | 1914 | EARLY_STOPPING | 0.5498 | 0.6378 | 0.4645 | 0.5245 | 52.8 |
| `H2__h1__s2__profiles` | profiles | 1.00 | 9608 | 1518 | EARLY_STOPPING | 0.5476 | 0.6484 | 0.4688 | 0.5625 | 44.7 |
| `H2__h1__s2__random_0` | random_0 | 1.00 | 9608 | 1617 | EARLY_STOPPING | 0.5390 | 0.6484 | 0.4688 | 0.5426 | 46.5 |
| `H2__h1__s2__random_1` | random_1 | 1.00 | 9608 | 1518 | EARLY_STOPPING | 0.5469 | 0.6484 | 0.4688 | 0.5494 | 45.0 |
| `H2__h1__s2__random_2` | random_2 | 1.00 | 9608 | 1584 | EARLY_STOPPING | 0.5494 | 0.6484 | 0.4688 | 0.5602 | 46.8 |
| `H2__h1__s3__profiles` | profiles | 1.00 | 9608 | 1716 | EARLY_STOPPING | 0.5637 | 0.6365 | 0.4645 | 0.5599 | 48.6 |
| `H2__h1__s3__random_0` | random_0 | 1.00 | 9608 | 1716 | EARLY_STOPPING | 0.5532 | 0.6365 | 0.4645 | 0.5589 | 48.2 |
| `H2__h1__s3__random_1` | random_1 | 1.00 | 9608 | 1914 | EARLY_STOPPING | 0.5556 | 0.6365 | 0.4645 | 0.5541 | 52.0 |
| `H2__h1__s3__random_2` | random_2 | 1.00 | 9608 | 1716 | EARLY_STOPPING | 0.5508 | 0.6365 | 0.4645 | 0.5544 | 48.3 |
| `H2__h2__s1__profiles` | profiles | 1.00 | 9608 | 1947 | EARLY_STOPPING | 0.5824 | 0.6864 | 0.4890 | 0.5605 | 52.5 |
| `H2__h2__s1__random_0` | random_0 | 1.00 | 9608 | 2013 | EARLY_STOPPING | 0.5801 | 0.6864 | 0.4890 | 0.5610 | 52.7 |
| `H2__h2__s1__random_1` | random_1 | 1.00 | 9608 | 2013 | EARLY_STOPPING | 0.5773 | 0.6864 | 0.4890 | 0.5659 | 52.5 |
| `H2__h2__s1__random_2` | random_2 | 1.00 | 9608 | 1980 | EARLY_STOPPING | 0.5927 | 0.6864 | 0.4890 | 0.5673 | 52.1 |
| `H2__h2__s2__profiles` | profiles | 1.00 | 9608 | 1518 | EARLY_STOPPING | 0.5913 | 0.6987 | 0.4949 | 0.6036 | 43.0 |
| `H2__h2__s2__random_0` | random_0 | 1.00 | 9608 | 1551 | EARLY_STOPPING | 0.5827 | 0.6987 | 0.4949 | 0.5878 | 43.8 |
| `H2__h2__s2__random_1` | random_1 | 1.00 | 9608 | 1518 | EARLY_STOPPING | 0.5819 | 0.6987 | 0.4949 | 0.5897 | 42.9 |
| `H2__h2__s2__random_2` | random_2 | 1.00 | 9608 | 1617 | EARLY_STOPPING | 0.5895 | 0.6987 | 0.4949 | 0.6007 | 45.0 |
| `H2__h2__s3__profiles` | profiles | 1.00 | 9608 | 1782 | EARLY_STOPPING | 0.5926 | 0.6862 | 0.4909 | 0.5950 | 47.9 |
| `H2__h2__s3__random_0` | random_0 | 1.00 | 9608 | 1716 | EARLY_STOPPING | 0.6025 | 0.6862 | 0.4909 | 0.5968 | 47.0 |
| `H2__h2__s3__random_1` | random_1 | 1.00 | 9608 | 1815 | EARLY_STOPPING | 0.5913 | 0.6862 | 0.4909 | 0.5996 | 50.0 |
| `H2__h2__s3__random_2` | random_2 | 1.00 | 9608 | 1881 | EARLY_STOPPING | 0.6015 | 0.6862 | 0.4909 | 0.5982 | 51.3 |
| `H2__h3__s1__profiles` | profiles | 1.00 | 9608 | 2079 | EARLY_STOPPING | 0.4377 | 0.5060 | 0.3635 | 0.4179 | 54.0 |
| `H2__h3__s1__random_0` | random_0 | 1.00 | 9608 | 2112 | EARLY_STOPPING | 0.4337 | 0.5060 | 0.3635 | 0.4207 | 55.1 |
| `H2__h3__s1__random_1` | random_1 | 1.00 | 9608 | 2112 | EARLY_STOPPING | 0.4287 | 0.5060 | 0.3635 | 0.4215 | 54.9 |
| `H2__h3__s1__random_2` | random_2 | 1.00 | 9608 | 2046 | EARLY_STOPPING | 0.4390 | 0.5060 | 0.3635 | 0.4249 | 53.4 |
| `H2__h3__s2__profiles` | profiles | 1.00 | 9608 | 1584 | EARLY_STOPPING | 0.4426 | 0.5085 | 0.3656 | 0.4462 | 44.3 |
| `H2__h3__s2__random_0` | random_0 | 1.00 | 9608 | 1650 | EARLY_STOPPING | 0.4290 | 0.5085 | 0.3656 | 0.4400 | 45.6 |
| `H2__h3__s2__random_1` | random_1 | 1.00 | 9608 | 1617 | EARLY_STOPPING | 0.4319 | 0.5085 | 0.3656 | 0.4389 | 44.7 |
| `H2__h3__s2__random_2` | random_2 | 1.00 | 9608 | 1617 | EARLY_STOPPING | 0.4350 | 0.5085 | 0.3656 | 0.4436 | 45.0 |
| `H2__h3__s3__profiles` | profiles | 1.00 | 9608 | 1848 | EARLY_STOPPING | 0.4521 | 0.5100 | 0.3673 | 0.4495 | 50.2 |
| `H2__h3__s3__random_0` | random_0 | 1.00 | 9608 | 1584 | EARLY_STOPPING | 0.4469 | 0.5100 | 0.3673 | 0.4451 | 45.5 |
| `H2__h3__s3__random_1` | random_1 | 1.00 | 9608 | 1914 | EARLY_STOPPING | 0.4471 | 0.5100 | 0.3673 | 0.4488 | 52.7 |
| `H2__h3__s3__random_2` | random_2 | 1.00 | 9608 | 1947 | EARLY_STOPPING | 0.4548 | 0.5100 | 0.3673 | 0.4498 | 53.5 |
| `H3__r0__s1__extractor` | extractor | 1.00 | 9608 | 1650 | EARLY_STOPPING | 0.6560 | 0.6907 | 0.4879 | 0.6393 | 47.4 |
| `H3__r0__s1__sequence` | sequence | 1.00 | 920 | 1518 | EARLY_STOPPING | 0.6553 | 0.6907 | 0.4879 | 0.6407 | 25.0 |
| `H3__r0__s1__summary` | summary | 1.00 | 808 | 2970 | UPDATE_BUDGET | 0.6830 | 0.6907 | 0.4879 | 0.6603 | 36.5 |
| `H3__r0__s2__extractor` | extractor | 1.00 | 9608 | 1353 | EARLY_STOPPING | 0.6558 | 0.6832 | 0.4880 | 0.6681 | 40.4 |
| `H3__r0__s2__sequence` | sequence | 1.00 | 920 | 1716 | EARLY_STOPPING | 0.6517 | 0.6832 | 0.4880 | 0.6650 | 28.7 |
| `H3__r0__s2__summary` | summary | 1.00 | 808 | 2970 | UPDATE_BUDGET | 0.6757 | 0.6832 | 0.4880 | 0.6920 | 38.7 |
| `H3__r0__s3__extractor` | extractor | 1.00 | 9608 | 1551 | EARLY_STOPPING | 0.6709 | 0.6968 | 0.4945 | 0.6793 | 44.7 |
| `H3__r0__s3__sequence` | sequence | 1.00 | 920 | 2046 | EARLY_STOPPING | 0.6695 | 0.6968 | 0.4945 | 0.6739 | 29.2 |
| `H3__r0__s3__summary` | summary | 1.00 | 808 | 2970 | UPDATE_BUDGET | 0.6858 | 0.6968 | 0.4945 | 0.6936 | 39.6 |
| `H3__r1__s1__extractor` | extractor | 1.00 | 9608 | 1947 | EARLY_STOPPING | 0.5824 | 0.6864 | 0.4890 | 0.5605 | 52.4 |
| `H3__r1__s1__sequence` | sequence | 1.00 | 920 | 1518 | EARLY_STOPPING | 0.5821 | 0.6864 | 0.4890 | 0.5587 | 25.3 |
| `H3__r1__s1__summary` | summary | 1.00 | 808 | 2970 | UPDATE_BUDGET | 0.6736 | 0.6864 | 0.4890 | 0.6440 | 36.0 |
| `H3__r1__s2__extractor` | extractor | 1.00 | 9608 | 1518 | EARLY_STOPPING | 0.5913 | 0.6987 | 0.4949 | 0.6036 | 43.6 |
| `H3__r1__s2__sequence` | sequence | 1.00 | 920 | 1419 | EARLY_STOPPING | 0.5914 | 0.6987 | 0.4949 | 0.5994 | 23.4 |
| `H3__r1__s2__summary` | summary | 1.00 | 808 | 2970 | UPDATE_BUDGET | 0.6875 | 0.6987 | 0.4949 | 0.6920 | 37.4 |
| `H3__r1__s3__extractor` | extractor | 1.00 | 9608 | 1782 | EARLY_STOPPING | 0.5926 | 0.6862 | 0.4909 | 0.5950 | 49.1 |
| `H3__r1__s3__sequence` | sequence | 1.00 | 920 | 1881 | EARLY_STOPPING | 0.5924 | 0.6862 | 0.4909 | 0.5929 | 28.5 |
| `H3__r1__s3__summary` | summary | 1.00 | 808 | 2970 | UPDATE_BUDGET | 0.6770 | 0.6862 | 0.4909 | 0.6780 | 40.2 |
| `pilot__H2_profiles` | profiles | 1.00 | 9608 | 297 | UPDATE_BUDGET | 0.4876 | 0.5060 | 0.3635 | — | 22.4 |
| `pilot__H3_extractor` | extractor | 1.00 | 9608 | 297 | UPDATE_BUDGET | 0.6456 | 0.6864 | 0.4890 | — | 18.7 |
| `pilot__H3_sequence` | sequence | 1.00 | 920 | 297 | UPDATE_BUDGET | 0.6341 | 0.6864 | 0.4890 | — | 13.3 |

## Effects (unit = replicate)

| h | e(h) = MASE(profiles) − MASE(random) | SD over replicates |
|---:|---:|---:|
| 0 | -0.0089 | 0.0056 |
| 1 | +0.0058 | 0.0042 |
| 2 | -0.0001 | 0.0063 |
| 3 | +0.0056 | 0.0043 |

slope of e(h): +0.0038 (negative = advantage grows with heterogeneity); replicates 3, random assignments 3

| r | d_r = MASE(sequence) − MASE(summary) | SD |
|---:|---:|---:|
| 0 | -0.0227 | 0.0058 |
| 1 | -0.0907 | 0.0058 |

gamma = d_1 − d_0 = -0.0680 (negative = advantage larger with lagged dependence); units 3

Bootstrap over replicates (95 % percentile, descriptive precision): H2_slope: [+0.0000, +0.0064], H3_d1: [-0.0961, -0.0846], H3_gamma: [-0.0721, -0.0637]

Negative favours the method. This pilot reports effects, dispersion, precision and cost; it does not support H2/H3.
