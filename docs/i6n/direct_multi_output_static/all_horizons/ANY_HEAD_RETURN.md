# Any-Horizon Patience: Closed

One paired R0 fit on the secondary 4090, runtime b33a5933, seed 6102026,
same model and byte-identical FIT/INNER/DEVELOPMENT data as the twelve-head
mean-stop control. Only the reset policy changed. Patience 10, absolute
min_delta 1e-8, max 200; monitor each head's FIT/INNER mean independently.
Gradient loss and coherent lowest-average checkpoint selection were unchanged.

The new fit completed 82 epochs, best-average epoch 49, 410.5356 seconds,
stopping with ten consecutive epochs without a significant new record in ANY
head. Every weight hash and monitored scalar in the first 25 epochs matches
the mean-stop control bitwise. This verifies the shared training prefix and
the intended stopping-only contrast, not prospective predictive performance.

| Horizon | Mean-stop relative error | Any-head-stop relative error |
|---|---:|---:|
| 2h | +1.153705% | +3.017288% |
| 3h | +1.673595% | +4.262495% |
| 4h | +2.812239% | +5.392481% |
| 5h | +3.622676% | +6.243259% |
| 6h | +3.874222% | +8.188270% |
| 12h | +6.176801% | +10.318932% |
| 18h | +10.478196% | +13.291716% |
| 24h | +13.611284% | +15.680121% |
| 36h | +16.764985% | +23.329725% |
| 48h | +18.472094% | +27.312129% |
| 60h | +24.823746% | +36.124231% |
| 72h | +21.957515% | +32.777927% |

All values use (MAE_h-naive_h)/naive_h on exactly the same business rows and
original-price scale per horizon as the control. No head beats its own naive;
the change allowed more training but worsened these DEVELOPMENT 2024 results.
Checkpoint selection remains a distinct mechanism; no conclusion that all
other stopping or selection policies are ineffective follows. TEST stayed
closed, and no strategy run was opened. One report read back exactly, zero
pending/mismatch, thirteen immutable files verified in each physical mirror;
the owned timers closed automatically. The preferred GPU continuation is
independent and was not changed.
