# Two production replays, and what separating the counters exposed

§3 of the 2026-09-15 order. Both runs are NON_GOVERNING, CPU only, bounded by
`crispdm-run -m 2G`, against the live services, each in a fresh run directory. No production
outage was induced and no service was restarted.

| run | requested | observed steps | observed updates | wall | status |
|---|---|---|---|---|---|
| `doin-offline-replay-prod-11` | 64 | — (summary absent) | — | 15.7 s | COMPLETED |
| `doin-offline-replay-prod-12` | 64 | **10,240** | 400 | 15.8 s | COMPLETED |
| `doin-offline-replay-prod-13` | 64 | **256** | 10 | 4.4 s | COMPLETED |

Reconciliation empty on all three lists for each; `prod-11` is retained intact as the earlier
mechanical evidence.

## What the separation caught

`prod-12` is the first run whose receipt carries counters read from the runtime itself
(`num_timesteps`, `_n_updates`). It reported **10,240** environment steps against a configured
budget of **64** — a factor of 160. The cause: the agent plugin resolves its parameters from
the **top level** of the configuration, and the runner wrote the budget only under `training`,
so the agent never saw it and used its own default of 10,000. Under the old receipt, which
echoed the configured number, this run would have been reported as "64 steps" and the
discrepancy would have been invisible.

`prod-13` is the same replay after the runner writes the budget where the plugin reads it.
Observed work drops to **256** steps and **10** updates, and the wall time falls from 15.8 s to
4.4 s — consistent with far less work being done.

## What is still not explained, and is not being explained away

256 observed steps against a 64-step budget is a factor of four, and this round did not
establish why. The resolved configuration names **no** agent, environment or pipeline plugin,
so defaults decide, and the relationship between a `learn()` budget and counted environment
steps depends on that plugin — vectorisation and `train_freq` are the usual reasons a step
counter advances faster than the budget. No interpretation is recorded in the receipt: the
requested budget, the observed work and the wrapper's duration are three separate numbers and
they stay separate.

That is the point of the requirement. A receipt that reported "64" would have been wrong twice
over and looked right both times.
