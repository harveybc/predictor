# Resource and plan reconciliation

Musashi, 2026-09-28 local; live readings 2026-09-29 02:31-02:35 UTC.
NO_NEW_MEASUREMENT. Read-only live inspection and documentary reconciliation,
not a new certification of every scientific result in the programme.

## Hosts and services

| Role | Finding | Disposition |
| --- | --- | --- |
| Coordinator | Rebooted today; about 22 GiB available RAM; no swap used; 4070 about 40 C; 126.5 GiB free disk | Lightweight orchestration; preserve desktop/services |
| Secondary worker | Rebooted today; about 19 GiB available RAM; no swap used; 4090 about 31 C; 662.9 GiB free disk | Preferred alternative for memory-heavy eligible workloads |
| Primary accelerator host | Up six days; external 5090 about 36 C; about 4.48 GiB MemAvailable, 4.37 GiB SUnreclaim, 0.185 GiB swap used; 174.1 GiB free disk | CUDA/headroom preflight required; not unconditional READY |

No compute applications appeared in nvidia-smi on any host; no admission lease
files were present. Two coordinator agent sessions existed, but process names
do not identify their conversation or prove completion. No agent was terminated.

The primary accelerator host logged NVIDIA NV_ERR_NO_MEMORY allocation failures
earlier today (last in the selected output at 15:06 local). No later Xid/OOM
matched the specific 15:07-onward query; repeated LTR-disabled messages remain.
The large unreclaimable slab is observed, its cause is NOT established. Do not
call this a proven driver leak or repair it by clearing caches/reloading drivers.
There were no matching current-boot OOM entries on the other two hosts.

The primary accelerator host has two failed p1lr decision units dated September
22, exit 4, plus a firmware notifier failure. These are not evidence of a newly
interrupted experiment after today's reboots. They were not restarted or reset.

The six probed service ports: /healthz returned 200 for governance, financial
lake, warehouse, synthetic lake and SOTA lake. The legacy public-panels port
refused connection: its unit is disabled/inactive, last stopped September 23.
It is NOT counted healthy and was not blindly reactivated. Initial /health
probes returned 404 because the actual endpoint is /healthz.

Chat and Alpaca runner on the coordinator, and both MT5 services on the
secondary worker, are active with NRestarts=0 after their hosts restarted.
No claim about new fills or quality follows. No broker mutation was made.

The installed launcher SHA256 is
`499fdc1877750337006de8aad6b30a943acfea7416aa157b7537c4b121c1dfc7`;
admission module SHA256 is
`8dc2c03b17e498697d634666309876686d051243ea4b640d65ec92c51ab8ee35`.
All three hosts match RR02 at `0ea5bff4`. This verifies deployment identity,
not bypass coverage or completion of interrupted verification selections.

Raw host snapshots are retained privately in the operator's
`health-20260928` state directory; machine identities are not published here.

## Where Satoshi stopped

Fetched predictor remote refs and inspected the delivered returns:

- RR01 `0ee09998`: restart manifest, interrupted attempts and sunk CPU retained.
- RR02 `0ea5bff4`: admission monitor delivered; now deployed identically.
- RR03/RR06 `ae82b189`: twelve-lane index, service identity and prepared pinning.
- RR04 `da22d7f9`: governed worker delivery and cube round-trip; mechanical,
  not a new positive forecasting result. Financial availability still absent.
- Q2 `7b0248f8`: historical censored/unanchored fits, withdrawn seal and successor;
  long-context comparison still missing, not a completed model-quality answer.
- H-CORE `c2d4388b`: prefix materialization delivered; remaining donor/consumer
  requirements must not be replaced with a duplicate prefix fit.
- M5PHET RR05: actual dirty worktree retained, not discarded. Isolated `44130ae`
  contains later router corrections, not yet evidence of integrated full scoring.

No new published predictor Satoshi tip beyond these deliveries was found by
this fetch. This does not exclude ongoing edits or unpublished work in an agent
session. A/B and R0/R1/R2 stay closed; they are not jobs to resume.

## Plan corrections made

Updated the existing research_dispatch_index.v2 at its canonical path, preserving
superseded observations. Reconciled monitor deployment, governed-worker proof,
M5PHET integration, post-reboot service observations and financial prerequisite.
Added ONE scientific lane for Weather/Traffic reference reproduction, linked to
the existing doctoral/business map; it is not a replacement for that programme.

Integrated the already-completed TSL lake/metric work: Electricity, Weather and
Traffic are registered and their governed bytes verified. The metric producer
contract and temporary deployed-provider tests exist. No new benchmark fit is
claimed, no fake production results were inserted, and the warehouse does not
magically enforce every future producer's scientific metadata.

The programme remains two concurrent maps: doctoral/business experiments and
M5PHET product. Full M4 acceptance and H-CORE's proposed large allocation are
not granted by this status update. Unit ambiguity in an inherited ECL row must
be settled against retained scorer evidence, not guessed during a health check.

Next dispatch is in
[RB01-RB07](../../handoffs/SATOSHI_RESUME_AND_BENCHMARK_ORDERS_2026_09_28.md).
No live service/checkpoint/presentation was changed by this reconciliation.
