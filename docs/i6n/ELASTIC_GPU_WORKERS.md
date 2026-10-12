# Elastic GPU workers

`tools.i6n_swarm_campaign prepare-late-workers` adds capacity to a running
campaign without changing its sealed scientific `PLAN.json` or any fit
identity. The amendment is operational only: it binds the original plan and
E-plan digests, creates one identity per added GPU, and writes a receipt under
`status/WORKER_AMENDMENTS/`.

Late workers may execute search through their DOIN node and fixed `annual` or
`multi_output` units. They cannot execute `freeze`, so adding a GPU cannot
duplicate data preparation. Every node retains the campaign seed, fold support,
class caps, evaluator, and ledger fencing rules.

The RTX 4070 path requires an explicit `desktop_workload` declaration, a
measured class peak, 8 GiB of host RAM reserve, and 2 GiB of GPU memory reserve.
The RTX 5070 Ti path requires measured peaks for itself and a declared sibling
class; admission includes both 1.25x peaks, both wrappers, a 6 GiB aggregate
reserve, and the live `crispdm-batch.slice` limit. Neither path learns a limit
by provoking an OOM or lowers an admitted cap.

Example:

```bash
python -m tools.i6n_swarm_campaign prepare-late-workers \
  --root <campaign-root> \
  --spec <private-worker-amendment.json>
```

The private specification contains machine addresses and GPU UUIDs and must not
be committed. Repeating the command with the identical specification is
idempotent. A different amendment must carry the current `PLAN.json` file
digest and the original E-plan digest.
