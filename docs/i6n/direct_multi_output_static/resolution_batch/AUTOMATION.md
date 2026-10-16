# Execution And Status

The owner approved this entire finite batch. No stage asks for approval and no
agent should poll logs. Prerequisites are the existing governed deployment
JSON, retained pre-TEST input, writable artifact storage and the pinned GPU
Python environment. Secrets remain in the existing systemd EnvironmentFile;
the scripts never print or copy their contents.

Private launch JSON (outside git):

```json
{
  "root": "/var/tmp/i6dmo/<new-batch-directory>",
  "source_deployment": "/absolute/path/to/existing/deployment.json",
  "input_dir": "/absolute/path/to/retained/pre-test/input"
}
```

Run once from the committed checkout, with the coordinator's configured
Python interpreter. Bootstrap authenticates or prepares data, replicates the
exact committed runtime, checks mounts and the 5090, and installs owned timers:

```bash
crispdm-run -m 4G -t 1800s -n resolution-bootstrap -- \
  python -B -m tools.i6dmo.resolution_batch bootstrap --spec <private-launch.json>
```

After successful bootstrap, use the generated `BATCH.private.json` in `root`:

```bash
python -B -m tools.i6dmo.resolution_batch status --spec <root>/BATCH.private.json
python -B -m tools.i6dmo.resolution_batch status --spec <root>/BATCH.private.json --cell long_ds4
```

Low-level commands `fit-next` and `tick` are owned by the installed timers;
do not run another fit process. The GPU timer admits exactly one pending cell
at a time. It waits for compute availability, never resets hardware, and never
uses the second GPU. The CPU timer copies authenticated terminals, enqueues
their reports idempotently, verifies warehouse readback, writes `RESULTS.md`
and status, verifies local/second-host copies, and stops its own timers only.
This uses the existing admission and outbox modules, not a replacement DOIN.

Repeated bootstrap with the same private config reinstalls owned timers, not
fits. A terminal cell is never trained again. A power-loss partial receives a
typed interrupted disposition; it is retained, not silently repeated. Failed
spacing prerequisites block the conditional sweep explicitly. Resource
refusal happens before any model attempt and is retried by the timer with the
same cap. It is not an excuse to lower the cap or launch on CPU.

Status ETA starts unmeasured. Once fits finish it is labeled a provisional,
mixed-model, fit-only projection; it is not an exact end-to-end ETA. Progress
and heartbeat are persisted per epoch. Scientific negative results complete
normally and do not stop the remaining independent experiments.

Storage: queues/locks/private specs and outbox SQLite remain on ext4. Large
arrays, predictions and weights use configured artifact storage; they are not
committed to git or uploaded as GitHub release assets. The final ledger,
code/config identities, MAE and naive per horizon, model size, fit duration,
memory peak and stopping history remain available for later analysis.
