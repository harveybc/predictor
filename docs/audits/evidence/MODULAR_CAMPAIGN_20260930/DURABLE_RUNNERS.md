# Durable campaign runners (GPU-IDLE-03 remedy)

**Problem.** At about 09:25Z an API session limit killed five agents. The campaign runner and loop processes they had started were children of those agent sessions, so they died with them. The 5090, 4090 and 5070 Ti went idle with cells still queued.

**Remedy.** Run every campaign loop as a **transient user-systemd unit**. Such a unit is owned by the user's systemd manager, not by any agent session, so it survives agent death. M06's own writer, watcher and GPU sampler already run this way and survived 09:25Z.

What this does **not** do:
- It adds no persistence mode.
- It adds no timers.
- It installs no unit files under `~/.config/systemd`.
- It survives agent death but not a reboot.
- Nothing is installed until the runner's owner confirms.

## Template (run by the owner of the runner, on the host where the loop runs)

```bash
NAME=<runner-name>                    # e.g. d2-runner-worker-a, f2-loop-worker-b, g2-cells-worker-b
WORKDIR=<checkout of the runner's pinned commit>
STOP=<path of the STOP file the loop honours>
systemd-run --user --unit "$NAME" --description "durable campaign runner $NAME" \
  -p Restart=on-failure -p RestartSec=30 -p WorkingDirectory="$WORKDIR" \
  -p StandardOutput=append:$HOME/.local/state/runners/$NAME.log -p StandardError=append:$HOME/.local/state/runners/$NAME.log \
  $HOME/.local/bin/crispdm-run -m 512M -t 24h -n "$NAME" -q -W 3600 -- \
  python3 -u <loop entry point> <args> --stop-file "$STOP"
```

- **Stopping deliberately.** Create `$STOP`. The loop finishes its current cell, then exits with status 0. `Restart=on-failure` does not restart a clean exit.
- **Crashes.** A crash or OOM exits non-zero, and the unit restarts after 30 s. On restart the loop's own `recover()` adopts any finished remote outcome. This is already in M04's runner since 1dd13e5d.
- **Heartbeat.** The loop must write `heartbeat.json` at least every 2 minutes into a directory named on its own command line (`--root`/`--out`). M06's probe picks it up automatically and flags NO_PROGRESS after 120 s without an update.
- **Cap and exemption.** The 512M cap matches the orchestrator's coordinator exemption for orchestration processes (`d*-runner-*`, under 64 MiB observed). The loop does ssh/launch/poll only, never data or model code.

## Rollback

```bash
touch "$STOP"                                   # graceful: finish the current cell
systemctl --user stop "$NAME"                   # or immediate: the unit's own scope only
systemctl --user reset-failed "$NAME" 2>/dev/null
```

Transient units leave no files behind, so the rollback is complete.

## Watcher coverage

- **New alarm `RUNNER_DOWN`.** It fires when a tracked campaign has runnable cells (queued, running or verifying) and no matching runner job is alive. The match pattern for each campaign comes from `runners` in `registry.json`.
- **Existing alarms that still apply:** the 120 s GPU-idle alarm (15 s sampling), and the 64 MiB alarm on exempt runners.

## Status per runner

| Runner | Owner | Confirmed | Installed |
|---|---|---|---|
| d2-runner-worker-a / -b | M04 | pending | no |
| M07 ETH loop | M07 (afcf115024ffa1381) | pending | no |
| lane G cell queue | lane G (aa7913479a97b45c0) | pending | no |
