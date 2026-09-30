# DEPLOY: ADM-DEADCACHE-01 admission fix on the preferred worker (the owner runs this)

The orchestrator reviewed and accepted the fix at commit `0928dc06` on branch `satoshi/m06-admission-dead-cache-20260930`. Deploying it writes to shared tooling, which is the owner's call; neither M06 nor the orchestrator deployed it.

What it changes:
- **The gate.** It charges the slice's `memory.current` minus its clean file cache. Clean cache is file − shmem − dirty − writeback − unevictable.
- **Scope-end reclaim.** Every new launch runs its command under `scope-exec`. When the command ends, scope-exec reclaims that job's own scope's clean cache through the scope's own `memory.reclaim`.
- **Only future launches.** Files are replaced by rename. A launcher that is already running keeps the text it opened, and no running child is touched.

| file | target | new sha256 (0928dc06) | rollback sha256 (deployed ddadf4a9) |
|---|---|---|---|
| launcher | `~/.local/bin/crispdm-run` | `056e207a1f36120cb24d8063932988082f50eca368bb03c6e2bb3de9df69d7d7` | `499fdc1877750337006de8aad6b30a943acfea7416aa157b7537c4b121c1dfc7` |
| admission module | `~/.local/libexec/crispdm/crispdm_admission.py` | `5cf92c8a9c1e2e42997eecb9141b90975312fbc5c1c98f5e400a00b0c4d1582b` | `8dc2c03b17e498697d634666309876686d051243ea4b640d65ec92c51ab8ee35` |

## One paste, on the preferred worker's own shell

```bash
set -euo pipefail
R=$HOME/Documents/GitHub/predictor
git -C "$R" fetch -q origin satoshi/m06-admission-dead-cache-20260930
T=$(mktemp -d)
git -C "$R" show 0928dc06:tools/crispdm-run          > "$T/crispdm-run"
git -C "$R" show 0928dc06:tools/crispdm_admission.py > "$T/crispdm_admission.py"
sha256sum -c <<EOF
056e207a1f36120cb24d8063932988082f50eca368bb03c6e2bb3de9df69d7d7  $T/crispdm-run
5cf92c8a9c1e2e42997eecb9141b90975312fbc5c1c98f5e400a00b0c4d1582b  $T/crispdm_admission.py
EOF
# rollback copies of the deployed bytes, verified before anything is replaced
B=$HOME/.local/state/crispdm-run/rollback_ddadf4a9; mkdir -p "$B"
cp -p "$HOME/.local/bin/crispdm-run"                       "$B/crispdm-run.499fdc18"
cp -p "$HOME/.local/libexec/crispdm/crispdm_admission.py"  "$B/crispdm_admission.py.8dc2c03b"
sha256sum -c <<EOF
499fdc1877750337006de8aad6b30a943acfea7416aa157b7537c4b121c1dfc7  $B/crispdm-run.499fdc18
8dc2c03b17e498697d634666309876686d051243ea4b640d65ec92c51ab8ee35  $B/crispdm_admission.py.8dc2c03b
EOF
# atomic replace (write beside, then rename): running launchers keep the text they opened
install -m 755 "$T/crispdm-run"          "$HOME/.local/bin/.crispdm-run.new"
mv -f "$HOME/.local/bin/.crispdm-run.new" "$HOME/.local/bin/crispdm-run"
install -m 755 "$T/crispdm_admission.py" "$HOME/.local/libexec/crispdm/.crispdm_admission.py.new"
mv -f "$HOME/.local/libexec/crispdm/.crispdm_admission.py.new" "$HOME/.local/libexec/crispdm/crispdm_admission.py"
sha256sum "$HOME/.local/bin/crispdm-run" "$HOME/.local/libexec/crispdm/crispdm_admission.py"
# smoke 1: the gate now reports charged vs in-use
python3 "$HOME/.local/libexec/crispdm/crispdm_admission.py" state | grep -E '"slice_(memory_current|charged_bytes)"'
# smoke 2: a 256M job runs through scope-exec and records its own-scope reclaim
"$HOME/.local/bin/crispdm-run" -m 256M -t 2m -n deploy-smoke-adm-deadcache -- true
grep SCOPE_CLEAN_CACHE_RECLAIM "$HOME/.local/state/crispdm/admission/ledger.jsonl" | tail -1
```

Expected results:
- Both `sha256sum -c` blocks print `OK`, and the final sha256 lines show `056e207a…` and `5cf92c8a…`.
- `slice_charged_bytes` is far below `slice_memory_current`. At the last reading, about 0.07 GiB was charged against 2.2 GiB in use.
- The smoke job exits 0, and the ledger's last `SCOPE_CLEAN_CACHE_RECLAIM` shows `"result": "RECLAIMED"` or `"NOTHING_CLEAN_TO_RECLAIM"` for a `crispdm-deploy-smoke-…scope` cgroup.

**After deploying, M04 must re-issue its queued request.** The `m04-modular-cost-pilot` acquire that is already waiting runs the OLD module in memory, and its in-process queue loop never reloads the gate. It waits until its own `-W` expires. M04 should stop its own queued launcher and launch again with the same honest `-m 6G`. That is not a lowered cap.

## Rollback (one paste)

```bash
set -euo pipefail
B=$HOME/.local/state/crispdm-run/rollback_ddadf4a9
install -m 755 "$B/crispdm-run.499fdc18"          "$HOME/.local/bin/.crispdm-run.new" && mv -f "$HOME/.local/bin/.crispdm-run.new" "$HOME/.local/bin/crispdm-run"
install -m 755 "$B/crispdm_admission.py.8dc2c03b" "$HOME/.local/libexec/crispdm/.crispdm_admission.py.new" && mv -f "$HOME/.local/libexec/crispdm/.crispdm_admission.py.new" "$HOME/.local/libexec/crispdm/crispdm_admission.py"
sha256sum "$HOME/.local/bin/crispdm-run" "$HOME/.local/libexec/crispdm/crispdm_admission.py"   # 499fdc18…, 8dc2c03b…
```

## The alternative: a one-time reclaim of the existing dead cache (also the owner's)

The worker_a slice held about 2.17 GiB of clean file cache with no live scope. A single write reclaims it without deploying anything:

```bash
echo 2G > /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/memory.reclaim
```

This is slice-level. It may return `EAGAIN` after a partial reclaim, and it does not stop the next finished job from leaving cache behind; the deployed fix does.

## Where the leases live

On every host the admission store is `~/.local/state/crispdm/admission/`, with `leases/`, `ledger.jsonl`, `queue.jsonl`, `requests/` and `retained/`. It is not `~/.local/state/crispdm-run/leases`. `crispdm_admission.py state` prints the path as `"store"`.
