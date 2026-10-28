"""Fake of the model lane's ``python -m tools.i6dmo.train`` CLI contract (no TensorFlow).

Env I6DMO_FAKE_TRAIN_RC: 0 (COMPLETED), 3 (FAILED_TYPED receipt), 2 (contract error, no receipt), 137 (killed).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _atomic(path, value):
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    for name in ("--design", "--bundle", "--out", "--device"):
        p.add_argument(name, required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    rc = int(os.environ.get("I6DMO_FAKE_TRAIN_RC", "0"))
    out = Path(args.out)
    if rc == 2:
        print(json.dumps({"status": "CONTRACT_ERROR", "code": "OUT_NOT_EMPTY"}))
        return 2
    out.mkdir(parents=True)
    for epoch in (1, 2):
        _atomic(out / "PROGRESS.json", dict(epoch=epoch, max_epochs=200, updated_at=time.time(), best_epoch=1))
    if rc == 137:
        return 137
    (out / "model").mkdir()
    (out / "model" / "MODEL.json").write_text("{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    receipt = dict(schema="i6dmo.train_receipt.v1", status="COMPLETED" if rc == 0 else "FAILED_TYPED",
                   code=None if rc == 0 else "ZERO_NAIVE_SCALE", device=args.device,
                   identity=dict(weights_sha256="w" * 64), artifacts={"model/MODEL.json": sha},
                   training=dict(best_epoch=1, stop_reason="patience", epochs_completed=2),
                   timings=dict(fit_s=1.5, predict_s=0.25, wall_s=2.0),
                   resources=dict(cgroup_memory_peak_bytes=123456, cpu_user_s=1.0, cpu_sys_s=0.1, device=args.device))
    _atomic(out / "RECEIPT.json", receipt)
    return rc


if __name__ == "__main__":
    sys.exit(main())
