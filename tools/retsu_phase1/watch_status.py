#!/usr/bin/env python3
"""Silent until the Dragon campaign fails or the training queue finishes."""
import json
import time
from pathlib import Path

STATUS = Path("/var/tmp/retsu-phase1/STATUS.json")
LOG = Path("/var/tmp/retsu-phase1/queue.log")


def load_status():
    try:
        return json.loads(STATUS.read_text())
    except Exception:
        return {}


def main():
    while True:
        rec = load_status()
        state = str(rec.get("state", ""))
        stage = str(rec.get("stage", ""))
        if state == "failed":
            print(
                f"FAILED: stage={stage} reason={rec.get('reason')} error={rec.get('error')}",
                flush=True,
            )
            raise SystemExit(1)
        if state == "queue_done":
            print("DONE: training queue finished", flush=True)
            raise SystemExit(0)
        text = LOG.read_text() if LOG.exists() else ""
        tail = text.rsplit("START ", 1)[-1]
        if "EXPORT_EXIT:" in tail and "EXPORT_EXIT:0" not in tail:
            print("FAILED: export exited nonzero", flush=True)
            raise SystemExit(1)
        if "TRAIN_EXIT:" in tail and "TRAIN_EXIT:0" not in tail:
            print("FAILED: train exited nonzero", flush=True)
            raise SystemExit(1)
        time.sleep(30)


if __name__ == "__main__":
    main()
