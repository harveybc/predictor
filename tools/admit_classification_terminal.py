#!/usr/bin/env python3
"""The producer-to-store gate, as a command a loader can put in front of a write.

    python tools/admit_classification_terminal.py TERMINAL.json [TERMINAL.json ...]

Exit 0 when every terminal is admitted, 2 when one is refused. The refusal is
printed with its name, so a loader's log says WHY a row was not taken rather
than that something went wrong.

It reads the terminal's tags only. A terminal of another contract is reported
`NOT_THIS_CONTRACT` and admitted: the general-purpose warehouse stays generic.
Nothing here is written anywhere, and no store is contacted.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import classification_provenance as provenance  # noqa: E402


def main(argv):
    if not argv:
        print(__doc__.strip())
        return 1
    refused = 0
    for name in argv:
        try:
            body = json.loads(Path(name).read_text())
        except (OSError, ValueError) as error:
            print(json.dumps({"file": name, "admitted": False,
                              "refusal": "TERMINAL_UNREADABLE", "detail": str(error)}))
            refused += 1
            continue
        try:
            verdict = provenance.admit_classification_terminal(body)
        except provenance.AdmissionRefused as error:
            print(json.dumps({"file": name, "admitted": False,
                              "refusal": error.refusal, "detail": error.detail}))
            refused += 1
            continue
        print(json.dumps({"file": name, **verdict}))
    return 2 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
