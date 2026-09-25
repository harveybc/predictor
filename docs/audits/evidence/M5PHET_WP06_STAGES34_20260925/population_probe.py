"""What the sealed population WOULD be if a candidate's own window were the sealing window.

No model is fitted here and no score is produced. The question is only whether the rows a stage could be scored on are
the rows `baseline_hand` was scored on: the sealing rule fixes the population as the holdout origins whose whole
sealing window and whole horizon lie inside the holdout, so a stage whose window is longer than 197 can only be scored
on FEWER rows -- a different population, a different seal, and a NOT_COMPARABLE row rather than a number.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
TOOLS = Path("/home/harveybc/Documents/GitHub/.worktrees/predictor-rp132/tools")
sys.path.insert(0, str(TOOLS))
EVAL = Path("/home/harveybc/Documents/GitHub/M5PHET/evaluation/src")
sys.path.insert(0, str(EVAL))

import fit_pipeline_spec as fit

data = fit.read_csv(Path(sys.argv[1]))
data["target_index"] = data["columns"].index("Global_active_power")
out = {}
for seal_window in (197, 1443, 2892):
    population = fit.sealed_population(data, holdout_fraction=0.2, seal_window=seal_window, horizon=60)
    protocol, seal = fit.build_protocol_and_seal(
        EVAL, population["rows"], population["labels"], data_path=Path(sys.argv[1]),
        sealed_at="2026-09-25T00:00:00Z", seal_window=seal_window, horizon=60,
        holdout_fraction=0.2, minimum_rows=1000)
    out[f"seal_window_{seal_window}"] = {"sealed_rows": len(population["rows"]), "seal": seal.seal,
                                         "protocol_digest": protocol.digest,
                                         "first_origin": population["rows"][0], "last_origin": population["rows"][-1]}
print(json.dumps(out, indent=2, sort_keys=True))
