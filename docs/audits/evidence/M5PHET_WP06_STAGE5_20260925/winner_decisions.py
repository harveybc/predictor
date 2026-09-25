"""WP23's clause applied to the stage this round ranked first: it must carry decision records like any other.

The search chose the representation and nothing else. Everything else in a searched stage -- the preprocessing, the
single block, the encoder -- was held at `baseline_hand`'s configuration by a PERSON, so those records are human ones
and are written with the same option sets, the same grounds and the same shapes as the ones the previous round wrote
for the hand-configured stages; only the stage in the state text differs. Two of the four questions refuse, exactly as
they refused there, and the refusals are the evidence that this pipeline cannot answer them from the declared options:

* `feature_grouping/grouping_cut` -- one block, and there is no `k=1` among the declared cuts;
* `group_extractor/extractor` -- the encoder is `tcn`, an inline family of `fused_branches`, and `feature-extractor`
  declares no such plugin.

A third refusal is new and is the reason the search needed a question of its own: the winning representation is not one
of the five `representation/candidate` options this comparison declared, and a search may not add an option to a set
the executing repository declared. Its own choice lives under `representation/searched_candidate`.

Only the rank-1 stage is given these records. The other searched stages are `COMPARABLE_BUT_NO_DECISION_RECORD` and
the calibration report prints that verdict per stage: they are comparable in the table and unusable for calibration,
which is WP23's own distinction and not a gap left by accident.

Usage: winner_decisions.py <table.json> <stage> <decisions dir> <out index.json>
"""
import json
import sys
from pathlib import Path

from m5phet import decide

AS_OF = "2026-09-25T09:00:00+00:00"

FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

#: the option sets and the grounds are read from the RECORDS the previous round wrote, not retyped: a record carries
#: the option set it was chosen from and the `why` it was made on, so reading them back is the only way to be sure the
#: two rounds' human records answer one question and land in one calibration group.
PREVIOUS_INDEX = Path(__file__).resolve().parents[1] / "M5PHET_WP06_STAGES34_20260925" / "human_decisions.json"
PREVIOUS_DECISIONS = Path("~/.local/state/m5phet/decisions-wp06-stage34-20260925").expanduser()

previous = json.loads(PREVIOUS_INDEX.read_text())


def from_previous(kind, question):
    """One earlier human record of this (kind, question): its option set and its ground, verbatim."""
    item = next(entry for entry in sorted(previous["written"], key=lambda e: e["record"])
                if entry["kind"] == kind and entry["question"] == question)
    record = json.loads((PREVIOUS_DECISIONS / f"{item['record']}.json").read_text())
    return record["options"], record["why"]


REPRESENTATION_OPTIONS, _representation_why = from_previous("representation", "candidate")
PREPROCESSING_OPTIONS, PREPROCESSING_WHY = from_previous("feature_preprocessing", "preprocessing")

#: these two questions the previous round could not answer from the declared options either, so no earlier record of
#: them exists to read; their option sets come from the Laya records the previous round read them from
LAYA_DECISIONS = Path("~/.local/state/m5phet/decisions").expanduser()
LAYA_RECORDS = {"grouping_cut": "ab09ba0c285346ed78612e6be2b87f3446dbb1beeb056ae6634d5e21fb5c6679",
                "extractor": "99ba1497853c554f2d61c43148824d6a34e2c4b3e212e582fee320f10d456633"}
OPTIONS = {name: json.loads((LAYA_DECISIONS / f"{digest}.json").read_text())["options"]
           for name, digest in LAYA_RECORDS.items()}
OPTIONS["preprocessing"] = PREPROCESSING_OPTIONS

table = json.loads(Path(sys.argv[1]).read_text())
stage = sys.argv[2]
decisions = Path(sys.argv[3]).expanduser()
rows = {row["stage"]: row for area in table["areas"] if area["area"] == "forecast" for row in area["rows"]}
if rows[stage]["rank"] != 1:
    raise SystemExit(f"{stage} is ranked {rows[stage]['rank']}, not first; this script exists for the rank-1 stage")

out = {"stage": stage, "written": [], "refused": []}


def state_for(kind, extra):
    payload = {"dataset": "household_dev_slice.csv (household DEV slice, 1-minute grid, 50400 rows)",
               "target": "Global_active_power", "horizon_steps": 60, "stage": stage,
               "fitting_harness": "predictor tools/fit_pipeline_spec.py, sealed holdout 33820b552ddf"}
    payload.update(extra)
    return decide.decision_state(kind, payload)


def keep(kind, question, entry, feature=None):
    item = {"stage": stage, "kind": kind, "question": question, "feature": feature}
    if entry["status"] == "OK":
        item.update({"record": Path(entry["record_path"]).stem, "chosen": entry["decision"]["chosen"]})
        out["written"].append(item)
    else:
        item.update({"refusal": entry.get("refusal"), "why": entry.get("why")})
        out["refused"].append(item)


keep("representation", "candidate", decide.human_choice(
    kind="representation", question="candidate", options=REPRESENTATION_OPTIONS, chosen="searched", as_of=AS_OF,
    why="the representation a search chose; it is not one of the five this comparison declared",
    state_text=state_for("representation", {"candidates_declared": [key for key, _ in REPRESENTATION_OPTIONS],
                                            "design_job": "m5phet.representation_design.v1 over the same file"}),
    record_dir=decisions))

for feature in FEATURES:
    keep("feature_preprocessing", "preprocessing", decide.human_choice(
        kind="feature_preprocessing", question="preprocessing", options=OPTIONS["preprocessing"],
        chosen="normalizer", why=PREPROCESSING_WHY, as_of=AS_OF,
        state_text=state_for("feature_preprocessing", {"feature": feature, "scale": "meter units"}),
        record_dir=decisions), feature=feature)

keep("feature_grouping", "grouping_cut", decide.human_choice(
    kind="feature_grouping", question="grouping_cut", options=OPTIONS["grouping_cut"], chosen="k=1",
    why="this stage feeds the columns the search selected to one encoder", as_of=AS_OF,
    state_text=state_for("feature_grouping", {"cut_used": "k=1"}), record_dir=decisions))

keep("group_extractor", "extractor", decide.human_choice(
    kind="group_extractor", question="extractor", options=OPTIONS["extractor"], chosen="tcn",
    why="the encoder this stage's core actually runs over the group", as_of=AS_OF,
    state_text=state_for("group_extractor", {"group_id": "all", "encoder_used": "tcn"}), record_dir=decisions))

Path(sys.argv[4]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps({"stage": stage, "written": len(out["written"]), "refused": len(out["refused"]),
                  "refusals": sorted({(item["kind"], item["refusal"]) for item in out["refused"]})}, default=list))
