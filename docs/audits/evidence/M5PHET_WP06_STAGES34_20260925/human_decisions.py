"""WP23's clause for the stages a PERSON configured: one decision record per (kind, question) the Laya stage carries.

Written BEFORE the fit, as a decision always is: a choice is a hypothesis, and the closure table judges it afterwards.
The option sets are copied verbatim from the Laya records that were asked of the same question on the same dataset, so
the two stages answer ONE question and the calibration report can put them in one group. Where the stage's own
configuration is not in the declared option set -- the hand pipeline's single block (there is no `k=1` cut) and its
`tcn` encoder (feature-extractor declares no such plugin) -- nothing is invented: `human_choice` refuses the key and
the refusal is kept as the evidence that the stage cannot supply that record.
"""
import json
import sys
from pathlib import Path

from m5phet import decide

LAYA_DECISIONS = Path("~/.local/state/m5phet/decisions").expanduser()
DECISIONS = Path("~/.local/state/m5phet/decisions-wp06-stage34-20260925").expanduser()
AS_OF = "2026-09-25T09:00:00+00:00"

FEATURES = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

# --- the option sets, verbatim from the linked Laya records of the same (kind, question) -------------------------------
LAYA_RECORDS = {"preprocessing": "c7cb869ea475658f704c280630a6166517945141df7817fe9dea210bc0d9a1bd",
                "grouping_cut": "ab09ba0c285346ed78612e6be2b87f3446dbb1beeb056ae6634d5e21fb5c6679",
                "extractor": "99ba1497853c554f2d61c43148824d6a34e2c4b3e212e582fee320f10d456633"}
OPTIONS = {name: json.loads((LAYA_DECISIONS / f"{digest}.json").read_text())["options"]
           for name, digest in LAYA_RECORDS.items()}

# --- the representations this comparison chooses between, declared by the design job and by the hand spec --------------
REPRESENTATION_OPTIONS = [
    ["hand_household_w60", "window [60], lags [1]: the window the household DEV resource e1_household_dev_pilot_v1 "
                           "declares, fixed by hand before this work"],
    ["short_memory", "windows [197], lags [1, 197]: the ACF of the levels falls inside the +-0.011206 band at lag 197"],
    ["seasonal_lag_74", "windows [74], lags [1, 74]: ACF peak at lag 74 (rho 0.356627)"],
    ["seasonal_lag_1443", "windows [197, 1443], lags [1, 1443], feature hour_of_day: ACF peak at lag 1443 "
                          "(rho 0.277748), period 86580 s"],
    ["seasonal_lag_2892", "windows [197, 2892], lags [1, 2892]: ACF peak at lag 2892 (rho 0.244256)"],
]

PREPROCESSING_WHY = (
    "every stage of this comparison is fitted by tools/fit_pipeline_spec.py, which standardises each column with a "
    "per-column z-score whose mean and standard deviation are fitted on the TRAIN rows only. Of the preprocessors the "
    "repositories declare, `normalizer` is the one whose declared job is exactly that scaling (its `method` parameter "
    "takes 'z-score'); `default_plugin`, `default_preprocessor` and `stl_preprocessor` are whole-dataset pipeline "
    "stages that trim and split a file, and `trimmer`, `cleaner`, `unbiaser` and `feature_selector` describe other "
    "operations this stage does not perform. The choice is what the stage USES, not what its spec left unnamed.")

STAGES = {
    "baseline_hand": {"representation": "hand_household_w60", "grouping": "k=1", "extractor": "tcn",
                      "representation_why": "the household DEV resource's own window, fixed by hand before this work; "
                                            "this stage exists to be the baseline the designed representations are "
                                            "measured against"},
    "quantile_hand": {"representation": "hand_household_w60", "grouping": "k=1",
                      "extractor": "NOT_APPLICABLE_SINGLE_WINDOW_CORE",
                      "representation_why": "WP07 fits a quantile head on the SAME hand representation, so that the "
                                            "interval is not confounded with a change of window"},
    "candidate_short_memory": {"representation": "short_memory", "grouping": "k=1", "extractor": "tcn",
                               "representation_why": "WP06 stage 3 fits one model per design candidate; this stage is "
                                                     "the candidate `short_memory` and nothing else about the "
                                                     "pipeline differs from baseline_hand"},
    "candidate_seasonal_lag_74": {"representation": "seasonal_lag_74", "grouping": "k=1", "extractor": "tcn",
                                  "representation_why": "WP06 stage 3 fits one model per design candidate; this stage "
                                                        "is the candidate `seasonal_lag_74` and nothing else about "
                                                        "the pipeline differs from baseline_hand"},
    "candidate_seasonal_lag_1443": {"representation": "seasonal_lag_1443", "grouping": "k=1", "extractor": "tcn",
                                    "representation_why": "WP06 stage 3 fits one model per design candidate; this "
                                                          "stage is the candidate `seasonal_lag_1443` and nothing "
                                                          "else about the pipeline differs from baseline_hand"},
    "candidate_seasonal_lag_2892": {"representation": "seasonal_lag_2892", "grouping": "k=1", "extractor": "tcn",
                                    "representation_why": "WP06 stage 3 fits one model per design candidate; this "
                                                          "stage is the candidate `seasonal_lag_2892` and nothing "
                                                          "else about the pipeline differs from baseline_hand"},
}


def state_for(stage, kind, extra):
    payload = {"dataset": "household_dev_slice.csv (household DEV slice, 1-minute grid, 50400 rows)",
               "target": "Global_active_power", "horizon_steps": 60, "stage": stage,
               "fitting_harness": "predictor tools/fit_pipeline_spec.py, sealed holdout 33820b552ddf"}
    payload.update(extra)
    return decide.decision_state(kind, payload)


def main():
    out = {"written": [], "refused": []}
    for stage, plan in STAGES.items():
        # 1. the representation this stage was given, out of the five this comparison declares
        entry = decide.human_choice(
            kind="representation", question="candidate", options=REPRESENTATION_OPTIONS,
            chosen=plan["representation"], why=plan["representation_why"], as_of=AS_OF,
            state_text=state_for(stage, "representation", {
                "candidates_declared": [key for key, _ in REPRESENTATION_OPTIONS],
                "design_job": "m5phet.representation_design.v1 over the same file"}),
            record_dir=DECISIONS)
        record(out, stage, "representation", "candidate", entry)

        # 2. one preprocessing choice per feature, as the Laya stage carries one per feature
        for feature in FEATURES:
            entry = decide.human_choice(
                kind="feature_preprocessing", question="preprocessing", options=OPTIONS["preprocessing"],
                chosen="normalizer", why=PREPROCESSING_WHY, as_of=AS_OF,
                state_text=state_for(stage, "feature_preprocessing", {"feature": feature, "scale": "meter units"}),
                record_dir=DECISIONS)
            record(out, stage, "feature_preprocessing", "preprocessing", entry, feature=feature)

        # 3 and 4: the two questions this pipeline's own configuration cannot answer from the declared option set
        entry = decide.human_choice(
            kind="feature_grouping", question="grouping_cut", options=OPTIONS["grouping_cut"],
            chosen=plan["grouping"], why="this stage feeds every meter column to one encoder", as_of=AS_OF,
            state_text=state_for(stage, "feature_grouping", {"cut_used": plan["grouping"]}), record_dir=DECISIONS)
        record(out, stage, "feature_grouping", "grouping_cut", entry)

        entry = decide.human_choice(
            kind="group_extractor", question="extractor", options=OPTIONS["extractor"],
            chosen=plan["extractor"], why="the encoder this stage's core actually runs over the group", as_of=AS_OF,
            state_text=state_for(stage, "group_extractor", {"group_id": "all", "encoder_used": plan["extractor"]}),
            record_dir=DECISIONS)
        record(out, stage, "group_extractor", "extractor", entry)

    Path(sys.argv[1]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"written": len(out["written"]), "refused": len(out["refused"]),
                      "refusals": sorted({(item["kind"], item["refusal"]) for item in out["refused"]})}, default=list))


def record(out, stage, kind, question, entry, feature=None):
    item = {"stage": stage, "kind": kind, "question": question, "feature": feature}
    if entry["status"] == "OK":
        item.update({"record": Path(entry["record_path"]).stem, "chosen": entry["decision"]["chosen"]})
        out["written"].append(item)
    else:
        item.update({"refusal": entry.get("refusal"), "why": entry.get("why")})
        out["refused"].append(item)


if __name__ == "__main__":
    main()
