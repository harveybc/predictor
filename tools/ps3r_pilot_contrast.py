"""PS3-R paired contrast table from pilot records (tests: tests/test_ps3r_pilot_contrast.py).

    python tools/ps3r_pilot_contrast.py --records <dir>/records --out <dir>/CONTRAST.json
        [--card-validator <feature-extractor checkout>]

Refuses, by name, before producing anything:
  OVERLAP                 probe fit and evaluation origins overlap or are not chronological;
  UNPAIRED_ROWS           the AE and contrastive arms were evaluated on different rows;
  SELF_FORECAST_REFUSED   a probe target that is not Y_s / Y_l / Y_b;
  POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT   a latent that is not (24, channels);
  RANDOM_ARM_NOT_PAIRED   the random or raw arm differs between the two families of one
                          (input, fold, seed) - they must be the same initial state and the same window.
Failed fits are listed, never dropped silently. The output carries per-row losses, Delta_probe of each
family, the paired contrast loss_ae - loss_contrastive on identical rows, the paired naive, and a
summary per input x target x horizon over folds and seeds. With ``--card-validator`` it also emits one
``representation_candidate_card.v1`` per input x family, validated by feature-extractor's
``app.representation_card.validate_card``.
"""
import argparse
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ALLOWED = ("Y_s", "Y_l", "Y_b")
TOL = 1e-9


def _check_record(r):
    shape = r["architecture"]["latent_shape"]
    if len(shape) != 2 or shape[0] != 24:
        raise ValueError(f"POOLED_OUTPUT_VIOLATES_TEMPORAL_CONTRACT: {r['input']} {r['arm']} latent {shape}")
    fit, ev = r["ranges"]["probe_fit_origins"], r["ranges"]["probe_eval_origins"]
    if not fit[0] <= fit[1] < ev[0] <= ev[1]:
        raise ValueError(f"OVERLAP: probe fit origins {fit} and eval origins {ev} for {r['input']} {r['fold']}")
    for p in r["probes"]:
        if p["target"] not in ALLOWED:
            raise ValueError(f"SELF_FORECAST_REFUSED: probe target {p['target']!r}")


def build_table(records):
    failed = [{k: r[k] for k in ("input", "fold", "seed", "arm", "reason")}
              for r in records if r.get("status") == "FIT_FAILED"]
    good = [r for r in records if r.get("status") != "FIT_FAILED"]
    by_key = defaultdict(dict)
    for r in good:
        _check_record(r)
        by_key[(r["input"], r["fold"], r["seed"])][r["arm"]] = r
    rows = []
    for (inp, fold, seed), arms in sorted(by_key.items()):
        if set(arms) != {"ae", "contrastive"}:
            continue
        ae = {(p["target"], p["horizon"]): p for p in arms["ae"]["probes"]}
        cl = {(p["target"], p["horizon"]): p for p in arms["contrastive"]["probes"]}
        for key in sorted(set(ae) & set(cl)):
            a, c = ae[key], cl[key]
            if a["eval_rows_sha256"] != c["eval_rows_sha256"]:
                raise ValueError(f"UNPAIRED_ROWS: {inp} {fold} {seed} {key}")
            for arm_name in ("loss_random", "loss_raw", "naive"):
                if abs(a[arm_name] - c[arm_name]) > TOL * max(1.0, abs(a[arm_name])):
                    raise ValueError(f"RANDOM_ARM_NOT_PAIRED: {arm_name} differs for {inp} {fold} {seed} {key}")
            rows.append({"input": inp, "fold": fold, "seed": seed, "target": key[0], "horizon": key[1],
                         "loss_name": a.get("loss_name"), "eval_rows": a.get("eval_rows"), "naive": a["naive"],
                         "loss_raw": a["loss_raw"], "loss_random": a["loss_random"],
                         "loss_ae": a["loss_trained"], "loss_contrastive": c["loss_trained"],
                         "delta_probe_ae": a["loss_random"] - a["loss_trained"],
                         "delta_probe_contrastive": c["loss_random"] - c["loss_trained"],
                         "contrast_ae_minus_contrastive": a["loss_trained"] - c["loss_trained"]})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["input"], row["target"], row["horizon"])].append(row)
    summary = []
    for (inp, target, horizon), items in sorted(groups.items()):
        def stats(key):
            vals = [i[key] for i in items]
            mean = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else None
            return mean, sd
        cm, csd = stats("contrast_ae_minus_contrastive")
        da, dasd = stats("delta_probe_ae")
        dc, dcsd = stats("delta_probe_contrastive")
        summary.append({"input": inp, "target": target, "horizon": horizon, "n": len(items),
                        "contrast_mean": cm, "contrast_sd": csd, "delta_ae_mean": da, "delta_ae_sd": dasd,
                        "delta_contrastive_mean": dc, "delta_contrastive_sd": dcsd,
                        "naive_mean": stats("naive")[0], "raw_mean": stats("loss_raw")[0],
                        "ae_beats_naive": sum(i["loss_ae"] < i["naive"] for i in items),
                        "contrastive_beats_naive": sum(i["loss_contrastive"] < i["naive"] for i in items)})
    return {"schema": "ps3r.pilot.contrast.v1", "status": "PAIRED_CONTRAST_GENERATED",
            "resolution": "values kept at full float precision; report at 1e-5 or finer",
            "rows": rows, "summary": summary, "failed_fits": failed,
            "limits": ["20 inputs (stratified sample, inclusion probabilities in the selection)", "2 seeds",
                       "inner-fold validation only; no outer validation or test read",
                       "single feature per branch; no grouping", "no winner is declared by this pilot"]}


def cards(records, validator_path):
    sys.path.insert(0, str(validator_path))
    validate = importlib.import_module("app.representation_card").validate_card
    out = {}
    for r in records:
        if r.get("status") == "FIT_FAILED":
            continue
        key = (r["input"], r["arm"])
        card = out.setdefault(key, {
            "schema": "representation_candidate_card.v1",
            "candidate_id": f"ps3r-pilot-{r['arm']}-{r['input']}".lower().replace("_", "-"),
            "family": "autoencoder_control" if r["arm"] == "ae" else "contrastive",
            "architecture": "conv1d_tcn", "objective": r["objective"]["name"] + "@" + r["objective"]["version"],
            "objective_identity": r["objective"], "declared_deviations": r["declared_deviations"],
            "input_identity": r.get("input_identity"),
            "corpus": {"kind": "TRAIN_ONLY", "pretrained_weights_source": None, "train_folds": []},
            "latent": {"layout": "temporal", "time_steps": r["architecture"]["latent_shape"][0],
                       "channels": r["architecture"]["latent_shape"][1], "grid_adapter": None},
            "conditioning_contract": "OPERATIONAL",
            "evaluation": {"reconstruction": r["reconstruction"], "probes": []},
            "status": "PILOT_ENGINEERING"})
        if r["fold"] not in card["corpus"]["train_folds"]:
            card["corpus"]["train_folds"].append(r["fold"])
        keys = ("target", "horizon", "fold", "seed", "probe", "loss_name", "loss_trained", "loss_random",
                "loss_raw", "naive", "delta_probe", "preservation", "raw_dimension_differs")
        card["evaluation"]["probes"] += [{k: p[k] for k in keys if k in p} for p in r["probes"]]
    validated = []
    for card in out.values():
        extra = {k: card.pop(k) for k in ("objective_identity", "declared_deviations", "input_identity", "status")}
        checked = validate(card)
        checked.update(extra)
        validated.append(checked)
    return validated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--card-validator")
    a = p.parse_args()
    records = [json.loads(f.read_text()) for f in sorted(Path(a.records).glob("*.json"))]
    table = build_table(records)
    if a.card_validator:
        table["cards"] = cards(records, a.card_validator)
    Path(a.out).write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(table["rows"]), "summary": len(table["summary"]),
                      "failed_fits": len(table["failed_fits"]), "cards": len(table.get("cards", []))}))


if __name__ == "__main__":
    main()
