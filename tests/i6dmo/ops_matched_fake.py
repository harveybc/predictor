"""ORCHESTRATION-ONLY fake child for the matched short contrast (fit and report kinds).

Stands in for the model lane's matched trainer/report until they land; it proves the runner's
queue/worker/terminal/outbox/mirror/CLOSE plumbing, NOT the science. The reviewer runs the real
adapters. The comparison unit itself is always the REAL ``unit_runner --kind comparison`` hook.

Env:
  I6DMO_FAKE_COUNTER            append '<kind>:<unit_id>' per invocation (counts fits)
  I6DMO_FAKE_REL_<ARM>          relative error of every head of that arm (default mg1 -0.02, mg2m -0.01)
  I6DMO_FAKE_FAIL_UNIT          unit_id that exits 3 (technical failure)
  I6DMO_FAKE_DROP_HEAD          '<arm>:<h>' omitted from that arm's METRICS (missing head)
  I6DMO_FAKE_POPULATION         population digest the report declares (default: the campaign pin)
  I6DMO_FAKE_ROWS_<ARM>         per-head row-id digest the report declares (default: shared)
  I6DMO_FAKE_PEAK               cgroup peak bytes reported by a fit (default 1 GiB)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HORIZONS = (2, 3, 4, 5, 6, 12)
REPO = Path(__file__).resolve().parents[2]
DEFAULT_REL = {"mg1": -0.02, "mg2m": -0.01}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _genuine_shaped_fit(spec, view, out, arm, population):
    """A trainer-shaped fit (R1 donor fixtures only): every field the donor verifier binds, built from
    the spec exactly as the real trainer records it, WITHOUT a fake marker. It trains nothing; its
    acceptance depends on the system's own custody (mirrors) and publication records."""
    import zipfile
    runtime = view["design"]["sealed_identities"]["runtime"]
    train = out / "train"
    (train / "model").mkdir(parents=True)
    capacity_doc = json.loads((REPO / "docs/i6n/direct_multi_output_static/matched_short/"
                                      "CAPACITY_MATCHED.json").read_text())
    family = {"mg1": "short_mg1", "mg2m": "short_mg2"}[arm]
    record = dict(schema="i6dmo.capacity.v1", arm=arm, family=family,
                  counts=capacity_doc["declared_capacity"][family])
    record["capacity_sha256"] = _canonical_sha(record)
    (train / "CAPACITY.json").write_text(json.dumps(record))
    with zipfile.ZipFile(train / "model" / "forecast_model.keras", "w") as z:
        z.writestr("metadata.json", json.dumps({"keras_version": runtime["keras"]}))
        z.writestr("model.weights.h5", hashlib.sha256(f"weights-{arm}".encode()).digest())
    weights = hashlib.sha256(f"weights-{arm}".encode()).hexdigest()
    (train / "model" / "MODEL.json").write_text(json.dumps(dict(weights_sha256=weights)))
    arts = {rel: _sha(train / rel) for rel in ("CAPACITY.json", "model/forecast_model.keras", "model/MODEL.json")}
    identity = dict(common_population_sha256=population, matched_arm=arm, design_sha256=view["design_sha256"],
                    effective_config_sha256=view["adapter_effective"]["effective_config_sha256"],
                    capacity_sha256=record["capacity_sha256"], weights_sha256=weights,
                    model_archive_sha256=arts["model/forecast_model.keras"])
    receipt = dict(schema="i6dmo.train_receipt.v1", status="COMPLETED", scientific=True, device="gpu", overrides={},
                   identity=identity, artifacts=arts, matched=dict(arm=arm, capacity=record,
                                                                  common_population_sha256=population),
                   resources=dict(tensorflow_version=runtime["tensorflow"], device="gpu"))
    (train / "RECEIPT.json").write_text(json.dumps(receipt))
    artifacts = {rel: dict(path=f"train/{rel}", sha256=sha) for rel, sha in arts.items()}
    artifacts["RECEIPT"] = dict(path="train/RECEIPT.json", sha256=_sha(train / "RECEIPT.json"))
    cost = dict(experiment_key=f"i6dmo:{spec['campaign_id']}:{spec['unit_id']}:cost",
                experiment_set_key=f"i6dmo:{spec['campaign_id']}", actor="predictor", lake="LOCAL_RETAINED",
                lineage="UNVERIFIED", config_sha256=spec["identity_sha256"], code_commit="fixture", project="predictor",
                phase="I6DMO", tags={"unit": spec["unit_id"], "test_read": "false", "weights_sha256": weights},
                metrics=[dict(metric="Fit_Seconds", value=0.5, split="train", horizon=None, unit="seconds")])
    return artifacts, dict(trainer_identity=identity, scientific=True), [cost]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True)
    parser.add_argument("command")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    if spec.get("kind") != args.kind:
        return 4
    if os.environ.get("I6DMO_FAKE_COUNTER"):
        with open(os.environ["I6DMO_FAKE_COUNTER"], "a") as handle:
            handle.write(f"{args.kind}:{spec['unit_id']}\n")
    Path(args.progress).write_text(json.dumps(dict(done=1, total=1, unit="epoch", phase="fake", updated_at=time.time())))
    if os.environ.get("I6DMO_FAKE_FAIL_UNIT") == spec["unit_id"]:
        return 3
    infra = os.environ.get("I6DMO_FAKE_INFRA_CODE", "")
    if infra and os.environ.get("I6DMO_FAKE_INFRA_UNIT", spec["unit_id"]) == spec["unit_id"]:
        # a typed refusal as unit_runner/train write it; fit_seconds is set only if training started
        started = os.environ.get("I6DMO_FAKE_INFRA_AFTER_FIT") == "1"
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "TERMINAL.json").write_text(json.dumps(dict(
            schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
            status="FAILED", disposition=infra, artifacts={}, eligibility={}, gates={}, reports=[],
            costs=dict(fit_seconds=12.5 if started else None))))
        return 0
    arm = spec["arm"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    view = spec["effective_config"]
    assert view.get("arm") == arm and view["design"] == view["arms"][arm]["design"], "arm view missing"
    assert Path(spec["pins"]["common_population"]["path"], "POPULATION.json").is_file(), "population pin unresolved"
    assert Path(spec["pins"]["matched_input"]["path"], "INPUT.npz").is_file(), "input pin unresolved"
    population = os.environ.get("I6DMO_FAKE_POPULATION") or view["data_pins"]["common_population"]["population_sha256"]
    artifacts, reports, lineage = {}, [], {}
    costs = dict(wall_seconds=0.1, fit_seconds=0.0, score_seconds=0.1, cgroup_peak_bytes=None)
    extra = {}
    if args.kind == "fit" and os.environ.get("I6DMO_FAKE_GENUINE_RECEIPT") == "1":
        artifacts, extra, reports = _genuine_shaped_fit(spec, view, out, arm, population)
        costs.update(fit_seconds=0.5, cgroup_peak_bytes=int(os.environ.get("I6DMO_FAKE_PEAK", 1 << 30)))
    elif args.kind == "fit":
        (out / "predictions.json").write_text(json.dumps(dict(arm=arm, fake=True)))
        artifacts["predictions"] = dict(path="predictions.json", sha256=_sha(out / "predictions.json"))
        # a receipt shaped like the trainer's (orchestration only; I6DMO_FAKE_SCIENTIFIC/DEVICE vary it)
        (out / "train").mkdir()
        (out / "train" / "RECEIPT.json").write_text(json.dumps(dict(
            status="COMPLETED", scientific=os.environ.get("I6DMO_FAKE_SCIENTIFIC", "true") == "true",
            device=os.environ.get("I6DMO_FAKE_DEVICE", "gpu"), overrides=None, fake=True)))
        artifacts["RECEIPT"] = dict(path="train/RECEIPT.json", sha256=_sha(out / "train" / "RECEIPT.json"))
        costs.update(fit_seconds=0.5, cgroup_peak_bytes=int(os.environ.get("I6DMO_FAKE_PEAK", 1 << 30)))
    else:
        rel = float(os.environ.get(f"I6DMO_FAKE_REL_{arm.upper()}", DEFAULT_REL[arm]))
        drop = os.environ.get("I6DMO_FAKE_DROP_HEAD", "")
        rows = os.environ.get(f"I6DMO_FAKE_ROWS_{arm.upper()}")
        per_h = {}
        for h in HORIZONS:
            if drop == f"{arm}:{h}":
                continue
            naive = 0.001 * h
            per_h[str(h)] = dict(n=100, mae=naive * (1 + rel), naive_mae=naive, relative_error=rel,
                                 row_ids_sha256=rows or hashlib.sha256(f"common-h{h}".encode()).hexdigest())
        # The model lane's METRICS "matched" block, shaped exactly as decide_matched consumes it.
        capacity_doc = json.loads((REPO / "docs/i6n/direct_multi_output_static/matched_short/"
                                          "CAPACITY_MATCHED.json").read_text())
        model_family = {"mg1": "short_mg1", "mg2m": "short_mg2"}[arm]
        failed = [dict(horizon=h, n=None, n_failed=None, relative_error=None, reason="missing_or_failed")
                  for h in HORIZONS if str(h) not in per_h]
        selection = dict(status="INCOMPLETE" if failed else "COMPLETE", score=None if failed else rel,
                         heads=list(HORIZONS), failed_heads=failed)
        fit = json.loads(Path(spec["upstream"][f"{arm}.fit"]["path"]).read_text())
        fit_receipt = json.loads((Path(spec["upstream"][f"{arm}.fit"]["path"]).parent / "out" / "train" /
                                  "RECEIPT.json").read_text())
        assert fit["unit_id"] == f"{arm}.fit"
        matched = dict(arm=arm, common_population_sha256=population, scientific_fit=fit_receipt["scientific"],
                       fit_device=fit_receipt["device"],
                       development_row_ids_sha256=rows or hashlib.sha256(b"common-rows").hexdigest(),
                       development_head_mask_sha256=hashlib.sha256(b"common-mask").hexdigest(),
                       development_counts={str(h): 100 for h in HORIZONS},
                       capacity=dict(schema="i6dmo.capacity.v1", arm=arm,
                                     counts=capacity_doc["declared_capacity"][model_family]),
                       selection=selection)
        (out / "METRICS.json").write_text(json.dumps(dict(per_horizon=per_h, matched=matched,
                                                          splits_read=["train", "validation"])))
        artifacts["metrics"] = dict(path="METRICS.json", sha256=_sha(out / "METRICS.json"))
        lineage = dict(common_population_sha256=population)
        reports.append(dict(experiment_key=f"i6dmo:fake:{spec['unit_id']}", experiment_set_key="i6dmo:fake",
                            actor="predictor", lake="LOCAL_RETAINED", lineage="UNVERIFIED",
                            config_sha256=spec["identity_sha256"], code_commit="fake", project="predictor",
                            phase="I6DMO", tags={"unit": spec["unit_id"], "test_read": "false"},
                            metrics=[dict(metric="Price_MAE", value=r["mae"], split="validation", horizon=int(h),
                                          unit="USD_per_EUR") for h, r in per_h.items()]))
    terminal = dict(schema="i6dmo.terminal.v1", unit_id=spec["unit_id"], identity_sha256=spec["identity_sha256"],
                    status="COMPLETE", disposition="FAKE_ORCHESTRATION_ONLY", artifacts=artifacts, costs=costs,
                    eligibility={}, gates={}, reports=reports, lineage=lineage, **extra)
    (out / "TERMINAL.json").write_text(json.dumps(terminal, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
