"""Tests for the FS-REP follower: synthetic PS3-R terminals, frozen rule, statuses."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fs_rep_dispositions as fsr  # noqa: E402

FOLDS = fsr.FOLDS
BASE_REV = "8987c573bd7e0819a54011e88ec9e7c309229350"
ALT_REV = "31add1270439e746edd99e98e20a95ab09c1cf71"
SERIES = {"batch_001": "a" * 64, "batch_002": "b" * 64, "batch_003": "c" * 64}
RECIPE = {"window": 168, "latent_dim": 8, "max_fit_windows": 16384, "max_val_windows": 0, "max_ref_windows": 256,
          "probe_lags": "0,1,2,23", "ridge_alpha": 1.0, "max_epochs": 200, "patience": 10, "seed": 0}


def _fold_family(feature, fold, fam, trained):
    rec = {"status": "NOT_APPLICABLE", "reason": "no decoder"}
    if fam in ("ae", "dae"):
        rec = {"status": "MEASURED", **{m: 0.01 for m in fsr.RECON_METRICS}}
    return {
        "kind": "fold_family", "feature_id": feature, "fold_id": fold, "family": fam, "seed": 0,
        "architecture_id": f"arch_{fam}", "latent_dim": 3 if fam == "identity" else 8,
        "train_row_ids_sha256": hashlib.sha256(fold.encode()).hexdigest(),
        "reconstruction": rec,
        "effective_dimension": {"dim": 8, "n_components_95": 3, "participation_ratio": 2.0},
        "cost": {"fit_wall_seconds": 10.0 if trained else 0.0, "updates": 100 if trained else 0,
                 "epochs_run": 5 if trained else 0, "params": 1000 if fam != "identity" else 0,
                 "encode_latency_ms_per_window": 0.3},
        "fit_report": {"stop_reason": "patience" if trained else "NOT_TRAINED"},
    }


def make_terminal(root: Path, result_dir: str, feature: str, role: str, batch: str, *,
                  losses: dict, rev=None, series=None, recipe_override=None, status="COMPLETED",
                  break_hash=False, drop_rows=0):
    """losses: family -> loss multiplier applied to a naive of 1.0 (identity 0.9 means skill)."""
    families = fsr.ROLE_FAMILIES[role]
    trained = [f for f in families if f not in ("identity", "random")]
    rows = []
    for fold in FOLDS:
        fold_jitter = 1e-6 * (FOLDS.index(fold) - 2)  # keeps differences tiny but signed
        for fam in families:
            rows.append(_fold_family(feature, fold, fam, fam in trained))
        for (target, h) in fsr.CELLS:
            spec = fsr.PROBE_CONTRACT[target]
            for fam in families:
                loss = losses[fam] + fold_jitter * (1 if fam != "identity" else 0)
                row = {"kind": "probe", "feature_id": feature, "fold_id": fold, "representation": fam,
                       "target": target, "horizon_index": h, "status": "MEASURED",
                       spec["loss"]: loss, spec["metrics"][1]: loss ** 2, "n_val": 100, "naive_n_val": 100}
                if target == "Y_b":
                    row["prior_log_loss"] = 1.0
                else:
                    row["naive_zero_mae"] = 1.0
                    row["naive_train_mean_mae"] = 1.0
                rows.append(row)
            for fam in trained:
                lt = losses[fam] + fold_jitter
                rows.append({"kind": "probe_delta", "feature_id": feature, "fold_id": fold, "trained": fam,
                             "target": target, "horizon_index": h, "loss": spec["loss"], "loss_trained": lt,
                             "delta_probe_random_minus_trained": losses["random"] - lt,
                             "preservation_raw_minus_trained": losses["identity"] - lt})
    rows.append({"kind": "feature_summary", "feature_id": feature, "reference_fold": FOLDS[-1],
                 "stability": {f: {"status": "MEASURED", "mean": 0.95, "min": 0.9, "pairs": 10} for f in families},
                 "probe_loss_across_folds": []})
    if drop_rows:
        rows = rows[:-drop_rows]
    directory = root / result_dir
    directory.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    (directory / "results.jsonl").write_bytes(body)
    args = dict(RECIPE, **(recipe_override or {}))
    manifest = {
        "schema": "ut_pilot_run.v1", "status": status, "features": [feature], "families": list(families),
        "folds": list(FOLDS), "seed": 0, "code_commit": rev or (BASE_REV if role == "baseline" else ALT_REV),
        "series_sha256": series or SERIES[batch], "args": args, "wall_seconds": 100.0,
        "peak_rss_bytes": 1, "cgroup_peak_bytes": 1,
        "results_sha256": ("0" * 64) if break_hash else hashlib.sha256(body).hexdigest(),
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def world(tmp_path: Path):
    repo = tmp_path / "repo"
    rule_dir = repo / "docs/audits/evidence/canonical_20261003/fs_closure/fs_rep"
    rule_dir.mkdir(parents=True)
    (rule_dir / "DECISION_RULE.md").write_text(f"# rule\n\n`RULE_VERSION = {fsr.RULE_VERSION}`\n")
    auto = repo / "docs/audits/evidence/canonical_20261003/automation"
    auto.mkdir(parents=True)
    heavy = ["f.win", "f.raw", "f.tie", "f.pend", "f.fail", "f.contra", "f.notrain"]
    cov = {"features": [{"feature_id": f, "batch": "batch_002", "in_extractibility_queue": True} for f in heavy]
           + [{"feature_id": "light.x", "batch": "batch_002", "in_extractibility_queue": False}]}
    (repo / "docs/audits/evidence/canonical_20261003/coverage_reconciliation").mkdir(parents=True)
    (repo / "docs/audits/evidence/canonical_20261003/coverage_reconciliation/coverage_reconciliation.json").write_text(json.dumps(cov))
    alt_rows = ["cell_id\tresult_dir"]
    base_rows = ["cell_id\tresult_dir"]
    for f in heavy:
        alt_rows.append(f"batch_002::{f}::masked_temporal_ae\tbatch_002/{f}")
        alt_rows.append(f"batch_002::{f}::past_to_current_siamese\tbatch_002/{f}/pass_p2c")
        base_rows.append(f"batch_002::{f}::baseline\tbatch_002/{f}")
    (auto / "alternative.tsv").write_text("\n".join(alt_rows) + "\n")
    (auto / "baseline_002.tsv").write_text("\n".join(base_rows) + "\n")
    (auto / "baseline_001_003.tsv").write_text("cell_id\tresult_dir\n")
    mirror = tmp_path / "mirror"
    alt, base, succ = mirror / "alt", mirror / "base", mirror / "succ"
    for d in (alt, base, succ):
        d.mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    (state / "local_paths.json").write_text(json.dumps({
        "mirror_roots": {"mirror_alternatives": str(alt), "mirror_baseline_002": str(base), "mirror_baseline_successor": str(succ)},
        "status_files": {"alternative": str(state / "alt_status.json")},
    }))
    (state / "alt_status.json").write_text(json.dumps({"counts": {"completed": 1, "pending": 2, "running": 0, "total": 3}, "durations_seconds": {"median": 100.0, "p90": 120.0, "sample_size": 1}, "workers": 1}))
    cfg = {
        "schema": fsr.SCHEMA_CONFIG, "rule_version": fsr.RULE_VERSION, "repo_root": str(repo),
        "decision_rule_doc": "docs/audits/evidence/canonical_20261003/fs_closure/fs_rep/DECISION_RULE.md",
        "output_dir": "docs/audits/evidence/canonical_20261003/fs_closure/fs_rep", "state_dir": str(state),
        "local_paths": str(state / "local_paths.json"),
        "heavy_candidates": "docs/audits/evidence/canonical_20261003/coverage_reconciliation/coverage_reconciliation.json",
        "heavy_denominator": len(heavy), "seed": 0,
        "plans": {
            "alternative": {"tsv": "docs/audits/evidence/canonical_20261003/automation/alternative.tsv", "roots": ["mirror_alternatives"]},
            "baseline_002": {"tsv": "docs/audits/evidence/canonical_20261003/automation/baseline_002.tsv", "roots": ["mirror_baseline_002", "mirror_baseline_successor"]},
            "baseline_001_003": {"tsv": "docs/audits/evidence/canonical_20261003/automation/baseline_001_003.tsv", "roots": ["mirror_baseline_successor"]},
        },
        "allowed_revisions": {"baseline": [BASE_REV], "alt_mtae": [ALT_REV], "alt_p2c": [ALT_REV]},
        "batch_series_sha256": SERIES, "recipe": RECIPE, "refit_input": None,
        "eta_inputs": {"fs_gpu_eta_candidates": ["docs/audits/evidence/canonical_20261003/fs_closure/fs_gpu/ps3r_eta.json"], "status_files": ["alternative"]},
    }
    cfg_path = rule_dir / "fs_rep_config.json"
    cfg_path.write_text(json.dumps(cfg))
    # f.win: dae learned and preserves (random 1.0, identity 0.9, dae 0.8, ae 0.95 loses to raw)
    make_terminal(base, "batch_002/f.win", "f.win", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.8})
    make_terminal(alt, "batch_002/f.win", "f.win", "alt_mtae", "batch_002", losses={"identity": 0.9, "random": 1.0, "masked_temporal_ae": 0.85})
    make_terminal(alt, "batch_002/f.win/pass_p2c", "f.win", "alt_p2c", "batch_002", losses={"identity": 0.9, "random": 1.0, "past_to_current_siamese": 0.95})
    # f.raw: every trained family loses to raw on every cell
    for role, d, fam in (("baseline", "batch_002/f.raw", None), ("alt_mtae", "batch_002/f.raw", "masked_temporal_ae"), ("alt_p2c", "batch_002/f.raw/pass_p2c", "past_to_current_siamese")):
        losses = {"identity": 0.9, "random": 1.0}
        if role == "baseline":
            losses.update(ae=0.97, dae=0.98)
        else:
            losses[fam] = 0.96
        make_terminal(base if role == "baseline" else alt, d, "f.raw", role, "batch_002", losses=losses)
    # f.tie: trained equal to raw up to the +-1e-6 fold jitter -> indistinguishable
    for role, d, fam in (("baseline", "batch_002/f.tie", None), ("alt_mtae", "batch_002/f.tie", "masked_temporal_ae"), ("alt_p2c", "batch_002/f.tie/pass_p2c", "past_to_current_siamese")):
        losses = {"identity": 0.9, "random": 1.0}
        if role == "baseline":
            losses.update(ae=0.9, dae=0.9)
        else:
            losses[fam] = 0.9
        make_terminal(base if role == "baseline" else alt, d, "f.tie", role, "batch_002", losses=losses)
    # f.pend: only the alternatives landed
    make_terminal(alt, "batch_002/f.pend", "f.pend", "alt_mtae", "batch_002", losses={"identity": 0.9, "random": 1.0, "masked_temporal_ae": 0.85})
    make_terminal(alt, "batch_002/f.pend/pass_p2c", "f.pend", "alt_p2c", "batch_002", losses={"identity": 0.9, "random": 1.0, "past_to_current_siamese": 0.95})
    # f.fail: baseline ok, mtae FAILED receipt, p2c recipe mismatch (cap measurement run)
    make_terminal(base, "batch_002/f.fail", "f.fail", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97})
    (alt / "batch_002/f.fail").mkdir(parents=True)
    (alt / "batch_002/f.fail/FAILED.codex.json").write_text(json.dumps({"status": "FAILED", "rc": 1}))
    make_terminal(alt, "batch_002/f.fail/pass_p2c", "f.fail", "alt_p2c", "batch_002", losses={"identity": 0.9, "random": 1.0, "past_to_current_siamese": 0.95}, recipe_override={"max_epochs": 1})
    # f.contra: two COMPLETED baselines with different bytes in two roots
    make_terminal(base, "batch_002/f.contra", "f.contra", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97})
    make_terminal(succ, "batch_002/f.contra", "f.contra", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.94, "dae": 0.97})
    make_terminal(alt, "batch_002/f.contra", "f.contra", "alt_mtae", "batch_002", losses={"identity": 0.9, "random": 1.0, "masked_temporal_ae": 0.85})
    make_terminal(alt, "batch_002/f.contra/pass_p2c", "f.contra", "alt_p2c", "batch_002", losses={"identity": 0.9, "random": 1.0, "past_to_current_siamese": 0.95})
    # f.notrain: every terminal FAILED because the series has no observed TRAIN values
    for d in (base / "batch_002/f.notrain", alt / "batch_002/f.notrain", alt / "batch_002/f.notrain/pass_p2c"):
        d.mkdir(parents=True)
        (d / "FAILED.codex.json").write_text(json.dumps({"status": "FAILED", "rc": 1}))
        (d / "FAILED.reason.txt").write_text("ContractError: no observed TRAIN values to fit normalization\n")
        (d / "results.jsonl").write_bytes(b"")
    return {"cfg_path": cfg_path, "repo": repo, "state": state, "alt": alt, "base": base, "succ": succ}


def _run(world):
    cfg = fsr.load_config(world["cfg_path"])
    return cfg, fsr.run_cycle(cfg, git=False)


def test_decisions_follow_frozen_rule(world):
    cfg, progress = _run(world)
    c = progress["candidates"]
    assert c["f.win"]["decision"] == "dae"
    assert c["f.raw"]["decision"] == "RAW"
    assert c["f.tie"]["decision"] == "NO_TRAINED_ADVANTAGE"
    assert c["f.pend"]["decision"] == "PENDING"
    assert c["f.fail"]["decision"] == "RAW" and "DECIDED_WITH_FAILED_FAMILIES" in c["f.fail"]["flags"]
    assert c["f.fail"]["terminals"]["alt_mtae"]["reason"] == "FAILED_RECEIPT"
    assert c["f.fail"]["terminals"]["alt_p2c"]["reason"] == "REJECTED_RECIPE_max_epochs"
    assert c["f.contra"]["terminals"]["baseline"]["reason"] == "CONTRADICTORY_TERMINAL"
    # the contradictory baseline is terminal (FAILED); the alternatives still contest and MTAE wins
    assert c["f.contra"]["decision"] == "masked_temporal_ae" and "DECIDED_WITH_FAILED_FAMILIES" in c["f.contra"]["flags"]
    assert progress["heavy_candidates_covered"] == 3
    assert progress["decisions"] == {"dae": 1, "RAW": 2, "NO_TRAINED_ADVANTAGE": 1, "PENDING": 1, "masked_temporal_ae": 1, "NOT_AVAILABLE_FOR_TRAIN": 1}
    assert "PROBE_ROWS_DIFFER" not in c["f.win"]["flags"]
    assert progress["safety"]["tensorflow_imported"] is False


def test_tiny_differences_are_not_rounded(world):
    cfg, progress = _run(world)
    out = cfg["_output_dir"] / "representation_dispositions.csv"
    rows = {(r["feature_id"], r["family"]): r for r in csv.DictReader(out.open())}
    tie = rows[("f.tie", "ae")]
    # jitter is +-1e-6 around zero: two folds positive, two negative, one zero -> indistinguishable
    assert tie["beats_raw_cells"] == "0" and tie["lost_to_raw_cells"] == "0" and tie["indistinguishable_cells"] == "14"
    assert tie["preservation_status"] == "MEASURED"
    assert abs(float(tie["preservation_mean_Ys"])) < 1e-9 and tie["preservation_mean_Ys"] != "0"


def test_every_metric_cell_has_a_status(world):
    cfg, progress = _run(world)
    out = cfg["_output_dir"] / "representation_dispositions.csv"
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 7 * 6
    allowed = {"MEASURED", "FAILED", "NOT_APPLICABLE", "PENDING"}
    for r in rows:
        for col in ("rec_status", "stability_status", "effdim_status", "probe_status", "delta_status", "preservation_status", "cost_status"):
            assert r[col] in allowed, (r["feature_id"], r["family"], col, r[col])
        if r["terminal_state"] == "COMPLETED":
            assert len(r["receipt_sha256"]) == 64
        if r["family"] in ("identity", "random"):
            assert r["delta_status"] == "NOT_APPLICABLE"
        if r["family"] in ("ae", "dae") and r["terminal_state"] == "COMPLETED":
            assert r["rec_status"] == "MEASURED"
        if r["family"] in ("masked_temporal_ae", "past_to_current_siamese") and r["terminal_state"] == "COMPLETED":
            assert r["rec_status"] == "NOT_APPLICABLE"
        assert r["refit_gain"] == "NOT_AVAILABLE" and r["refit_gate_applied"] == "false"
    pend = [r for r in rows if r["feature_id"] == "f.pend" and r["family"] == "ae"][0]
    assert pend["terminal_state"] == "PENDING" and pend["probe_status"] == "PENDING" and pend["G1_learned"] == "PENDING"
    fail = [r for r in rows if r["feature_id"] == "f.fail" and r["family"] == "masked_temporal_ae"][0]
    assert fail["terminal_state"] == "FAILED" and fail["probe_status"] == "FAILED" and len(fail["receipt_sha256"]) == 64


def test_catalog_long_format_and_digests(world):
    cfg, progress = _run(world)
    gz = cfg["_output_dir"] / "representation_metric_catalog.csv.gz"
    rows = list(csv.DictReader(gzip.open(gz, "rt")))
    assert set(rows[0]) == set(fsr.CATALOG_COLUMNS)
    win = [r for r in rows if r["feature_id"] == "f.win" and r["family"] == "dae"]
    metrics = {r["metric"] for r in win}
    assert {"probe.mae", "probe.log_loss", "probe.naive_zero_mae", "probe.prior_log_loss", "probe.skill_strict_vs_all_naives",
            "probe_delta.random_minus_trained", "probe_delta.preservation_raw_minus_trained", "probe_delta.verdict_beats_raw",
            "reconstruction.mae_rel_train_constant", "stability.linear_cka_mean", "effective_dimension.participation_ratio", "cost.fit_wall_seconds"} <= metrics
    assert all(len(r["receipt_sha256"]) == 64 for r in win)
    assert progress["outputs"]["representation_metric_catalog.csv.gz"]["sha256"] == hashlib.sha256(gz.read_bytes()).hexdigest()


def test_refit_gate_applies_when_table_present(world):
    refit = world["state"] / "refit.csv"
    refit.write_text("feature_id,family,refit_gain\nf.win,dae,-0.01\n")
    cfg = json.loads(world["cfg_path"].read_text())
    cfg["refit_input"] = str(refit)
    world["cfg_path"].write_text(json.dumps(cfg))
    cfg, progress = _run(world)
    assert progress["refit_gate_applied"] is True
    # dae fails G4; ae loses to raw; mtae (0.85) passes -> winner becomes masked_temporal_ae
    assert progress["candidates"]["f.win"]["decision"] == "masked_temporal_ae"


def test_identity_only_refit_export_is_feature_level_evidence(world):
    """FS-CLOSE export: family=identity only, rows fill over time, trained refits not materialised."""
    refit = world["state"] / "refit_gain_export.csv"
    refit.write_text("feature_id,family,refit_gain,cells,positive_cells,head,rows,trained_families\n"
                     "f.win,identity,0.002,14,9,ridge,100,NOT_MATERIALISED\n"
                     "f.raw,identity,,0,0,ridge,0,NOT_MATERIALISED\n")
    cfg = json.loads(world["cfg_path"].read_text())
    cfg["refit_input"] = str(refit)
    world["cfg_path"].write_text(json.dumps(cfg))
    cfg, progress = _run(world)
    c = progress["candidates"]
    assert c["f.win"]["decision"] == "dae"  # trained G4 not applied: no trained refit rows
    assert c["f.notrain"]["decision"] == "NOT_AVAILABLE_FOR_TRAIN"
    assert "RAW_REFIT_GAIN_POSITIVE" in c["f.win"]["flags"]
    assert not any(f.startswith("RAW_REFIT") for f in c["f.raw"]["flags"])  # empty gain skipped
    rows = {(r["feature_id"], r["family"]): r for r in csv.DictReader((cfg["_output_dir"] / "representation_dispositions.csv").open())}
    assert rows[("f.win", "identity")]["refit_gate_applied"] == "true" and rows[("f.win", "identity")]["refit_gain"] == "0.002"
    assert rows[("f.win", "dae")]["refit_gate_applied"] == "false" and rows[("f.win", "dae")]["refit_gain"] == "NOT_AVAILABLE"
    assert rows[("f.win", "dae")]["G4_refit"] == "NOT_APPLICABLE"
    assert rows[("f.raw", "identity")]["refit_gate_applied"] == "false"


def test_rejections_for_hash_revision_and_digest(world, tmp_path):
    cfg = fsr.load_config(world["cfg_path"])
    cache = tmp_path / "cache"
    cache.mkdir()
    cell = {"cell_id": "batch_002::g::baseline", "batch": "batch_002", "feature_id": "g", "role": "baseline", "result_dir": "batch_002/g"}
    root = tmp_path / "r1"
    make_terminal(root, "batch_002/g", "g", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97}, break_hash=True)
    assert fsr.examine_cell(cell, [root], cfg, cache)["reason"] == "REJECTED_RESULTS_HASH"
    root = tmp_path / "r2"
    make_terminal(root, "batch_002/g", "g", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97}, rev="f" * 40)
    assert fsr.examine_cell(cell, [root], cfg, cache)["reason"] == "REJECTED_REVISION"
    root = tmp_path / "r3"
    make_terminal(root, "batch_002/g", "g", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97}, series="d" * 64)
    assert fsr.examine_cell(cell, [root], cfg, cache)["reason"] == "REJECTED_INPUT_DIGEST"
    root = tmp_path / "r4"
    make_terminal(root, "batch_002/g", "g", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97}, drop_rows=3)
    assert fsr.examine_cell(cell, [root], cfg, cache)["reason"] == "REJECTED_ROW_KINDS"
    root = tmp_path / "r5"
    make_terminal(root, "batch_002/g", "g", "baseline", "batch_002", losses={"identity": 0.9, "random": 1.0, "ae": 0.95, "dae": 0.97}, status="RUNNING")
    assert fsr.examine_cell(cell, [root], cfg, cache)["state"] == "PENDING"


def test_rule_version_must_match_frozen_doc(world):
    rule = world["repo"] / "docs/audits/evidence/canonical_20261003/fs_closure/fs_rep/DECISION_RULE.md"
    rule.write_text("# rule\n\n`RULE_VERSION = fs_rep_rule.v0`\n")
    with pytest.raises(fsr.ConfigError):
        fsr.load_config(world["cfg_path"])


def test_eta_from_status_files(world):
    cfg, progress = _run(world)
    eta = progress["eta"]
    assert eta["queues"]["alternative"]["remaining"] == 2
    assert eta["queues"]["alternative"]["eta_median_utc"] > progress["cycle_started_at"]
    assert eta["full_coverage_eta_median_utc"] == eta["queues"]["alternative"]["eta_median_utc"]


def test_no_train_support_never_defaults_to_raw(world):
    """Musashi 2026-10-05: RAW and every trained family lack TRAIN observations -> NOT_AVAILABLE_FOR_TRAIN."""
    cfg, progress = _run(world)
    c = progress["candidates"]["f.notrain"]
    assert c["decision"] == "NOT_AVAILABLE_FOR_TRAIN" and c["cause"] == "NO_OBSERVED_TRAIN_VALUES"
    assert c["flags"] == ["TRAIN_SUPPORT_ABSENT"]
    assert not any(f in c["flags"] for f in ("DECIDED_WITH_FAILED_FAMILIES", "ALL_TRAINED_FAMILIES_FAILED", "RAW_NO_PROBE_SKILL"))
    rows = [r for r in csv.DictReader((cfg["_output_dir"] / "representation_dispositions.csv").open()) if r["feature_id"] == "f.notrain"]
    assert len(rows) == 6
    for r in rows:
        assert r["provisional_extractibility_disposition"] == "NOT_AVAILABLE_FOR_TRAIN"
        assert r["disposition_class"] == "PROVISIONAL_EXTRACTIBILITY_DISPOSITION"
        assert r["terminal_state"] == "FAILED" and len(r["receipt_sha256"]) == 64
        for col in ("rec_status", "probe_status", "delta_status", "preservation_status", "cost_status"):
            assert r[col] == "NOT_APPLICABLE"
    dec = {r["feature_id"]: r for r in csv.DictReader((cfg["_output_dir"] / "candidate_decisions.csv").open())}
    assert dec["f.notrain"]["PLUS_EXTRACTIBILITY_EVIDENCE"] == "NOT_AVAILABLE_FOR_TRAIN"
    # the valid case stays separate: raw has support, only trained families failed -> RAW with the flag
    assert progress["candidates"]["f.fail"]["decision"] == "RAW"
    assert "DECIDED_WITH_FAILED_FAMILIES" in progress["candidates"]["f.fail"]["flags"]
    assert dec["f.fail"]["PLUS_EXTRACTIBILITY_EVIDENCE"] == "RAW"


def test_all_failed_without_declared_cause_is_still_not_raw(world, tmp_path):
    cfg = fsr.load_config(world["cfg_path"])
    terminals = {role: {"state": "FAILED", "reason": "FAILED_RECEIPT", "receipt_sha256": "0" * 64} for role in fsr.ROLE_FAMILIES}
    d = fsr.decide("x", terminals, None)
    assert d["decision"] == "NOT_AVAILABLE_FOR_TRAIN" and d["cause"] == "ALL_TERMINALS_FAILED_CAUSE_UNDECLARED"
    d = fsr.decide("x", terminals, None, {"x": "NO_OBSERVED_TRAIN_VALUES"})
    assert d["cause"] == "NO_OBSERVED_TRAIN_VALUES"
