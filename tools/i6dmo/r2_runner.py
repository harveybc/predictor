"""R0-versus-R2 branch-pretraining contrast: the OPS unit bodies (``unit_runner`` hooks).

The finite graph lives in ``campaign`` (mode ``R2_CONTRAST``) and reuses the queue, governed worker,
stored-terminal verifier, outbox/readback, mirrors, automatic CLOSE and TIMERS_STOPPED:

    r2.pretrain.00 .. r2.pretrain.18  (one per feature, GPU, own admission class)
        -> r2.fit (imports the 19 authenticated donors, then the R0 recipe) -> r2.report
    r0.fit (fresh, no import binding) -> r0.report
    r2.probe (frozen predictive probe, REPORT ONLY: nothing depends on it)
    r0.report + r2.report + r2.fit -> r2.comparison (r2_design.decide_r2) -> publish -> CLOSE

Every body calls the model lane's REAL functions: ``branch_pretraining_corpus``,
``pretrain_branch`` (seed bound before each build), ``write_branch_donor_terminal``,
``branch_donor_context`` / ``import_branch_donors`` (inside the R2 fit, ``train.r2_import_into``)
and ``r2_design.decide_r2``. Pretraining reads FIT and INNER only: DEVELOPMENT and TEST are never
requested from the population accessor (``FEATURE_SPLITS``).
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

#: The only population splits pretraining and the probe may open.
FEATURE_SPLITS = ("FIT", "INNER")
R2_VERDICTS = {("R2_LOWER_SCORE", "R2"): "R2_SELECTED", ("R0_RETAINED", "R0"): "R0_RETAINED",
               ("INCOMPLETE_COMPARISON", None): "INCOMPLETE_COMPARISON"}
DISPOSITIONS = ("R2_SELECTED", "R0_RETAINED", "INCOMPLETE_COMPARISON")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False, default=float))
    os.replace(tmp, path)


def _progress(path, done, total, phase):
    _atomic(path, dict(done=done, total=total, unit="step", phase=phase, updated_at=time.time()))


# --------------------------------------------------------------------- shared helpers
def scaler_sha256(population):
    from tools.i6dmo import model as M
    s = population.scaler
    return M.scaler_identity_sha256(s.feature_names, s.mean, s.std, s.fit_row_digest, s.n_rows)


def split_tensors(population, design, split):
    """The arm tensors of ONE split; anything but FIT/INNER is refused (DEVELOPMENT/TEST never opened)."""
    from tools.i6dmo import train as T
    if split not in FEATURE_SPLITS:
        raise ValueError(f"R2_SPLIT_FORBIDDEN:{split}")
    return T.arm_split(population, design, split)


def feature_corpus(population, effective, feature_index, tensors=None):
    """``model.branch_pretraining_corpus`` for one feature from the authenticated population."""
    from tools.i6dmo import model as M
    from tools.i6dmo import train as T
    design = effective["design"]
    fit, inner = tensors or (split_tensors(population, design, "FIT"), split_tensors(population, design, "INNER"))
    return M.branch_pretraining_corpus(
        fit.x, fit.row_id, inner.x, inner.row_id, inner.slot_instants, feature_index=feature_index,
        scaler_mean=population.scaler.mean, scaler_std=population.scaler.std,
        inner_start_epoch=T._epoch(effective["splits"]["inner"]["start"]))


def corpora(population, effective):
    """{feature: learned-corpus sha256} of all branches (FIT/INNER tensors read once)."""
    design = effective["design"]
    tensors = (split_tensors(population, design, "FIT"), split_tensors(population, design, "INNER"))
    return {f: feature_corpus(population, effective, i, tensors)[2]["sha256"]
            for i, f in enumerate(effective["feature_manifest"]["features"])}


def _context(spec):
    """Design doc, its model-lane effective config and the authenticated population for a unit."""
    from tools.i6dmo import train as T
    view = spec["effective_config"]
    folder = Path(spec.get("_folder") or ".")
    design_path = folder / "design.json"
    _atomic(design_path, view["design"])
    effective = T.effective_from_design(str(design_path))
    pop_pin, inp = spec["pins"]["common_population"], spec["pins"]["matched_input"]
    population = T.load_matched_population(pop_pin["path"], pop_pin["population_sha256"],
                                           effective["horizons_hours"], effective["feature_manifest"]["features"],
                                           input_dir=inp["path"])
    return view, effective, population


def _report(spec, suffix, metrics, tags=None):
    view = spec["effective_config"]
    base = {"unit": spec["unit_id"], "family": spec.get("family"), "kind": spec["kind"], "test_read": "false",
            "contrast": "R0_VS_R2", "regime": str(spec.get("regime")),
            "engineering": str(bool(view.get("engineering_cpu_fixture"))).lower()}
    return dict(experiment_key=f"i6dmo:{spec['campaign_id']}:{spec['unit_id']}:{suffix}"[:128],
                experiment_set_key=f"i6dmo:{spec['campaign_id']}", actor="predictor", lake="LOCAL_RETAINED",
                lineage="UNVERIFIED", config_sha256=spec["identity_sha256"], code_commit=view.get("code_commit"),
                project="predictor", phase="I6DMO", tags={**base, **(tags or {})},
                metrics=[m for m in metrics if m["value"] is not None and math.isfinite(float(m["value"]))])


def _metric(name, value, unit, split="train", horizon=None):
    return dict(metric=name, value=None if value is None else float(value), split=split, horizon=horizon, unit=unit)


def _terminal(spec, status, disposition, *, artifacts=None, reports=None, costs=None, extra=None, gates=None):
    return dict(schema="i6dmo.terminal.v1", unit_id=spec.get("unit_id"), identity_sha256=spec.get("identity_sha256"),
                status=status, disposition=disposition, artifacts=artifacts or {}, eligibility={}, gates=gates or {},
                reports=reports or [], costs=costs or {}, **(extra or {}))


# --------------------------------------------------------------------- pretraining unit
def run_pretrain(spec, out_dir, progress_path):
    """ONE branch-pretraining unit: the ruled objective on this feature's FIT/purged-INNER windows."""
    import tensorflow as tf
    from tools.i6dmo import model as M
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(spec, _folder=str(out.parent))
    if spec.get("test_open"):
        return _terminal(spec, "FAILED", "TEST_FORBIDDEN")
    device = "gpu" if (spec.get("device") or {}).get("gpu_uuid") else "cpu"
    receipt_env = M.enable_determinism()
    if device == "gpu" and not tf.config.list_physical_devices("GPU"):
        return _terminal(spec, "FAILED", "NO_GPU_VISIBLE", costs=dict(fit_seconds=None))
    try:
        view, effective, population = _context(spec)
    except Exception as exc:  # pre-fit refusal: typed with the data lane's code
        return _terminal(spec, "FAILED", getattr(exc, "code", type(exc).__name__), costs=dict(fit_seconds=None))
    r2 = spec["r2"]  # ruled objective + this campaign's provenance (worker.r2_unit_spec)
    feature, index = spec["feature"], int(spec["feature_index"])
    _progress(progress_path, 1, 4, "corpus")
    fit_w, sel_w, corpus = feature_corpus(population, effective, index)
    placement = "/GPU:0" if device == "gpu" else "/CPU:0"
    with tf.device(placement):
        bundle = M.build_dmo(effective["model"])
        if list(bundle.config["features"]).index(feature) != index:
            return _terminal(spec, "FAILED", "R2_FEATURE_ORDER")
        provenance = dict(r2["provenance"], corpus={feature: corpus["sha256"]})
        context = M.branch_donor_context(bundle, population_sha256=population.digest,
                                         scaler_sha256=scaler_sha256(population), objective=r2["objective"],
                                         provenance=provenance)
        _progress(progress_path, 2, 4, "pretrain")
        t0 = time.monotonic()
        try:
            pretrained = M.pretrain_branch(bundle, feature, fit_w, sel_w, r2["objective"])
        except M.DonorImportError as exc:  # a budget stop never yields a donor (typed terminal failure)
            return _terminal(spec, "FAILED", exc.code,
                             costs=dict(fit_seconds=time.monotonic() - t0 or 1e-9, wall_seconds=time.time() - started))
        fit_seconds = time.monotonic() - t0
    ref = M.write_branch_donor_terminal(out / "donor", bundle=bundle, feature=feature, context=context,
                                        unit_id=spec["unit_id"], corpus=corpus, pretrained=pretrained)
    _atomic(out / "CORPUS.json", corpus)
    artifacts = {"donor_terminal": dict(path="donor/TERMINAL.json", sha256=ref["sha256"]),
                 "donor.npz": dict(path="donor/donor.npz", sha256=_sha(out / "donor" / "donor.npz")),
                 "corpus": dict(path="CORPUS.json", sha256=_sha(out / "CORPUS.json"))}
    fit = pretrained["fit"]
    history = fit.get("history") or []
    loss_metrics = []
    for epoch, row in enumerate(history, start=1):
        if isinstance(row, dict):
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    loss_metrics.append(_metric(f"Pretrain_{key}_epoch_{epoch:03d}", value, "loss", "inner"))
    loss_metrics.append(_metric("Pretrain_Best_Validation_Loss", fit.get("best_validation_loss"), "loss", "inner"))
    loss_metrics.append(_metric("Pretrain_Selected_Epoch", fit.get("selected_epoch"), "count"))
    recon = pretrained.get("reconstruction") or {}
    recon_metrics = []
    for split in ("selection", "fit", "inner"):
        block = recon.get(split) if isinstance(recon.get(split), dict) else None
        for key, value in (block or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                recon_metrics.append(_metric(f"Reconstruction_{split}_{key}", value, "z", "inner"))
    for key, value in recon.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            recon_metrics.append(_metric(f"Reconstruction_{key}", value, "z", "inner"))
    costs = dict(fit_seconds=fit_seconds, wall_seconds=time.time() - started, score_seconds=0.0,
                 cgroup_peak_bytes=None, updates=fit.get("observed_updates"))
    reports = [
        _report(spec, "pretrain_loss", loss_metrics, {"fact": "r2.pretraining.loss_curve", "feature": feature}),
        _report(spec, "reconstruction", recon_metrics or [_metric("Reconstruction_Reported", 0, "bool", "inner")],
                {"fact": "r2.pretraining.reconstruction", "feature": feature, "use": "REPORT_ONLY_NEVER_SELECTS",
                 "state": str(recon.get("state"))}),
        _report(spec, "compute", [_metric("Pretrain_Fit_Seconds", fit_seconds, "seconds"),
                                  _metric("Pretrain_Updates", fit.get("observed_updates"), "count")],
                {"fact": "r2.pretraining.compute_cost", "feature": feature, "device": device})]
    _progress(progress_path, 4, 4, "done")
    return _terminal(spec, "COMPLETE", "DONOR_WRITTEN", artifacts=artifacts, reports=reports, costs=costs,
                     extra=dict(feature=feature, feature_index=index, applied_seed=pretrained["applied_seed"],
                                learned_corpus_sha256=corpus["sha256"], determinism=receipt_env,
                                donor_terminal_sha256=ref["sha256"]))


# --------------------------------------------------------------------- frozen predictive probe
def _branch_features(bundle, x_scaled):
    """Frozen branch outputs, flattened (n, 19*13*4)."""
    from tools.i6dmo import model as M
    parts = []
    for j, feature in enumerate(bundle.config["features"]):
        y = np.asarray(bundle.branch_models[feature](np.asarray(x_scaled[:, :, j:j + 1], np.float32),
                                                     training=False), np.float64)
        parts.append(y.reshape(len(y), -1))
    return np.concatenate(parts, axis=1)


def _ridge_probe(phi_fit, y_fit, m_fit, phi_in, y_in, m_in, alpha=1.0):
    """Closed-form ridge per head on FIT rows (common masks); INNER MAE per head."""
    out = []
    for j in range(y_fit.shape[1]):
        a, b = m_fit[:, j], m_in[:, j]
        if not a.any() or not b.any():
            out.append(None)
            continue
        X = np.concatenate([phi_fit[a], np.ones((a.sum(), 1))], axis=1)
        w = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y_fit[a, j])
        pred = np.concatenate([phi_in[b], np.ones((b.sum(), 1))], axis=1) @ w
        out.append(float(np.mean(np.abs(pred - y_in[b, j]))))
    return out


def run_probe(spec, out_dir, progress_path):
    """The frozen predictive probe (R2-07): REPORT ONLY. Nothing in the graph depends on it; it never
    gates and never selects (the comparison reads only the two arm reports and the import record)."""
    import tensorflow as tf
    from tools.i6dmo import model as M
    from tools.i6dmo import train as T
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(spec, _folder=str(out.parent))
    if spec.get("test_open"):
        return _terminal(spec, "FAILED", "TEST_FORBIDDEN")
    M.enable_determinism()
    try:
        view, effective, population = _context(spec)
    except Exception as exc:
        return _terminal(spec, "FAILED", getattr(exc, "code", type(exc).__name__), costs=dict(fit_seconds=None))
    design = effective["design"]
    fit, inner = split_tensors(population, design, "FIT"), split_tensors(population, design, "INNER")
    scale = np.asarray([float(population.naive_loss_scale[h]) if population.naive_loss_scale.get(h) else np.nan
                        for h in effective["horizons_hours"]])

    def target(t):
        return (np.asarray(t.target_price, np.float64) - np.asarray(t.origin_price, np.float64)[:, None]) / scale

    def scaled(t):
        return (np.asarray(t.x, np.float64) - population.scaler.mean) / population.scaler.std
    y_fit, y_in = np.nan_to_num(target(fit)), np.nan_to_num(target(inner))
    m_fit = np.asarray(fit.head_mask, bool) & np.isfinite(scale)[None, :]
    m_in = np.asarray(inner.head_mask, bool) & np.isfinite(scale)[None, :]
    donors = spec.get("r2_donors") or {}
    with tf.device("/CPU:0"):
        r0 = M.build_dmo(effective["model"])
        r0_probe = _ridge_probe(_branch_features(r0, scaled(fit)), y_fit, m_fit,
                                _branch_features(r0, scaled(inner)), y_in, m_in)
        pre = M.build_dmo(effective["model"])
        try:
            record = T.r2_import_into(pre, {"path": spec["pins"]["common_population"]["path"],
                                            "expected_digest": spec["pins"]["common_population"]["population_sha256"],
                                            "input_dir": spec["pins"]["matched_input"]["path"]}, effective, donors)
        except Exception as exc:
            return _terminal(spec, "FAILED", getattr(exc, "code", type(exc).__name__), costs=dict(fit_seconds=None))
        pre_probe = _ridge_probe(_branch_features(pre, scaled(fit)), y_fit, m_fit,
                                 _branch_features(pre, scaled(inner)), y_in, m_in)
    naive = [float(np.mean(np.abs(y_in[m_in[:, j], j]))) if m_in[:, j].any() else None
             for j in range(y_in.shape[1])]
    doc = dict(schema="i6dmo.r2_probe.v1", use="REPORT_ONLY_NEVER_A_GATE_NEVER_SELECTS", alpha=1.0,
               population="INNER_COMMON_ORIGINS", coordinate="delta/naive_loss_scale",
               r0_init_mae=r0_probe, pretrained_mae=pre_probe, naive_mae=naive,
               import_sha256=record["import_sha256"])
    ratio = lambda v, n: None if v is None or not n else v / n  # noqa: E731
    doc["r0_ratio"] = [ratio(v, n) for v, n in zip(r0_probe, naive)]
    doc["pretrained_ratio"] = [ratio(v, n) for v, n in zip(pre_probe, naive)]
    valid = [(a, b) for a, b in zip(doc["r0_ratio"], doc["pretrained_ratio"]) if a is not None and b is not None]
    doc["probe_gain_equal_head_mean"] = (sum(a for a, _ in valid) - sum(b for _, b in valid)) / len(valid) if valid else None
    _atomic(out / "PROBE.json", doc)
    metrics = [_metric("Probe_Gain_Equal_Head_Mean", doc["probe_gain_equal_head_mean"], "ratio", "inner")]
    for h, a, b in zip(effective["horizons_hours"], doc["r0_ratio"], doc["pretrained_ratio"]):
        metrics += [_metric("Probe_R0_Init_MAE_Ratio", a, "ratio", "inner", h),
                    _metric("Probe_Pretrained_MAE_Ratio", b, "ratio", "inner", h)]
    report = _report(spec, "probe", metrics, {"fact": "r2.pretraining.predictive_probe",
                                              "use": "REPORT_ONLY_NEVER_A_GATE_NEVER_SELECTS"})
    return _terminal(spec, "COMPLETE", "PROBE_REPORTED",
                     artifacts={"probe": dict(path="PROBE.json", sha256=_sha(out / "PROBE.json"))},
                     reports=[report], costs=dict(fit_seconds=None, score_seconds=time.time() - started))


# --------------------------------------------------------------------- comparison
def disposition_of(verdict):
    key = (verdict.get("decision"), verdict.get("selected"))
    if key not in R2_VERDICTS:
        raise ValueError(f"UNKNOWN_R2_VERDICT:{key}")
    return R2_VERDICTS[key]


def _import_record(spec):
    """The R2 fit's retained import record, authenticated through its stored terminal + receipt."""
    from tools.i6dmo import matched_runner as MR
    up = (spec.get("upstream") or {}).get("r2.fit")
    if not up:
        return None
    path = Path(up["path"])
    if not path.is_file() or _sha(path) != up.get("sha256"):
        raise MR._Refusal("UPSTREAM_DIGEST_MISMATCH:r2.fit")
    stored = json.loads(path.read_text())
    child = stored["terminal"]
    art = (child.get("artifacts") or {}).get("R2_IMPORT.json")
    receipt_art = (child.get("artifacts") or {}).get("RECEIPT")
    if not art or not receipt_art:
        raise MR._Refusal("R2_IMPORT_RECORD_MISSING")
    record_path = path.parent / "out" / art["path"]
    if not record_path.is_file() or _sha(record_path) != art["sha256"]:
        raise MR._Refusal("R2_IMPORT_RECORD_DIGEST")
    receipt = json.loads((path.parent / "out" / receipt_art["path"]).read_text())
    if (receipt.get("artifacts") or {}).get("R2_IMPORT.json") != art["sha256"]:
        raise MR._Refusal("R2_IMPORT_RECORD_NOT_IN_RECEIPT")
    return json.loads(record_path.read_text())


def run_comparison(spec, out_dir, progress_path):
    """r2.comparison: authenticate both reports and the import record, then ``r2_design.decide_r2``."""
    from tools.i6dmo import matched_runner as MR
    from tools.i6dmo import r2_design as R2
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    view = spec.get("effective_config") or {}
    expected = ((view.get("data_pins") or {}).get("common_population") or {}).get("population_sha256")
    require = False if view.get("engineering_cpu_fixture") is True else True
    metrics, inputs = {}, {}
    try:
        for regime in ("R0", "R2"):
            unit_id = f"{regime.lower()}.report"
            up = (spec.get("upstream") or {}).get(unit_id)
            if up:
                metrics[regime], inputs[unit_id] = MR._authenticated_report(unit_id, up, expected)
        record = _import_record(spec)
        if len(metrics) < 2 or record is None:
            verdict = dict(schema="i6dmo.r2_decision.v1", decision="INCOMPLETE_COMPARISON", selected=None,
                           missing=[f"ARM_MISSING:{r}" for r in ("R0", "R2") if r not in metrics]
                           + ([] if record is not None else ["R2_IMPORT_RECORD_MISSING"]), scores=None, per_head={})
        else:
            try:
                verdict = R2.decide_r2(metrics["R0"], metrics["R2"], r2_import_record=record,
                                       require_scientific=require)
            except R2.R2DesignError as exc:
                raise MR._Refusal(f"{exc.code}:{str(exc).split(': ', 1)[-1][:120]}") from None
        disposition = disposition_of(verdict)
    except MR._Refusal as exc:
        return _terminal(spec, "FAILED", exc.code, costs=dict(fit_seconds=None, score_seconds=time.time() - started))
    doc = dict(schema="i6dmo.r2_comparison.v1", disposition=disposition, rule="tools.i6dmo.r2_design.decide_r2",
               require_scientific=require, verdict=verdict, inputs=inputs, test_read=False,
               common_population_sha256=expected,
               note="one seed; a recipe decision, not superiority; the probe/reconstruction never select")
    _atomic(out / "COMPARISON.json", doc)
    rows = []
    for h, values in (verdict.get("per_head") or {}).items():
        for regime, value in values.items():
            rows.append(_metric(f"{regime}_Relative_Error", value, "ratio", "validation", int(h)))
    for regime, score in (verdict.get("scores") or {}).items():
        rows.append(_metric(f"{regime}_Equal_Head_Mean_Relative_Error", score, "ratio", "validation"))
    rows.append(_metric("R2_Selected", 1 if disposition == "R2_SELECTED" else 0, "bool", "validation"))
    tags = {"fact": "r0_r2.decision", "disposition": disposition, "rule": "decide_r2",
            "relative_error": "(MAE_h-naive_h)/naive_h; negative is better"}
    report = _report(spec, "decision", rows, tags)
    per_head = _report(spec, "per_head", [r for r in rows if r["horizon"] is not None],
                       {"fact": "r0_r2.development.per_head_metrics"})
    return _terminal(spec, "COMPLETE", disposition,
                     artifacts={"comparison": dict(path="COMPARISON.json", sha256=_sha(out / "COMPARISON.json"))},
                     reports=[report, per_head], costs=dict(fit_seconds=None, score_seconds=time.time() - started),
                     gates=dict(comparison_status=disposition, selected=verdict.get("selected")))


def run_unit(spec, out_dir, progress_path):
    """``unit_runner`` hook for the R2 kinds."""
    kind = spec.get("kind")
    if kind == "pretrain":
        return run_pretrain(spec, out_dir, progress_path)
    if kind == "evaluation":
        return run_probe(spec, out_dir, progress_path)
    if kind == "comparison":
        return run_comparison(spec, out_dir, progress_path)
    return _terminal(spec, "FAILED", "KIND_INVALID")


# --------------------------------------------------------------------- operator path (prepare/validate/preflight)
R2_PREPARED_SCHEMA = "i6dmo.r2_prepared.v1"
RULED_CONTRACT = "R2_CONTRAST_DESIGN.ruled.json"


def _contract(path=None):
    from tools.i6dmo import matched_runner as MR
    from tools.i6dmo import r2_design as R2
    path = Path(path or MR.MATCHED_DIR / RULED_CONTRACT)
    if not path.is_file():
        raise MR.PrepareRefused("MISSING_ARTIFACT:ruled R2 contract")
    doc = json.loads(path.read_text())
    try:
        R2.validate_r2_design(doc, for_execution=True)
    except R2.R2DesignError as exc:
        raise MR.PrepareRefused(f"R2_CONTRACT_NOT_EXECUTABLE:{exc.code}") from None
    return path, doc


def _bind(contract, matched_doc):
    """Scientific binding: the prepared mg2m design, population and runtime are the ruled contract's."""
    from tools.i6dmo import matched_runner as MR
    shared = contract["shared"]
    design = matched_doc["designs"]["mg2m"]
    problems = [name for name, ok in (
        ("design_seal", design["seal_sha256"] == shared["experiment_seal_sha256"]),
        ("population", matched_doc["data_pins"]["common_population"]["population_sha256"]
         == shared["population_sha256"]),
        ("runtime", design["runtime"] == shared["fit_pinning"]["runtime"])) if not ok]
    if problems:
        raise MR.PrepareRefused(f"R2_CONTRACT_BINDING:{problems}")


def prepare(*, deployment, input_dir, out, role="coordinator", contract=None, mg2m_design=None,
            population_seal=None, population_dir=None, test=False, probe=None, env=None):
    """``r2 prepare``: the matched preparation (pins by path, population verified, memory/disk, no
    TEST) plus the RULED R2 contract and its binding. Writes ``<out>`` and its matched companion
    ``<out>.matched.json`` (both private, 0600). The R0 reference is always fitted fresh here."""
    from tools.i6dmo import matched_runner as MR
    if test:
        raise MR.PrepareRefused("TEST_REQUEST: the R2 contrast has no TEST stage (R2-12)")
    contract_path, doc = _contract(contract)
    out = Path(out).resolve()
    companion = out.with_name(out.name + ".matched.json")
    matched = MR.prepare(deployment=deployment, input_dir=input_dir, out=companion, role=role,
                         designs={"mg2m": mg2m_design} if mg2m_design else None, population_seal=population_seal,
                         population_dir=population_dir, prior_roots=(), test=False, probe=probe, env=env)
    _bind(doc, matched)
    prepared = dict(schema=R2_PREPARED_SCHEMA, matched=dict(path=str(companion), sha256=_sha(companion)),
                    contract=dict(path=str(contract_path.resolve()), sha256=_sha(contract_path),
                                  seal_sha256=doc["seal_sha256"], contrast_id=doc["contrast_id"]),
                    r0_policy="FRESH_R0_FIT (the R0 reference is fitted inside this graph; nothing is adopted)",
                    test_open=False, prepared_at=time.time())
    _atomic(out, prepared)
    os.chmod(out, 0o600)
    return prepared


def _load_prepared(prepared_path):
    from tools.i6dmo import matched_runner as MR
    doc = json.loads(Path(prepared_path).read_text())
    if doc.get("schema") != R2_PREPARED_SCHEMA or doc.get("test_open") is not False:
        raise MR.PrepareRefused("PREPARED_SCHEMA")
    for key in ("matched", "contract"):
        ref = doc[key]
        if not Path(ref["path"]).is_file() or _sha(ref["path"]) != ref["sha256"]:
            raise MR.PrepareRefused(f"MISSING_ARTIFACT:{key} changed since prepare")
    return doc


def validate(prepared_path):
    """``r2 validate``: re-verify the prepared pins, population and contract binding; launches nothing."""
    from tools.i6dmo import matched_runner as MR
    doc = _load_prepared(prepared_path)
    matched = MR.validate(doc["matched"]["path"])
    _, contract = _contract(doc["contract"]["path"])
    _bind(contract, json.loads(Path(doc["matched"]["path"]).read_text()))
    return dict(state="VALID", contract_seal_sha256=contract["seal_sha256"], mg2m_design=matched["designs"]["mg2m"],
                runtime=matched["runtime"]["mg2m"], common_population_sha256=matched["common_population_sha256"],
                r0_policy=doc["r0_policy"])


def preflight(prepared_path, out_dir, *, probe=None):
    """``r2 preflight``: the matched per-role probe (pins, disk, TF/Keras runtime pin, admission)."""
    from tools.i6dmo import matched_runner as MR
    doc = _load_prepared(prepared_path)
    validate(prepared_path)
    result = MR.preflight(doc["matched"]["path"], out_dir, probe=probe)
    return dict(result, contract_seal_sha256=doc["contract"]["seal_sha256"])


def init(prepared_path, root, *, code_commit=None, entrypoints=None):
    """``r2 init``: seal the scientific R0-vs-R2 graph from a validated preparation."""
    from tools.i6dmo import campaign as C
    validate(prepared_path)
    doc = _load_prepared(prepared_path)
    matched = json.loads(Path(doc["matched"]["path"]).read_text())
    return C.init_r2_contrast(doc["contract"]["path"], matched["designs"]["mg2m"]["path"], matched["deployment"],
                              root, code_commit=code_commit, data_pins=matched["data_pins"], entrypoints=entrypoints)
