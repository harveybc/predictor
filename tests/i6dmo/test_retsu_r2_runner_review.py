"""Retsu review of the R2 runner at a5ac7607: adversarial seams on the REAL adapters (CPU engineering
fixture on the lane's synthetic population; no GPU, no TEST, no real systemd, no production OLAP).

``xfail(strict=True)`` marks a REQUIRED behaviour that is missing (a minimal reproduction of a finding).
"""
import json
import shutil

import pytest

pytest.importorskip("tensorflow")

from tests.i6dmo.test_r2_real_chain import (  # noqa: E402,F401  (fixtures and helpers)
    CONTRACT, REAL_MG2M, Recorder, _campaign, _drive, _pins, _provider, _real_systemctl_forbidden, _spec,
    _stored, _unit_spec, chain, world)


def _cp(spec):
    return dict(path=spec["pins"]["common_population"]["path"],
                expected_digest=spec["pins"]["common_population"]["population_sha256"],
                input_dir=spec["pins"]["matched_input"]["path"])


def _effective(spec, tmp_path):
    from tools.i6dmo import train as T
    design = tmp_path / "design.json"
    design.write_text(json.dumps(spec["effective_config"]["design"]))
    return T.effective_from_design(str(design))


def _import(spec, effective, donors):
    from tools.i6dmo import model as M, train as T
    bundle = M.build_dmo(effective["model"])
    before = [w.copy() for w in bundle.forecast_model.get_weights()]
    try:
        T.r2_import_into(bundle, _cp(spec), effective, donors)
        code = "ACCEPTED"
    except T.TrainDisposition as exc:
        code = exc.code
    untouched = all((a == b).all() for a, b in zip(before, bundle.forecast_model.get_weights()))
    return code, untouched


@pytest.fixture(scope="module")
def other(world):
    """A SECOND, independently initialized campaign on the same population, driven to completion."""
    from tools.i6dmo import campaign
    root, _ = _campaign(world, "other_campaign", )
    # a different code pin => a different campaign id; same population, same ruled objective
    _drive(root, world)
    q = campaign.open_queue(root)
    try:
        terms = {u["unit_id"]: q.terminal(u["unit_id"]) for u in q.units()}
    finally:
        q.close()
    return dict(root=root, terms=terms)


def test_donor_from_another_campaign_is_refused_before_any_weight_is_set(chain, other, tmp_path):
    from tools.i6dmo import campaign
    spec = _unit_spec(chain, "r2.fit")
    effective = _effective(spec, tmp_path)
    assert campaign.load_plan(other["root"])["campaign_id"] != campaign.load_plan(chain["root"])["campaign_id"]
    theirs = campaign.r2_donor_references(other["root"], campaign.open_queue(other["root"]),
                                          campaign.load_effective(other["root"]))
    assert theirs is not None
    # (a) their whole donor set under OUR provenance
    code, untouched = _import(spec, effective, dict(spec["r2_donors"], donors=theirs))
    assert code == "R2_DONOR_PROVENANCE_MISMATCH" and untouched
    # (b) ONE substituted feature among our own
    mixed = json.loads(json.dumps(spec["r2_donors"]))
    f0 = next(iter(mixed["donors"]))
    mixed["donors"][f0] = theirs[f0]
    code, untouched = _import(spec, effective, mixed)
    assert code == "R2_DONOR_PROVENANCE_MISMATCH" and untouched


def test_stale_or_substituted_donor_bytes_are_refused(chain, tmp_path):
    spec = _unit_spec(chain, "r2.fit")
    effective = _effective(spec, tmp_path)
    donors = json.loads(json.dumps(spec["r2_donors"]))
    f0 = next(iter(donors["donors"]))
    ref = donors["donors"][f0]["terminal"]
    # copy the donor directory; then (1) edit the terminal JSON (stale/rewritten), (2) swap donor.npz
    src = __import__("pathlib").Path(ref["path"]).parent
    work = tmp_path / "donor"
    shutil.copytree(src, work)
    term = json.loads((work / "TERMINAL.json").read_text())
    term["fit"]["selected_epoch"] = 99
    (work / "TERMINAL.json").write_text(json.dumps(term))
    donors["donors"][f0]["terminal"] = dict(path=str(work / "TERMINAL.json"), sha256=ref["sha256"])
    assert _import(spec, effective, donors) == ("R2_DONOR_TERMINAL_DIGEST", True)
    shutil.rmtree(work)
    shutil.copytree(src, work)
    other_feature = list(donors["donors"])[1]
    other_npz = __import__("pathlib").Path(spec["r2_donors"]["donors"][other_feature]["terminal"]["path"]).parent
    shutil.copyfile(other_npz / "donor.npz", work / "donor.npz")
    donors["donors"][f0]["terminal"] = dict(path=str(work / "TERMINAL.json"), sha256=ref["sha256"])
    assert _import(spec, effective, donors) == ("R2_DONOR_WEIGHTS_DIGEST", True)


def test_eighteen_donors_hold_the_r2_fit_and_never_launch_it(chain, tmp_path):
    import sqlite3
    from tools.i6dmo import campaign, worker as W
    copy = tmp_path / "copy"
    shutil.copytree(chain["root"], copy)
    db = sqlite3.connect(copy / "state" / "queue.sqlite3")
    try:
        db.execute("DELETE FROM terminals WHERE unit_id='r2.pretrain.11'")
        db.commit()
    finally:
        db.close()
    effective = campaign.load_effective(copy)
    unit = next(u for u in campaign.load_plan(copy)["units"] if u["unit_id"] == "r2.fit")
    assert W.r2_donor_problem(copy, effective, unit).startswith("R2_DONOR_REFERENCES_INCOMPLETE")
    q = campaign.open_queue(copy)
    try:
        assert campaign.r2_donor_references(copy, q, effective) is None
    finally:
        q.close()


def test_r0_metrics_with_an_import_binding_and_cross_class_fits_are_refused(chain, tmp_path):
    from tools.i6dmo import r2_design as R2
    root, terms = chain["root"], chain["terms"]
    r0 = json.loads(_stored(root, terms["r0.report"], "metrics").read_text())
    r2 = json.loads(_stored(root, terms["r2.report"], "metrics").read_text())
    record = json.loads(_stored(root, terms["r2.fit"], "R2_IMPORT.json").read_text())
    bad = json.loads(json.dumps(r0))
    bad["matched"]["fit_identity"]["r2_import_sha256"] = record["import_sha256"]
    with pytest.raises(R2.R2DesignError, match="R2_R0_HAS_IMPORT"):
        R2.decide_r2(bad, r2, r2_import_record=record, require_scientific=False)
    a, b = json.loads(json.dumps(r0)), json.loads(json.dumps(r2))
    for doc, name in ((a, "NVIDIA GeForce RTX 5090"), (b, "NVIDIA GeForce RTX 4090 Laptop GPU")):
        doc["matched"].update(scientific_fit=True, fit_device="gpu", fit_runtime="2.21.0", fit_gpu_class=name)
    with pytest.raises(R2.R2DesignError, match="R2_FIT_PINNING"):
        R2.decide_r2(a, b, r2_import_record=record)


def test_cross_class_placement_is_held_at_claim_time(world, tmp_path):
    """Was the C1 reproduction (r2.fit placed on the secondary 4090 role while r0.fit held the
    preferred 5090 host). Since the C1 fix the secondary role never picks r2.fit: it is held typed
    WAIT_RESOURCE FIT_GPU_CLASS_HOLD for the sealed class."""
    from tools.i6dmo import campaign, worker as W
    contract = json.loads(CONTRACT.read_text())
    pins = _pins(world)
    pins["common_population"]["population_sha256"] = contract["shared"]["population_sha256"]
    root = tmp_path / "sci"
    spec_path = _spec(world, tmp_path)
    campaign.init_r2_contrast(CONTRACT, REAL_MG2M, spec_path, root, code_commit="c" * 40, data_pins=pins)
    plan, spec = campaign.load_plan(root), campaign.load_deployment(root)
    q = campaign.open_queue(root)
    try:
        for i in range(19):
            q.set_state(f"r2.pretrain.{i:02d}", "COMPLETE", "SIMULATED")
        q.set_state("r2.fit", "QUEUED")
        held = q.claim("r0.fit", owner="w", role="preferred", host_key="host-preferred", boot_id="b",
                       device_class="rtx5090", heavy=True, ttl=600)
        assert held["result"] == "CLAIMED"
        picked = W._pick(q, plan, spec, "secondary")
        row = q.unit("r2.fit")
    finally:
        q.close()
    assert picked is None or picked["unit_id"] != "r2.fit"
    assert row["state"] == "WAIT_RESOURCE" and row["cause"].startswith("FIT_GPU_CLASS_HOLD")


def test_r2_fit_must_wait_for_the_r0_gpu_class(world, tmp_path):
    from tools.i6dmo import campaign, worker as W
    contract = json.loads(CONTRACT.read_text())
    pins = _pins(world)
    pins["common_population"]["population_sha256"] = contract["shared"]["population_sha256"]
    root = tmp_path / "sci"
    campaign.init_r2_contrast(CONTRACT, REAL_MG2M, _spec(world, tmp_path), root, code_commit="c" * 40,
                              data_pins=pins)
    plan, spec = campaign.load_plan(root), campaign.load_deployment(root)
    q = campaign.open_queue(root)
    try:
        for i in range(19):
            q.set_state(f"r2.pretrain.{i:02d}", "COMPLETE", "SIMULATED")
        q.set_state("r2.fit", "QUEUED")
        q.claim("r0.fit", owner="w", role="preferred", host_key="host-preferred", boot_id="b",
                device_class="rtx5090", heavy=True, ttl=600)
        picked = W._pick(q, plan, spec, "secondary")
    finally:
        q.close()
    assert picked is None or picked["unit_id"] != "r2.fit"


def test_r0_fit_is_reached_after_the_pretraining_claims_not_starved(world, tmp_path):
    """Plan order puts 19 pretraining units first; on one GPU role r0.fit is the 20th claim (bounded)."""
    from tools.i6dmo import campaign, worker as W
    contract = json.loads(CONTRACT.read_text())
    pins = _pins(world)
    pins["common_population"]["population_sha256"] = contract["shared"]["population_sha256"]
    root = tmp_path / "sci"
    campaign.init_r2_contrast(CONTRACT, REAL_MG2M, _spec(world, tmp_path), root, code_commit="c" * 40,
                              data_pins=pins)
    plan, spec = campaign.load_plan(root), campaign.load_deployment(root)
    order = []
    q = campaign.open_queue(root)
    try:
        for _ in range(21):
            row = W._pick(q, plan, spec, "preferred")
            if row is None:
                break
            order.append(row["unit_id"])
            q.set_state(row["unit_id"], "COMPLETE", "SIMULATED")
    finally:
        q.close()
    assert order.index("r0.fit") == 19 and "r2.fit" not in order[:19]
