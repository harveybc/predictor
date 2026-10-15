"""R1 donor hardening (MUSASHI_TO_SATOSHI_MATCHED_AUDIT_FOLLOWUP_20261015 section 2).

A donor root may be reused only when ONE common verifier accepts it for both interfaces:
prepare (``matched_runner._check_donor``) and actual adoption (``worker._adoptable`` through
``campaign.donor_problem``). Every case below must be refused by both, with ZERO adoption and
ZERO fit; a genuine donor (published, mirrored, trainer-shaped receipts bound to this campaign's
population/arm/design/runtime/capacity/weights) must still be accepted with zero new fits.

Fixtures: the arm children are the orchestration fake; its ``I6DMO_FAKE_GENUINE_RECEIPT=1`` mode
writes a trainer-shaped receipt without a fake marker (it trains nothing). The trust anchors are the
system's OWN custody (mirrors written by ``publish``) and publication records (outbox exact readback),
so a hand-assembled or rewritten root is refused even when its own hashes are self-consistent.
"""
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from tests.i6dmo.ops_fakes import FakeWarehouse
from tests.i6dmo.test_ops_matched_runner import BOGUS_RUNTIME, drive, make_matched, provider, states, worker

GENUINE = {"I6DMO_FAKE_GENUINE_RECEIPT": "1", "I6DMO_FAKE_SCIENTIFIC": "true"}
UNITS = ("mg1.fit", "mg1.report", "mg2m.fit", "mg2m.report", "comparison")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def genuine_donor(tmp_path, name="donor", counter=None):
    """A matched donor driven to CLOSE: published (exact readback) and mirrored on two targets."""
    from tools.i6dmo import campaign
    root = make_matched(tmp_path / name, name=name)
    env = dict(GENUINE, **({"I6DMO_FAKE_COUNTER": str(counter)} if counter else {}))
    drive(root, env=env)
    campaign.publish(root, warehouse=FakeWarehouse())
    assert (root / "CLOSE.json").exists(), states(root)
    return root


def _mirrors(root):
    spec = json.loads((root / "private" / "deployment.json").read_text())
    return [Path(spec["mirrors"]["local"]), Path(spec["mirrors"]["second_host_path"])]


def _fit_folder(root, unit_id="mg1.fit"):
    plan = json.loads((root / "PLAN.json").read_text())
    unit = next(u for u in plan["units"] if u["unit_id"] == unit_id)
    return root / "terminals" / unit["identity_sha256"]


def rewrite(root, unit_id="mg1.fit", *, receipt=None, files=None, drop=(), mirrors=True, queue=True):
    """A COHERENT rewrite of one fit terminal: every in-root digest is recomputed (artifacts, child,
    stored envelope, queue backup) and, with ``mirrors``, copied into both mirror targets too."""
    folder = _fit_folder(root, unit_id)
    train = folder / "out" / "train"
    for rel, data in (files or {}).items():
        (train / rel).write_bytes(data)
    for rel in drop:
        (train / rel).unlink()
    rec_path = train / "RECEIPT.json"
    if rec_path.exists() and receipt is not None:
        doc = json.loads(rec_path.read_text())
        receipt(doc)
        doc["artifacts"] = {rel: _sha(train / rel) for rel in doc.get("artifacts", {}) if (train / rel).exists()}
        rec_path.write_text(json.dumps(doc))
    child = json.loads((folder / "CHILD_TERMINAL.json").read_text())
    arts = {}
    for name, art in child["artifacts"].items():
        path = folder / "out" / art["path"]
        if path.exists():
            arts[name] = dict(art, sha256=_sha(path))
    child["artifacts"] = arts
    if rec_path.exists():
        child["trainer_identity"] = json.loads(rec_path.read_text()).get("identity", child.get("trainer_identity"))
    raw = json.dumps(child).encode()
    (folder / "CHILD_TERMINAL.json").write_bytes(raw)
    stored = json.loads((folder / "TERMINAL.json").read_text())
    stored.update(terminal=child, child_terminal_sha256=hashlib.sha256(raw).hexdigest())
    (folder / "TERMINAL.json").write_text(json.dumps(stored))
    if queue:
        con = sqlite3.connect(root / "backup" / "queue.sqlite3")
        con.execute("UPDATE terminals SET terminal=?, terminal_sha256=? WHERE unit_id=?",
                    (json.dumps(child, sort_keys=True, separators=(",", ":")), hashlib.sha256(raw).hexdigest(), unit_id))
        con.commit()
        con.close()
    if mirrors:
        for target in _mirrors(root):
            for path in [*folder.rglob("*"), root / "backup" / "queue.sqlite3"]:
                if path.is_file():
                    dest = target / path.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest)
            for rel in drop:
                (target / (folder / "out" / "train" / rel).relative_to(root)).unlink(missing_ok=True)


def _prepare_refusal(donor, target):
    """Prepare's verdict for exactly the target the worker sees (same designs, pins, entry points)."""
    from tools.i6dmo import campaign, matched_runner as MR
    effective = campaign.load_effective(target)
    sealed = {a: dict(seal_sha256=effective["arms"][a]["design"]["seal_sha256"]) for a in ("mg1", "mg2m")}
    try:
        try:
            MR._check_donor(donor, sealed, effective["data_pins"], entrypoints=effective["entrypoints"])
        except TypeError:  # pre-R1 signature (no entry-point argument): its own fixed target
            MR._check_donor(donor, sealed, effective["data_pins"])
    except MR.PrepareRefused as exc:
        return str(exc)
    return None


def assert_refused_without_adoption_or_fit(tmp_path, donor, *, expect=None, entries="fakes"):
    from tools.i6dmo import campaign
    counter = tmp_path / f"fits-{donor.name}.txt"
    target = make_matched(tmp_path / f"target-{donor.name}", name="target", prior_roots=[donor], entries=entries)
    worker_problem = campaign.donor_problem(donor, campaign.load_effective(target))
    prepare_problem = _prepare_refusal(donor, target)
    assert worker_problem and prepare_problem, (worker_problem, prepare_problem)
    if expect:
        assert expect in worker_problem and expect in prepare_problem, (worker_problem, prepare_problem)
    campaign.tick(target)
    for role in ("preferred", "secondary", "coordinator"):  # a refused donor never launches anything here
        worker(target, role, {"I6DMO_FAKE_COUNTER": str(counter)}, facts_provider=provider(BOGUS_RUNTIME))
    q = campaign.open_queue(target)
    try:
        adopted = [u for u in UNITS if (q.terminal(u) or {}).get("source") == "ADOPTED"]
    finally:
        q.close()
    assert adopted == [], adopted
    assert not counter.exists() or not [x for x in counter.read_text().splitlines() if x.startswith("fit:")]
    assert not list((target / "attempts").glob("*/a*")) or all(
        not (p / "out" / "TERMINAL.json").exists() for p in (target / "attempts").glob("*/a*"))
    return worker_problem


# ------------------------------------------------------------------------------- acceptance
def test_genuine_donor_is_accepted_by_both_interfaces_and_reused_with_zero_new_fits(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "fits.txt"
    donor = genuine_donor(tmp_path, counter=counter)
    fits = lambda: [line for line in counter.read_text().splitlines() if line.startswith("fit:")]  # noqa: E731
    fitted = fits()
    assert sorted(fitted) == ["fit:mg1.fit", "fit:mg2m.fit"]
    target = make_matched(tmp_path / "reuser", name="reuser", prior_roots=[donor])
    assert campaign.donor_problem(donor, campaign.load_effective(target)) is None
    assert _prepare_refusal(donor, target) is None
    drive(target, env=dict(GENUINE, I6DMO_FAKE_COUNTER=str(counter)))
    assert fits() == fitted  # zero new fits
    q = campaign.open_queue(target)
    try:
        assert all((q.terminal(u) or {}).get("source") == "ADOPTED" for u in UNITS)
    finally:
        q.close()


# --------------------------------------------------------------------------------- refusals
def test_fake_marker_is_refused(tmp_path):
    donor = genuine_donor(tmp_path)
    rewrite(donor, receipt=lambda r: r.update(fake=True))
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_FAKE_MARKER")


def test_absent_scientific_or_gpu_metadata_is_refused(tmp_path):
    donor = genuine_donor(tmp_path)
    rewrite(donor, receipt=lambda r: (r.pop("scientific"), r.pop("device")))
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_FIT_NOT_SCIENTIFIC")


def test_terminal_copied_from_another_campaign_is_refused(tmp_path):
    from tools.i6dmo import campaign
    origin = genuine_donor(tmp_path, name="origin")
    forged = make_matched(tmp_path / "forged", name="forged")
    shutil.copytree(origin / "terminals", forged / "terminals", dirs_exist_ok=True)
    campaign.publish(forged, warehouse=FakeWarehouse())  # even with its own mirrors of the copies
    assert_refused_without_adoption_or_fit(tmp_path, forged, expect="DONOR_SOURCE_TERMINAL_UNTRUSTED")


@pytest.mark.parametrize("field,value,code", [
    ("common_population_sha256", "e" * 64, "DONOR_POPULATION_MISMATCH"),
    ("matched_arm", "mg2m", "DONOR_ARM_MISMATCH"),
])
def test_mismatched_population_or_arm_is_refused(tmp_path, field, value, code):
    donor = genuine_donor(tmp_path)
    rewrite(donor, receipt=lambda r: r["identity"].update({field: value}))
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect=code)


def test_changed_weights_are_refused_even_after_a_coherent_rewrite_of_root_and_mirrors(tmp_path):
    import io
    import zipfile
    donor = genuine_donor(tmp_path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        pin = json.loads((donor / "EFFECTIVE_CONFIG.json").read_text())["arms"]["mg1"]["design"]["sealed_identities"]
        z.writestr("metadata.json", json.dumps({"keras_version": pin["runtime"]["keras"]}))
        z.writestr("model.weights.h5", b"other weights")
    new_weights = hashlib.sha256(b"other weights").hexdigest()

    def mutate(r):
        r["identity"].update(weights_sha256=new_weights)
    rewrite(donor, files={"model/forecast_model.keras": buf.getvalue(),
                          "model/MODEL.json": json.dumps(dict(weights_sha256=new_weights)).encode()},
            receipt=lambda r: (mutate(r), r["identity"].update(
                model_archive_sha256=hashlib.sha256(buf.getvalue()).hexdigest())))
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_PUBLICATION_MISMATCH")


def test_changed_weights_in_the_root_only_are_refused_by_custody(tmp_path):
    donor = genuine_donor(tmp_path)
    rewrite(donor, files={"model/MODEL.json": json.dumps(dict(weights_sha256="0" * 64)).encode()},
            receipt=lambda r: None, mirrors=False)
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_CUSTODY_MISMATCH")


def test_changed_capacity_is_refused(tmp_path):
    donor = genuine_donor(tmp_path)

    def mutate(r):
        r["matched"]["capacity"]["counts"] = dict(r["matched"]["capacity"]["counts"], heads=1)
    rewrite(donor, receipt=mutate)
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_CAPACITY_MISMATCH")


def test_changed_design_is_refused(tmp_path):
    from tools.i6dmo import campaign
    donor = genuine_donor(tmp_path)
    eff = json.loads((donor / "EFFECTIVE_CONFIG.json").read_text())
    eff["arms"]["mg1"]["design"]["seal_sha256"] = "d" * 64
    body = {k: v for k, v in eff.items() if k != "seal_sha256"}
    (donor / "EFFECTIVE_CONFIG.json").write_text(json.dumps(campaign._seal(body)))
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_DESIGN")


def test_runtime_seal_mismatch_is_an_incompatible_donor(tmp_path):
    donor = genuine_donor(tmp_path)
    rewrite(donor, receipt=lambda r: r["resources"].update(tensorflow_version="2.18.0"))
    problem = assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_RUNTIME_MISMATCH")
    assert "2.18.0" in problem


def test_missing_fit_receipt_is_refused(tmp_path):
    donor = genuine_donor(tmp_path)
    rewrite(donor, drop=("RECEIPT.json",))
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_FIT_RECEIPT_MISSING")


def test_missing_trusted_source_terminal_reference_is_refused(tmp_path):
    donor = genuine_donor(tmp_path)
    con = sqlite3.connect(donor / "backup" / "queue.sqlite3")
    con.execute("DELETE FROM terminals WHERE unit_id='mg1.fit'")
    con.commit()
    con.close()
    for target in _mirrors(donor):
        shutil.copy2(donor / "backup" / "queue.sqlite3", target / "backup" / "queue.sqlite3")
    assert_refused_without_adoption_or_fit(tmp_path, donor, expect="DONOR_SOURCE_TERMINAL_UNTRUSTED")


# --------------------------------------------- the reviewer's hand-forged roots (re-check p6 A)
@pytest.mark.parametrize("variant", ["no_receipt", "cpu_donor", "engineering_donor", "override_receipt",
                                     "forged_scientific_receipt_not_in_artifacts", "fake_scientific_receipt"])
def test_reviewer_hand_forged_roots_are_refused(tmp_path, variant):
    """r_recheck_796ec8f4.d1: a real-entry root into which another campaign's (fake) terminals were
    copied by hand, with the reviewer's receipt/effective variants."""
    from tools.i6dmo import campaign
    a_root = make_matched(tmp_path / "A", name="A")
    drive(a_root, env={"I6DMO_FAKE_SCIENTIFIC": "true"})
    root = make_matched(tmp_path / f"D_{variant}", name=f"D_{variant}", entries=None)
    shutil.copytree(a_root / "terminals", root / "terminals", dirs_exist_ok=True)
    if variant in ("cpu_donor", "engineering_donor"):
        eff = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
        eff.update(fit_device="cpu") if variant == "cpu_donor" else eff.update(engineering_cpu_fixture=True)
        (root / "EFFECTIVE_CONFIG.json").write_text(json.dumps(campaign._seal(
            {k: v for k, v in eff.items() if k != "seal_sha256"})))
    if variant in ("override_receipt", "forged_scientific_receipt_not_in_artifacts", "fake_scientific_receipt"):
        for t in (root / "terminals").glob("*/TERMINAL.json"):
            if json.loads(t.read_text())["unit_id"].endswith(".fit"):
                (t.parent / "out" / "train").mkdir(parents=True, exist_ok=True)
                rec = {"override_receipt": {"scientific": False, "device": "gpu", "overrides": {"max_epochs": 2}},
                       "forged_scientific_receipt_not_in_artifacts": {"scientific": True, "device": "gpu",
                                                                     "overrides": {}},
                       "fake_scientific_receipt": {"scientific": True, "device": "gpu", "overrides": {},
                                                   "fake": True}}[variant]
                (t.parent / "out" / "train" / "RECEIPT.json").write_text(json.dumps(rec))
    campaign.publish(root, warehouse=FakeWarehouse())
    assert_refused_without_adoption_or_fit(tmp_path, root, entries=None)


# ---------------------------------------------------------------- one verifier, two interfaces
def test_prepare_and_adoption_call_the_same_verifier_with_the_same_verdict(tmp_path, monkeypatch):
    from tools.i6dmo import campaign, donor_trust
    donor = genuine_donor(tmp_path)
    target = make_matched(tmp_path / "t", name="t", prior_roots=[donor])
    calls = []
    real = donor_trust.verify_donor

    def spy(prior_root, target_doc, **kw):
        out = real(prior_root, target_doc, **kw)
        calls.append((str(prior_root), donor_trust.target_digest(target_doc), out["verdict"], tuple(out["problems"])))
        return out
    monkeypatch.setattr(donor_trust, "verify_donor", spy)
    cases = {"genuine": None, "fake": lambda r: r.update(fake=True),
             "runtime": lambda r: r["resources"].update(tensorflow_version="0.0.0")}
    for name, mutate in cases.items():
        calls.clear()
        if mutate:
            rewrite(donor, receipt=mutate)
        worker_problem = campaign.donor_problem(donor, campaign.load_effective(target))
        prepare_problem = _prepare_refusal(donor, target)
        assert len(calls) == 2, name
        assert calls[0][1:] == calls[1][1:], (name, calls)  # same target, same verdict, same problems
        assert (worker_problem is None) == (prepare_problem is None) == (name == "genuine"), name
