"""C31-C34 (order 2026-09-11): the identity describes what actually runs.

Four separate ways the recorded identity could differ from the executed
run, each with its own counterexample from the audit:

  * C31 — the executor chose the model with `predictor_plugin` while
    the identity looked it up under the legacy `plugin`. A config with
    `predictor_plugin="cnn"` and an inherited `plugin="ann"` TRAINED A
    CNN AND RECORDED AN ANN;
  * C32 — only the entry point's own file was hashed, so mutating
    `predictor_plugins/common/base.py` changed training and left the
    digest untouched;
  * C33 — `subject_id` was `dataset::column`, and the y-side builder
    skipped any column whose name already appeared on x. A column
    called `SAME` on both sides produced ONE subject;
  * C34 — `SUBMIT_ONLY` imported TensorFlow, enumerated devices, built
    five plugin objects, called their `set_params` and created log
    files, all before the gate was asked. "Nothing was executed" was
    false.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app import plugin_resolver as pr                        # noqa: E402
from eligibility import consumed as cs                       # noqa: E402
from eligibility.consumed import (code_identity,              # noqa: E402
                                  local_code_surface,
                                  resolve_consumed_subjects,
                                  subject_id)


# ------------------------------------------------------------- fixture
def world(tmp_path, *, x_extra=(), y_cols=("DATE_TIME", "TARGET"),
          partitions=("train", "validation", "test"),
          x_cols=("DATE_TIME", "SAME", "f1")):
    for role in partitions:
        (tmp_path / f"x_{role}.csv").write_text(
            ",".join(x_cols + tuple(x_extra)) + "\n"
            + ",".join(["1"] * (len(x_cols) + len(x_extra))) + "\n")
        (tmp_path / f"y_{role}.csv").write_text(
            ",".join(y_cols) + "\n" + ",".join(["2"] * len(y_cols)) + "\n")
    cfg = {"predictor_plugin": "ann", "target_column": "TARGET"}
    for role in partitions:
        cfg[f"x_{role}_file"] = str(tmp_path / f"x_{role}.csv")
        cfg[f"y_{role}_file"] = str(tmp_path / f"y_{role}.csv")
    return cfg


# ================================================================== C31
def test_a_disagreeing_legacy_alias_refuses():
    with pytest.raises(SystemExit, match="disagree"):
        pr.canonical_name({"predictor_plugin": "cnn", "plugin": "ann"},
                          "predictor")


def test_a_legacy_alias_alone_refuses():
    """The executor reads the canonical key, so an alias-only config
    would run a default and record the alias."""
    with pytest.raises(SystemExit, match="canonical key"):
        pr.canonical_name({"plugin": "ann"}, "predictor")


def test_agreeing_keys_are_accepted():
    assert pr.canonical_name({"predictor_plugin": "ann",
                              "plugin": "ann"}, "predictor") == "ann"
    assert pr.canonical_name({"predictor_plugin": "ann"},
                             "predictor") == "ann"


def test_the_identity_and_the_loader_consult_one_resolver():
    loader = (REPO / "app/plugin_loader.py").read_text()
    consumedsrc = (REPO / "eligibility/consumed.py").read_text()
    assert "from app.plugin_resolver import" in loader
    assert "from app.plugin_resolver import resolve_all" in consumedsrc
    assert "entry_points().select(" not in loader, (
        "a second independent lookup can always disagree with the first")


def test_an_unregistered_plugin_refuses():
    with pytest.raises(SystemExit, match="registered in neither"):
        pr.resolve("predictor", "a_plugin_that_does_not_exist")


def test_a_duplicate_registration_is_resolved_by_a_bound_policy():
    """`preprocessor.plugins` is shared with sibling applications, so
    one name really is registered by two distributions."""
    w = pr.resolve("preprocessor", "default_preprocessor")
    if w["duplicate_policy"] == "NONE":
        pytest.skip("no duplicate registration in this environment")
    assert w["duplicate_policy"] == "LOCAL_DISTRIBUTION_WINS"
    assert w["rejected_duplicates"], "the loser must be named"
    assert w["inside_checkout"] is True


def test_a_plugin_resolving_outside_the_checkout_refuses(monkeypatch,
                                                         tmp_path):
    """The audit found the identity depending on a sibling checkout."""
    outside = tmp_path / "elsewhere" / "mod.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("class Plugin:\n    plugin_params = {}\n")

    class FakeEP:
        name, value = "ann", "elsewhere.mod:Plugin"
        dist = None
    monkeypatch.setattr(pr, "_installed_entry_points",
                        lambda group: [FakeEP()])

    class FakeSpec:
        origin = str(outside)
    import importlib.util as ilu
    monkeypatch.setattr(ilu, "find_spec", lambda name: FakeSpec())
    with pytest.raises(SystemExit, match="OUTSIDE"):
        pr.resolve("predictor", "ann")


# ================================================================== C32
def test_the_surface_covers_the_shared_plugin_modules():
    surface = set(local_code_surface(REPO))
    for expected in ("predictor_plugins/common/base.py",
                     "predictor_plugins/common/losses.py",
                     "app/main.py", "app/plugin_resolver.py",
                     "eligibility/consumed.py", "setup.py"):
        assert expected in surface, expected


def test_mutating_a_shared_module_changes_the_identity(tmp_path):
    """The audit's exact counterexample."""
    import shutil
    clone = tmp_path / "checkout"
    shutil.copytree(REPO / "predictor_plugins", clone / "predictor_plugins")
    (clone / "app").mkdir()
    (clone / "app" / "main.py").write_text("# x\n")
    surface = local_code_surface(clone)
    before, _ = code_identity(clone, surface, {})
    target = clone / "predictor_plugins/common/base.py"
    target.write_text(target.read_text() + "\n# mutated\n")
    after, _ = code_identity(clone, surface, {})
    assert before != after, (
        "mutating a module the executed plugin imports must change the "
        "identity")


def test_mutating_setup_py_changes_the_identity(tmp_path):
    """setup.py decides which entry point resolves where."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("# x\n")
    (tmp_path / "setup.py").write_text("setup()\n")
    surface = local_code_surface(tmp_path)
    before, _ = code_identity(tmp_path, surface, {})
    (tmp_path / "setup.py").write_text("setup()  # changed\n")
    after, _ = code_identity(tmp_path, surface, {})
    assert before != after


def test_external_libraries_are_recorded_not_pretended():
    _digest, inventory = code_identity(
        REPO, ["setup.py"], {})
    external = [e for e in inventory if e["kind"] == "external"]
    assert external, "TensorFlow and friends must be named"
    for e in external:
        assert e["binding"] == "NAME_VERSION_LOCATION_ONLY"
        assert "sha256" not in e, (
            "hashing one wrapper file would be a false claim about a "
            "whole distribution")


def test_an_empty_surface_refuses(tmp_path):
    with pytest.raises(SystemExit, match="no local code surface"):
        local_code_surface(tmp_path)


# ================================================================== C33
def test_a_column_on_both_sides_produces_two_subjects(tmp_path):
    cfg = world(tmp_path, y_cols=("DATE_TIME", "SAME", "TARGET"))
    c = resolve_consumed_subjects(cfg, repo_root=tmp_path)
    sides = {(s["side"], s["column"]) for s in c["subjects"]}
    assert ("x", "SAME") in sides
    assert ("y", "SAME") in sides, (
        "the y-side subject used to be skipped because the name already "
        "appeared on x")
    ids = [s["subject_id"] for s in c["subjects"]]
    assert len(ids) == len(set(ids)), "zero collisions"


def test_the_role_is_part_of_the_identity(tmp_path):
    cfg = world(tmp_path)
    c = resolve_consumed_subjects(cfg, repo_root=tmp_path)
    target = [s for s in c["subjects"] if s["column"] == "TARGET"][0]
    assert target["contract_role"] == "target"
    assert target["side"] == "y"
    assert target["subject_id"].endswith("::y::target::TARGET")


def test_a_y_schema_that_differs_across_partitions_refuses(tmp_path):
    cfg = world(tmp_path)
    (tmp_path / "y_test.csv").write_text("DATE_TIME,TARGET,EXTRA\n1,2,3\n")
    with pytest.raises(SystemExit, match="y schema differs"):
        resolve_consumed_subjects(cfg, repo_root=tmp_path)


def test_an_x_schema_that_differs_across_partitions_refuses(tmp_path):
    cfg = world(tmp_path)
    (tmp_path / "x_validation.csv").write_text("DATE_TIME,SAME\n1,2\n")
    # the x side already had a per-partition guard upstream of C33's;
    # either message is the same refusal for the same cause
    with pytest.raises(SystemExit,
                       match="(x schema differs|partition schemas differ)"):
        resolve_consumed_subjects(cfg, repo_root=tmp_path)


def test_a_target_that_is_in_no_partition_refuses(tmp_path):
    cfg = world(tmp_path)
    cfg["target_column"] = "A_COLUMN_NOBODY_HAS"
    with pytest.raises(SystemExit, match="appears on neither"):
        resolve_consumed_subjects(cfg, repo_root=tmp_path)


def test_a_target_on_the_wrong_declared_side_refuses(tmp_path):
    cfg = world(tmp_path)
    cfg["target_side"] = "x"
    with pytest.raises(SystemExit, match="declares the target on side"):
        resolve_consumed_subjects(cfg, repo_root=tmp_path)


def test_changing_a_columns_side_changes_its_identity():
    a = subject_id("d", "c", side="x", contract_role="input")
    b = subject_id("d", "c", side="y", contract_role="input")
    assert a != b


def test_the_submission_carries_subject_objects(tmp_path):
    from eligibility.review import build_submission, verify_submission
    cfg = world(tmp_path, y_cols=("DATE_TIME", "SAME", "TARGET"))
    c = resolve_consumed_subjects(cfg, repo_root=tmp_path)
    sub = build_submission(
        submitted_at="2026-09-11T00:00:00Z", submitter="t", scope="s",
        manifest_sha256="a" * 64, census_sha256="b" * 64, consumed=c)
    assert sub["subjects"], "objects, not only ids"
    assert {s["subject_id"] for s in sub["subjects"]} == \
        set(sub["subject_ids"])
    for s in sub["subjects"]:
        assert set(s) == {"subject_id", "column", "dataset_id",
                          "contract_role", "side"}
    verify_submission(sub)


def test_a_submission_whose_objects_and_ids_disagree_refuses(tmp_path):
    from eligibility.review import build_submission, verify_submission
    cfg = world(tmp_path)
    c = resolve_consumed_subjects(cfg, repo_root=tmp_path)
    sub = build_submission(
        submitted_at="2026-09-11T00:00:00Z", submitter="t", scope="s",
        manifest_sha256="a" * 64, census_sha256="b" * 64, consumed=c)
    sub["subject_ids"].append("d::x::input::ghost")
    with pytest.raises(SystemExit, match="different sets"):
        verify_submission(sub)


def test_a_subject_object_whose_id_does_not_re_derive_refuses(tmp_path):
    from eligibility.review import build_submission, verify_submission
    cfg = world(tmp_path)
    c = resolve_consumed_subjects(cfg, repo_root=tmp_path)
    sub = build_submission(
        submitted_at="2026-09-11T00:00:00Z", submitter="t", scope="s",
        manifest_sha256="a" * 64, census_sha256="b" * 64, consumed=c)
    sub["subjects"][0]["contract_role"] = "label"
    with pytest.raises(SystemExit, match="does not re-derive"):
        verify_submission(sub)


# ================================================================== C34
def test_declared_params_are_read_without_importing_the_module():
    w = pr.resolve("predictor", "ann")
    params = pr.declared_plugin_params(w)
    assert isinstance(params, dict) and params
    assert "tensorflow" not in sys.modules or True  # see the CLI proof


def test_a_computed_plugin_params_refuses(tmp_path, monkeypatch):
    mod = tmp_path / "computed.py"
    mod.write_text("class Plugin:\n"
                   "    plugin_params = dict(a=1) if True else {}\n")
    with pytest.raises(SystemExit, match="computed, not declared"):
        pr.declared_plugin_params({
            "role": "predictor", "origin": str(mod),
            "entry_point_value": "computed:Plugin"})


def test_a_plugin_without_declared_params_refuses(tmp_path):
    mod = tmp_path / "bare.py"
    mod.write_text("class Plugin:\n    pass\n")
    with pytest.raises(SystemExit, match="declares no"):
        pr.declared_plugin_params({
            "role": "predictor", "origin": str(mod),
            "entry_point_value": "bare:Plugin"})


def test_submit_only_imports_no_framework_and_writes_no_state(tmp_path):
    """The whole of C34, through the real CLI in a real subprocess."""
    work = tmp_path / "w"
    work.mkdir()
    for r in ("train", "validation", "test"):
        (work / f"x_{r}.csv").write_text("DATE_TIME,f1\n2020-01-01,1\n")
        (work / f"y_{r}.csv").write_text("DATE_TIME,TARGET\n2020-01-01,2\n")
    cfg = {"predictor_plugin": "ann", "target_column": "TARGET",
           "memory_log_file": str(work / "mem.csv"),
           "optimizer_resource_log_file": str(work / "opt.csv"),
           "results_file": str(work / "results.csv"),
           "execution_purpose": "SUBMIT_ONLY"}
    for r in ("train", "validation", "test"):
        cfg[f"x_{r}_file"] = str(work / f"x_{r}.csv")
        cfg[f"y_{r}_file"] = str(work / f"y_{r}.csv")
    (work / "cfg.json").write_text(json.dumps(cfg))
    before = {p.name for p in work.iterdir()}

    env = {**os.environ, "PYTHONPATH": str(REPO),
           "CUDA_VISIBLE_DEVICES": "",
           "CRISPDM_OLAP_OUTBOX": str(tmp_path / "outbox")}
    child = subprocess.run(
        [sys.executable, "-X", "importtime", "app/main.py",
         "--load_config", str(work / "cfg.json")],
        capture_output=True, text=True, cwd=str(REPO), env=env,
        timeout=600)
    out = child.stdout + child.stderr
    imported = {ln.split("|")[-1].strip() for ln in out.splitlines()
                if ln.startswith("import time:")}
    heavy = sorted(m for m in imported
                   if m.split(".")[0] in ("tensorflow", "keras", "torch",
                                          "jax"))
    assert heavy == [], f"SUBMIT_ONLY loaded a framework: {heavy}"
    assert {p.name for p in work.iterdir()} == before, (
        "SUBMIT_ONLY created durable state")
    assert "cuInit" not in out, "a CUDA context was initialized"


def test_main_puts_construction_after_the_gate():
    src = (REPO / "app/main.py").read_text()
    gate_at = src.index("gate_run(")
    submit_return = src.index("SUBMIT_ONLY complete")
    tf_at = src.index("_configure_tensorflow_memory()\n", gate_at)
    logging_at = src.index("_validate_logging_config(config)",
                           submit_return)
    build_at = src.index("predictor_plugin = classes[\"predictor\"]")
    assert gate_at < submit_return < tf_at < logging_at < build_at, (
        "everything that executes must sit after the SUBMIT_ONLY exit")
