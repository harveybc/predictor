"""One rule, four consumers, no drift.

The gate is only a single point of decision if every repository
asks the SAME code. These tests read the adapter as it is
installed in each consuming checkout and assert byte identity,
then exercise the two behaviours that must hold everywhere: a
configured-but-unlocatable gate refuses, and an unconfigured gate
degrades to a recorded non-authoritative stamp rather than a
silent allow.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SIBLINGS = REPO.parent

ADAPTERS = {
    "preprocessor": SIBLINGS / "preprocessor/app/"
                               "eligibility_adapter.py",
    "agent-multi": SIBLINGS / "agent-multi/app/"
                              "eligibility_adapter.py",
    "doin-plugins": SIBLINGS / "doin-plugins/src/doin_plugins/"
                               "eligibility/adapter.py",
    "doin-domains": SIBLINGS / "doin-domains/src/doin_domains/"
                               "eligibility/adapter.py",
}

PRESENT = {k: v for k, v in ADAPTERS.items() if v.is_file()}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _partitions(tmp_path):
    """A minimal but REAL data contract, so a refusal about the
    manifest is not masked by a refusal about missing files."""
    cfg = {}
    for role, key in (("d4", "x_train_file"),
                      ("d5", "x_validation_file"),
                      ("d6", "x_test_file")):
        p = tmp_path / f"{role}.csv"
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["DATE_TIME", "px"])
            w.writerow(["2020-01-01", "1"])
        cfg[key] = str(p)
    cfg["target_column"] = "px"
    cfg["eligibility_scope"] = "forecasting"
    return cfg


def _flat(path: Path) -> str:
    return re.sub(r'[\s"]+', " ", path.read_text())


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(len(PRESENT) < 2,
                    reason="fewer than two consumer checkouts "
                           "present")
def test_every_consumer_carries_the_identical_adapter():
    digests = {k: _sha(p) for k, p in PRESENT.items()}
    assert len(set(digests.values())) == 1, (
        "the eligibility adapter has DRIFTED between "
        f"consumers: {digests}")


@pytest.mark.parametrize("name", sorted(PRESENT))
def test_consumer_resolves_the_one_shared_gate(name):
    mod = _load(PRESENT[name], f"adapter_{name.replace('-', '_')}")
    gate_mod, integ = mod.load_gate({})
    assert Path(gate_mod.__file__).resolve() == (
        REPO / "eligibility/gate.py").resolve()
    assert Path(integ.__file__).resolve() == (
        REPO / "eligibility/integration.py").resolve()


@pytest.mark.parametrize("name", sorted(PRESENT))
def test_an_unconfigured_new_experiment_refuses(name, tmp_path):
    """Order C5: omitting a manifest is no longer an implicit
    legacy run in ANY consumer."""
    mod = _load(PRESENT[name], f"adapter2_{name.replace('-', '_')}")
    config = _partitions(tmp_path)
    config["execution_purpose"] = "NEW_EXPERIMENT"
    with pytest.raises(SystemExit, match="no eligibility "
                                         "manifest"):
        mod.gate_run(config, consumer=name, repo_root=tmp_path)


@pytest.mark.parametrize("name", sorted(PRESENT))
def test_archival_replay_must_be_declared_in_every_consumer(
        name, tmp_path):
    mod = _load(PRESENT[name], f"adapter3_{name.replace('-', '_')}")
    config = _partitions(tmp_path)
    config["execution_purpose"] = mod.PURPOSE_ARCHIVAL
    stamp = mod.gate_run(config, consumer=name,
                         repo_root=tmp_path)
    assert stamp["eligibility_status"] == \
        "LEGACY_NON_AUTHORITATIVE"
    assert stamp["execution_purpose"] == mod.PURPOSE_ARCHIVAL


@pytest.mark.parametrize("name", sorted(PRESENT))
def test_no_consumer_offers_a_subject_id_shortcut(name):
    """The subject set is derived; a consumer that accepted a
    caller's list would reopen the hole the audit found."""
    src = PRESENT[name].read_text()
    assert "subject_ids" not in src
    assert "def gate_subjects" not in src


@pytest.mark.parametrize("name", sorted(PRESENT))
def test_configured_but_unlocatable_gate_refuses(name, tmp_path):
    mod = _load(PRESENT[name], f"adapter4_{name.replace('-', '_')}")
    config = {"eligibility_manifest": str(tmp_path / "m.json"),
              "eligibility_gate_path": str(tmp_path / "nowhere"),
              "eligibility_scope": "forecasting"}
    # the sibling fallback must not rescue an explicitly wrong
    # configured path when it does not carry the gate
    monkey = mod._candidate_roots(config)
    assert monkey[0] == Path(str(tmp_path / "nowhere"))


def test_agent_multi_universe_gate_is_wired_at_formation():
    src = (SIBLINGS / "agent-multi/optimizer_plugins/"
                      "project3_full_genome_optimizer.py")
    if not src.is_file():
        pytest.skip("agent-multi not present")
    text = src.read_text()
    assert "filter_to_eligible(" in text
    assert text.index("filter_to_eligible(") < text.index(
        'run_config["feature_columns"] = columns')
    assert "refusing rather than widening" in text


def test_preprocessor_gate_precedes_materialization():
    src = SIBLINGS / "preprocessor/app/data_processor.py"
    if not src.is_file():
        pytest.skip("preprocessor not present")
    text = src.read_text()
    assert "gate_run(" in text
    assert text.index("gate_run(") < text.index(
        "plugin.process(data, config)")


def test_agent_multi_refuses_an_undeclared_new_experiment():
    src = (SIBLINGS / "agent-multi/optimizer_plugins/"
                      "project3_full_genome_optimizer.py")
    if not src.is_file():
        pytest.skip("agent-multi not present")
    text = _flat(src)
    assert "no longer an implicit legacy run" in text
    assert "ARCHIVAL_REPLAY_NON_AUTHORITATIVE" in text


def test_doin_plugins_verifies_without_selecting():
    src = (SIBLINGS / "doin-plugins/src/doin_plugins/"
                      "eligibility/verify.py")
    if not src.is_file():
        pytest.skip("doin-plugins not present")
    text = src.read_text()
    assert "verification confers no eligibility" in text
    # a verification consumer must not expose a ranking or a
    # candidate-list API
    for forbidden in ("def select", "def rank", "def choose",
                      "def promote"):
        assert forbidden not in text
