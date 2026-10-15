"""R0-vs-R2 operator path without TensorFlow: runbook commands, preparation refusals, init refusals."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATCHED = ROOT / "docs/i6n/direct_multi_output_static/matched_short"
CONTRACT = MATCHED / "R2_CONTRAST_DESIGN.ruled.json"


def test_run_r2_document_names_only_existing_commands():
    """Every `python -m tools.i6dmo ...` line in RUN_R2.md is a parser-accepted command."""
    import re
    import shlex
    from tools.i6dmo import cli
    text = (MATCHED / "RUN_R2.md").read_text().replace("\\\n", " ")
    commands = re.findall(r"python -m tools\.i6dmo (?!\.)([^\n`]+)", text)
    assert len(commands) >= 10
    assert {c.split()[0] for c in commands} >= {"r2", "deploy", "status"}
    assert {c.split()[1] for c in commands if c.startswith("r2 ")} == {"prepare", "validate", "preflight", "init",
                                                                       "status"}
    parser = cli.build_parser()
    for command in commands:
        parser.parse_args(shlex.split(command.split("#")[0]))  # SystemExit on any unknown command/flag


def test_r2_prepare_refuses_test_and_unbound_preparations(tmp_path):
    from tools.i6dmo import matched_runner as MR, r2_runner as R
    with pytest.raises(MR.PrepareRefused, match="TEST_REQUEST"):
        R.prepare(deployment=tmp_path / "d.json", input_dir=tmp_path, out=tmp_path / "o.json", test=True)
    contract = json.loads(CONTRACT.read_text())
    shared = contract["shared"]
    bound = dict(designs={"mg2m": dict(seal_sha256=shared["experiment_seal_sha256"],
                                       runtime=shared["fit_pinning"]["runtime"])},
                 data_pins={"common_population": {"population_sha256": shared["population_sha256"]}})
    R._bind(contract, bound)  # the committed contract binds the committed mg2m identities
    for mutate, name in ((lambda d: d["designs"]["mg2m"].update(seal_sha256="0" * 64), "design_seal"),
                         (lambda d: d["data_pins"]["common_population"].update(population_sha256="1" * 64),
                          "population"),
                         (lambda d: d["designs"]["mg2m"].update(runtime={"tensorflow": "2.0", "keras": "3.0"}),
                          "runtime")):
        doc = json.loads(json.dumps(bound))
        mutate(doc)
        with pytest.raises(MR.PrepareRefused, match=f"R2_CONTRACT_BINDING:.*{name}"):
            R._bind(contract, doc)


def test_r2_contract_must_be_the_executable_ruled_one(tmp_path):
    from tools.i6dmo import matched_runner as MR, r2_runner as R
    path, doc = R._contract()
    assert doc["seal_sha256"].startswith("0cad8817") and path.name == "R2_CONTRAST_DESIGN.ruled.json"
    tampered = dict(doc, pretraining=dict(doc["pretraining"], objective="UNDECIDED_REQUIRES_REVIEW"))
    (tmp_path / "c.json").write_text(json.dumps(tampered))
    with pytest.raises(MR.PrepareRefused, match="R2_CONTRACT_NOT_EXECUTABLE"):
        R._contract(tmp_path / "c.json")
    with pytest.raises(MR.PrepareRefused, match="MISSING_ARTIFACT"):
        R._contract(tmp_path / "absent.json")


def test_r2_init_refuses_other_arms_reuse_and_existing_roots(tmp_path):
    from tests.i6dmo.ops_fakes import private_spec
    from tools.i6dmo import campaign
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(private_spec(tmp_path)))
    contract = json.loads(CONTRACT.read_text())
    pins = dict(common_population=dict(base="experiment_storage", relpath="population", sha256="a" * 64,
                                       population_sha256=contract["shared"]["population_sha256"]),
                matched_input=dict(relpath="input", sha256="b" * 64))
    with pytest.raises(campaign.Refused, match="R2_ARM_CHANGE"):
        campaign.init_r2_contrast(CONTRACT, MATCHED / "EXPERIMENT.mg1.json", spec, tmp_path / "a",
                                  code_commit="c" * 40, data_pins=pins)
    with pytest.raises(campaign.Refused, match="R0_REUSE_NOT_SUPPORTED_BY_THIS_RUNNER"):
        campaign.init_r2_contrast(CONTRACT, MATCHED / "EXPERIMENT.mg2m.json", spec, tmp_path / "b",
                                  code_commit="c" * 40, data_pins=pins, r0_reuse={"decision": "REUSE_R0"})
    with pytest.raises(campaign.Refused, match="UNFROZEN_COMPARISON_OR_MISSING_PIN"):
        campaign.init_r2_contrast(CONTRACT, MATCHED / "EXPERIMENT.mg2m.json", spec, tmp_path / "c",
                                  code_commit="c" * 40, data_pins={"matched_input": pins["matched_input"]})
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "x").write_text("x")
    with pytest.raises(campaign.Refused, match="ROOT_NOT_NEW"):
        campaign.init_r2_contrast(CONTRACT, MATCHED / "EXPERIMENT.mg2m.json", spec, tmp_path / "d",
                                  code_commit="c" * 40, data_pins=pins)
    out = campaign.init_r2_contrast(CONTRACT, MATCHED / "EXPERIMENT.mg2m.json", spec, tmp_path / "e",
                                    code_commit="c" * 40, data_pins=pins, r0_reuse={"decision": "FRESH_R0_REQUIRED"})
    assert len(out["units"]) == 25 and campaign.load_effective(tmp_path / "e")["r2"]["r0_reuse"] == "FRESH_R0_REQUIRED"
