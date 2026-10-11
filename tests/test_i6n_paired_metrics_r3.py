"""Correction round 3 for I6-N C2: result-level checks (independent verifier, 2026-10-11).

Each test demands a TYPED rejection or a NON-SUCCESS exit for a result whose
groups do not match its contract, or whose contract does not match its digest.
None of them encodes the defective behaviour observed on ``cdc85fb5``.

A forgery here is self-consistent: groups are edited and resealed, the OLAP
rows are regenerated from the edited groups and the result is resealed, so only
a result-level check can refuse it.
"""

import json

import pytest

from tools.i6n_contracts import paired_metrics as pm
from tools.i6n_contracts import paired_metrics_cli as cli
from tests.test_i6n_paired_metrics import (
    ENTRY, EXAMPLE, _completed, _resealed, _snapshot, _status, document, group, rec)

# The unverified row builder: ``_olap_rows`` after the fix, the then-unverified
# public ``olap_records`` on the audited commit (so PRE forgeries stay consistent).
_ROWS = getattr(pm, "_olap_rows", pm.olap_records)


def _reseal_result(body):
    body = {k: v for k, v in body.items() if k != "sha256"}
    if "contract" in body:
        body["olap_records"] = _ROWS(body)
    return dict(body, sha256=pm.canonical_sha256(body))


def _with_groups(result, edit):
    return _reseal_result(dict(result, groups=edit(list(result["groups"]))))


def _moved(g, h=None, role=None):
    """A copy of ``g`` relabelled to another horizon or role, row keys and seal recomputed."""
    h = g["horizon_hours"] if h is None else h
    role = g["role"] if role is None else role
    rows = [[r[0], r[1], h, *([role] if len(r) == 4 else [])] for r in g["row_keys"]]
    return _resealed(g, horizon_hours=h, role=role, row_keys=rows, row_keys_sha256=pm.canonical_sha256(rows))


def _honest():
    return pm.score(document([rec(0, 2)]))


def _no_groups(result):
    return _with_groups(result, lambda gs: [])


def _duplicate(result):
    return _with_groups(result, lambda gs: gs + [gs[0]])


def _missing(result):
    return _with_groups(result, lambda gs: gs[:-1])


def _extra_horizon(result):
    return _with_groups(result, lambda gs: gs + [_moved(gs[0], h=24)])


def _extra_role(result):
    return _with_groups(result, lambda gs: gs + [_moved(gs[0], role="undeclared_role")])


def _replaced_by_undeclared(result):
    """Same group COUNT as the contract, but one group swapped for an undeclared horizon."""
    return _with_groups(result, lambda gs: gs[:-1] + [_moved(gs[0], h=24)])


def _contract_changed_digest_kept(result):
    return _reseal_result(dict(result, contract=dict(result["contract"], unit="ANOTHER_UNIT")))


def _contract_changed_digests_recomputed(result):
    """contract.unit changed AND every contract digest recomputed; groups keep the old unit."""
    contract = dict(result["contract"], unit="ANOTHER_UNIT")
    digest = pm.canonical_sha256(contract)
    groups = [_resealed(g, contract_sha256=digest) for g in result["groups"]]
    return _reseal_result(dict(result, contract=contract, contract_sha256=digest, groups=groups))


def _without_contract(result):
    return _reseal_result({k: v for k, v in result.items() if k != "contract"})


def _doubled_olap_rows(result):
    body = {k: v for k, v in result.items() if k != "sha256"}
    body["olap_records"] = body["olap_records"] + body["olap_records"][:2]
    return dict(body, sha256=pm.canonical_sha256(body))


FORGERIES = {
    "no_groups": (_no_groups, "GROUP_MISSING"),
    "duplicate_group": (_duplicate, "DUPLICATE_GROUP"),
    "missing_group": (_missing, "GROUP_MISSING"),
    "extra_horizon_group": (_extra_horizon, "GROUP_UNDECLARED"),
    "extra_role_group": (_extra_role, "GROUP_UNDECLARED"),
    "undeclared_group_replacing_a_declared_one": (_replaced_by_undeclared, "GROUP_UNDECLARED"),
    "contract_changed_digest_kept": (_contract_changed_digest_kept, "CONTRACT_DIGEST_INVALID"),
    "contract_changed_digests_recomputed": (_contract_changed_digests_recomputed, "RESULT_INCONSISTENT"),
    "missing_contract": (_without_contract, "SCHEMA_INVALID"),
    "doubled_embedded_olap_rows": (_doubled_olap_rows, "RESULT_INCONSISTENT"),
}


def _code(call, *args):
    with pytest.raises(pm.PairedMetricsError) as caught:
        call(*args)
    return caught.value.code


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_r3_verify_result_refuses_a_result_whose_groups_or_contract_disagree(name):
    forge, code = FORGERIES[name]
    assert _code(pm.verify_result, forge(_honest())) == code


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_r3_compare_configs_refuses_with_a_typed_code_never_key_error_or_comparable(name):
    forge, code = FORGERIES[name]
    a = _honest()
    b = forge(pm.score(document([rec(0, 2)], config_id="synthetic-config-b")))
    assert _code(pm.compare_configs, a, b) == code
    assert _code(pm.compare_configs, b, a) == code


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_r3_olap_records_refuses_with_a_typed_code_and_never_double_counts(name):
    forge, code = FORGERIES[name]
    assert _code(pm.olap_records, forge(_honest())) == code


def test_r3_duplicate_group_is_not_published_twice():
    honest = _honest()
    with pytest.raises(pm.PairedMetricsError):
        pm.olap_records(_duplicate(honest))
    assert pm.olap_records(honest) == honest["olap_records"]  # control: one row set per group


def test_r3_expected_group_set_is_kinds_x_horizons_x_roles_plus_all_roles():
    result = pm.score(document([rec(0, 2), rec(0, 72)]))
    keys = [(g["population_kind"], g["horizon_hours"], g["role"]) for g in result["groups"]]
    assert len(keys) == len(set(keys)) == 2 * 2 * 3
    assert set(keys) == {(k, h, r) for k in ("S", "U") for h in (2, 72)
                         for r in ("entry_origin", "management_origin", pm.ALL_ROLES)}
    assert pm.verify_result(result) is result  # control


def test_r3_permuted_groups_of_an_honest_result_still_verify():
    honest = _honest()
    permuted = _with_groups(honest, lambda gs: list(reversed(gs)))
    assert pm.verify_result(permuted)["state"] == "COMPLETE"


def test_r3_contract_with_a_description_verifies_and_its_digest_is_the_input_digest():
    doc = json.loads(EXAMPLE.read_text())
    assert "description" in doc["contract"]
    result = pm.score(doc)
    assert result["contract_sha256"] == pm.input_digests(doc)["config_sha256"]
    assert result["contract_sha256"] == pm.canonical_sha256(result["contract"])
    pm.verify_result(result)
    changed = _reseal_result(dict(result, contract=dict(result["contract"], description="edited")))
    assert _code(pm.verify_result, changed) == "CONTRACT_DIGEST_INVALID"


def test_r3_honest_pair_still_compares():
    a = _honest()
    b = pm.score(document([rec(0, 2, pred=1.5)], config_id="synthetic-config-b"))
    assert pm.compare_configs(a, b)["state"] == "COMPARABLE"


def _forge_output(out, forge):
    wrapped = json.loads((out / cli.RESULT_NAME).read_text())
    wrapped["result"] = forge(wrapped["result"])
    wrapped = _resealed(wrapped)
    (out / cli.RESULT_NAME).write_text(json.dumps(wrapped, indent=2))
    status = json.loads((out / cli.STATUS_NAME).read_text())
    groups = wrapped["result"].get("groups", [])
    status["result"] = {
        "result_state": wrapped["result"]["state"], "result_file": cli.RESULT_NAME,
        "result_sha256": wrapped["result"]["sha256"],
        "result_file_sha256": cli.file_hash(out / cli.RESULT_NAME),
        "groups_total": len(groups),
        "groups_by_state": {s: sum(1 for g in groups if g["state"] == s)
                            for s in sorted({g["state"] for g in groups})}}
    (out / cli.STATUS_NAME).write_text(json.dumps(status, indent=2))


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_r3_status_never_reports_success_for_a_resealed_forgery(name, tmp_path, capsys):
    forge, code = FORGERIES[name]
    config, out = _completed(tmp_path, document([rec(0, 2)]))
    assert _status(out, capsys)[0] == cli.EXIT_COMPLETE  # control before the forgery
    _forge_output(out, forge)
    before = _snapshot(out)
    exit_code, payload = _status(out, capsys)
    assert exit_code == cli.EXIT_DAMAGED
    assert payload["disposition"] == "RESULT_INVALID" and payload["result"]["verified"] is False
    assert payload["error"]["detail"].startswith(code)
    capsys.readouterr()
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_CORRUPT"
    assert _snapshot(out) == before


def test_r3_group_entry_role_helper_matches_fixture():
    # Guards the fixture assumption used above: the first group is S / h2 / entry role.
    g = _honest()["groups"][0]
    assert (g["population_kind"], g["horizon_hours"], g["role"]) == ("S", 2, ENTRY)
    assert g == group(_honest(), "S", 2)
