"""R4: a stored result must be bound to the identity of the config it claims.

The run path and ``--status`` hold the wrapper identity (``input_sha256`` and
``config_sha256`` recomputed from the config file). A result whose own
``input_sha256``/``contract_sha256`` differ from that identity is refused, even
when every digest inside the result has been recomputed coherently.
"""

import json

import pytest

from tools.i6n_contracts import paired_metrics as pm
from tools.i6n_contracts import paired_metrics_cli as cli
from tests.test_i6n_paired_metrics import _completed, _resealed, _snapshot, _status, document, rec
from tests.test_i6n_paired_metrics_r3 import _forge_output, _reseal_result


def _input_digest_replaced(result):
    return _reseal_result(dict(result, input_sha256="0" * 64))


def _contract_narrowed(result):
    """Keep only the smallest horizon, with contract, group digests and seals recomputed."""
    contract = dict(result["contract"])
    keep = min(contract["horizons_hours"])
    contract["horizons_hours"] = [keep]
    digest = pm.canonical_sha256(contract)
    groups = [_resealed(g, contract_sha256=digest) for g in result["groups"] if g["horizon_hours"] == keep]
    return _reseal_result(dict(result, contract=contract, contract_sha256=digest, groups=groups))


FORGERIES = {"input_digest_replaced": _input_digest_replaced, "contract_narrowed": _contract_narrowed}


def test_r4_forgeries_verify_in_isolation_so_only_the_identity_binding_can_refuse_them():
    """Precondition: the forged results are internally coherent (verify_result accepts them)."""
    honest = pm.score(document([rec(0, 2)]))
    assert len(honest["contract"]["horizons_hours"]) > 1
    for forge in FORGERIES.values():
        pm.verify_result(forge(honest))


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_r4_status_and_rerun_refuse_a_result_not_bound_to_the_config_identity(name, tmp_path, capsys):
    config, out = _completed(tmp_path, document([rec(0, 2)]))
    assert _status(out, capsys)[0] == cli.EXIT_COMPLETE  # control before the forgery
    _forge_output(out, FORGERIES[name])
    before = _snapshot(out)
    exit_code, payload = _status(out, capsys)
    assert exit_code == cli.EXIT_DAMAGED
    assert payload["disposition"] == "RESULT_INVALID" and payload["result"]["verified"] is False
    assert payload["error"]["detail"].startswith("RESULT_IDENTITY_BINDING_INVALID")
    capsys.readouterr()
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "OUTPUT_CORRUPT" and "RESULT_IDENTITY_BINDING_INVALID" in error["detail"]
    assert _snapshot(out) == before
