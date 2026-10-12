"""Lane B lease service: the coordinator ledger reached by two worker nodes over HTTP (loopback)."""
import pytest

from tools.i6n_swarm import common, lease_service as S, ledger as L


@pytest.fixture
def served(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("s3cret-token\n")
    token_file.chmod(0o600)
    led = L.Ledger(tmp_path / "ledger.sqlite3")
    server = S.start_server(led, "127.0.0.1", 0, token_file)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield led, url, token_file
    server.shutdown()
    server.server_close()


def spec():
    return dict(stage="search", dag_node="search:h12", horizon=12, fold="F1", experiment="e",
                genotype_canonical_sha256="a" * 64, effective_config_sha256="c" * 64,
                search_candidate_sha256="d" * 64, device_class="rtx5090", fixture=True)


def test_two_nodes_share_one_authority(served):
    led, url, token = served
    n1, n2 = S.LedgerClient(url, token), S.LedgerClient(url, token)
    fit = common.sha256_json({"fit": 1})
    n1.ensure_unit(fit, spec())
    a = n1.claim(fit, "n1", "preferred", "b1", "rtx5090", 60)
    assert a["result"] == "CLAIMED"
    assert n2.claim(fit, "n2", "secondary", "b2", "rtx4090", 60)["result"] == "HELD"
    assert n1.renew(fit, a["token"], ttl=60, progress={"epoch": 1}) is True
    doc = dict(fit_id=fit, fold="F1", experiment="e", effective_config_sha256="c" * 64,
               genotype_canonical_sha256="a" * 64, shared_receipt={"status": "ok"},
               candidate_receipt=dict(fold="F1", experiment="e", horizon_hours=12, effective_config_sha256="c" * 64,
                                      genotype_canonical_sha256="a" * 64, seed=6102026))
    assert n1.complete(fit, a["token"], doc) == "ACCEPTED"
    assert n2.claim(fit, "n2", "secondary", "b2", "rtx4090", 60)["result"] == "ADOPT"
    assert led.unit(fit)["state"] == "COMPLETE"


def test_wrong_token_is_refused(served, tmp_path):
    led, url, _ = served
    bad = tmp_path / "bad"
    bad.write_text("nope")
    with pytest.raises(S.LedgerAuthError):
        S.LedgerClient(url, bad).unit("0" * 64)


def test_receipt_mismatch_crosses_the_wire(served):
    led, url, token = served
    client = S.LedgerClient(url, token)
    fit = common.sha256_json({"fit": 2})
    client.ensure_unit(fit, spec())
    a = client.claim(fit, "n1", "preferred", "b1", "rtx5090", 60)
    with pytest.raises(L.ReceiptMismatch):
        client.complete(fit, a["token"], dict(fit_id="f" * 64, fold="F1", experiment="e",
                                              effective_config_sha256="c" * 64, genotype_canonical_sha256="a" * 64,
                                              shared_receipt={}, candidate_receipt={}))


def test_unreachable_ledger_raises_unavailable(tmp_path):
    token = tmp_path / "t"
    token.write_text("x")
    client = S.LedgerClient("http://127.0.0.1:9", token, timeout=0.5)
    with pytest.raises(S.LedgerUnavailable):
        client.unit("0" * 64)


def test_only_whitelisted_methods_are_served(served):
    led, url, token = served
    with pytest.raises(S.LedgerRequestError):
        S.LedgerClient(url, token)._call("_conn", {})
