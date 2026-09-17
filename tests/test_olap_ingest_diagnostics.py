"""K4: ingestion errors are observable. A known-invalid envelope is refused at the boundary
with a typed, permanent 400; a transient unavailability stays retryable; what the store
answered is persisted beside the pending entry (status, class, bounded reason, attempts)
without secrets; a store that answers 5xx to the same bytes every cycle raises attention;
recovery loads and a second drain duplicates nothing.

Two fixtures: a local HTTP stand-in that answers whatever the rule needs, and the REAL
warehouse service on a disposable DuckDB cube with the packaged loader from this checkout
(`PYTHONPATH=olap/store/src`, the same lever `disposable_route_stack --host-pythonpath` uses).
The real-service rules skip when the store-hosts interpreter is absent.
"""
import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ob = _load("outbox", REPO / "olap" / "outbox.py")
loader = _load("olap_loader_duckdb", REPO / "tools" / "olap_loader_duckdb.py")
CE = _load("campaign_envelope", REPO / "olap" / "campaign_envelope.py")
HOSTS_PYTHON = Path(os.environ.get("STORE_HOSTS_PYTHON",
                                   Path.home() / ".venvs/store-hosts-duckdb-prod/bin/python"))


def item(id_, digest="d" * 64):
    return {"id": id_, "digest": digest, "eligibility_state": "MECHANICAL_EVIDENCE_NON_GOVERNING"}


def envelope(*, consumed=None, key="k4-test"):
    return CE.build_envelope(
        campaign_key=key, producer="predictor", result_class="MECHANICAL",
        identity={"run_id": "k4", "code_identity": "0" * 40, "design_sha256": "1" * 64},
        data_consumed=consumed if consumed is not None else
        {"datasets": [item("u1")], "variables": [item("v0", "UNAVAILABLE")],
         "operators": [item("op", "e" * 64)]},
        partitions={"exposure": "MECHANICAL_NO_SCIENTIFIC_EXPOSURE", "splits": "UNAVAILABLE"},
        budget={"device": "cpu", "wall_seconds": 1.0, "cost_units": 1},
        terminal={"state": "COMPLETED", "adjudication": "MECHANICAL_EVIDENCE_NO_ADJUDICATION"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL", "freeze": "f" * 64},
        units=[{"candidate_key": "c", "cell_key": "u1/v0/op", "metric_name": "d3.verdict",
                "metric_value": 1.0, "terminal_state": "COMPLETED",
                "uncertainty_kind": "NONE"}])


def malformed():
    """The d3mech-v1 first envelope: bare strings under data_consumed."""
    doc = envelope()
    doc["data_consumed"] = {"datasets": ["u1"], "variables": ["v0"], "operators": ["op"]}
    return doc


# --- the boundary --------------------------------------------------------------------------------

def test_bare_string_consumption_items_are_refused_at_the_boundary_as_a_typed_refusal():
    with pytest.raises(CE.EnvelopeRefusal, match="data_consumed.datasets\\[0\\] must be an object"):
        CE.validate_envelope(malformed())
    assert issubclass(CE.EnvelopeRefusal, SystemExit)   # the store's web maps it to 400


def test_units_that_are_not_objects_are_refused_at_the_boundary():
    doc = envelope()
    doc["units"] = ["not an object"]
    with pytest.raises(CE.EnvelopeRefusal, match="units\\[0\\] must be an object"):
        CE.validate_envelope(doc)


def test_a_well_formed_envelope_passes_the_boundary():
    CE.validate_envelope(envelope())


# --- a stand-in store: what the loader records for each answer ------------------------------------

class Answering(http.server.BaseHTTPRequestHandler):
    status = 503
    body = {"error": "database error: whatever"}
    posts = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        type(self).posts.append(json.loads(raw))
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).body).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def stand_in():
    Answering.posts = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answering)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def drain(root, url, token="tok-secret-" + "x" * 50):
    return loader.drain_once(root, url=url, token=token)


def test_a_permanent_refusal_moves_the_entry_to_failed_with_its_typed_reason(tmp_path, stand_in):
    _, url = stand_in
    Answering.status, Answering.body = 400, {"error": "data_consumed.datasets[0] must be an object"}
    ob.emit(malformed(), kind="envelope", root=tmp_path)
    out = drain(tmp_path, url)
    assert out == {"loaded": 0, "failed": 1, "retryable": 0, "counts": {}}
    failed = ob._entries(tmp_path / "failed")
    assert len(failed) == 1
    reason = (tmp_path / "failed" / (failed[0].stem + ".reason")).read_text()
    assert reason.startswith("http 400: data_consumed.datasets[0]")


def test_a_server_error_stays_pending_with_status_class_reason_and_attempts(tmp_path, stand_in):
    _, url = stand_in
    Answering.status = 503
    Answering.body = {"error": "database error: 'str' object has no attribute 'get' Bearer abc"}
    entry = ob.emit(envelope(), kind="envelope", root=tmp_path)
    for n in (1, 2):
        out = drain(tmp_path, url)
        assert out["retryable"] == 1 and out["failed"] == 0
    retries = ob.retrying(tmp_path)
    assert len(retries) == 1
    r = retries[0]
    assert r["entry"] == entry["outbox_entry"]
    assert r["attempts"] == 2 and r["last_status"] == 503 and r["class"] == ob.RETRY_SERVER_ERROR
    assert "no attribute 'get'" in r["reason"] and "Bearer abc" not in r["reason"]
    assert ob.counts(tmp_path)["pending"] == 1          # the sidecar is not an entry
    health = ob.health(tmp_path)
    assert health["retrying"] == 1 and health["attention_required"] is False


def test_the_same_bytes_answered_5xx_every_cycle_raise_attention(tmp_path, stand_in):
    _, url = stand_in
    Answering.status, Answering.body = 503, {"error": "database error: internal"}
    ob.emit(envelope(), kind="envelope", root=tmp_path)
    for _ in range(ob.RETRY_ATTENTION_ATTEMPTS):
        drain(tmp_path, url)
    health = ob.health(tmp_path)
    assert health["retrying_server_error"] == 1 and health["attention_required"] is True
    assert ob.counts(tmp_path)["pending"] == 1          # nothing deleted, nothing disguised


def test_transport_failure_and_bad_credential_are_retryable_with_their_own_class(tmp_path, stand_in):
    server, url = stand_in
    ob.emit(envelope(), kind="envelope", root=tmp_path)
    Answering.status, Answering.body = 401, {"error": "unauthorized"}
    drain(tmp_path, url)
    assert ob.retrying(tmp_path)[0]["class"] == ob.RETRY_AUTH
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    closed_port = sock.getsockname()[1]
    sock.close()
    drain(tmp_path, f"http://127.0.0.1:{closed_port}")
    r = ob.retrying(tmp_path)[0]
    assert r["class"] == ob.RETRY_TRANSPORT and r["last_status"] is None and r["attempts"] == 2


def test_recovery_loads_takes_the_diagnosis_to_receipts_and_a_second_drain_posts_nothing(
        tmp_path, stand_in):
    _, url = stand_in
    Answering.status, Answering.body = 503, {"error": "database error: down"}
    entry = ob.emit(envelope(), kind="envelope", root=tmp_path)
    drain(tmp_path, url)
    Answering.status, Answering.body = 201, {"campaigns": 1, "units": 1}
    out = drain(tmp_path, url)
    assert out["loaded"] == 1 and out["counts"] == {"campaigns": 1, "units": 1}
    assert ob.retrying(tmp_path) == []
    receipts = list((tmp_path / "receipts").glob("*.retry.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["attempts"] == 1
    posts_before = len(Answering.posts)
    again = drain(tmp_path, url)
    assert again == {"loaded": 0, "failed": 0, "retryable": 0, "counts": {}}
    assert len(Answering.posts) == posts_before            # nothing was posted twice
    assert ob.counts(tmp_path) == {"pending": 0, "loaded": 1, "failed": 0}


# --- the real service on a disposable cube --------------------------------------------------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def warehouse(tmp_path):
    if not HOSTS_PYTHON.is_file():
        pytest.skip("store-hosts interpreter absent")
    port = _free_port()
    config = {"store_id": "olap_cube", "title": "k4 disposable cube", "kind": "warehouse",
              "web_host": "127.0.0.1", "web_port": port,
              "backend": {"entry_point": "predictor_duckdb",
                          "distribution": "predictor-duckdb-store",
                          "settings": {"duckdb_path": str(tmp_path / "cube.duckdb"),
                                       "schema": "main", "memory_limit": "512MB", "threads": 1,
                                       "min_free_bytes": 1, "holdout_start": None,
                                       "lake_id": "olap_cube"}}}
    (tmp_path / "warehouse.json").write_text(json.dumps(config))
    token = "k4-disposable-token"
    env = dict(os.environ, DATA_GOV_LAKE_TOKEN=token, PYTHONUNBUFFERED="1",
               PYTHONPATH=str(REPO / "olap" / "store" / "src"))
    for name in ("PGDATABASE", "PGUSER", "PGPASSWORD", "PGHOST", "PGPORT"):
        env.pop(name, None)
    log = (tmp_path / "warehouse.log").open("w")
    proc = subprocess.Popen([str(HOSTS_PYTHON), "-m", "data_warehouse_service.main",
                             "--load_config", str(tmp_path / "warehouse.json")],
                            cwd=str(tmp_path), env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.skip("the disposable warehouse did not start: "
                        + (tmp_path / "warehouse.log").read_text()[-400:])
        try:
            urllib.request.urlopen(f"{url}/healthz", timeout=2)
            break
        except Exception:
            time.sleep(0.2)
    else:
        os.killpg(os.getpgid(proc.pid), 9)
        pytest.skip("the disposable warehouse never answered /healthz")
    yield url, token, proc
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), 9)
    log.close()


def test_the_real_service_refuses_the_malformed_envelope_with_400_not_503(tmp_path, warehouse):
    url, token, _ = warehouse
    status, answer = loader.post_envelope(url, token, malformed())
    assert status == 400, answer
    assert "data_consumed.datasets[0] must be an object" in answer["error"]
    status, answer = loader.post_envelope(url, token, envelope())
    assert status == 201, answer


def test_the_real_service_and_loader_permanent_recovery_and_no_duplicate(tmp_path, warehouse):
    url, token, proc = warehouse
    root = tmp_path / "outbox"
    ob.emit(malformed(), kind="envelope", root=root)
    good = ob.emit(envelope(key="k4-real"), kind="envelope", root=root)
    # 1. permanent refusal typed, the good one loaded
    out = loader.drain_once(root, url=url, token=token)
    assert out["failed"] == 1 and out["loaded"] == 1, out
    # 2. the service goes away: transport failure, retry diagnosis, nothing lost
    later = ob.emit(envelope(key="k4-later"), kind="envelope", root=root)
    os.killpg(os.getpgid(proc.pid), 15)
    proc.wait(timeout=20)
    out = loader.drain_once(root, url=url, token=token)
    assert out["retryable"] == 1
    assert ob.retrying(root)[0]["class"] == ob.RETRY_TRANSPORT
    # 3. recovery: the same service, the same cube; the second drain duplicates nothing
    env = dict(os.environ, DATA_GOV_LAKE_TOKEN=token, PYTHONUNBUFFERED="1",
               PYTHONPATH=str(REPO / "olap" / "store" / "src"))
    proc2 = subprocess.Popen([str(HOSTS_PYTHON), "-m", "data_warehouse_service.main",
                              "--load_config", str(tmp_path / "warehouse.json")],
                             cwd=str(tmp_path), env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                urllib.request.urlopen(f"{url}/healthz", timeout=2)
                break
            except Exception:
                time.sleep(0.2)
        out = loader.drain_once(root, url=url, token=token)
        assert out["loaded"] == 1 and ob.retrying(root) == []
        status, answer = loader.post_envelope(url, token, envelope(key="k4-real"))
        assert status == 201 and answer.get("skipped_existing", 0) >= 1, answer
        assert loader.drain_once(root, url=url, token=token) == \
            {"loaded": 0, "failed": 0, "retryable": 0, "counts": {}}
        assert ob.counts(root) == {"pending": 0, "loaded": 2, "failed": 1}
    finally:
        if proc2.poll() is None:
            os.killpg(os.getpgid(proc2.pid), 9)
