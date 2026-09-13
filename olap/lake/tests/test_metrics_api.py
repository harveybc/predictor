"""POST /api/v1/metrics through the Flask test client."""

from app.config import DEFAULT_VALUES
from app.main import assemble

TOKEN = "test-lake-token"


def _client(tmp_path):
    config = dict(DEFAULT_VALUES)
    config.update({
        "sqlite_path": str(tmp_path / "cube.sqlite"),
        "secret_key": "t",
        "holdout_start": None,
        "lake_service_token": TOKEN,
    })
    plugins = assemble(config)
    app = plugins["web"].create_app({"config": config, "plugins": plugins})
    app.config["TESTING"] = True
    return app.test_client()


def _h():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_metrics_requires_token(tmp_path, make_report):
    client = _client(tmp_path)
    assert client.post("/api/v1/metrics", json=make_report()).status_code == 401
    wrong = client.post(
        "/api/v1/metrics", json=make_report(), headers={"Authorization": "Bearer nope"}
    )
    assert wrong.status_code == 401


def test_metrics_stored_already_stored_and_invalid(tmp_path, make_report):
    client = _client(tmp_path)
    report = make_report()

    first = client.post("/api/v1/metrics", json=report, headers=_h())
    assert first.status_code == 201
    assert first.get_json() == {
        "stored": True, "already_stored": False, "lineage": "VERIFIED",
        "report_sha256": report["report_sha256"],
    }

    second = client.post("/api/v1/metrics", json=dict(report, lineage="UNVERIFIED"), headers=_h())
    assert second.status_code == 200
    assert second.get_json()["already_stored"] is True
    assert second.get_json()["lineage"] == "VERIFIED"

    mismatch = client.post(
        "/api/v1/metrics", json=dict(report, report_sha256="0" * 64), headers=_h()
    )
    assert mismatch.status_code == 400
    assert "mismatch" in mismatch.get_json()["error"]

    empty = client.post("/api/v1/metrics", json=dict(report, metrics=[]), headers=_h())
    assert empty.status_code == 400

    not_json = client.post("/api/v1/metrics", data="[]", headers=_h(),
                           content_type="application/json")
    assert not_json.status_code == 400

    # the SELECT path still sees the gov_* tables and the stored report
    listed = client.get("/api/v1/discover", headers=_h()).get_json()["resources"]
    assert {"gov_report", "gov_metric", "gov_dataset"} <= {r["resource_id"] for r in listed}
    rows = client.get(
        "/api/v1/query",
        query_string={"sql": "SELECT experiment_key, lineage FROM gov_report LIMIT 5"},
        headers=_h(),
    ).get_json()["rows"]
    assert rows == [{"experiment_key": "toy-ann-1d", "lineage": "VERIFIED"}]


def test_metrics_body_limit_is_16_mib(tmp_path):
    client = _client(tmp_path)
    too_big = client.post(
        "/api/v1/metrics", data=b"{" + b" " * (16 * 1024 * 1024) + b"}",
        headers=_h(), content_type="application/json",
    )
    assert too_big.status_code == 413
