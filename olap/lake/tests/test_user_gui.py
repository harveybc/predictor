import sqlite3

from app.config import DEFAULT_VALUES
from app.main import assemble

TOKEN = "test-lake-token"


def _client(tmp_path):
    db = tmp_path / "c.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE fact_performance (ts TEXT, metric TEXT, value REAL)")
    conn.execute("INSERT INTO fact_performance VALUES ('2024-01-01','MAE',1)")
    conn.commit()
    conn.close()
    config = dict(DEFAULT_VALUES)
    config.update(
        {
            "sqlite_path": str(db),
            "secret_key": "t",
            "time_column": "ts",
            "lake_service_token": TOKEN,
        }
    )
    plugins = assemble(config)
    app = plugins["web"].create_app({"config": config, "plugins": plugins})
    app.config["TESTING"] = True
    return app.test_client()


def _h():
    return {"Authorization": f"Bearer {TOKEN}"}


def test_gui_lists_tables(tmp_path):
    client = _client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    assert b"fact_performance" in page.data


def test_api_requires_token(tmp_path):
    client = _client(tmp_path)
    assert client.get("/api/v1/discover").status_code == 401


def test_api_query(tmp_path):
    client = _client(tmp_path)
    ok = client.get(
        "/api/v1/query",
        query_string={"sql": "SELECT ts, value FROM fact_performance LIMIT 10"},
        headers=_h(),
    )
    assert ok.status_code == 200
    assert ok.get_json()["rows"]
    bad = client.get(
        "/api/v1/query",
        query_string={"sql": "DROP TABLE fact_performance"},
        headers=_h(),
    )
    assert bad.status_code == 400
    nolimit = client.get(
        "/api/v1/query",
        query_string={"sql": "SELECT ts FROM fact_performance"},
        headers=_h(),
    )
    assert nolimit.status_code == 400
    sleep = client.get(
        "/api/v1/query",
        query_string={"sql": "SELECT pg_sleep(1) LIMIT 1"},
        headers=_h(),
    )
    assert sleep.status_code == 400
