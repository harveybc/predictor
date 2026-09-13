import sqlite3

from app.config import DEFAULT_VALUES
from app.main import assemble


def _client(tmp_path):
    db = tmp_path / "c.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE fact_performance (ts TEXT, metric TEXT, value REAL)")
    conn.execute("INSERT INTO fact_performance VALUES ('2024-01-01','MAE',1)")
    conn.commit()
    conn.close()
    config = dict(DEFAULT_VALUES)
    config.update({"sqlite_path": str(db), "secret_key": "t", "time_column": "ts"})
    plugins = assemble(config)
    app = plugins["web"].create_app({"config": config, "plugins": plugins})
    app.config["TESTING"] = True
    return app.test_client()


def test_gui_lists_tables(tmp_path):
    client = _client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    assert b"fact_performance" in page.data
    assert b"read-only" in page.data.lower() or b"OLAP" in page.data


def test_api_query(tmp_path):
    client = _client(tmp_path)
    ok = client.get(
        "/api/v1/query",
        query_string={"sql": "SELECT ts, value FROM fact_performance"},
    )
    assert ok.status_code == 200
    assert ok.get_json()["rows"]
    bad = client.get("/api/v1/query", query_string={"sql": "DROP TABLE fact_performance"})
    assert bad.status_code == 400
