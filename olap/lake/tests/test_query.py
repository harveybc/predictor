import sqlite3

from query_plugins.sql_query import Plugin


def test_select_limit_holdout_and_reject_write(tmp_path):
    db = tmp_path / "c.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE fact_performance (ts TEXT, metric TEXT, value REAL)"
    )
    conn.execute("INSERT INTO fact_performance VALUES ('2024-01-01','MAE',1)")
    conn.execute("INSERT INTO fact_performance VALUES ('2025-06-01','MAE',9)")
    conn.commit()
    conn.close()
    q = Plugin()
    q.set_params(sqlite_path=str(db), holdout_start="2025-01-01", time_column="ts")
    names = {r["resource_id"] for r in q.discover()}
    assert "fact_performance" in names
    out = q.query("SELECT ts, value FROM fact_performance WHERE ts < '2025-01-01' LIMIT 10")
    assert out["rows"][0]["value"] == 1
    try:
        q.query("DELETE FROM fact_performance")
        assert False
    except ValueError:
        pass
    try:
        q.query("SELECT ts FROM fact_performance")
        assert False
    except ValueError as exc:
        assert "LIMIT" in str(exc)
    try:
        q.query("SELECT pg_sleep(1) LIMIT 1")
        assert False
    except ValueError:
        pass
    try:
        q.query("SELECT ts, value FROM fact_performance WHERE ts >= '2025-01-01' LIMIT 10")
        assert False
    except PermissionError:
        pass
