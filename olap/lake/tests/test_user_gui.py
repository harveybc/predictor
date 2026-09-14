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
            "operator_config_path": str(tmp_path / "pending.json"),
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


def test_schema_page_and_api(tmp_path):
    client = _client(tmp_path)
    page = client.get('/resources/fact_performance')
    assert page.status_code == 200
    assert b'metric' in page.data and b'TEXT' in page.data
    assert client.get('/api/v1/schema?resource=fact_performance').status_code == 401
    response = client.get('/api/v1/schema?resource=fact_performance', headers=_h())
    assert response.status_code == 200
    assert len(response.get_json()['columns']) == 3
    assert client.get('/resources/missing').status_code == 404


def test_configuration_validates_then_stages(tmp_path):
    import json
    client = _client(tmp_path)
    response = client.post('/config', data={'holdout_start': 'not-a-date'}, follow_redirects=True)
    assert b'Invalid configuration' in response.data
    assert not (tmp_path / 'pending.json').exists()
    response = client.post('/config', data={
        'holdout_start': '2026-01-01', 'title': 'Research warehouse',
        'schema': 'public', 'data_gov_url': 'http://127.0.0.1:15055',
    }, follow_redirects=True)
    saved = json.loads((tmp_path / 'pending.json').read_text())
    assert saved['data_gov_url'] == 'http://127.0.0.1:15055'
    assert saved['title'] == 'Research warehouse'
    assert b'Pending restart' in response.data
    assert b'2025-01-01' in response.data


def test_query_results_are_visible(tmp_path):
    client = _client(tmp_path)
    response = client.post('/ops/query', data={'sql': 'SELECT metric, value FROM fact_performance LIMIT 2'})
    assert response.status_code == 200
    assert b'MAE' in response.data
    assert b'Result' in response.data


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
