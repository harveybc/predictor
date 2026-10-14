"""Independent regression: honor the deployed query limit without truncation."""
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from tools.i6dmo.outbox import HttpWarehouse
from predictor_olap_store.query import MAX_LIMIT


def client(monkeypatch, count=2, fetched=2):
    monkeypatch.setenv("I6DMO_TEST_TOKEN", "synthetic-only")
    calls = []

    def transport(method, url, **kwargs):
        sql = parse_qs(urlsplit(url).query)["sql"][0]
        calls.append(sql)
        limit = int(sql.rsplit("LIMIT ", 1)[1])
        if limit > MAX_LIMIT:
            return 400, b'{}'
        if "COUNT(*)" in sql:
            rows = [{"row_count": count}]
        else:
            rows = [{"metric": "fixture", "value": i} for i in range(fetched)]
        return 200, json.dumps({"rows": rows}).encode()

    return HttpWarehouse("http://fixture.invalid", "I6DMO_TEST_TOKEN", transport), calls


def test_readback_obeys_shared_query_limit_and_checks_cardinality(monkeypatch):
    api, calls = client(monkeypatch)
    assert len(api.query_metrics("a" * 64)) == 2
    assert len(calls) == 2
    assert "COUNT(*)" in calls[0]
    assert calls[1].endswith(f"LIMIT {MAX_LIMIT}")


def test_readback_overflow_is_refused_not_truncated(monkeypatch):
    api, calls = client(monkeypatch, count=MAX_LIMIT + 1)
    with pytest.raises(RuntimeError, match="WAREHOUSE_READBACK_OVERFLOW"):
        api.query_metrics("a" * 64)
    assert len(calls) == 1


def test_readback_wrong_cardinality_never_closes(monkeypatch):
    api, _ = client(monkeypatch, count=3, fetched=2)
    with pytest.raises(RuntimeError, match="WAREHOUSE_READBACK_COUNT_MISMATCH"):
        api.query_metrics("a" * 64)


def test_readback_rejects_invalid_count(monkeypatch):
    api, _ = client(monkeypatch, count=True)
    with pytest.raises(RuntimeError, match="WAREHOUSE_READBACK_COUNT_INVALID"):
        api.query_metrics("a" * 64)
