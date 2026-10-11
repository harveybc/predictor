"""No live requests: exact canonical redirect and frozen holdout scope."""

from datetime import date
from copy import deepcopy
import time

import pytest

from tools import i6n_external_clock_audit_v2_resume as resume

URL = "http://datafeed.dukascopy.com/datafeed/EURUSD/2024/09/28/BID_candles_min_1.bi5"
HTTPS = URL.replace("http:", "https:")


@pytest.mark.parametrize("target", ["https://example.com/x", HTTPS.replace("datafeed.", "www."),
    HTTPS + "?other=1", HTTPS + "#x", HTTPS.replace("/28/", "/29/"),
    HTTPS.replace("https://", "https://user@"), URL, ""])
def test_only_exact_canonical_https_allowed(target):
    with pytest.raises(ValueError):
        resume.redirect_target(URL, target)


class Response:
    def __init__(self, status, location=None):
        self.status_code = status
        self.headers = {"Location": location} if location else {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def iter_content(self, chunk_size):
        yield b"payload"


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_one_redirect_only_and_timeouts():
    session = Session([Response(301, HTTPS), Response(200)])
    trace = []
    assert resume.fetch_holdout(session, date(2024, 10, 28), time.monotonic() + 60, trace) == b"payload"
    assert [c[0] for c in session.calls] == [URL, HTTPS]
    assert all(c[1] == dict(timeout=(5, 15), stream=True, allow_redirects=False) for c in session.calls)
    assert trace[0]["location"] == HTTPS


@pytest.mark.parametrize("second", [Response(301, HTTPS), Response(403), TimeoutError("timeout")])
def test_no_retry_or_alternate_after_failure(second):
    session = Session([Response(301, HTTPS), second])
    with pytest.raises((ValueError, TimeoutError)):
        resume.fetch_holdout(session, date(2024, 10, 28), time.monotonic() + 60, [])
    assert len(session.calls) == 2


def test_cannot_redownload_replays_or_test():
    session = Session([])
    for day in (date(2024, 7, 15), date(2025, 10, 27)):
        with pytest.raises(ValueError):
            resume.fetch_holdout(session, day, time.monotonic() + 60, [])
    assert session.calls == []


def prior_transport():
    planned = [d.isoformat() for d in resume.v1.planned_dates() if d.isoformat() != "2019-01-21"]
    planned += [d.isoformat() for d in resume.v2.holdout_dates()]
    cases = [dict(date=d, status="PASS", evidence_role="replay" if i < 17 else "holdout")
             for i, d in enumerate(planned)]
    cases[17].update(status="INCONCLUSIVE", dukascopy_sha256=None, reason="ReadTimeout: timeout")
    return dict(planned_dates=planned, cases=cases)


def test_only_missing_transport_case_selected_and_22_carried():
    prior = prior_transport()
    before = deepcopy(prior)
    assert [c["date"] for c in resume.pending_cases(prior, ["2019-10-28"])] == ["2019-10-28"]
    assert prior == before
    assert sum(c["status"] == "PASS" for c in prior["cases"]) == 22
    with pytest.raises(ValueError, match="authorization"):
        resume.pending_cases(prior, ["2024-10-28"])


@pytest.mark.parametrize("updates", [
    dict(reason="price mismatch", available_endpoints=24, matched_exact=23),
    dict(reason="ValueError: HTTP 403"), dict(reason="ValueError: malformed binary"),
    dict(dukascopy_sha256="a" * 64), dict(evidence_role="replay"),
])
def test_numeric_and_nontransport_failures_never_retried(updates):
    prior = prior_transport()
    prior["cases"][17].update(updates)
    with pytest.raises(ValueError):
        resume.pending_cases(prior, ["2019-10-28"])
