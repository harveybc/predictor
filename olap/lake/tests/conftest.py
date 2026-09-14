import pytest

from query_plugins.sql_query import report_sha256


@pytest.fixture
def make_report():
    """A valid §3 report as data-gov forwards it, with report_sha256 set from
    the canonical body. Keyword overrides replace top-level fields."""

    def _make(**over):
        body = {
            "experiment_key": "toy-ann-1d",
            "experiment_set_key": None,
            "actor": "predictor",
            "lake": "olap_cube",
            "config_sha256": "a" * 64,
            "code_commit": "0123abc-dirty",
            "project": "predictor",
            "phase": "phase_1_daily",
            "tags": {"plugin": "ann"},
            "datasets": [
                {
                    "lake": "predictor_examples",
                    "resource": "phase_1/normalized_d4.csv",
                    "sha256": "1" * 64,
                    "role": "x_train_file",
                    "lineage": "VERIFIED",
                    "reason": None,
                    "event_id": 17,
                    "source_sha256": "2" * 64,
                    "from": None,
                    "to": None,
                    "delivery": "AS_IS",
                    "time_column": "DATE_TIME",
                }
            ],
            "metrics": [
                {
                    "metric": "MAE",
                    "value": 0.0065,
                    "split": "train",
                    "horizon": 24,
                    "std_dev": 0.0007,
                    "min_value": 0.0055,
                    "max_value": 0.0071,
                    "unit": None,
                }
            ],
            "lineage": "VERIFIED",
        }
        body.update(over)
        body["report_sha256"] = report_sha256(body)
        return body

    return _make
