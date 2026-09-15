import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "store_flow", Path(__file__).parents[1] / "tools/verify_deployed_store_flow.py")
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


def test_measured_values():
    raw = b"value\n1\n2\n"
    assert flow.measure(raw, flow.sha(raw)) == {
        "rows": 2, "mean": 1.5, "minimum": 1.0, "maximum": 2.0}


def test_wrong_input_identity():
    with pytest.raises(ValueError, match="identity"):
        flow.measure(b"value\n999\n", flow.sha(b"value\n1\n"))


@pytest.mark.parametrize("raw", [b"value\n", b"value\nnan\n", b"value\ninf\n"])
def test_empty_or_nonfinite_input(raw):
    with pytest.raises(ValueError, match="finite"):
        flow.measure(raw, flow.sha(raw))
