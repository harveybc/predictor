"""Offline counterexamples for the A1-A5 candidate; no services or real data written."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.financial_checkout.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "1b4a23431a0607eb130a3ac7b25cee0a66a6e378"
    sys.path.insert(0, str(root / "store/src"))
    from financial_data_store.inventory import availability_scope
    tests = load("contract_candidate_tests", root / "store/tests/test_ethusdt_4h_contract.py")
    derive = load("derive_candidate", root / "store/tools/derive_availability_contract.py")
    live = dict(tests.CONTRACT["availability"], use_class="LIVE_EQUIVALENT")
    live_result = availability_scope(live)
    record = {"revision": revision, "scope": "OFFLINE_SYNTHETIC_REVIEW_NOT_SCIENTIFIC",
              "live_class_is_accepted_by_scope_validator": live_result["use_class"] == "LIVE_EQUIVALENT"}
    invalid = {"measured": False, "event_column": {"timezone_in_schema": "UTC", "is_datetime": True},
               "event_duplicates": 9, "event_monotonic_increasing": False,
               "available_at_or_after_event_always": False}
    result = derive.candidate_contract("open_time", "close_time", "4h", invalid)
    record["invalid_facts_still_generate_zero_lag_contract"] = result["availability"]["completion_lag_max"] == "0s"
    with tempfile.TemporaryDirectory(prefix="contract-review-") as tmp:
        work = Path(tmp)
        row = tests.bar("2024-01-01T20:00:00Z")
        row["received_time"] = tests.pd.Timestamp("2024-01-02T00:10:00Z")
        dataset_root, resource = tests.write(work / "delayed", [row])
        backend = tests.store(dataset_root, resource, work / "delayed")
        raw, info = tests.delivered(backend, resource, start="2024-01-01", end="2024-01-01")
        selected = tests.rows_of(raw, work, "delayed.parquet")
        record["day_one_cut_includes_row_received_on_day_two"] = len(selected) == 1
        record["delayed_cut_sha256"] = info["sha256"]
        partial = tests.bar("2024-01-01T20:00:00Z", span=tests.pd.Timedelta(0))
        partial["is_final"] = False
        dataset_root, resource = tests.write(work / "partial", [partial])
        raw, info = tests.delivered(tests.store(dataset_root, resource, work / "partial"),
                                    resource, start="2024-01-01", end="2024-01-01")
        selected = tests.rows_of(raw, work, "partial.parquet")
        record["zero_span_explicitly_nonfinal_bar_is_delivered"] = len(selected) == 1
    checks = [v for k, v in record.items() if isinstance(v, bool)]
    assert len(checks) == 4 and all(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
