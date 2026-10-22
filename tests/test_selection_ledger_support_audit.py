import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("support_audit", Path(__file__).parents[1] / "tools/selection_ledger_support_audit.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(key, arm, **extra):
    return dict(key=key, arm=arm, stage="contrast", task="task", model="hgb", family="fam", ctx="core15", fold=0,
                support_hash="same", val_hash="same", rows_train=100, rows_val=50, seed=1, k_folds=3, **extra)


def test_equal_population_is_paired():
    r = module.audit('\n'.join(map(json.dumps, [row("a", "none"), row("b", "with")])))
    assert r["stages"]["contrast"]["mismatched_or_unverifiable_groups"] == 0


def test_different_dates_detected_even_with_same_row_counts():
    a, b = row("a", "none"), row("b", "with")
    b["val_hash"] = "different"
    r = module.audit('\n'.join(map(json.dumps, [a, b])))
    assert r["stages"]["contrast"]["mismatched_or_unverifiable_groups"] == 1


def test_missing_hash_does_not_prove_pairing():
    a, b = row("a", "none"), row("b", "with")
    del a["support_hash"]
    del b["support_hash"]
    r = module.audit('\n'.join(map(json.dumps, [a, b])))
    assert r["stages"]["contrast"]["mismatched_or_unverifiable_groups"] == 1


def test_incomplete_encoding_is_not_misreported_as_complete():
    a = row("a", "base")
    a["stage"] = "encoding"
    r = module.audit(json.dumps(a))
    assert r["stages"]["encoding"]["groups_with_multiple_arms"] == 0


def test_last_attempt_is_used_but_attempt_count_preserved():
    a, b = row("a", "none"), row("b", "with")
    broken = {**b, "val_hash": "old"}
    r = module.audit('\n'.join(map(json.dumps, [a, broken, b])))
    assert r["attempt_rows"] == 3
    assert r["unique_units"] == 2
    assert r["stages"]["contrast"]["mismatched_or_unverifiable_groups"] == 0
