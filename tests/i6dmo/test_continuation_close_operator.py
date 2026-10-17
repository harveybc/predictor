import importlib.util
from pathlib import Path


def module():
    path=Path(__file__).parents[2]/"tools/i6dmo/continuation_close_operator.py"
    s=importlib.util.spec_from_file_location("close_operator",path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def test_missing_head_support_is_displayed_not_multiplied_or_fabricated():
    m=module()
    result={"a":{"state":"COMPLETE","metrics":{"VALIDATION":[
        {"horizon":2,"relative_error":-.01,"n":10},{"horizon":12,"relative_error":None,"n":0}]}}}
    out=m.render([{"id":"a"}],result)
    assert "h2: -1.000000%" in out and "h12: UNAVAILABLE (n=0)" in out
    assert result["a"]["metrics"]["VALIDATION"][1]["relative_error"] is None


def test_pending_and_failed_cells_are_not_given_zero_error():
    out=module().render([{"id":"pending"},{"id":"failed"}],{"failed":{"state":"FAILED","reason":"fixture"}})
    assert "PENDING" in out and "FAILED" in out and "0.000000%" not in out


def test_checksum_queries_are_bounded_and_retain_every_file():
    m=module();files={f"results/{i}/artifact":str(i) for i in range(300)}
    chunks=m.checksum_chunks(files)
    assert max(map(len,chunks))<=64 and [p for c in chunks for p in c]==sorted(files)


def test_remote_checksum_missing_or_wrong_file_is_not_accepted():
    import pytest
    m=module();expected={"a":"abc","b":"def"}
    m.verify_checksums(expected,"abc  a\ndef  b\n")
    with pytest.raises(ValueError,match="MIRROR"):
        m.verify_checksums(expected,"abc  a\n")
