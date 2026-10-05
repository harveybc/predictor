"""Behavioral tests for the immutable manifest-backed weekly row provider."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from tools.business_asof_window import SupportSpec, resolve_asof_window
from tools.business_weekly_protocol import EvaluationSplit, WeekSpec
from tools.manifest_weekly_rows import ManifestContractError, ManifestWeeklyRows


UTC = timezone.utc
FIELDS = (
    "record_id",
    "event_time",
    "available_time",
    "target_available_time",
    "row_digest",
)


def dt(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=UTC)


def week() -> WeekSpec:
    return WeekSpec(
        split=EvaluationSplit.VALIDATION,
        ordinal=0,
        start=dt(2024, 1, 8),
        end=dt(2024, 1, 15),
        cutoff=dt(2024, 1, 8),
        fit_start=dt(2020, 1, 8),
        retrain_due=True,
    )


def record(
    record_id: object,
    event_time: object,
    *,
    available_time: object | None = None,
    target_available_time: object | None = None,
    row_digest: object | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "event_time": event_time,
        "available_time": available_time if available_time is not None else event_time,
        "target_available_time": (
            target_available_time
            if target_available_time is not None
            else "2023-01-03T00:00:00+00:00"
        ),
        "row_digest": row_digest or f"sha256:{str(record_id):0>64}"[-71:],
    }


def digest_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    path.write_bytes(payload)
    return digest_bytes(payload)


def write_csv(path: Path, rows: list[dict[str, object]]) -> str:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return digest_bytes(path.read_bytes())


def valid_rows() -> list[dict[str, object]]:
    return [
        record("later", "2023-01-02T00:00:00+00:00", row_digest="sha256:" + "b" * 64),
        record("outside", "2019-12-31T00:00:00+00:00", row_digest="sha256:" + "c" * 64),
        record(
            "earlier",
            "2023-01-01T01:00:00+01:00",
            row_digest="sha256:" + "a" * 64,
        ),
    ]


@pytest.mark.parametrize("suffix,writer", [(".jsonl", write_jsonl), (".csv", write_csv)])
def test_returns_window_candidates_in_deterministic_order_and_preserves_digest(
    tmp_path: Path,
    suffix,
    writer,
):
    path = tmp_path / f"rows{suffix}"
    expected_digest = writer(path, valid_rows())

    provider = ManifestWeeklyRows(path, expected_digest)
    first = provider(week())
    second = provider(week())

    assert provider.source_digest == expected_digest
    assert first == second
    assert [row.record_id for row in first] == ["earlier", "later"]
    assert first[0].event_time == dt(2023, 1, 1)
    assert all(week().fit_start <= row.event_time < week().cutoff for row in first)


def test_provider_does_not_hide_rows_that_asof_window_must_reject(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    rows = [
        record(
            "late-feature",
            "2023-01-01T00:00:00+00:00",
            available_time="2023-01-01T00:00:01+00:00",
            row_digest="sha256:" + "d" * 64,
        )
    ]
    provider = ManifestWeeklyRows(path, write_jsonl(path, rows))
    support = SupportSpec(
        input_lookback=timedelta(0),
        target_horizon=timedelta(days=1),
        maximum_holding_support=timedelta(0),
        inner_validation_weeks=1,
    )

    assert [row.record_id for row in provider(week())] == ["late-feature"]
    with pytest.raises(ValueError, match="feature bytes unavailable"):
        resolve_asof_window(week(), provider(week()), support)


def test_rejects_wrong_digest_and_changed_bytes_after_construction(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    source_digest = write_jsonl(path, valid_rows())

    with pytest.raises(ManifestContractError, match="digest mismatch"):
        ManifestWeeklyRows(path, "sha256:" + "0" * 64)

    provider = ManifestWeeklyRows(path, source_digest)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ManifestContractError, match="source changed"):
        provider(week())


def test_rejects_symlink_initially_and_symlink_substitution(tmp_path: Path):
    target = tmp_path / "target.jsonl"
    source_digest = write_jsonl(target, valid_rows())
    link = tmp_path / "link.jsonl"
    link.symlink_to(target)

    with pytest.raises(ManifestContractError, match="symlink"):
        ManifestWeeklyRows(link, source_digest)

    provider = ManifestWeeklyRows(target, source_digest)
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement)
    with pytest.raises(ManifestContractError, match="symlink|source changed"):
        provider(week())


def test_rejects_naive_parent_traversal_even_when_target_exists(tmp_path: Path):
    child = tmp_path / "child"
    child.mkdir()
    path = tmp_path / "rows.jsonl"
    source_digest = write_jsonl(path, valid_rows())
    traversing = child / ".." / "rows.jsonl"

    with pytest.raises(ManifestContractError, match="traversal"):
        ManifestWeeklyRows(traversing, source_digest)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda row: row.pop("available_time"), "availability"),
        (lambda row: row.__setitem__("target_available_time", None), "availability"),
        (lambda row: row.__setitem__("event_time", "2023-01-01T00:00:00"), "timezone"),
        (lambda row: row.__setitem__("record_id", True), "record_id"),
        (lambda row: row.__setitem__("available_time", float("nan")), "non-finite"),
        (lambda row: row.__setitem__("row_digest", "not-a-digest"), "row_digest"),
    ],
)
def test_rejects_missing_boolean_nonfinite_and_malformed_fields(
    tmp_path: Path,
    mutation,
    reason,
):
    path = tmp_path / "rows.jsonl"
    row = valid_rows()[0]
    mutation(row)
    source_digest = write_jsonl(path, [row])

    with pytest.raises(ManifestContractError, match=reason):
        ManifestWeeklyRows(path, source_digest)


@pytest.mark.parametrize("duplicate_field", ["record_id", "row_digest"])
def test_rejects_duplicate_record_or_content_identity(tmp_path: Path, duplicate_field):
    path = tmp_path / "rows.jsonl"
    rows = valid_rows()[:2]
    rows[1][duplicate_field] = rows[0][duplicate_field]

    with pytest.raises(ManifestContractError, match=f"duplicate {duplicate_field}"):
        ManifestWeeklyRows(path, write_jsonl(path, rows))


def test_rejects_atomic_regular_file_substitution_even_with_identical_bytes(tmp_path: Path):
    path = tmp_path / "rows.csv"
    source_digest = write_csv(path, valid_rows())
    provider = ManifestWeeklyRows(path, source_digest)
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)

    with pytest.raises(ManifestContractError, match="source changed"):
        provider(week())
