"""CPU audit against normally installed news-signal and calendar source.

Public installed CLI uses the declared NON_MODEL_FIXTURE. No GPU, feeds,
brokers, weights or historical scientific arrays are used. Corruption probes
write only inside TemporaryDirectory. --out contains measurements, not fixes.
"""
import argparse
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    for name in ("news", "feature", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    from news_signal.collector import DurableQueue, RecordedDirectorySource, collect
    from news_signal.shadow import ShadowStore
    corpus = args.news / "examples/eurusd/events"
    event = json.loads((corpus / "00_relevant.json").read_text())
    event.pop("received_at")
    received = "2026-09-24T11:50:30Z"
    out = {"scope": "CPU fixtures, clean installed CLI, no model inference or live services"}
    with tempfile.TemporaryDirectory(prefix="musashi-cl14-") as tmp:
        root = Path(tmp)
        env = {k: v for k, v in os.environ.items() if not k.startswith("NEWS_SIGNAL_") and k != "PYTHONPATH"}
        env.update(NEWS_SIGNAL_BACKEND="fixture", NEWS_SIGNAL_DEVICE="cpu", CUDA_VISIBLE_DEVICES="")
        cli = str(Path(sys.executable).with_name("news-signal"))
        command = [cli, "ask", "--input", str(corpus / "00_relevant.json"),
                   "--question", "Which economy is named?", "--option=us=United States",
                   "--option=eu=Euro area", "--option=none=Neither", "--as-of", "2026-09-24T12:00:00Z",
                   "--store", str(root / "shadow")]
        def ask():
            done = subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=30)
            return json.loads(done.stdout)
        response = ask()
        recovered = subprocess.run([cli, "replay", "--store", str(root / "shadow")], env=env,
                                   capture_output=True, text=True, check=True, timeout=30)
        out["clean_installed_ask"] = {"status": response["status"],
            "fixture_marked": "NON_MODEL_FIXTURE" in json.dumps(response),
            "fresh_process_replay": json.loads(recovered.stdout),
            "dependency": json.loads(importlib.metadata.distribution("m5phet").read_text("direct_url.json"))}
        store = ShadowStore(root / "shadow")
        record = store.find_record(response["stored"]["record_sha256"])
        path = store._path(record["record_id"])
        damaged = json.loads(path.read_text())
        damaged["features"]["answer"]["label"] = "CORRUPTED"
        bad_bytes = json.dumps(damaged).encode()
        path.write_bytes(bad_bytes)
        repaired_once = ask()
        once = store._read(path)
        path.write_bytes(bad_bytes)
        repaired_twice = ask()
        twice = store._read(path)
        out["repeated_shadow_corruption"] = {
            "first_disposition": repaired_once["stored"]["disposition"], "first_disk_integrity": once["integrity"],
            "second_disposition": repaired_twice["stored"]["disposition"], "second_disk_integrity": twice["integrity"],
            "returned_digest_matches_disk": repaired_twice["stored"]["record_sha256"] == twice["record_sha256"]}

        queue = DurableQueue(root / "queue")
        entry, _ = queue.offer(event, received_at=received)
        qpath = queue._path(entry["entry_id"])
        damaged = json.loads(qpath.read_text())
        damaged["event"]["headline"] = "CORRUPTED"
        qpath.write_text(json.dumps(damaged))
        blocked = len(queue.pending())
        offered, disposition = queue.offer(event, received_at=received)
        out["corrupt_queue_reoffer"] = {"disposition": disposition,
            "returned_matches_disk": offered["entry_sha256"] == queue.get(entry["entry_id"])["entry_sha256"],
            "disk_integrity": queue.get(entry["entry_id"])["integrity"], "pending": len(queue.pending())}
        queue.retry(entry["entry_id"])
        pending = queue.pending()
        out["retry_resigns_corruption"] = {"pending_before_retry": blocked, "pending_after_retry": len(pending),
            "headline_dispatched": pending[0]["event"]["headline"] if pending else None,
            "integrity_after_retry": queue.get(entry["entry_id"])["integrity"]}

        queue = DurableQueue(root / "substitution")
        original, _ = queue.offer(event, received_at=received)
        other, _ = queue.offer({**event, "event_id": "other-story", "headline": "Different story"}, received_at=received)
        queue._path(original["entry_id"]).write_bytes(queue._path(other["entry_id"]).read_bytes())
        reused, disposition = queue.offer(event, received_at=received)
        out["queue_wrong_key"] = {"disposition": disposition, "requested_event": event["event_id"],
            "returned_event": reused["event_id"], "integrity_failures": queue.report()["integrity_failures"]}
        queue = DurableQueue(root / "sources")
        queue.offer({**event, "source": "wire-a"}, received_at=received)
        second, disposition = queue.offer({**event, "source": "wire-b"}, received_at=received)
        out["source_collision"] = {"disposition": disposition, "revises_another_source": bool(second["revises"]),
                                   "events": queue.report()["events"], "entries": queue.report()["entries"]}
        queue = DurableQueue(root / "recorded")
        first = collect(RecordedDirectorySource(corpus), queue, received_at=received)
        second = collect(RecordedDirectorySource(corpus), queue, received_at=received)
        out["recorded_positive"] = {"first": {k: first[k] for k in ("accepted", "revisions", "duplicates", "refused")},
                                    "second": {k: second[k] for k in ("accepted", "revisions", "duplicates", "refused")},
                                    "report": queue.report()}

    spec = importlib.util.spec_from_file_location("calendar_candidate", args.feature / "app/economic_calendar.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    def row(kind, observed, published=None, **extra):
        result = {"schema": cal.SCHEMA, "event_key": "CPI.fixture", "kind": kind, "observed_at": observed,
                  "event_time": "2026-03-03T13:30:00Z", "historical_availability": "KNOWN",
                  "period": "2026-02", "unit": "percent_yoy", **extra}
        if published is not None:
            result["published_at"] = published
        return result
    def surprise(rows):
        book = cal.PointInTimeCalendar()
        book.add_all(rows)
        return book.surprise("CPI.fixture", "2026-03-03T14:00:00Z")
    actual = row("ACTUAL", "2026-03-03T13:45:00Z", "2026-03-03T13:30:00Z", actual=2.9)
    new_consensus = row("CONSENSUS", "2026-03-03T13:21:00Z", "2026-03-03T13:20:00Z", consensus=2.7)
    old_consensus = row("CONSENSUS", "2026-03-03T13:25:00Z", "2026-03-03T13:00:00Z", consensus=2.5)
    result = surprise([old_consensus, new_consensus, actual])
    out["publication_vs_receipt_order"] = {"release_consensus": result["release_consensus"],
        "release_surprise": result["release_surprise"], "expected_latest_published_consensus": 2.7,
        "expected_release_surprise": 2.9 - 2.7}
    no_publication = copy.deepcopy(actual)
    no_publication.pop("published_at")
    later_consensus = row("CONSENSUS", "2026-03-03T13:35:00Z", "2026-03-03T13:34:00Z", consensus=2.9)
    result = surprise([old_consensus, later_consensus, no_publication])
    out["unknown_publication_substituted"] = {"release_clock_was_supplied": False,
        "reported_release_published_at": result["release_actual_published_at"],
        "release_surprise": result["release_surprise"], "available_surprise": result["available_surprise"]}
    revision = row("REVISION", "2026-03-03T13:41:00Z", "2026-03-03T13:40:00Z", actual=3.1)
    result = surprise([old_consensus, actual, revision])
    out["revision_arrived_first"] = {"release_actual": result["release_actual"],
        "expected_original_actual": 2.9, "revised_actual": result.get("revised_actual"),
        "revised_kind": result.get("revised_actual_kind")}
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
