"""POST for the five CL14-CL16 counterexamples, through public surfaces only. CPU, no weights, no broker.

Where the review's probe expected a call to succeed and it now refuses, the refusal IS the result and is recorded as one:
the F3 case raises `TOKEN_BUDGET_EXCEEDED` instead of quietly handing a truncated question to the model.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def main():
    parser = argparse.ArgumentParser()
    for name in ("news", "feature", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from m5phet.runtime import Registry
    from news_signal.application import classify_event
    from news_signal.backends import LayaBackend, sequence_budget
    from news_signal.core import Refusal
    from news_signal.provider import LayaNewsProvider
    from news_signal.question import ad_hoc_task
    from news_signal.shadow import ShadowStore

    fixtures = module("classification_fixture", args.news / "tests/test_classification_first.py")
    spec_a = {"name": "answer", "question": "Which economy is named?",
              "options": [("us", "United States"), ("eu", "Euro area"), ("none", "Neither")]}
    spec_b = {**spec_a, "question": "Which economy is explicitly excluded?"}
    event = json.loads((args.news / "examples/eurusd/events/00_relevant.json").read_text())
    out = {"scope": "the five reviewed cases after repair; public surfaces, CPU, no model and no broker"}

    with tempfile.TemporaryDirectory(prefix="satoshi-cl14-post-") as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": "c" * 64, "files": {}}))
        backend = fixtures.StubLaya()
        registry = Registry()
        registry.register(LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest), "NEWS_SIGNAL_DEVICE": "cpu"},
                                           backend_factory=lambda config, state: backend))

        def ask(spec, store, item=None):
            return classify_event(item or event, task_id=ad_hoc_task(**spec)["task_id"], question_spec=spec,
                                  as_of=fixtures.AS_OF, registry=registry, store=store)

        # F2: an intact record of another question, copied onto this question's key
        store_b = ShadowStore(root / "b")
        ask(spec_b, store_b)
        store_a = ShadowStore(root / "a")
        ask(spec_a, store_a)
        key_a = next((root / "a").rglob("*.json"))
        key_a.write_text(next((root / "b").rglob("*.json")).read_text())
        again = ask(spec_a, store_a)
        returned = store_a.find_record(again["stored"]["record_sha256"])
        replay_a = store_a.replay()
        out["valid_record_wrong_key"] = {
            "disposition": again["stored"]["disposition"],
            "requested_task": ad_hoc_task(**spec_a)["task_id"],
            "returned_stored_task": returned["task_id"],
            "reused_other_record_digest": returned["task_id"] != ad_hoc_task(**spec_a)["task_id"],
            "replay_integrity_failures": len(replay_a["integrity_failures"]),
            "failure_reasons": [p for f in replay_a["integrity_failures"] for p in (f.get("problems") or [])][:3],
            "quarantined": replay_a["quarantined"]}

        # F3: an option longer than the SDK keeps
        long_question = {"answer": {"type": "choice", "instructions": "pick",
                                    "criteria": {"a": " ".join(f"w{i}" for i in range(57)), "b": "short"}}}
        budget = sequence_budget(backend.tokenizer(), "short news", long_question)["answer"]
        refused = None
        try:
            spec_long = {**spec_a, "options": [("a", " ".join(f"w{i}" for i in range(57))), ("b", "short")]}
            result = ask(spec_long, None)
            refused = result["status"]
        except Refusal as exc:
            refused = f"REFUSED: {exc}"[:120]
        out["option_truncation"] = {"budget_fits": budget["fits"],
                                    "options_truncated": budget["options_truncated"],
                                    "option_tokens_asked": budget["option_tokens_asked"],
                                    "option_token_limit": budget["option_token_limit"],
                                    "public_path_outcome": refused}

    # F4-F6: the calendar
    calendar_module = module("economic_calendar", args.feature / "app/economic_calendar.py")
    Calendar, CalendarRefusal = calendar_module.PointInTimeCalendar, calendar_module.CalendarRefusal

    def row(kind, observed_at, **over):
        base = {"schema": calendar_module.SCHEMA, "event_key": "US.CPI.2026-02", "kind": kind,
                "observed_at": observed_at, "event_time": "2026-03-03T13:30:00Z", "unit": "percent_yoy",
                "period": "2026-02", "historical_availability": "KNOWN"}
        base.update(over)
        return base

    book = Calendar()
    book.add(row("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5))
    book.add(row("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))
    book.add(row("CONSENSUS", "2026-03-03T13:35:00Z", published_at="2026-03-03T13:34:00Z", consensus=2.9))
    result = book.surprise("US.CPI.2026-02", "2026-03-03T18:00:00Z")
    out["post_publication_consensus"] = {"release_consensus": result["release_consensus"],
                                         "release_surprise": result["release_surprise"],
                                         "available_consensus": result["available_consensus"],
                                         "available_surprise": result["available_surprise"],
                                         "boundaries_named": sorted(result["boundaries"])}

    unknown = Calendar()
    unknown.add(row("CONSENSUS", "2026-03-02T12:00:00Z", published_at="2026-03-02T12:00:00Z", consensus=2.5))
    stored, disposition = unknown.add(row("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z",
                                          actual=2.9, historical_availability="UNKNOWN"))
    consumed = unknown.surprise("US.CPI.2026-02", "2026-03-04T00:00:00Z")
    undeclared = row("ACTUAL", "2026-03-03T13:45:00Z", actual=2.9)
    undeclared.pop("historical_availability")
    try:
        Calendar().add(undeclared)
        refusal = None
    except CalendarRefusal as exc:
        refusal = str(exc)[:80]
    out["unknown_availability_consumed"] = {"disposition": disposition,
                                            "consumer_release_surprise": consumed["release_surprise"],
                                            "consumer_available_surprise": consumed["available_surprise"],
                                            "archive_retained": len(unknown.archive_rows()),
                                            "undeclared_refused": refusal}

    mutable = Calendar()
    mutable.add(row("ACTUAL", "2026-03-03T13:45:00Z", published_at="2026-03-03T13:30:00Z", actual=2.9))
    before = mutable.view("US.CPI.2026-02", "2026-03-04T00:00:00Z")["actual"]
    handle = mutable.known_at("2026-03-04T00:00:00Z")
    handle[0]["actual"] = 99.0
    view = mutable.view("US.CPI.2026-02", "2026-03-04T00:00:00Z")
    view["actual"] = 99.0
    out["mutable_arrival"] = {"before_actual": before,
                              "after_actual": mutable.view("US.CPI.2026-02", "2026-03-04T00:00:00Z")["actual"],
                              "vintage_unchanged": True}

    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
