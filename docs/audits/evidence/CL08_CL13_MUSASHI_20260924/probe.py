"""CPU-only public-boundary fixtures and independent retained-question parity.

No real models, market feeds, service mutations or brokers. The SDK sequence
builder is loaded by AST from the pinned public source, not reimplemented.
Use --packaging-only in the dependency-resolved clean venv as well as the
candidate-runtime venv. All test writes are confined to TemporaryDirectory.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import tempfile
import typing
from urllib.request import urlopen

SDK_URL = "https://raw.githubusercontent.com/NandhaKishorM/laya/1e28ac20c0896b1c37a744cd11f740eb98f8b178/laya/common.py"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ("news", "feature", "predictor", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--packaging-only", action="store_true")
    args = p.parse_args()
    from m5phet.runtime import Registry
    from news_signal.application import classify_event
    from news_signal.backends import LayaBackend, sequence_budget
    from news_signal.provider import LayaNewsProvider
    from news_signal.question import ad_hoc_task, build_question
    from news_signal.shadow import ShadowStore

    fixtures = module("classification_fixture", args.news / "tests/test_classification_first.py")
    spec_a = {"name": "answer", "question": "Which economy is named?",
              "options": [("us", "United States"), ("eu", "Euro area"), ("none", "Neither")]}
    spec_b = {**spec_a, "question": "Which economy is explicitly excluded?"}
    event = json.loads((args.news / "examples/eurusd/events/00_relevant.json").read_text())
    out = {"scope": "CPU fixtures; retained real-weight JSON parity; no fresh model inference"}
    with tempfile.TemporaryDirectory(prefix="musashi-cl08-") as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema": "news_checkpoint.v1", "sha256": "c" * 64, "files": {}}))
        registry = Registry()
        backend = fixtures.StubLaya()
        registry.register(LayaNewsProvider(environ={"NEWS_SIGNAL_MANIFEST": str(manifest), "NEWS_SIGNAL_DEVICE": "cpu"},
                          backend_factory=lambda config, state: backend))
        def ask(spec, store=None):
            return classify_event(event, task_id=ad_hoc_task(**spec)["task_id"], question_spec=spec,
                                  as_of=fixtures.AS_OF, registry=registry, store=store)
        first = ask(spec_a)
        origin = json.loads(importlib.metadata.distribution("m5phet").read_text("direct_url.json"))
        if origin.get("url", "").startswith("file:"):
            origin["url"] = "LOCAL_CANDIDATE_CHECKOUT"
        out["installed_question_path"] = {
            "status": first["status"], "reason": first["result"].get("why"),
            "backend_calls": len(backend.seen),
            "installed_m5phet_origin": origin}
        if args.packaging_only:
            args.out.write_text(json.dumps(out, indent=2) + "\n")
            print(json.dumps(out, indent=2))
            return
        assert first["status"] == "SHADOW_ONLY", "Full probes require the proposed runtime, not the declared old pin"
        store = ShadowStore(root / "store")
        a = ask(spec_a, store)
        b = ask(spec_b, store)
        record_a = store.find_record(a["stored"]["record_sha256"])
        record_b = store.find_record(b["stored"]["record_sha256"])
        path_a, path_b = store._path(record_a["record_id"]), store._path(record_b["record_id"])
        path_a.write_bytes(path_b.read_bytes())
        reused = ask(spec_a, store)
        out["valid_record_wrong_key"] = {
            "disposition": reused["stored"]["disposition"],
            "requested_task": reused["receipt"]["task_id"],
            "returned_stored_task": store.find_record(reused["stored"]["record_sha256"])["task_id"],
            "reused_other_record_digest": reused["stored"]["record_sha256"] == b["stored"]["record_sha256"],
            "replay_integrity_failures": len(store.replay()["integrity_failures"])}

    # A deterministic tokenizer isolates sequence assembly from learned weights.
    class Tokens:
        mask_token = "[MASK]"
        mask_token_id, cls_token_id, sep_token_id = -1, -2, -3
        def encode(self, text, **kwargs):
            return text.split()
        def __call__(self, text, **kwargs):
            return {"input_ids": self.encode(text, **kwargs)}

    source = urlopen(SDK_URL, timeout=30).read()
    names = {"serialize_state", "render_criterion", "_resolve_noul_labels", "render_options", "build_sequence"}
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    scope = {**vars(typing), "json": json, "_DEFAULT_NOUL_LABELS": {"false": "false", "true": "true"}}
    exec(compile(ast.Module(body=functions, type_ignores=[]), SDK_URL, "exec"), scope)
    long_question = build_question(name="answer", question="Which alternative applies?",
                                  options=[("a", "word " * 55 + "TAIL_SEMANTICS"), ("b", "different")])
    q = long_question["answer"]
    token_report = sequence_budget(Tokens(), "short news", long_question)["answer"]
    upstream_ids, _ = scope["build_sequence"](Tokens(), "short news",
                                             {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]})
    class Agent:
        device = "cpu"
        tok = Tokens()
        calls = 0
        def system_one(self, *a, **k):
            self.calls += 1
            return {"fixture": True}
    agent = Agent()
    LayaBackend.from_agent(agent, {"kind": "TOKEN_BUDGET_FIXTURE"}, "cpu").predict("short news", long_question)
    out["option_truncation"] = {"budget_fits": token_report["fits"], "backend_called": agent.calls,
        "tail_present_in_requested_option": "TAIL_SEMANTICS" in q["criteria"]["a"],
        "tail_present_in_upstream_sequence": "TAIL_SEMANTICS" in upstream_ids,
        "source_sha256": hashlib.sha256(source).hexdigest(), "source_url": SDK_URL,
        "tokenizer": "DETERMINISTIC_WORD_FIXTURE_NOT_REAL_CHECKPOINT_TOKENIZER"}

    cal = module("calendar_candidate", args.feature / "app/economic_calendar.py")
    def arrival(kind, observed, **extra):
        return {"schema": cal.SCHEMA, "event_key": "US.CPI.fixture", "kind": kind,
                "observed_at": observed, "event_time": "2026-03-03T13:30:00Z",
                "unit": "percent_yoy", "period": "2026-02", **extra}
    earlier = arrival("CONSENSUS", "2026-03-03T13:00:00Z", consensus=2.5)
    later = arrival("CONSENSUS", "2026-03-03T13:35:00Z", consensus=2.9, published_at="2026-03-03T13:34:00Z")
    actual = arrival("ACTUAL", "2026-03-03T13:45:00Z", actual=2.9, published_at="2026-03-03T13:30:00Z")
    book = cal.PointInTimeCalendar()
    book.add_all([earlier, later, actual])
    late_result = book.surprise("US.CPI.fixture", "2026-03-03T14:00:00Z")
    out["post_publication_consensus"] = {"selected_consensus": late_result["consensus"],
        "computed_surprise": late_result["surprise"], "pre_publication_consensus": 2.5,
        "expected_release_surprise": 2.9 - 2.5}
    unknown = {**actual, "historical_availability": "UNKNOWN"}
    archived = cal.PointInTimeCalendar()
    archived.add_all([earlier, unknown])
    out["unknown_availability_consumed"] = {
        "helper": cal.availability_checked(unknown, historical_availability="UNKNOWN")["point_in_time_usable"],
        "consumer_surprise": archived.surprise("US.CPI.fixture", "2026-03-03T14:00:00Z")["surprise"]}
    before_id = book.vintage_identity("2026-03-03T14:00:00Z")
    before_actual = book.view("US.CPI.fixture", "2026-03-03T14:00:00Z")["actual"]
    book.known_at("2026-03-03T14:00:00Z")[-1]["actual"] = 99.0
    out["mutable_arrival"] = {"before_actual": before_actual,
        "after_actual": book.view("US.CPI.fixture", "2026-03-03T14:00:00Z")["actual"],
        "vintage_unchanged": book.vintage_identity("2026-03-03T14:00:00Z") == before_id}

    evidence = args.predictor / "docs/audits/evidence/CL09_CL10_20260924"
    parity = {}
    for name in ("rate_cut_risk", "who_is_affected"):
        native = json.loads((evidence / f"direct_{name}.json").read_text())
        left = {r["input_sha256"]: r["response"]["answers"] for r in native["single"]}
        receipts = [json.loads(p.read_text()) for p in (evidence / "wrapper/receipts").glob(f"{name}__*.json")]
        right = {r["input_sha256"]: {q: a["sdk_answer"] for q, a in r["features"].items()} for r in receipts}
        parity[name] = {"population_equal": set(left) == set(right), "distinct_inputs": len(left),
            "exact_answers": sum(json.dumps(left[k]) == json.dumps(right[k]) for k in left.keys() & right.keys()),
            "labels_over_13_rows": dict(Counter(r["features"][name]["label"] for r in receipts))}
    out["retained_real_question_parity"] = parity
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
