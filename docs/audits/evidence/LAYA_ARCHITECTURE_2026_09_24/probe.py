"""Exercise the upstream schema helpers without weights, network or GPU.

PYTHONPATH must point to upstream Laya 84ca7240348e6fb3b8145d2d9f6266d5125747d8.
These are boundary counterexamples, not model-quality measurements.
"""
import argparse
import json
from pathlib import Path

from laya.structured import answers_to_json, questions_from_json_schema

parser = argparse.ArgumentParser()
parser.add_argument("--output")
args = parser.parse_args()
schema = {"type": "object", "properties": {"flag": {"const": False}}, "required": ["flag"]}
collision = {"type": "object", "properties": {"value": {"enum": [1, "1"]}}}
result = {
    "source_revision": "84ca7240348e6fb3b8145d2d9f6266d5125747d8",
    "constant_false_questions": questions_from_json_schema(schema),
    "constant_false_projection": answers_to_json({"flag": {"type": "noul", "noul": 0.9}}, schema),
    "missing_required_projection": answers_to_json({}, schema),
    "colliding_enum_questions": questions_from_json_schema(collision),
    "colliding_enum_projection": answers_to_json({"value": {"choice": "1"}}, collision),
    "scope": "actual upstream helper functions; no neural forward pass or package substitution",
}
text = json.dumps(result, indent=2, sort_keys=True) + "\n"
if args.output:
    Path(args.output).write_text(text)
print(text)
