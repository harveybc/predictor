"""CB03: seal the evidence directory, so a later reader can tell what produced what.

Every artefact gets its sha256 and a one-line statement of what it is and what it
is NOT. The smoke files in particular are cost measurements and say so, because a
number in a directory beside a reproduction will otherwise be read as a result.

  python tools/df_cb03_evidence_manifest_20260929.py --dir docs/audits/evidence/cb03_20260929
"""
import argparse
import hashlib
import json
from pathlib import Path

WHAT = {
    "native_published400.json": (
        "THE REPRODUCTION. The author's own harness, laya 0.2.1, the checkpoint's own "
        "budget, the author's state envelope, on the pinned 400-row published population. "
        "accuracy 0.9525 against the published 0.9525."),
    "smoke_train32.json": (
        "A COST MEASUREMENT, not a result. 32 rows of the official TRAIN split, run before "
        "the evaluation population was opened, to size the memory cap and the wall clock. "
        "Its accuracy is on 32 same-class train rows and is not a quality claim."),
    "parity_smoke8.json": (
        "A COST MEASUREMENT, not a result. 8 rows through the framework, to catch contract "
        "errors before the 400-row run. Its parity fields are computed against a 400-row "
        "reference and are therefore incomplete by construction."),
    "parity_400.json": (
        "THE PARITY RUN. The same 400 rows through m5phet.runtime.run with the "
        "distribution's laya_news provider. accuracy 0.9350, 387/400 label agreement "
        "against the reproduction, 0 refusals, coverage 400/400."),
    "v_sdk0311_ckptbudget_author.json": (
        "ATTRIBUTION N2. The author's recipe under the SDK the framework pins "
        "(laya 0.3.11), everything else unchanged. Bit-identical to the reproduction."),
    "v_sdk0311_fwbudget_author.json": (
        "ATTRIBUTION N3. N2 plus the sequence budget the framework hard-codes (512/192). "
        "Bit-identical to N2 on these rows, whose longest sequence is 232 tokens."),
    "v_sdk0311_fwbudget_fwenvelope.json": (
        "ATTRIBUTION N4. N3 plus the state envelope the framework serialises. THIS is "
        "where the gap appears: accuracy 0.9350, 13 label flips, max |dp| 0.346653."),
    "CB03_PARITY_ATTRIBUTION.json": (
        "The ladder, computed from the artefacts above: one change per rung, so each "
        "step's disagreement belongs to that change alone."),
    "CB03_CLASSIFICATION_RECEIPTS.json": (
        "Three classification_metrics.v1 receipts built by CB04's own contract: our "
        "accuracy, our macro-F1, and the author's published value as a "
        "PUBLISHED_REFERENCE receipt of its own. Includes the two refusals exercised "
        "and compare() returning difference 0.0 against the published value."),
    "cb03_checkpoint_manifest.json": (
        "news_checkpoint.v1: the sealed checkpoint directory the framework loaded, every "
        "file with its size and sha256. The framework refuses to run if it changes."),
}

NOT_HERE = [
    "no full 7,600-row AG News test evaluation: a separate experiment, and its denominator "
    "is never attached to the 400-row published score",
    "no BANKING77, FOMC, MASSIVE or Financial PhraseBank measurement",
    "no GPU run: the published cell is device cpu and the author's harness hard-codes CPU",
    "no business held-out measurement and therefore no provider quality badge",
    "nothing from the 19-prompt router corpus, at any point, in any file here",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    root = Path(args.dir)
    files = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        h = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        files[path.name] = {"bytes": path.stat().st_size, "sha256": h.hexdigest(),
                            "what_it_is": WHAT.get(path.name, "UNDESCRIBED")}
    manifest = {
        "schema": "cb03_evidence_manifest.v1",
        "package": "CB03 native AG News reproduction, then M5PHET parity",
        "date": "2026-09-29",
        "author": "Satoshi, successor technical lead",
        "population": {
            "id": "agnews_test_first400_laya_published",
            "rows": 400,
            "population_sha256":
                "b4c5f991060bcefcc69fac9339b32086dcd0674ac15e44f41eeeeb7c9e782324",
            "governed_resource_sha256":
                "71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb",
            "held_out": False,
            "held_out_note": "the benchmark's own results file marks this suite in_training: true",
            "paired_naive_accuracy_same_rows": 0.18,
        },
        "checkpoint": {
            "id": "convaiinnovations/laya-typed-decisions",
            "revision": "1a793eb568e6718f15941d08f85432581df534e3",
            "weights_sha256":
                "4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e",
            "inference_files_identical_to_revision_live_at_the_published_run": "f9ab0b22",
        },
        "evaluator": {
            "repository": "NandhaKishorM/laya",
            "commit": "ee760389dc69e28c66893b717fa87c84c0b6063a",
            "bench_local_sha256":
                "08862fb4dba102db873e3e3ad428dedf92c12832b4e8ef71b35e433cce4eea22",
            "bench_apps_sha256":
                "b25e7fb1228fb7d4d51d121ea08b3c3471b9c9af67d5a40fe415a2f33983e960",
        },
        "files": files,
        "what_is_not_here": NOT_HERE,
        "authorises_broker_deployment": False,
        "execution_authority": "NONE",
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({"files": len(files), "undescribed":
                      [k for k, v in files.items() if v["what_it_is"] == "UNDESCRIBED"]},
                     indent=1))


if __name__ == "__main__":
    main()
