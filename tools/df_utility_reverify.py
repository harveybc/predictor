#!/usr/bin/env python3
"""Successor verification of a utility run from its conserved results (O1–O3).

Re-reads every calibration and contrast attempt of a run root without repeating a single
simulation: bytes against the declared and runner-verified digests, the recorded job against
the freeze (protocol identity, operator, contrast id), then the calibration counts, rate and
Clopper–Pearson bound DERIVED from the per-simulation records under the declared failure policy,
under the harness code that applied to the run (the file at the run's frozen commit). Each
contrast is re-decided from its conserved delta and the derived support; the delta of decisions
against the original outcomes is reported. A readable table per contrast (raw loss, transformed
loss, delta, interval, paired rows, coverage, cost, null scope) is written next to the receipt.
Also records the limited diff between the frozen and the resuming commit when a run was resumed
under new code (O2). Nothing is re-run; missing evidence is INCONCLUSIVE, never rebuilt.

    python tools/df_utility_reverify.py --root RUN_ROOT [--repo .] [--out REVERIFY.json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


H = _load("df_utility_harness")
ops = _load("df_d3_operators")

SCIENTIFIC_FILES = ("tools/df_utility_harness.py", "tools/df_d3_operators.py", "tools/df_d3_contract.py",
                    "tools/df_d3_acceptance.py", "tools/df_d3_design.py")


def file_at(repo: Path, commit: str, path: str) -> bytes | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                              check=True, capture_output=True).stdout
    except subprocess.CalledProcessError:
        return None


def protocol_from(doc: dict, *, calibration=None) -> "H.Protocol":
    return H.Protocol(**{**{k: (tuple(v) if isinstance(v, list) else v) for k, v in doc.items()
                           if k not in ("protocol_sha256", "comparisons", "alpha_adjusted")},
                        "calibration": calibration})


def verified_bytes(attempt: Path) -> tuple:
    """(doc, digest, problems): the output file the child named, re-hashed against the digest
    it declared and the one the runner verified."""
    problems = []
    result_path = attempt / "result.json"
    outcome_path = attempt / "outcome.json"
    if not outcome_path.is_file():
        return None, None, ["no outcome.json (the attempt never ended)"]
    recorded = json.loads(outcome_path.read_text())
    if recorded.get("status") != "COMPLETED":
        # an attempt that ended under a ceiling: carried by its recorded outcome, no score
        return {"outcome": (recorded.get("summary") or {}).get("outcome"), "not_completed": True,
                "reason": (recorded.get("summary") or {}).get("reason")}, None, []
    if not result_path.is_file():
        return None, None, ["COMPLETED without result.json"]
    result = json.loads(result_path.read_text())
    name = result.get("output_file")
    path = attempt / str(name)
    if not name or not path.is_file():
        return None, None, [f"output {name!r} absent"]
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    verified = (recorded.get("verified") or {}).get("output_sha256")
    if digest != result.get("output_sha256"):
        problems.append("bytes differ from the digest the child declared")
    if digest != verified:
        problems.append("bytes differ from the digest the runner verified")
    try:
        doc = json.loads(body)
    except ValueError:
        return None, digest, problems + ["output is not JSON"]
    return doc, digest, problems


def recorded_job(attempt: Path) -> tuple:
    path = attempt / "job.json"
    if not path.is_file():
        return None, ["no recorded job"]
    return json.loads(path.read_text()), []


def reverify(root: Path, repo: Path) -> dict:
    freeze = json.loads((root / "FREEZE.json").read_text())
    pre = json.loads((root / "FREEZE.pre.json").read_text())
    report = json.loads((root / "REPORT.json").read_text()) if (root / "REPORT.json").is_file() else {}
    frozen_commit = (freeze.get("code_identity") or {}).get("value")
    now_commit = (report.get("code_identity_now") or {}).get("value") or frozen_commit
    harness_at = {}
    for c in {frozen_commit, now_commit}:
        body = file_at(repo, c, "tools/df_utility_harness.py") if c else None
        harness_at[c] = hashlib.sha256(body).hexdigest() if body else None
    applicable = harness_at.get(frozen_commit)
    out = {"schema": "df_utility_reverify.v1", "run_id": freeze["run_id"], "root": str(root),
           "code_identity_frozen": frozen_commit, "code_identity_resumed": now_commit,
           "harness_sha256_at_frozen_commit": applicable,
           "harness_sha256_at_resumed_commit": harness_at.get(now_commit),
           "harness_sha256_now": H.harness_sha256(),
           "failure_policy": H.FAILURE_POLICY,
           "calibrations": {}, "contrasts": {}, "decision_delta": [], "table": [],
           "resume_diff": None, "all_verified": True}
    if now_commit and frozen_commit and now_commit != frozen_commit:
        files = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", frozen_commit, now_commit],
                               check=True, capture_output=True, text=True).stdout.split()
        digests = {}
        for f in SCIENTIFIC_FILES:
            a, b = file_at(repo, frozen_commit, f), file_at(repo, now_commit, f)
            digests[f] = {"frozen": hashlib.sha256(a).hexdigest() if a else None,
                          "resumed": hashlib.sha256(b).hexdigest() if b else None}
        out["resume_diff"] = {"files_changed": files, "scientific_files": digests,
                              "scientific_code_unchanged": all(d["frozen"] == d["resumed"] for d in digests.values()),
                              "harness_unchanged": harness_at.get(frozen_commit) == harness_at.get(now_commit)}
    protocols = freeze["protocols"]
    bare = {k: protocol_from(doc) for k, doc in protocols.items()}
    # --- calibrations -------------------------------------------------------------------------
    records = {}
    for attempt in sorted((root / "attempts").glob("calibrate__*")):
        kind = attempt.name.split("__", 1)[1]
        entry = {"attempt": attempt.name, "problems": [], "derived": None, "supports": None}
        doc, digest, problems = verified_bytes(attempt)
        entry["problems"] += problems
        job, jp = recorded_job(attempt)
        entry["problems"] += jp
        if doc is not None and doc.get("not_completed"):
            entry["problems"].append(f"calibration ended {doc.get('outcome')}: no record")
        elif doc is not None and job is not None:
            if job.get("operator") != kind:
                entry["problems"].append("recorded job is for another operator")
            base_doc = job.get("protocol") or {}
            try:
                job_base = protocol_from(base_doc).base_sha256()
            except Exception as e:  # noqa: BLE001
                job_base, entry["problems"] = None, entry["problems"] + [f"recorded protocol unsealable: {e}"]
            if job_base and job_base != bare[kind].base_sha256():
                entry["problems"].append("recorded job's protocol base is not the freeze's")
            if doc.get("protocol_base_sha256") != bare[kind].base_sha256():
                entry["problems"].append("record's protocol base is not the freeze's")
            if dict(job.get("plan") or {}) != dict(pre.get("plan") or {}):
                entry["problems"].append("recorded job's plan is not the sealed plan")
            derived = H.derive_calibration(doc)
            entry["derived"] = {k: v for k, v in derived.items() if k != "problems"}
            entry["problems"] += derived["problems"]
            entry["record_problems"] = H.calibration_record_problems(doc)
            entry["stated"] = {k: doc.get(k) for k in ("scored", "failed", "advances", "false_advance_rate",
                                                       "upper_bound", "bound_confidence", "alpha_adjusted",
                                                       "generator", "n_sims", "n", "harness_sha256")}
            ok, why = H.calibration_supports(bare[kind], ops.build(kind), doc["n"], record=doc,
                                             harness_sha256=applicable)
            entry["supports"] = {"decision": ok, "why": why, "under_harness": applicable}
            entry["record_sha256"] = digest
            if not entry["problems"] and not entry["record_problems"]:
                records[kind] = doc
        if entry["problems"]:
            out["all_verified"] = False
        out["calibrations"][kind] = entry
    # --- contrasts --------------------------------------------------------------------------------
    for attempt in sorted(p for p in (root / "attempts").iterdir()
                          if not p.name.startswith(("calibrate__", "mechanics__"))):
        entry = {"attempt": attempt.name, "problems": [], "original_outcome": None, "reverified_outcome": None}
        doc, digest, problems = verified_bytes(attempt)
        entry["problems"] += problems
        job, jp = recorded_job(attempt)
        entry["problems"] += jp
        kind = (job or {}).get("operator")
        if doc is not None and doc.get("not_completed"):
            entry["original_outcome"] = entry["reverified_outcome"] = doc.get("outcome")
            entry["carried"] = f"attempt ended {doc.get('outcome')}: {doc.get('reason')}"
        elif doc is not None and job is not None:
            entry["original_outcome"] = doc.get("outcome")
            if doc.get("contrast_id") not in (None, attempt.name) or job.get("contrast_id") != attempt.name:
                entry["problems"].append("contrast identity differs from the attempt")
            sealed = (job.get("protocol") or {}).get("protocol_sha256")
            if kind not in protocols or sealed != protocols[kind]["protocol_sha256"]:
                entry["problems"].append("recorded job's protocol is not the frozen one for this operator")
            if "protocol_sha256" in doc and doc["protocol_sha256"] != sealed:
                entry["problems"].append("file's protocol identity is not the job's")
            n = len((job.get("series") or {}).get("values") or [])
            if "delta_lower" in doc and not entry["problems"]:
                rec = records.get(kind)
                if rec is None:
                    ok, why = False, "no verified calibration record for this operator"
                else:
                    ok, why = H.calibration_supports(bare[kind], ops.build(kind), n, record=rec,
                                                     harness_sha256=applicable)
                margin = bare[kind].margin
                entry["reverified_outcome"] = (H.ADVANCES if doc["delta_lower"] > margin else H.DOES_NOT_ADVANCE) \
                    if ok else H.INCONCLUSIVE_UNCALIBRATED
                entry["support"] = {"decision": ok, "why": why}
                losses_a = [b["loss_a"] for b in doc.get("blocks", [])]
                losses_b = [b["loss_b"] for b in doc.get("blocks", [])]
                cov = doc.get("coverage") or {}
                entry_table = {
                    "contrast": attempt.name, "unit": job.get("unit"), "variable": job.get("variable"),
                    "operator": kind, "loss": doc.get("loss_name"),
                    "loss_raw_mean": sum(losses_a) / len(losses_a) if losses_a else None,
                    "loss_transformed_mean": sum(losses_b) / len(losses_b) if losses_b else None,
                    "delta_mean": doc["delta_mean"], "delta_lower": doc["delta_lower"], "delta_se": doc["delta_se"],
                    "alpha_adjusted": doc.get("alpha_adjusted"), "margin": margin,
                    "blocks_used": doc.get("blocks_used"),
                    "rows_paired": cov.get("rows_paired"), "n": cov.get("n"), "inputs_missing": cov.get("inputs_missing"),
                    "train_rows": sum(b["train_rows"] for b in doc.get("blocks", [])),
                    "validation_rows": sum(b["validation_rows"] for b in doc.get("blocks", [])),
                    "purge": doc.get("purge"), "cpu_seconds": (doc.get("cost") or {}).get("cpu_seconds"),
                    "null_scope": ({"generator": rec["generator"], "n_sims": rec["n_sims"], "n": rec["n"],
                                    "bound_confidence": rec["bound_confidence"],
                                    "derived_decision_bound": out["calibrations"][kind]["derived"]["decision_bound"],
                                    "failed": rec["failed"]} if rec else None),
                    "original_outcome": doc.get("outcome"), "reverified_outcome": entry["reverified_outcome"]}
                out["table"].append(entry_table)
            elif not entry["problems"]:
                entry["reverified_outcome"] = doc.get("outcome")       # REFUSED / INSUFFICIENT: carried
        if entry["problems"]:
            out["all_verified"] = False
            entry["reverified_outcome"] = "INCONCLUSIVE_EVIDENCE_UNVERIFIED"
        if entry["original_outcome"] != entry["reverified_outcome"]:
            out["decision_delta"].append({"contrast": attempt.name, "original": entry["original_outcome"],
                                          "reverified": entry["reverified_outcome"],
                                          "why": (entry.get("support") or {}).get("why") or "; ".join(entry["problems"])})
        out["contrasts"][attempt.name] = entry
    return out


def markdown(out: dict) -> str:
    lines = [f"# Utility run `{out['run_id']}` — successor re-verification", "",
             f"Frozen commit `{(out['code_identity_frozen'] or '')[:7]}`, resumed `{(out['code_identity_resumed'] or '')[:7]}`; "
             f"harness that applied `{(out['harness_sha256_at_frozen_commit'] or '')[:12]}…`; failure policy `{out['failure_policy']}`.", "",
             "## Calibrations (derived from every simulation)", "",
             "| operator | stated advances/scored (+failed) | derived | stated bound | derived bound | decision bound | supports | why |",
             "|---|---|---|---:|---:|---:|---|---|"]
    for k, e in out["calibrations"].items():
        d, s = e.get("derived") or {}, e.get("stated") or {}
        sup = e.get("supports") or {}
        lines.append(f"| `{k}` | {s.get('advances')}/{s.get('scored')} (+{s.get('failed')}) | "
                     f"{d.get('advances')}/{d.get('scored')} (+{d.get('failed')}) | {s.get('upper_bound', float('nan')):.5f} | "
                     f"{d.get('upper_bound', float('nan')):.5f} | {d.get('decision_bound', float('nan')):.5f} | "
                     f"{'YES' if sup.get('decision') else 'NO'} | {sup.get('why') or '; '.join(e['problems']) or '—'} |")
    lines += ["", "## Contrasts", "",
              "| unit | variable | operator | loss | raw | transformed | Δ (raw−transf.) | lower (1−α/m) | blocks | paired rows | train/val rows | cpu s | null scope | original | re-verified |",
              "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|"]
    for t in out["table"]:
        ns = t["null_scope"] or {}
        scope = (f"{ns.get('generator')} ×{ns.get('n_sims')} @n={ns.get('n')}, bound {ns.get('derived_decision_bound', float('nan')):.5f}"
                 if ns else "none")
        lines.append(f"| `{t['unit']}` | {t['variable']} | `{t['operator']}` | {t['loss']} | {t['loss_raw_mean']:.5f} | "
                     f"{t['loss_transformed_mean']:.5f} | {t['delta_mean']:+.5f} | {t['delta_lower']:+.5f} | {t['blocks_used']} | "
                     f"{t['rows_paired']}/{t['n']} | {t['train_rows']}/{t['validation_rows']} | {t['cpu_seconds']} | {scope} | "
                     f"{t['original_outcome']} | {t['reverified_outcome']} |")
    lines += ["", f"Decision delta: {len(out['decision_delta'])} contrast(s) changed; all evidence verified: {out['all_verified']}.",
              "", "DOES_NOT_ADVANCE is not equivalence and says nothing about other domains, horizons or models; "
              "INCONCLUSIVE_UNCALIBRATED is descriptive (its record's derived bound exceeds α/m).", ""]
    if out.get("resume_diff"):
        rd = out["resume_diff"]
        lines += [f"Resumed under new code: files changed {rd['files_changed']}; scientific code unchanged: "
                  f"{rd['scientific_code_unchanged']}; harness unchanged: {rd['harness_unchanged']}.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--out", default="REVERIFY.json")
    args = parser.parse_args(argv)
    out = reverify(args.root, args.repo)
    target = args.root / args.out
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; a verification is never written over")
    target.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    target.with_suffix(".md").write_text(markdown(out))
    print(json.dumps({"run_id": out["run_id"], "all_verified": out["all_verified"],
                      "calibrations": {k: (e["supports"] or {}).get("decision") for k, e in out["calibrations"].items()},
                      "decision_delta": out["decision_delta"],
                      "resume_diff": out["resume_diff"] and {k: v for k, v in out["resume_diff"].items() if k != "scientific_files"}},
                     indent=1))
    return 0 if out["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
