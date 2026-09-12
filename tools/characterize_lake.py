#!/usr/bin/env python3
"""C49-C51 (order 2026-09-12): a disposition for every one of the
1,965 conceptual variables.

The v2 run measured 107 subjects and every one of them succeeded, so
the ledger read as complete while 1,858 conceptual variables of the
lake had no disposition at all. A denominator you choose after the fact
is not a denominator.

This processes the WHOLE census in deterministic, resumable batches.
Each `variable_id` ends exactly once in

    MEASURED | NOT_IDENTIFIABLE | UNAVAILABLE | FAILED

and absence is a result: a variable whose file is gone is UNAVAILABLE,
a variable with too few finite values is NOT_IDENTIFIABLE, and a
producer that crashes on one variable does not cancel its batch.

Custody (C50):

  * a PRE-RESULT ledger lists all 1,965 identities before anything is
    measured, so the denominator cannot drift;
  * a write-once terminal per variable makes the run resumable: a
    second invocation skips what already has one, and can neither
    duplicate nor lose an outcome;
  * every measured row binds source, bytes, window, code, protocol and
    cost through the C39 contract;
  * the outbox receives the batch envelope whatever the outcomes were —
    successes, absences, failures and inconclusives alike.

Conceptual variables and physical appearances are counted separately
and never multiplied together.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from olap import characterization as ch                      # noqa: E402
from eligibility.consumed import (code_identity,              # noqa: E402
                                  local_code_surface)

MEASURED = "MEASURED"
NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
UNAVAILABLE = "UNAVAILABLE"
FAILED = "FAILED"
OUTCOMES = (MEASURED, NOT_IDENTIFIABLE, UNAVAILABLE, FAILED)

LEDGER_SCHEMA = "crispdm.lake_characterization_ledger.v1"
DEV_FRACTION = 0.7
#: a row cap keeps one pathological file from costing the whole run;
#: it is part of the declared window, not a silent truncation.
ROW_CAP = 200_000

TEMPORAL = {"date_time", "datetime", "date", "timestamp", "time",
            "close_time", "open_time", "index", "period"}


def sha_obj(o) -> str:
    return hashlib.sha256(
        json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


def is_temporal(column: str) -> bool:
    return column.strip().lower() in TEMPORAL


# ------------------------------------------------------- pre-result
def build_pre_ledger(census: dict, state_dir: Path) -> dict:
    """C50: the denominator, written down BEFORE anything is measured."""
    variables = census["variables"]
    doc = {
        "schema": "crispdm.lake_characterization_pre_ledger.v1",
        "census_sha256": census["census_sha256"],
        "censused_at": census["censused_at"],
        "conceptual_variables": len(variables),
        "physical_appearances": len(census["appearances"]),
        "identities": sorted(v["variable_id"] for v in variables),
        "rule": "every identity listed here must end in exactly one of "
                f"{list(OUTCOMES)}; the denominator cannot be chosen "
                "after the results are in",
        "written_at": utc_now(),
    }
    doc["pre_ledger_sha256"] = sha_obj(
        {k: doc[k] for k in sorted(doc) if k != "written_at"})
    state_dir.mkdir(parents=True, exist_ok=True)
    out = state_dir / "PRE_LEDGER.json"
    if out.is_file():
        prior = json.loads(out.read_text())
        if prior["pre_ledger_sha256"] != doc["pre_ledger_sha256"]:
            raise SystemExit(
                "REFUSED: a pre-result ledger already exists for a "
                "DIFFERENT census; a run may not change its own "
                "denominator mid-flight")
        return prior
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def terminal_path(state_dir: Path, variable_id: str) -> Path:
    safe = hashlib.sha256(variable_id.encode()).hexdigest()[:32]
    return state_dir / "terminals" / f"{safe}.json"


def write_terminal_once(state_dir: Path, doc: dict) -> bool:
    """Write-once. A second attempt at the same variable is refused by
    the filesystem, which is what makes resumption safe."""
    p = terminal_path(state_dir, doc["variable_id"])
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    try:
        os.write(fd, (json.dumps(doc, sort_keys=True, default=str)
                      + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def existing_terminals(state_dir: Path) -> dict:
    out = {}
    d = state_dir / "terminals"
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except ValueError:
            continue
        out[doc["variable_id"]] = doc
    return out


# ------------------------------------------------------------ reading
def read_column(path: Path, column: str, cap: int) -> tuple:
    """One pass over the file, one column, capped and declared.

    The lake is parquet; the registry views are CSV. Reading a parquet
    file with a CSV reader reports every column as absent, which is how
    a first run turned 1,965 genuine variables into "the column is not
    in this file's header".
    """
    if path.suffix.lower() in (".parquet", ".pq"):
        import pandas as pd
        try:
            frame = pd.read_parquet(path, columns=[column])
        except Exception as exc:                          # noqa: BLE001
            if "not" in str(exc).lower() and "column" in str(exc).lower():
                return None, {"reason": "the column is not in this "
                                        "file's schema", "rows": 0}
            raise
        series = frame[column]
        rows = int(series.size)
        values = pd.to_numeric(series.iloc[:cap],
                               errors="coerce").astype("float64").tolist()
        return values, {"rows": rows, "capped": rows > cap}

    values, rows = [], 0
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            return None, {"reason": "the column is not in this file's "
                                    "header", "rows": 0}
        for row in reader:
            rows += 1
            raw = row.get(column)
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
            if rows >= cap:
                break
    return values, {"rows": rows, "capped": rows >= cap}


def normalize_concept(name: str) -> str:
    """The census's own normalization. A conceptual variable named
    `Volume` is spelled `volume` in one file and `Volume` in another;
    matching the literal spelling would report the data as absent."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def pick_appearance(variable: dict, appearances: dict) -> tuple:
    """Deterministic: the first appearance, in sorted id order, that
    actually carries a column for this concept — with the exact
    spelling that appearance uses."""
    want = normalize_concept(variable.get("concept_name", ""))
    first = None
    for aid in sorted(variable.get("appearances", [])):
        a = appearances.get(aid)
        if a is None:
            continue
        first = first or a
        for column in a.get("declared_columns", []):
            if normalize_concept(column) == want:
                return a, column
    return first, None


def emit_batch_envelope(batch_doc: dict, *, census: dict,
                       measured_at: str, terminal_attempt: str,
                       code_digest: str) -> dict:
    """One batch, one campaign envelope.

    The first version emitted these as outbox EVENTS. The outbox
    accepted them and the loader refused every one — "only envelopes
    are loadable today" — so fourteen batches went straight to
    dead-letter. An outcome the cube cannot receive is not a reported
    outcome, so a batch is now what it actually is: one unit of the
    lake-characterization campaign.
    """
    from olap import outbox as ob
    from olap.campaign_envelope import build_envelope

    outcomes = batch_doc["outcomes"]
    terminal_state = "COMPLETE" if not outcomes.get(FAILED) else "FAILED"
    doc = build_envelope(
        campaign_key="crispdm::lake_characterization",
        producer="predictor",
        result_class="DEVELOPMENT",
        identity={"run_id": terminal_attempt,
                  "code_identity": code_digest,
                  "design_sha256": census["census_sha256"]},
        data_consumed={"variables": [], "operators": [],
                       "datasets": [{"id": "financial-data lake",
                                     "digest": census["census_sha256"],
                                     "eligibility_state":
                                         "FINANCIAL_DOMAIN_"
                                         "DEVELOPMENT_ONLY"}]},
        partitions={"exposure": "DEVELOPMENT_ONLY_NON_CONFIRMATORY",
                    "splits": "first 70% of rows in file order"},
        budget={"device": "cpu", "cost_units": "wall_seconds",
                "wall_seconds": batch_doc["seconds"]},
        terminal={"state": terminal_state,
                  "adjudication": "MEASUREMENT_ONLY_NO_SELECTION"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL",
                   "batch": batch_doc["batch"]},
        units=[{"cell_key": batch_doc["batch"],
                "candidate_key": outcome,
                "metric_name": "conceptual_variables",
                "metric_value": float(count),
                "terminal_state": terminal_state}
               for outcome, count in sorted(outcomes.items())])
    return ob.emit(doc, kind="envelope")


# ---------------------------------------------------------------- run
def run(census: dict, lake_root: Path, state_dir: Path, *,
        measured_at: str, terminal_attempt: str, batch_size: int,
        limit: int | None, load: bool, emit_outbox: bool) -> dict:
    started = time.perf_counter()
    pre = build_pre_ledger(census, state_dir)
    appearances = {a["appearance_id"]: a for a in census["appearances"]}
    variables = {v["variable_id"]: v for v in census["variables"]}
    done = existing_terminals(state_dir)

    code_digest, _inv = code_identity(REPO, local_code_surface(REPO), {})
    pending = [vid for vid in pre["identities"] if vid not in done]
    if limit is not None:
        pending = pending[:limit]

    batches, rows_loaded = [], 0
    digest_cache: dict[str, str] = {}
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        batch_id = f"batch_{start // batch_size:05d}"
        batch_rows, outcomes = [], {o: 0 for o in OUTCOMES}
        t_batch = time.perf_counter()
        for vid in chunk:
            var = variables[vid]
            terminal = {"variable_id": vid, "batch": batch_id,
                        "measured_at": measured_at,
                        "concept_name": var.get("concept_name"),
                        "entity": var.get("entity")}
            try:
                app, column = pick_appearance(var, appearances)
                if app is None:
                    terminal.update(outcome=UNAVAILABLE,
                                    reason="the census lists no "
                                           "appearance for this variable")
                elif column is None:
                    terminal.update(
                        outcome=UNAVAILABLE,
                        appearance=app["appearance_id"],
                        reason="no appearance of this variable declares "
                               "a column for its concept, so there are "
                               "no bytes to measure")
                elif is_temporal(var.get("concept_name", "")):
                    terminal.update(
                        outcome=NOT_IDENTIFIABLE,
                        reason="a temporal identifier is the axis the "
                               "variables are observed on, not a "
                               "variable; it receives an axis contract "
                               "and no numeric descriptor")
                else:
                    path = lake_root / app["relative_path"]
                    if not path.is_file():
                        terminal.update(
                            outcome=UNAVAILABLE,
                            reason=f"the appearance file "
                                   f"{app['relative_path']} is not "
                                   "present in this checkout")
                    else:
                        values, meta = read_column(path, column, ROW_CAP)
                        if values is None:
                            terminal.update(
                                outcome=UNAVAILABLE,
                                reason=meta["reason"], appearance=
                                app["appearance_id"])
                        else:
                            n = len(values)
                            end = int(n * DEV_FRACTION)
                            used = values[:end]
                            window = {
                                "rule": f"first {DEV_FRACTION:.0%} of "
                                        f"rows in file order",
                                "rows_total": n, "rows_used": end,
                                "row_cap": ROW_CAP,
                                "capped": meta.get("capped", False)}
                            if app["relative_path"] not in digest_cache:
                                digest_cache[app["relative_path"]] = (
                                    app.get("physical_sha256")
                                    or sha_file(path))
                            binding = {
                                "source_id": app["relative_path"],
                                "source_sha256":
                                    digest_cache[app["relative_path"]],
                                "window_sha256": sha_obj(window),
                                "window_contract":
                                    json.dumps(window, sort_keys=True),
                                "code_identity": code_digest,
                                "protocol_version": ch.PROTOCOL_VERSION,
                                "side": "x", "contract_role": "input",
                                "units": var.get("unit") or "UNDECLARED",
                                "terminal_attempt": terminal_attempt}
                            measured = ch.characterize_series(
                                used, variable_id=vid,
                                partition_key=f"{var['entity']}::"
                                              "development",
                                bank_authority=ch.BANK_FINANCIAL,
                                measured_at=measured_at, binding=binding)
                            identifiable = [r for r in measured
                                            if r["identifiable"]]
                            batch_rows.extend(measured)
                            terminal.update(
                                outcome=(MEASURED if identifiable
                                         else NOT_IDENTIFIABLE),
                                appearance=app["appearance_id"],
                                descriptors=len(measured),
                                not_identifiable=len(measured)
                                - len(identifiable),
                                rows_used=end)
            except Exception as exc:                      # noqa: BLE001
                # C50: one variable's crash is that variable's outcome,
                # never the batch's.
                terminal.update(outcome=FAILED,
                                reason=f"{type(exc).__name__}: "
                                       f"{str(exc)[:160]}")
            if not write_terminal_once(state_dir, terminal):
                terminal["outcome"] = "ALREADY_TERMINAL"
            else:
                outcomes[terminal["outcome"]] = \
                    outcomes.get(terminal["outcome"], 0) + 1

        ch.assert_no_selection(batch_rows)
        if load and batch_rows:
            from tools.backfill_campaign_envelopes import _engine
            rows_loaded += ch.load_rows(_engine(), batch_rows).get(
                "fact_variable_characterization", 0)
        batch_doc = {
            "batch": batch_id, "variables": len(chunk),
            "outcomes": outcomes, "rows": len(batch_rows),
            "seconds": round(time.perf_counter() - t_batch, 2)}
        batches.append(batch_doc)
        if emit_outbox:
            emit_batch_envelope(batch_doc, census=census,
                                measured_at=measured_at,
                                terminal_attempt=terminal_attempt,
                                code_digest=code_digest)
        print(json.dumps(batch_doc, sort_keys=True), flush=True)

    final = existing_terminals(state_dir)
    tally = {o: 0 for o in OUTCOMES}
    for doc in final.values():
        tally[doc["outcome"]] = tally.get(doc["outcome"], 0) + 1
    covered = len(final)
    ledger = {
        "schema": LEDGER_SCHEMA,
        "measured_at": measured_at,
        "terminal_attempt": terminal_attempt,
        "census_sha256": census["census_sha256"],
        "pre_ledger_sha256": pre["pre_ledger_sha256"],
        "code_identity": code_digest,
        "protocol_version": ch.PROTOCOL_VERSION,
        "coverage": {
            "conceptual_variables_declared": pre["conceptual_variables"],
            "conceptual_variables_with_a_terminal": covered,
            "complete": covered == pre["conceptual_variables"],
            "physical_appearances_declared": pre["physical_appearances"],
            "note": "conceptual variables and physical appearances are "
                    "different populations; neither count is the other",
        },
        "outcomes": tally,
        "batches": batches,
        "rows_loaded": rows_loaded,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "selection_emitted": "NONE",
        "confirmation_used": "NONE",
        "gpu_used": "NONE",
        "previous_run_kept": "the 107-subject v2 observation is a prior "
                             "observation and is not edited to feign "
                             "coverage",
    }
    ledger["ledger_sha256"] = sha_obj(ledger)
    return ledger


def disposition_rows(terminals: dict, *, measured_at: str,
                     terminal_attempt: str, code_digest: str,
                     census_sha: str) -> list[dict]:
    """One row per conceptual variable recording its DISPOSITION.

    The descriptor rows only exist for variables that were measured, so
    the cube showed 1,612 variables while 1,965 had a terminal. A
    disposition that lives only in a state directory is not a reported
    outcome: the cube must be able to answer "did every variable end
    somewhere" without being handed a file.
    """
    rows = []
    for vid, t in sorted(terminals.items()):
        outcome = t["outcome"]
        binding = {
            "source_id": t.get("appearance") or "NO_APPEARANCE",
            "source_sha256": census_sha,
            "window_sha256": census_sha,
            "window_contract": json.dumps(
                {"rule": f"first {DEV_FRACTION:.0%} of rows in file "
                         "order", "row_cap": ROW_CAP},
                sort_keys=True),
            "code_identity": code_digest,
            "protocol_version": ch.PROTOCOL_VERSION,
            "side": "x", "contract_role": "input",
            "units": "DISPOSITION",
            "terminal_attempt": terminal_attempt,
        }
        row = {
            "variable_id": vid,
            "partition_key": f"{t.get('entity') or 'UNKNOWN'}"
                             "::development",
            "bank_authority": ch.BANK_FINANCIAL,
            "descriptor": "characterization_disposition",
            "value": None,
            "value_text": outcome,
            "descriptor_contract": (
                "the terminal disposition of this conceptual variable "
                "in the full-census characterization: exactly one of "
                f"{list(OUTCOMES)}. "
                + (t.get("reason") or "measured")),
            "identifiable": outcome == MEASURED,
            "cost_seconds": 0.0,
            "measured_at": measured_at,
            "binding_state": "BOUND_TO_SOURCE_BYTES",
            **binding,
        }
        row["observation_sha256"] = ch._sha({
            "variable_id": vid, "partition_key": row["partition_key"],
            "descriptor": row["descriptor"],
            "value_text": row["value_text"],
            "identifiable": row["identifiable"],
            "contract": row["descriptor_contract"],
            "bank": row["bank_authority"], "binding": binding})
        row["measurement_sha256"] = ch._sha(
            {"observation": row["observation_sha256"],
             "measured_at": measured_at, "cost_seconds": 0.0})
        rows.append(row)
    return rows


def emit_from_terminals(census: dict, state_dir: Path, *,
                        measured_at: str,
                        terminal_attempt: str) -> dict:
    """Emit what the write-once terminals already record.

    Measurement is not repeated: the terminals ARE the outcome, and a
    second measurement would be a second observation of the same bytes
    for no reason. This exists because the first emission used a kind
    the loader cannot carry.
    """
    from eligibility.consumed import code_identity, local_code_surface
    code_digest, _ = code_identity(REPO, local_code_surface(REPO), {})
    terminals = existing_terminals(state_dir)
    pre = json.loads((state_dir / "PRE_LEDGER.json").read_text())
    by_batch: dict[str, dict] = {}
    for doc in terminals.values():
        b = by_batch.setdefault(doc.get("batch", "batch_unknown"),
                                {o: 0 for o in OUTCOMES})
        b[doc["outcome"]] = b.get(doc["outcome"], 0) + 1
    emitted = []
    for batch, outcomes in sorted(by_batch.items()):
        doc = {"batch": batch, "outcomes": outcomes,
               "variables": sum(outcomes.values()), "rows": 0,
               "seconds": 0.0}
        emitted.append(emit_batch_envelope(
            doc, census=census, measured_at=measured_at,
            terminal_attempt=terminal_attempt,
            code_digest=code_digest))
    tally = {o: 0 for o in OUTCOMES}
    for d in terminals.values():
        tally[d["outcome"]] = tally.get(d["outcome"], 0) + 1

    rows = disposition_rows(terminals, measured_at=measured_at,
                            terminal_attempt=terminal_attempt,
                            code_digest=code_digest,
                            census_sha=census["census_sha256"])
    ch.assert_no_selection(rows)
    from tools.backfill_campaign_envelopes import _engine
    loaded = ch.load_rows(_engine(), rows).get(
        "fact_variable_characterization", 0)

    ledger = {
        "schema": LEDGER_SCHEMA,
        "mode": "EMIT_ONLY_FROM_WRITE_ONCE_TERMINALS",
        "disposition_rows_loaded": loaded,
        "measured_at": measured_at,
        "terminal_attempt": terminal_attempt,
        "census_sha256": census["census_sha256"],
        "pre_ledger_sha256": pre["pre_ledger_sha256"],
        "coverage": {
            "conceptual_variables_declared": pre["conceptual_variables"],
            "conceptual_variables_with_a_terminal": len(terminals),
            "complete": len(terminals) == pre["conceptual_variables"],
            "physical_appearances_declared": pre["physical_appearances"],
        },
        "outcomes": tally,
        "emitted": len(emitted),
        "measurement_repeated": False,
    }
    ledger["ledger_sha256"] = sha_obj(ledger)
    return ledger


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--lake-root", required=True, type=Path)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--measured-at", required=True)
    ap.add_argument("--terminal-attempt", required=True)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--emit-outbox", action="store_true")
    ap.add_argument("--emit-only", action="store_true",
                    help="rebuild the batch summaries from the "
                         "write-once terminals and emit them, without "
                         "measuring anything again")
    a = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        os.nice(15)
    except OSError:
        pass
    census = json.loads(a.census.read_text())
    if a.emit_only:
        ledger = emit_from_terminals(
            census, a.state_dir.expanduser(),
            measured_at=a.measured_at,
            terminal_attempt=a.terminal_attempt)
        a.ledger.parent.mkdir(parents=True, exist_ok=True)
        a.ledger.write_text(json.dumps(ledger, indent=1,
                                       sort_keys=True) + "\n")
        print(json.dumps({k: ledger[k] for k in
                          ("coverage", "outcomes", "emitted",
                           "disposition_rows_loaded",
                           "ledger_sha256")}, indent=1,
                         sort_keys=True))
        return 0
    ledger = run(census, a.lake_root.expanduser(),
                 a.state_dir.expanduser(),
                 measured_at=a.measured_at,
                 terminal_attempt=a.terminal_attempt,
                 batch_size=a.batch_size, limit=a.limit,
                 load=a.load, emit_outbox=a.emit_outbox)
    a.ledger.parent.mkdir(parents=True, exist_ok=True)
    a.ledger.write_text(json.dumps(ledger, indent=1, sort_keys=True)
                        + "\n")
    print(json.dumps({k: ledger[k] for k in
                      ("coverage", "outcomes", "rows_loaded",
                       "wall_seconds", "ledger_sha256")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
