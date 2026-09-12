"""POST for C31-C44 (order 2026-09-11): each PRE counterexample, dead.

Same twelve items, same public APIs, against the CORRECTED code. Each
one must now refuse, or produce the right answer, for its own exact
reason — a generic failure is not a correction.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
HOME = Path.home()
PY = sys.executable

RESULTS: list[tuple[int, str, bool, str]] = []


def check(n: int, title: str, ok: bool, detail: str) -> None:
    RESULTS.append((n, title, ok, detail))
    print(f"\n=== POST {n}: {title} ===")
    print(f"  [{'CORRECTED' if ok else 'NOT-CORRECTED'}] {detail}")


def world(tmp: Path, *, x_cols="DATE_TIME,SAME,f1",
          y_cols="DATE_TIME,SAME,TARGET",
          partitions=("train", "validation", "test")) -> dict:
    cfg = {"predictor_plugin": "ann", "target_column": "TARGET"}
    for role in partitions:
        (tmp / f"x_{role}.csv").write_text(
            x_cols + "\n" + ",".join(["1"] * len(x_cols.split(","))) + "\n")
        (tmp / f"y_{role}.csv").write_text(
            y_cols + "\n" + ",".join(["2"] * len(y_cols.split(","))) + "\n")
        cfg[f"x_{role}_file"] = str(tmp / f"x_{role}.csv")
        cfg[f"y_{role}_file"] = str(tmp / f"y_{role}.csv")
    return cfg


# --------------------------------------------------------------- 1
from app.plugin_resolver import (canonical_name,                # noqa: E402
                                 declared_plugin_params, resolve)
try:
    canonical_name({"predictor_plugin": "cnn", "plugin": "ann"},
                   "predictor")
    ok1, why1 = False, "the disagreement was accepted"
except SystemExit as exc:
    ok1 = "disagree" in str(exc)
    why1 = str(exc)[:150]
check(1, "a disagreeing legacy alias refuses before any model exists",
      ok1, why1)

# --------------------------------------------------------------- 2
from eligibility.consumed import (code_identity,                # noqa: E402
                                  local_code_surface,
                                  resolve_consumed_subjects,
                                  resolve_plugin_modules)
surface = local_code_surface(REPO)
covered = "predictor_plugins/common/base.py" in surface
with tempfile.TemporaryDirectory() as d:
    clone = Path(d) / "c"
    import shutil
    shutil.copytree(REPO / "predictor_plugins", clone / "predictor_plugins")
    (clone / "app").mkdir(parents=True)
    (clone / "app/main.py").write_text("# x\n")
    surf = local_code_surface(clone)
    a, _ = code_identity(clone, surf, {})
    t = clone / "predictor_plugins/common/base.py"
    t.write_text(t.read_text() + "\n# mutated\n")
    b, _ = code_identity(clone, surf, {})
check(2, "mutating a shared plugin module changes the identity",
      covered and a != b,
      f"surface={len(surface)} files; base.py bound={covered}; "
      f"digest changed={a != b}")

# --------------------------------------------------------------- 3
from app.plugin_resolver import _declared_entry_points           # noqa: E402
declared = _declared_entry_points("predictor.plugins")
w = resolve("predictor", "ann")
check(3, "resolution falls back to this repository's own declaration",
      bool(declared) and w["inside_checkout"] is True,
      f"setup.py declares {len(declared)} predictor entry points; "
      f"resolution stays inside this checkout "
      f"(source={w['resolution_source']})")

# --------------------------------------------------------------- 4
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    c = resolve_consumed_subjects(world(tmp), repo_root=tmp)
    sides = {(s["side"], s["column"]) for s in c["subjects"]}
    ids = [s["subject_id"] for s in c["subjects"]]
check(4, "x::SAME and y::SAME are distinct subjects",
      ("x", "SAME") in sides and ("y", "SAME") in sides
      and len(ids) == len(set(ids)),
      f"subjects={sorted(sides)}; collisions="
      f"{len(ids) - len(set(ids))}")

# --------------------------------------------------------------- 5
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    cfg = world(tmp)
    (tmp / "y_test.csv").write_text("DATE_TIME,TARGET,EXTRA\n1,2,3\n")
    try:
        resolve_consumed_subjects(cfg, repo_root=tmp)
        ok5, why5 = False, "a divergent y schema was accepted"
    except SystemExit as exc:
        ok5 = "y schema differs" in str(exc)
        why5 = str(exc)[:150]
check(5, "a y schema that differs across partitions refuses", ok5, why5)

# --------------------------------------------------------------- 6
from olap import campaign_envelope as ce                         # noqa: E402


def _pg_env():
    f = HOME / ".config/crispdm/olap-loader.env"
    if not f.is_file():
        return None
    env = dict(os.environ)
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env if env.get("PGUSER") else None


def _envelope(**over):
    base = dict(campaign_key="q::post", producer="predictor",
                result_class="DEVELOPMENT", run="r1", state="COMPLETE",
                code="c" * 64, design="d" * 64, cell="cell-1")
    base.update(over)
    return ce.build_envelope(
        campaign_key=base["campaign_key"], producer=base["producer"],
        result_class=base["result_class"],
        identity={"run_id": base["run"], "code_identity": base["code"],
                  "design_sha256": base["design"]},
        data_consumed={"variables": [], "operators": [], "datasets": []},
        partitions={"exposure": "DEV", "splits": "t"},
        budget={"device": "cpu", "wall_seconds": 1.0,
                "cost_units": "wall_seconds"},
        terminal={"state": base["state"], "adjudication": "NONE"},
        artifacts={"verification": "BORN_AT_PRODUCER_TERMINAL"},
        units=[{"cell_key": base["cell"], "candidate_key": "ann",
                "metric_name": "m", "metric_value": 1.0,
                "terminal_state": base["state"]}])


PG = _pg_env()
if PG is None:
    check(6, "a run dimension carries the execution identity", False,
          "no PostgreSQL credentials on this host; behaviour not "
          "exercised")
else:
    import uuid
    from sqlalchemy import create_engine, text
    name = f"crispdm_post_{uuid.uuid4().hex[:10]}"
    admin_url = (f"postgresql+psycopg2://{PG['PGUSER']}:"
                 f"{PG['PGPASSWORD']}@{PG.get('PGHOST', 'localhost')}:"
                 f"{PG.get('PGPORT', '5432')}/postgres")
    admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'CREATE DATABASE "{name}"'))
    eng = create_engine(admin_url.rsplit("/", 1)[0] + "/" + name)
    try:
        ce.load_envelope(eng, _envelope(run="r1", state="FAILED"))
        ce.load_envelope(eng, _envelope(run="r2", state="COMPLETE",
                                        code="e" * 64, cell="cell-2"))
        with eng.connect() as c:
            runs = [dict(r) for r in c.execute(text(
                "SELECT run_id, terminal_state, code_identity FROM "
                "public.dim_campaign_run ORDER BY run_id")).mappings()]
            campaigns = c.execute(text(
                "SELECT count(*) FROM public.dim_campaign")).scalar()
        reused_refused = False
        try:
            ce.load_envelope(eng, _envelope(run="r1", code="f" * 64,
                                            cell="cell-3"))
        except SystemExit as exc:
            reused_refused = "DIFFERENT identity" in str(exc)
        foreign_refused = False
        try:
            ce.load_envelope(eng, _envelope(run="r9",
                                            producer="someone-else"))
        except SystemExit as exc:
            foreign_refused = "belongs to producer" in str(exc)
    finally:
        eng.dispose()
        with admin.connect() as c:
            c.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()
    check(6, "a run dimension carries the execution identity",
          len(runs) == 2 and campaigns == 1
          and [r["terminal_state"] for r in runs] == ["FAILED",
                                                      "COMPLETE"]
          and reused_refused and foreign_refused,
          f"one campaign, {len(runs)} attempts (FAILED then COMPLETE, "
          f"different code); a REUSED run_id with a different "
          f"execution refuses ({reused_refused}); a foreign producer "
          f"refuses ({foreign_refused})")

# --------------------------------------------------------------- 7
from olap import inventory_rows as ir                            # noqa: E402
# Strip SQL comments: the DDL DOCUMENTS the old ordering in prose, and
# a check that reads prose as code is a test of my writing, not of the
# views.
_sql = "\n".join(ln for ln in ir.INVENTORY_DDL.splitlines()
                 if not ln.strip().startswith("--"))
check(7, "`current` is the latest OBSERVATION, not the latest load",
      "loaded_at DESC" not in _sql
      and _sql.count("observed_at DESC") >= 4
      and _sql.count("_ambiguous") >= 4,
      f"no statement orders by load time; {_sql.count('observed_at DESC')} "
      f"orderings by observed_at with a supersession override, and "
      f"{_sql.count('CREATE VIEW') } views including the companions "
      f"that expose incomparable branches")

# --------------------------------------------------------------- 8
from olap import characterization as ch                          # noqa: E402
B = {"source_id": "a.csv", "source_sha256": "a" * 64,
     "window_sha256": "b" * 64, "window_contract": "rows 0..9",
     "code_identity": "c" * 64,
     "protocol_version": ch.PROTOCOL_VERSION, "side": "x",
     "contract_role": "input", "units": "u",
     "terminal_attempt": "run-1"}
vals = [float(i) for i in range(64)]
r1 = ch.characterize_series(vals, variable_id="v", partition_key="p",
                            bank_authority=ch.BANK_FINANCIAL,
                            measured_at="2026-09-12T00:00:00Z",
                            binding=B)
r2 = ch.characterize_series(vals, variable_id="v", partition_key="p",
                            bank_authority=ch.BANK_FINANCIAL,
                            measured_at="2026-09-12T00:00:00Z",
                            binding=dict(B, source_sha256="f" * 64))
try:
    ch.characterize_series(vals, variable_id="v", partition_key="p",
                           bank_authority=ch.BANK_FINANCIAL,
                           measured_at="2026-09-12T00:00:00Z",
                           binding={})
    refused_unbound = False
except SystemExit:
    refused_unbound = True
check(8, "different source bytes are different observations",
      r1[0]["observation_sha256"] != r2[0]["observation_sha256"]
      and refused_unbound,
      "the same descriptive value over different source bytes yields "
      "different observation digests, and an unbound measurement "
      "refuses outright")

# --------------------------------------------------------------- 9
import numpy as np                                               # noqa: E402
# One hole in 400 dilutes the defect to nothing, so the gap pattern is
# made SYSTEMATIC: every third sample is missing. Compacting then pairs
# samples that are two positions apart on most of the series, and the
# reported "lag 1" becomes a lag-1.5 of a series that does not exist.
t = np.arange(600.0)
clean = np.sin(2 * np.pi * t / 12.0)
holed = clean.copy()
holed[2::3] = np.nan


def desc(v, **kw):
    return {r["descriptor"]: (r["value"], r["identifiable"])
            for r in ch.characterize_series(
                v, variable_id="v", partition_key="p",
                bank_authority=ch.BANK_SYNTHETIC,
                measured_at="2026-09-12T00:00:00Z", binding=B, **kw)}


def pre_acf1(v):
    """What the PRE code computed: compact, THEN shift."""
    f = v[np.isfinite(v)]
    a_, b_ = f[:-1], f[1:]
    return float(np.corrcoef(a_, b_)[0, 1])


def true_acf1(v):
    """Adjacent ORIGINAL positions, both finite."""
    a_, b_ = v[:-1], v[1:]
    k = np.isfinite(a_) & np.isfinite(b_)
    return float(np.corrcoef(a_[k], b_[k])[0, 1])


a, b = desc(clean), desc(holed)
c_imp = desc(holed, imputation="linear_interpolation")
post_val = b["autocorrelation_lag1"][0]
pre_val = pre_acf1(holed)
truth = true_acf1(holed)
check(9, "a gap is skipped, not closed, and the spectrum refuses",
      abs(post_val - truth) < 1e-9
      # 0.18 on a correlation is not a rounding difference: it is the
      # third significant figure of the answer being wrong.
      and abs(pre_val - truth) > 0.1
      and b["spectral_centroid"] == (None, False)
      and c_imp["spectral_centroid"][1] is True,
      f"with every third sample missing: PRE reports {pre_val:+.4f}, "
      f"POST reports {post_val:+.4f}, the truth over adjacent finite "
      f"pairs is {truth:+.4f}; the spectrum is NOT identifiable "
      f"without a predeclared imputation and IS with one")

# --------------------------------------------------------------- 10
runner = (REPO / "tools/run_characterization_v2.py").read_text()
ledger_p = (REPO /
            "docs/audits/evidence/characterization_v2_ledger_2026_09_12.json")
ledger = json.loads(ledger_p.read_text()) if ledger_p.is_file() else {}
axis = ledger.get("axis_contracts", [])
temporal_measured = any(
    "DATE_TIME" in str(att.get("variable_id", ""))
    for att in ledger.get("attempts", []))
check(10, "a timestamp is an axis contract, never a numeric variable",
      "def is_temporal" in runner and axis and not temporal_measured
      and all(c["state"] in ("DECLARED_AS_AXIS_NOT_MEASURED",
                             "NO_TEMPORAL_IDENTIFIER") for c in axis),
      f"{len(axis)} axis contracts recorded; zero temporal columns "
      f"received a descriptor")

# --------------------------------------------------------------- 11
from olap import terminal as T                                   # noqa: E402
main_src = (REPO / "app/main.py").read_text()
with tempfile.TemporaryDirectory() as d:
    tmp = Path(d)
    os.environ["CRISPDM_OLAP_OUTBOX"] = str(tmp / "outbox")
    from olap import outbox as ob
    ob.ensure_outbox(tmp / "outbox")
    shared: dict = {}
    late = tmp / "results/phase"

    def boom(*_a, **_k):
        raise OSError("outbox unwritable")
    real_emit = T.ob.emit
    T.ob.emit = boom
    try:
        def body():
            shared["results_dir"] = late
            return {"ok": True}
        T.terminal_run(body, campaign_key=lambda: "k",
                       producer="predictor", config=lambda: {},
                       results_dir=lambda: shared.get("results_dir"))
        T.terminal_run(lambda: {"terminal_state": "INCONCLUSIVE"},
                       campaign_key=lambda: "k", producer="predictor",
                       config=lambda: {},
                       results_dir=lambda: shared.get("results_dir"))
    finally:
        T.ob.emit = real_emit
    gaps = sorted(late.glob(f"{T.OPERATIONAL_GAP_STEM}-*.json"))
check(11, "an outbox failure leaves ONE gap per run beside the results",
      "results_dir=lambda:" in main_src and len(gaps) == 2,
      f"{len(gaps)} gap files written for two runs into one directory; "
      f"the call site binds results_dir lazily")

# --------------------------------------------------------------- 12
from olap import outbox as ob                                    # noqa: E402
with tempfile.TemporaryDirectory() as d:
    root = Path(d) / "ob"
    ob.ensure_outbox(root)
    ob.emit({"x": 1}, kind="event", root=root)
    p = next(iter(ob.pending_entries(root)))
    ob.mark(p, ob.FAILED, reason="REFUSED: synthetic", root=root)
    hb = ob.heartbeat(root)
    healthy_with_dl = hb["healthy"]
    attention = hb["attention_required"]
    import time as _t
    stale = ob.health(root, now=_t.time() + 10_000)["healthy"]
check(12, "health is the service; a dead-letter is a separate signal",
      healthy_with_dl is True and attention is True and stale is False,
      "a known refusal leaves `healthy` true and raises "
      "`attention_required`; a stale heartbeat makes `healthy` false")

# ---------------------------------------------------------------
bad = [n for n, _t2, ok, _d in RESULTS if not ok]
print(f"\n=== POST SUMMARY: {len(RESULTS) - len(bad)}/{len(RESULTS)} "
      f"CORRECTED ===")
for n in bad:
    print(f"  NOT-CORRECTED: item {n}")
sys.exit(1 if bad else 0)
