"""PRE for B4 R11-R14, T2 R7-R10 and CRISP-DM C45-C52 (order 2026-09-12).

Frozen BEFORE any edit this time. The previous return had to confess
that the C-block PRE was written after the corrections; this file was
written and run first, against the audited tips, and nothing in the
working trees had been touched when it produced its output.

The B4 and T2 items work on COPIES of their roots under
~/.local/state/crispdm-pre-copies. The real evidence roots are never
opened for writing here; their digests are checked before and after.

Private paths are redacted to `~`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
GH = HOME / "Documents/GitHub"
COPIES = HOME / ".local/state/crispdm-pre-copies"
B4_COPY = COPIES / "b4_v7"
B4_REAL = HOME / ".local/share/agent-multi/b4_campaign_results_v7_20260907"
T2_REAL = (HOME / ".local/share/agent-multi"
           / "t2_confirmatory_results_resource_successor_v1_20260909")
AM_B4 = GH / ".worktrees/am-data-first"
AM_T2 = GH / ".worktrees/am-t0t1"
PREDICTOR = GH / "predictor"
FINANCIAL = GH / "financial-data"

FAILURES: list[str] = []


def redact(v) -> str:
    return str(v).replace(str(HOME), "~")


def item(n: int, title: str) -> None:
    print(f"\n=== PRE {n}: {title} ===")


def expect(cond: bool, msg: str) -> None:
    print(f"  [{'PRE-REPRODUCED' if cond else 'NOT-REPRODUCED'}] {msg}")
    if not cond:
        FAILURES.append(f"{n_current}: {msg}")


n_current = 0


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root: Path) -> str:
    """Cheap structural digest: names, sizes and mtimes. Enough to
    prove the real roots were not written to."""
    entries = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            st = p.stat()
            entries.append((str(p.relative_to(root)), st.st_size,
                            st.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


REAL_BEFORE = {"b4": tree_digest(B4_REAL), "t2": tree_digest(T2_REAL)}

sys.path.insert(0, str(AM_B4 / "tools"))
import b4_campaign_closure as B4C                              # noqa: E402


# ------------------------------------------------------------ helpers
def synth_b4(tmp: Path):
    """A miniature B4 root with the same record shapes, built through
    the shipped fixture builder so the PRE drives real code."""
    sys.path.insert(0, str(AM_B4 / "tests"))
    import test_b4_campaign_closure as fixture                 # noqa: E402
    return fixture.build_root(tmp, cells=12, completed=2, partial=1)


def swap_after(module, name: str, replacement_path: Path,
               new_bytes: bytes):
    """Wrap `module.name` so the file is REPLACED right after the
    wrapped call returns — the classic time-of-check/time-of-use
    window between two independent path opens."""
    original = getattr(module, name)
    state = {"swapped": False}

    def wrapper(*a, **k):
        out = original(*a, **k)
        if not state["swapped"]:
            state["swapped"] = True
            replacement_path.write_bytes(new_bytes)
        return out

    setattr(module, name, wrapper)
    return original, state


# ============================================================ B4: 1-5
n_current = 1
item(1, "the B4 terminal is parsed from one open and hashed from "
        "another, so the record can describe bytes that are not on disk")
with tempfile.TemporaryDirectory() as d:
    root, mat = synth_b4(Path(d))
    cell = root / "o2022_seed101"
    term_p = cell / "B4_CELL_TERMINAL.json"
    honest = json.loads(term_p.read_text())
    forged_bytes = json.dumps(dict(honest, wall_seconds=999999.0),
                              indent=1).encode()
    forged_digest = hashlib.sha256(forged_bytes).hexdigest()
    # An adversary that can replace one file can replace the seal pair
    # too; the point is that the closure's own two reads disagree.
    intent_p = next(cell.glob("SEAL_INTENT_*.json"))
    seal_p = next(cell.glob("SEAL_COMPLETE_*.json"))
    intent_doc = json.loads(intent_p.read_text())
    intent_doc["terminal_sha256"] = forged_digest
    intent_bytes = json.dumps(intent_doc, indent=1).encode()
    intent_p.write_bytes(intent_bytes)
    seal_doc = json.loads(seal_p.read_text())
    seal_doc["intent_sha256"] = hashlib.sha256(intent_bytes).hexdigest()
    seal_doc["terminal_sha256"] = forged_digest
    seal_p.write_text(json.dumps(seal_doc, indent=1))

    original_read = Path.read_text
    state = {"swapped": False}

    def read_then_swap(self, *a, **k):
        out = original_read(self, *a, **k)
        if self == term_p and not state["swapped"]:
            state["swapped"] = True
            term_p.write_bytes(forged_bytes)   # between parse and hash
        return out

    Path.read_text = read_then_swap
    try:
        closure = B4C.build_closure(root, mat)
        done = [c for c in closure["cells"]
                if c["classification"] == B4C.COMPLETED
                and c["cell"] == "o2022_seed101"][0]
        consumed_wall = done["wall_seconds_declared"]
        verified = all(x["verified"] for x in done["descriptors_verified"])
    except SystemExit as exc:
        consumed_wall, verified = f"REFUSED: {str(exc)[:90]}", False
    finally:
        Path.read_text = original_read
    on_disk = json.loads(term_p.read_text())["wall_seconds"]
print(f"  every descriptor reported verified : {verified}")
print(f"  wall_seconds the closure recorded  : {consumed_wall}")
print(f"  wall_seconds actually on disk      : {on_disk}")
expect(verified and consumed_wall == honest["wall_seconds"]
       and on_disk == 999999.0,
       "the adjudication is built from the FIRST open and every digest "
       "from a LATER one: it certifies bytes it did not consume")

n_current = 2
item(2, "the per-bar ledger is hashed from one open and counted from "
        "another")
with tempfile.TemporaryDirectory() as d:
    root, mat = synth_b4(Path(d))
    per_bar = root / "o2022_seed101/per_bar_o2022_seed101.csv"
    honest_bytes = per_bar.read_bytes()
    # SAME row count, different numbers: nothing downstream can notice
    forged = b"net_return\n9.9\n9.9\n"
    original_sha = B4C.sha_file

    def sha_then_swap(p):
        out = original_sha(p)
        if Path(p) == per_bar and per_bar.read_bytes() == honest_bytes:
            per_bar.write_bytes(forged)
        return out

    B4C.sha_file = sha_then_swap
    try:
        closure = B4C.build_closure(root, mat)
        done = [c for c in closure["cells"]
                if c["cell"] == "o2022_seed101"][0]
        accepted = all(x["verified"]
                       for x in done["descriptors_verified"])
        recorded_digest = done["per_bar_sha256"]
    except SystemExit as exc:
        accepted, recorded_digest = False, f"REFUSED: {str(exc)[:80]}"
    finally:
        B4C.sha_file = original_sha
    now = per_bar.read_bytes()
print(f"  bytes hashed  : {honest_bytes!r}")
print(f"  bytes on disk : {now!r}")
print(f"  closure accepted the cell : {accepted}")
print(f"  digest recorded for evidence that is no longer there : "
      f"{str(recorded_digest)[:32]}")
expect(accepted and now != honest_bytes,
       "the digest check and the row count are independent opens, so a "
       "replacement that preserves the row count is consumed without "
       "any guard noticing")

n_current = 3
item(3, "a claim, intent or seal can be replaced between parse and hash")
with tempfile.TemporaryDirectory() as d:
    root, mat = synth_b4(Path(d))
    cell = root / "o2022_seed101"
    intent = next(cell.glob("SEAL_INTENT_*.json"))
    honest_intent = json.loads(intent.read_text())
    seal = next(cell.glob("SEAL_COMPLETE_*.json"))
    # The completion binds sha256(intent). The intent is PARSED first
    # (for terminal_sha256) and HASHED afterwards (for the seal
    # binding); replacing it in between binds a different object.
    src = (AM_B4 / "tools/b4_campaign_closure.py").read_text()
    parse_at = src.index('intent_doc = json.loads(intent.read_text())')
    hash_at = src.index('seal_doc.get("intent_sha256") == sha_file(intent)')
print(f"  intent parsed at source offset : {parse_at}")
print(f"  intent hashed at source offset : {hash_at}  "
      f"(a later, INDEPENDENT open)")
print(f"  same file opened twice         : {parse_at < hash_at}")
expect(parse_at < hash_at,
       "the seal chain parses the intent and then re-opens it to hash "
       "it; the two reads are not guaranteed to see the same bytes")

n_current = 4
item(4, "objects of the partial cell can change after the inventory")
with tempfile.TemporaryDirectory() as d:
    root, mat = synth_b4(Path(d))
    part = root / "o2022_seed103"   # cells=12, completed=2
    status = part / "cell_runtime/status.json"
    honest_status = json.loads(status.read_text())
    original_inv = B4C.inventory

    def inv_then_swap(r, **k):
        out = original_inv(r, **k)
        if Path(r) == part and status.is_file():
            status.write_text(json.dumps(
                {"epoch_completed": 1999, "num_timesteps": 1,
                 "stop_reason": "FORGED",
                 "last_durable_artifact": None}))
        return out

    B4C.inventory = inv_then_swap
    try:
        closure = B4C.build_closure(root, mat)
        p = [c for c in closure["cells"]
             if c["classification"] == B4C.PARTIAL][0]
        consumed = p["last_durable_progress"]
    except SystemExit as exc:
        consumed = f"REFUSED: {str(exc)[:80]}"
    finally:
        B4C.inventory = original_inv
print(f"  epoch when inventoried : {honest_status['epoch_completed']}")
print(f"  epoch adjudicated      : {consumed}")
expect(isinstance(consumed, dict)
       and consumed.get("epoch_completed") == 1999,
       "the inventory hashes the artifacts and the classification then "
       "RE-READS status.json, so the adjudicated progress is not the "
       "inventoried progress")

n_current = 5
item(5, "an artifact created after the absence check keeps NOT_STARTED")
with tempfile.TemporaryDirectory() as d:
    root, mat = synth_b4(Path(d))
    ghost = root / "o2022_seed104"
    original_ns = B4C.record_not_started

    def ns_then_create(r, cell):
        out = original_ns(r, cell)
        if cell == "o2022_seed104" and not ghost.exists():
            ghost.mkdir(parents=True)
            (ghost / "B4_CELL_TERMINAL.json").write_text("{}")
        return out

    B4C.record_not_started = ns_then_create
    try:
        closure = B4C.build_closure(root, mat)
        states = {c["cell"]: c["classification"] for c in closure["cells"]}
        ghost_state = states.get("o2022_seed104")
    except SystemExit as exc:
        ghost_state = f"REFUSED: {str(exc)[:80]}"
    finally:
        B4C.record_not_started = original_ns
    ghost_exists = ghost.exists()
    ghost_terminal = (ghost / "B4_CELL_TERMINAL.json").is_file()
print(f"  the cell directory exists afterwards : {ghost_exists}")
print(f"  it even carries a terminal           : {ghost_terminal}")
print(f"  its recorded classification          : {ghost_state}")
expect(ghost_state == B4C.NOT_STARTED and ghost_exists
       and ghost_terminal,
       "absence is decided by a bare existence check, so a cell that "
       "appears afterwards keeps the classification NOT_STARTED")

# ========================================================= C42-C44: 6-12
sys.path.insert(0, str(FINANCIAL / "_scripts"))
import derive_availability_bridge as BRIDGE                    # noqa: E402
import derive_feature_lineage as LINEAGE                       # noqa: E402

demand = BRIDGE.demand_columns(PREDICTOR, GH / "agent-multi")

n_current = 6
item(6, "demand is derived from a research registry, not from a "
        "consumer")
inv = json.loads((PREDICTOR /
                  "examples/research/crispdm_dataset_inventory.v1.json"
                  ).read_text())
paths = [d["relative_path"] for d in inv["datasets"]]
bridge_src = (FINANCIAL / "_scripts/derive_availability_bridge.py"
              ).read_text()
sup_block = bridge_src[bridge_src.index("supervised, targets, sources"):
                       bridge_src.index("rl, rl_configs")]
consults_config = "config" in sup_block.lower()
print(f"  registry datasets : {[Path(x).name for x in paths]}")
print(f"  the supervised branch consults any config : "
      f"{consults_config}")
config_roots = (PREDICTOR / "examples/config",
                GH / "agent-multi/examples/config")
for rel in paths:
    hits = subprocess.run(
        ("grep", "-rl", Path(rel).name, *[str(r) for r in config_roots]),
        capture_output=True, text=True).stdout.strip().splitlines()
    bound = []
    for h in hits:
        try:
            doc = json.loads(Path(h).read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(doc, dict) and "observation_contract" in doc:
            bound.append(h)
    print(f"  {Path(rel).name:52s} configs={len(hits):3d} "
          f"with observation_contract={len(bound)}")
print(f"  supervised demand published        : "
      f"{len(demand['supervised_columns'])}")
print(f"  active RL configs                  : "
      f"{len(demand['rl_configs'])}")
expect(not consults_config and len(demand["supervised_columns"]) > 0
       and len(demand["rl_configs"]) == 0,
       "supervised demand is read straight from the registry headers "
       "without ever consulting a config, so a positive count is "
       "published while no active consumer has been demonstrated")

n_current = 7
item(7, "an empty RL set produces a positive subset claim")
bridge_doc = json.loads((FINANCIAL /
                         "features/census/AVAILABILITY_BRIDGE.v2.json"
                         ).read_text())
claim = bridge_doc["demand"]["rl_is_subset_of_subset"] \
    if "rl_is_subset_of_subset" in bridge_doc["demand"] \
    else bridge_doc["demand"].get("rl_is_subset_of_supervised")
print(f"  rl columns        : {bridge_doc['demand']['rl_columns']}")
print(f"  subset claim      : {claim}")
expect(bridge_doc["demand"]["rl_columns"] == 0 and claim is True,
       "the empty set satisfies every subset test, so the report "
       "states a positive relation that means nothing")

n_current = 8
item(8, "two different grains are reported side by side as one")
lineage_doc = json.loads((FINANCIAL /
                          "features/census/FEATURE_LINEAGE.v1.json"
                          ).read_text())
print(f"  unique column NAMES (bridge)          : "
      f"{bridge_doc['demand']['supervised_columns']}")
print(f"  (dataset, column) SUBJECTS (lineage)  : "
      f"{lineage_doc['columns_examined']}")
names = set(demand["supervised_columns"])
subjects = {(n["dataset_id"], n["column"]) for n in
            lineage_doc["lineage"]}
print(f"  recomputed: names={len(names)} subjects={len(subjects)}")
expect(len(names) != len(subjects)
       and "dataset_column_subjects" not in bridge_doc["demand"],
       "93 unique names and 97 (dataset, column) subjects are "
       "published without a field that distinguishes the two grains")

n_current = 9
item(9, "a column with neither a target declaration nor a consuming "
        "config enters as an active input")
legacy = [n for n in lineage_doc["lineage"]
          if "eurusd" in n["dataset_id"]]
print(f"  legacy view columns treated as inputs : {len(legacy)}")
print(f"  declared targets in the registry      : "
      f"{bridge_doc['demand']['supervised_targets']}")
sample = legacy[0] if legacy else {}
print(f"  example                               : "
      f"{sample.get('column')} kind={sample.get('kind')}")
expect(len(legacy) > 0 and bridge_doc["demand"]["supervised_targets"] == [],
       "with no target metadata every column becomes an input, so a "
       "label could be consumed as a feature and nothing would say so")

n_current = 10
item(10, "every derived feature receives the same upstream sentence")
upstreams = {n.get("upstream") for n in lineage_doc["lineage"]
             if n["kind"] == "DERIVED_FEATURE"}
derived = [n for n in lineage_doc["lineage"]
           if n["kind"] == "DERIVED_FEATURE"]
print(f"  derived features            : {len(derived)}")
print(f"  DISTINCT upstream sentences : {len(upstreams)}")
for u in list(upstreams)[:1]:
    print(f"    {u[:110]}")
# Scope to the ONE view whose features actually resolved: the legacy
# view is UNAVAILABLE for a different reason and would mask the point.
resolved_ds = "financial_data.project3.ethusdt_4h_tech_stat.model_ready.v1"
in_view = [n for n in derived if n["dataset_id"] == resolved_ds]
avail = {n["earliest_available_time"] for n in in_view}
print(f"  derived features in the resolved view : {len(in_view)}")
print(f"  DISTINCT availability formulas there  : {len(avail)}")
expect(len(in_view) > 1 and len(upstreams) == 1 and len(avail) == 1,
       "a rolling mean, a lagged return and an externally published "
       "series all receive one generic provenance and one generic "
       "availability")

n_current = 11
item(11, "changing the real producer of a feature does not change its "
         "lineage")
src = (FINANCIAL / "_scripts/derive_feature_lineage.py").read_text()
inspects = [tok for tok in ("feature-eng", "feature_eng",
                            "feature-extractor", "feature_extractor")
            if tok in src]
print(f"  producer repositories inspected by the tool : "
      f"{inspects or 'NONE'}")
for field in ("commit", "function", "lookback", "shift", "alignment",
              "producer_code_sha256"):
    print(f"  node field {field:22s}: "
          f"{any(field in n for n in lineage_doc['lineage'])}")
expect(not inspects,
       "the tool never opens a producer repository, so mutating the "
       "code that computes a feature cannot change its lineage")

n_current = 12
item(12, "the characterization ledger ends green with 107 of 1,965")
ledger = json.loads((PREDICTOR / "docs/audits/evidence"
                     / "characterization_v2_ledger_2026_09_12.json"
                     ).read_text())
cov = ledger["coverage"]
census_total = 1965
print(f"  conceptual variables attempted : "
      f"{cov['conceptual_variables_attempted']}")
print(f"  outcomes                        : {ledger['outcomes']}")
print(f"  lake conceptual variables       : {census_total}")
missing = census_total - cov["conceptual_variables_attempted"]
print(f"  variables with NO outcome       : {missing}")
expect(ledger["outcomes"]["FAILED"] == 0
       and ledger["outcomes"]["UNAVAILABLE"] == 0
       and missing == 1858,
       "every attempted subject succeeded, so the ledger reads as "
       "complete while 1,858 conceptual variables have no disposition "
       "at all")

# ============================================================ T2: 13-16
n_current = 13
item(13, "a T2 record can be verified and a different one scored")
recon_src = (AM_T2 / "tools/t2_completion_reconstruction.py").read_text()
adj_at = recon_src.index("counts = ex.final_adjudication(")
reread_at = recon_src.index('root / "units" / f"RECORD_{safe}.json"')
score_at = recon_src.index("screen = conf.adjudicate_screen(")
print(f"  final_adjudication at offset : {adj_at}")
print(f"  RECORD re-opened by path at  : {reread_at}")
print(f"  adjudicate_screen at         : {score_at}")
print(f"  verify -> reopen -> score    : "
      f"{adj_at < reread_at < score_at}")
expect(adj_at < reread_at < score_at,
       "the units are verified, then EVERY record is re-opened by "
       "path to build the list that is scored")

n_current = 14
item(14, "the closure's own code is not part of the identity it calls "
         "reviewed")
sys.path.insert(0, str(AM_T2 / "tools"))
record_path = (HOME / ".config/agent-multi/reviewer_authority"
               / "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json")
record = json.loads(record_path.read_text())
pinned = sorted(record["executor_code_identity"])
for extra in ("tools/t2_completion_reconstruction.py",
              "tools/t2_campaign_closure.py"):
    print(f"  {extra:44s} pinned: {extra in pinned}")
closure_src = (AM_T2 / "tools/t2_campaign_closure.py").read_text()
mixes = ("sys.path.insert(0, str(TIP_REPO" in closure_src
         or 'str(TIP_REPO / "tools")' in closure_src)
print(f"  the closure imports the reconstructor from the TIP : {mixes}")
expect("tools/t2_completion_reconstruction.py" not in pinned
       and "tools/t2_campaign_closure.py" not in pinned and mixes,
       "the reconstruction driver and the closure itself come from the "
       "branch tip while the output is labelled a reviewed identity")

n_current = 15
item(15, "no dedicated battery exercises the T2 closure")
candidates = sorted((AM_T2 / "tests").rglob("*t2_campaign_closure*"))
print(f"  test files naming t2_campaign_closure : "
      f"{[p.name for p in candidates] or 'NONE'}")
expect(not candidates,
       "there is no battery that could make the two defects above bite")

n_current = 16
item(16, "T2 publishes a p-value of 1.3125")
conf_src = (AM_T2 / "tools/t2_confirmatory.py").read_text()
snippet = "2 * (0.5 ** 6) * sum(" in conf_src
table = {}
for s in range(7):
    if s >= 3:
        table[s] = round(2 * (0.5 ** 6)
                         * sum(math.comb(6, k) for k in range(s, 7)), 5)
    else:
        table[s] = None
print(f"  doubled one-tail formula present : {snippet}")
print(f"  published table for 0..6 positives:")
for s, v in table.items():
    flag = "  <-- outside [0,1]" if isinstance(v, float) and v > 1 else ""
    print(f"    {s} positives -> {v}{flag}")
closure_log = T2_REAL / "T2_CAMPAIGN_CLOSURE.jsonl"
if closure_log.is_file():
    last = json.loads(closure_log.read_text().splitlines()[-1])
    published = last["screen_adjudication"].get(
        "sign_test_exact_p_two_sided")
    print(f"  value in the durable closure record : {published}")
expect(snippet and table[3] > 1.0 and table[0] is None,
       "the doubled one-tail exceeds 1 at three positives and the "
       "formula is not defined below three, so it is neither bounded "
       "nor symmetric")

# ---------------------------------------------------------------
real_after = {"b4": tree_digest(B4_REAL), "t2": tree_digest(T2_REAL)}
print(f"\n=== REAL EVIDENCE ROOTS ===")
for k in REAL_BEFORE:
    same = REAL_BEFORE[k] == real_after[k]
    print(f"  {k}: unchanged={same}")
    if not same:
        FAILURES.append(f"the real {k} root was modified by the PRE")

print(f"\n=== PRE SUMMARY: 16 items; {len(FAILURES)} NOT reproduced ===")
for f in FAILURES:
    print(f"  NOT-REPRODUCED: {f}")
print("Copies live under ~/.local/state/crispdm-pre-copies; every "
      "destructive step above ran on a temporary synthetic root.")
sys.exit(1 if FAILURES else 0)
