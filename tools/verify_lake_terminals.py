#!/usr/bin/env python3
"""C59-C61 (order 2026-09-12): verify the 1,965 terminals from the
OUTSIDE, then supersede them ADDITIVELY.

C59 — INDEPENDENCE. This module imports nothing from
`characterize_lake`. It does not reuse its census reader, its
normalization, its digest helper or its path builder. A verifier that
borrows the producer's code verifies that the code agrees with itself,
which is the one thing never in doubt. Everything here is re-derived
from bytes, under descriptor-first custody: one path resolution, one
retained descriptor per directory, one read per artifact, and every
fact from that read.

What it checks, in order, refusing rather than repairing:

  1. the PRE ledger names N identities and is self-consistent;
  2. the terminals directory holds exactly those N, with no extra,
     no duplicate and no missing;
  3. every terminal's file NAME is derivable from its own
     `variable_id`, so a terminal cannot be filed under another
     variable's name;
  4. every terminal's `variable_id` appears in the PRE ledger;
  5. every MEASURED terminal names a source that still exists, and
     the source's bytes are re-digested HERE;
  6. the directory instance (device, inode, uid, mode) is published,
     so a directory swapped between the ledger read and the terminal
     reads is visible rather than invisible.

C60-C61 — the v1 terminal carries an outcome and a count. It does not
carry the schema it was written against, the digest of the bytes it
measured, the window contract it used, or a digest of itself. Four
absences, each of which lets a terminal be silently wrong. The v2
terminal carries all four and is written BESIDE v1 in a separate
directory: v1 is never opened for writing, never moved, never
rewritten. Supersession is additive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from descriptor_custody import Custody, CustodyRefusal  # noqa: E402

V2_SCHEMA = "crispdm.lake_characterization_terminal.v2"
REPORT_SCHEMA = "crispdm.lake_terminal_verification.v1"
MEASURED = "MEASURED"


class VerificationRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")
        self.reason = msg


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_obj(o) -> str:
    return sha_bytes(
        json.dumps(o, sort_keys=True, separators=(",", ":"),
                   default=str).encode())


#: candidate naming rules. The verifier does not KNOW which one the
#: producer used and does not import its code to find out — my first
#: attempt simply guessed md5 and declared all 1,965 terminals
#: misfiled. What matters is not which rule was chosen but that ONE
#: fixed rule explains every filename: that is what makes it
#: impossible for a terminal to be filed under another variable's
#: name. So every candidate is tested against every terminal, and the
#: rules that survive are published.
NAMING_RULES = {
    "sha256[:32]": lambda v: hashlib.sha256(v.encode()).hexdigest()[:32],
    "sha256": lambda v: hashlib.sha256(v.encode()).hexdigest(),
    "sha256[:16]": lambda v: hashlib.sha256(v.encode()).hexdigest()[:16],
    "md5": lambda v: hashlib.md5(v.encode()).hexdigest(),
    "sha1": lambda v: hashlib.sha1(v.encode()).hexdigest(),
    "blake2b[:32]": lambda v: hashlib.blake2b(
        v.encode()).hexdigest()[:32],
    "identity": lambda v: v,
}


def surviving_naming_rules(pairs) -> list[str]:
    """Rules consistent with EVERY (variable_id, filename) pair."""
    return sorted(
        name for name, fn in NAMING_RULES.items()
        if all(fn(vid) + ".json" == fname for vid, fname in pairs))


# ------------------------------------------------------------ C59
def load_census(census_path: Path) -> dict:
    """Read the census under its OWN custody and check that its
    content address is the name it is filed under. The verifier needs
    the census to know which FILE each variable came from; it reads
    the artifact, never the producer's reader."""
    census_custody = Custody(census_path.parent, require_owner=True)
    try:
        art = census_custody.root_snapshot().read(census_path.name)
        doc = art.json()
    finally:
        census_custody.close()
    claimed = census_path.name.partition("census-")[2][:64]
    # `census_sha256` is a digest of the census CONTENT, computed by
    # the producer over its own canonical form; it is NOT the sha256 of
    # the file's bytes, and the census receipt records the two as
    # separate fields. I conflated them, and the verifier refused a
    # census that was perfectly intact. What can be checked from the
    # outside is that the content digest the census DECLARES is the
    # name it is filed under; the byte digest is published beside it,
    # recomputed here, and never presented as the same thing.
    if doc.get("census_sha256") != claimed:
        raise VerificationRefusal(
            f"the census declares content digest "
            f"{str(doc.get('census_sha256'))[:16]} but is filed under "
            f"{claimed[:16]}")
    appearances = {a["appearance_id"]: a for a in doc["appearances"]}
    sources = {}
    for v in doc["variables"]:
        for app_id in v.get("appearances", ()):
            app = appearances.get(app_id)
            rel = app and (app.get("relative_path") or app.get("path")
                           or app.get("logical_id"))
            if rel:
                sources.setdefault(v["variable_id"], rel)
                break
    return {"census_sha256": claimed,
            "census_sha256_state": "DECLARED_BY_THE_PRODUCER — a "
                                   "content digest, not the file's "
                                   "bytes; not recomputed here",
            "file_sha256": art.sha256,
            "file_sha256_state": "RECOMPUTED_HERE_FROM_THE_BYTES_READ",
            "sources": sources,
            "conceptual_variables": len(doc["variables"]),
            "physical_appearances": len(appearances)}


def verify(state_dir: Path, lake_root: Path, census_path: Path, *,
           strict_mode: bool = False) -> dict:
    custody = Custody(state_dir, require_owner=True,
                      strict_mode=strict_mode)
    findings: list[dict] = []
    try:
        root = custody.root_snapshot()
        ledger = root.read("PRE_LEDGER.json").json()
        ids = ledger["identities"]
        if len(set(ids)) != len(ids):
            raise VerificationRefusal(
                "the pre-result ledger lists a duplicate identity, so "
                "its own denominator is not a set")
        if len(ids) != ledger["conceptual_variables"]:
            raise VerificationRefusal(
                f"the ledger lists {len(ids)} identities but declares "
                f"{ledger['conceptual_variables']} conceptual variables")
        census = load_census(census_path)
        if census["census_sha256"] != ledger.get("census_sha256"):
            raise VerificationRefusal(
                "the census supplied is NOT the census the pre-result "
                "ledger was built from; the denominator would be a "
                "different population")
        if census["conceptual_variables"] != len(ids):
            raise VerificationRefusal(
                f"the census holds {census['conceptual_variables']} "
                f"variables and the ledger lists {len(ids)}")
        declared = {vid: {"variable_id": vid,
                          "relative_path": census["sources"].get(vid)}
                    for vid in ids}
        recomputed = sha_obj(sorted(ids))

        terminals = root.subdir("terminals")
        names = sorted(terminals.files)

        seen: dict[str, dict] = {}
        #: EVERY (variable_id, filename) pair actually found on disk.
        #: Keeping only `seen` let a duplicate variable_id overwrite —
        #: and so ERASE — the evidence of the file it displaced, which
        #: is precisely the misfiling the naming check exists to catch.
        all_pairs: list[tuple[str, str]] = []
        unknown, unreadable = [], []
        for name in names:
            if not name.endswith(".json"):
                findings.append({"kind": "NON_TERMINAL_FILE",
                                 "name": name})
                continue
            try:
                art = terminals.read(name)
                doc = art.json()
            except (CustodyRefusal, ValueError) as exc:
                unreadable.append({"name": name, "error": str(exc)[:120]})
                continue
            vid = doc.get("variable_id")
            if vid is None:
                unreadable.append({"name": name,
                                   "error": "no variable_id"})
                continue
            if vid not in declared:
                unknown.append(vid)
            all_pairs.append((vid, name))
            if vid in seen:
                findings.append({"kind": "DUPLICATE_VARIABLE_ID",
                                 "variable_id": vid,
                                 "files": [seen[vid]["_file"], name]})
            seen[vid] = doc | {"_file": name,
                               "_bytes_sha256": sha_bytes(art.raw())}

        missing = sorted(set(declared) - set(seen))
        extra = sorted(set(seen) - set(declared))

        pairs = sorted(all_pairs)
        rules = surviving_naming_rules(pairs)
        collisions = len(pairs) - len({f for _, f in pairs})
        misfiled = []
        if not rules:
            # no single rule explains every name, so a name proves
            # nothing about which variable a terminal belongs to
            probe = dict(NAMING_RULES)
            best, best_n = None, -1
            for rname, fn in probe.items():
                n = sum(1 for vid, f in pairs if fn(vid) + ".json" == f)
                if n > best_n:
                    best, best_n = rname, n
            misfiled = [{"variable_id": vid, "name": f,
                         "closest_rule": best,
                         "expected_name": NAMING_RULES[best](vid)
                                          + ".json"}
                        for vid, f in pairs
                        if NAMING_RULES[best](vid) + ".json" != f][:20]

        # source re-digestion for every MEASURED terminal
        sources, source_absent, digested = {}, [], 0
        for vid, doc in seen.items():
            if doc.get("outcome") != MEASURED:
                continue
            entity = doc.get("entity")
            decl = declared.get(vid, {})
            rel = decl.get("relative_path") or decl.get("source") \
                or doc.get("source")
            if not rel:
                source_absent.append(
                    {"variable_id": vid, "variables_affected": 1,
                     "why": "neither the census nor the terminal names "
                            "a source file"})
                continue
            if rel in sources:
                continue
            p = lake_root / rel
            if not p.is_file():
                # a missing FILE is one absence, however many variables
                # it carried. Counting it once per variable inflated
                # one gone file into hundreds of findings
                for a in source_absent:
                    if a.get("source") == rel:
                        a["variables_affected"] += 1
                        break
                else:
                    source_absent.append({"source": rel,
                                          "variables_affected": 1,
                                          "why": "the file is gone"})
                continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            sources[rel] = {"sha256": h.hexdigest(),
                            "bytes": p.stat().st_size}
            digested += 1

        exact = not (missing or extra or unknown or unreadable
                     or findings or collisions) and bool(rules)
        report = {
            "schema": REPORT_SCHEMA,
            "independence": "this verifier imports nothing from "
                            "characterize_lake; every fact is "
                            "re-derived from bytes",
            "custody": {
                "root_logical": "crispdm_lake_characterization_state",
                "physical_paths": "WITHHELD — a physical path is "
                                  "private evidence, never a public "
                                  "record",
                "directory_instances": {
                    "root": root.facts(),
                    "terminals": terminals.facts()},
                "reads": len(custody.reads()),
                "retained_descriptors": custody.open_descriptors(),
                "strict_mode": strict_mode},
            "pre_ledger": {
                "declared": len(declared),
                "recorded_sha256": ledger.get("pre_ledger_sha256"),
                "recorded_digest_state":
                    "DECLARED_BY_THE_PRODUCER_NOT_RECOMPUTED — the "
                    "verifier does not know the producer's formula and "
                    "does not import it to find out",
                "identities_sha256": recomputed,
                "identities_digest_definition":
                    "sha256 of the sorted identity list, computed here",
                "census_sha256": ledger.get("census_sha256"),
                "census_file_sha256": census["file_sha256"],
                "census_file_sha256_state": census["file_sha256_state"],
                "census_sha256_state": census["census_sha256_state"],
                "census_conceptual_variables":
                    census["conceptual_variables"],
                "census_physical_appearances":
                    census["physical_appearances"],
                "sources_resolved_from_census":
                    sum(1 for v in declared.values()
                        if v["relative_path"])},
            "terminals": {
                "files_present": len(names),
                "read": len(seen),
                "missing": missing, "missing_count": len(missing),
                "extra": extra, "extra_count": len(extra),
                "naming_rules_consistent_with_every_file": rules,
                "naming_rule_state": (
                    "ONE_FIXED_RULE_EXPLAINS_EVERY_FILENAME" if rules
                    else "NO_SINGLE_RULE_EXPLAINS_EVERY_FILENAME"),
                "filename_collisions": collisions,
                "misfiled": misfiled,
                "unknown_variable_ids": unknown,
                "unreadable": unreadable},
            "sources": {
                "distinct_files_digested": digested,
                "absent": source_absent,
                "absent_count": len(source_absent),
                "absent_count_definition":
                    "distinct absences, not variables affected; each "
                    "entry names how many variables it costs",
                "variables_affected_by_absence":
                    sum(a["variables_affected"] for a in source_absent),
                "digests": sources},
            "outcomes": {
                k: sum(1 for d in seen.values()
                       if d.get("outcome") == k)
                for k in sorted({d.get("outcome")
                                 for d in seen.values()})},
            "other_findings": findings,
            "verdict": ("TERMINALS_VERIFIED_EXACT" if exact
                        else "TERMINALS_DIVERGE"),
            "grants_nothing": "a verified terminal set is a verified "
                              "set of outcomes. It confers no "
                              "eligibility on any variable",
        }
        report["verification_sha256"] = sha_obj(report)
        return report | {"_terminals": seen, "_declared": declared,
                         "_sources": sources}
    finally:
        custody.close()


# ------------------------------------------------------------ C60-C61
def supersede(state_dir: Path, report: dict, *, v2_dirname: str,
              window_contract: dict, code_identity: dict,
              superseded_at: str) -> dict:
    """Write v2 terminals BESIDE v1. v1 is never opened for writing."""
    v1_dir = state_dir / "terminals"
    v2_dir = state_dir / v2_dirname
    if v2_dir.exists():
        raise VerificationRefusal(
            f"{v2_dirname} already exists. History is superseded "
            "additively, and a supersession is written once")
    before = {p.name: p.stat() for p in sorted(v1_dir.iterdir())}

    wc_digest = sha_obj(window_contract)
    v2_dir.mkdir(parents=True)
    written = 0
    for vid, doc in sorted(report["_terminals"].items()):
        decl = report["_declared"].get(vid, {})
        rel = decl.get("relative_path") or decl.get("source") \
            or doc.get("source")
        src = report["_sources"].get(rel)
        body = {
            "schema": V2_SCHEMA,
            "supersedes": {
                "schema": "crispdm.lake_characterization_terminal.v1",
                "file": doc["_file"],
                "sha256": doc["_bytes_sha256"],
                "unchanged": True},
            "variable_id": vid,
            "concept_name": doc.get("concept_name"),
            "entity": doc.get("entity"),
            "appearance": doc.get("appearance"),
            "batch": doc.get("batch"),
            "outcome": doc.get("outcome"),
            "descriptors": doc.get("descriptors"),
            "not_identifiable": doc.get("not_identifiable"),
            "rows_used": doc.get("rows_used"),
            "measured_at": doc.get("measured_at"),
            "source": {
                "logical_id": rel or "UNDECLARED",
                "sha256": (src or {}).get("sha256", "UNAVAILABLE"),
                "bytes": (src or {}).get("bytes"),
                "state": ("DIGESTED_BY_THE_VERIFIER" if src
                          else "NOT_DIGESTED")},
            "window_contract": window_contract,
            "window_contract_sha256": wc_digest,
            "code_identity": code_identity,
            "superseded_at": superseded_at,
            "grants_nothing": "a descriptor describes a variable. It "
                              "confers no eligibility",
        }
        body["terminal_sha256"] = sha_obj(body)
        (v2_dir / doc["_file"]).write_text(
            json.dumps(body, indent=1, sort_keys=True) + "\n")
        written += 1

    after = {p.name: p.stat() for p in sorted(v1_dir.iterdir())}
    changed = sorted(
        n for n in set(before) | set(after)
        if n not in before or n not in after
        or before[n].st_mtime_ns != after[n].st_mtime_ns
        or before[n].st_size != after[n].st_size
        or before[n].st_ino != after[n].st_ino)
    if changed:
        raise VerificationRefusal(
            f"the supersession touched {len(changed)} v1 terminals; "
            "history is never rewritten")
    index = {
        "schema": "crispdm.lake_terminal_supersession.v1",
        "superseded_at": superseded_at,
        "v1_directory": "terminals",
        "v2_directory": v2_dirname,
        "v1_terminals_unchanged": len(before),
        "v2_terminals_written": written,
        "window_contract_sha256": wc_digest,
        "verification_sha256": report["verification_sha256"],
        "what_v2_adds": [
            "schema — the contract the terminal was written against",
            "source.sha256 — the BYTES the outcome describes, "
            "re-digested by an independent verifier",
            "window_contract + its digest — what a descriptor means",
            "terminal_sha256 — the terminal's digest of ITSELF, so a "
            "terminal that is edited stops matching",
        ],
        "rule": "v1 is never opened for writing, never moved and never "
                "rewritten; the run refuses outright if any v1 file's "
                "inode, size or mtime moves",
    }
    index["supersession_sha256"] = sha_obj(index)
    (state_dir / "TERMINAL_SUPERSESSION.v1.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--lake-root", required=True, type=Path)
    ap.add_argument("--census", required=True, type=Path)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--v2-dirname", default="terminals_v2")
    ap.add_argument("--superseded-at", default=None)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)

    report = verify(a.state_dir.expanduser(),
                    a.lake_root.expanduser(),
                    a.census.expanduser(), strict_mode=a.strict)
    public = {k: v for k, v in report.items() if not k.startswith("_")}
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(public, indent=1,
                                       sort_keys=True) + "\n")
    out = {"verdict": public["verdict"],
           "declared": public["pre_ledger"]["declared"],
           "read": public["terminals"]["read"],
           "missing": public["terminals"]["missing_count"],
           "extra": public["terminals"]["extra_count"],
           "naming_rules": public["terminals"]
               ["naming_rules_consistent_with_every_file"],
           "misfiled": len(public["terminals"]["misfiled"]),
           "sources_digested":
               public["sources"]["distinct_files_digested"],
           "outcomes": public["outcomes"],
           "reads": public["custody"]["reads"],
           "retained_descriptors":
               public["custody"]["retained_descriptors"]}
    if a.supersede:
        if public["verdict"] != "TERMINALS_VERIFIED_EXACT":
            raise VerificationRefusal(
                "a divergent terminal set is not superseded; the "
                "divergence is reported and adjudicated first")
        idx = supersede(
            a.state_dir.expanduser(), report,
            v2_dirname=a.v2_dirname,
            window_contract={
                "definition": "descriptors are computed over the "
                              "variable's FULL observed series in file "
                              "order; no resampling, no centring and "
                              "no forward fill",
                "missing_policy": "non-finite values are excluded from "
                                  "the descriptor and counted",
                "minimum_finite_values": 2,
                "ordering": "the file's own row order, which is not "
                            "asserted to be chronological"},
            code_identity={
                "verifier": Path(__file__).name,
                "verifier_sha256": sha_bytes(
                    Path(__file__).read_bytes()),
                "custody": "descriptor_custody.py",
                "custody_sha256": sha_bytes(
                    (Path(__file__).parent /
                     "descriptor_custody.py").read_bytes())},
            superseded_at=a.superseded_at or "UNDECLARED")
        out["supersession"] = {
            "v2_written": idx["v2_terminals_written"],
            "v1_unchanged": idx["v1_terminals_unchanged"],
            "sha256": idx["supersession_sha256"][:16]}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
