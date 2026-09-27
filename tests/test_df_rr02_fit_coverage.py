"""RR02 (order 2026-09-26): the four seal-pinned fit runners, closed by VERSIONED INTEGRATION.

The DR01 follow-on closed four of the eight per-cell fit runners and deliberately deferred four --
`df_e1_block`, `df_e1_huber`, `df_fin_runner`, `df_d2_unit_worker` -- because each is pinned by a
gate that recomputes the digest of the very file the guard call would have to live in.  Editing them
would refuse every design already sealed against them and invalidate every cached replay whose
`replay_code_sha256` is that file.

The order requires them closed "by versioned integration, not by invalidating old seals".  The
integration point is `df_benchmark_contract.bind()`:

  * it is reached on the fit path of `df_e1_block`, `df_e1_huber`, `df_fin_runner` and
    `df_e1_phase1`, at the moment the contract is checked against the prepared data the runner is
    about to consume -- so a refusal there costs no compute;
  * `df_benchmark_contract.py` is pinned by NO design's `source_code` set and is not a member of
    `df_d2_design.D2_CODE_FILES`, so requiring the proof there changes not one sealed byte;
  * the requirement carries a version, `BIND_ADMISSION_GUARD_VERSION`.

Three of the four are therefore closed.  `df_d2_unit_worker` is NOT, and this file proves why
rather than implying otherwise: every module on its fit path is inside `D2_CODE_FILES`, so there is
no unpinned place on that path to put the proof.  That gap is named in the RR02 return.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
GUARD_EXIT = 75


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- the integration point does not disturb any seal ------------------------------------------

SEALED_SOURCE_SETS = (
    ("df_fin_runner.py", "df_fin_task.py", "df_e1_block.py", "df_mod_e0.py", "df_e1_governed.py"),
    ("df_e1_huber.py", "df_e1_phase1.py", "df_e1_pilot.py", "df_mod_e0.py", "df_e1_governed.py"),
    ("df_e1_block.py", "df_gru_reference.py", "df_e1_calendar.py", "df_e1_pilot.py",
     "df_mod_e0.py", "df_e1_governed.py"),
)


def test_RR02_the_integration_point_is_pinned_by_no_seal():
    """The load-bearing claim of this whole approach, checked rather than asserted."""
    for sealed in SEALED_SOURCE_SETS:
        assert "df_benchmark_contract.py" not in sealed
    D = _load("df_d2_design")
    assert "df_benchmark_contract" not in D.D2_CODE_FILES
    # and the three sealed sets really are the sets the runners seal, read from their own source
    fin = (TOOLS / "df_fin_runner.py").read_text()
    assert all(n in fin for n in SEALED_SOURCE_SETS[0])
    huber = (TOOLS / "df_e1_huber.py").read_text()
    assert all(n in huber for n in SEALED_SOURCE_SETS[1])


def test_RR02_not_one_byte_of_the_four_seal_pinned_runners_changed():
    """`git show` against the follow-on tip this branch is based on.  Versioned integration means
    the seals still verify; it does not mean the seals were rewritten."""
    base = "f66ee25d"
    for name in ("df_e1_block.py", "df_e1_huber.py", "df_fin_runner.py", "df_d2_unit_worker.py"):
        r = subprocess.run(["git", "show", f"{base}:tools/{name}"], capture_output=True,
                           cwd=str(TOOLS.parent), timeout=120)
        if r.returncode != 0:
            pytest.skip(f"{base} is not reachable from this checkout")
        assert r.stdout == (TOOLS / name).read_bytes(), f"{name} was edited"


def test_RR02_the_versioned_requirement_is_declared_and_called_from_bind():
    B = _load("df_benchmark_contract")
    assert B.BIND_ADMISSION_GUARD_VERSION.startswith("rr02")
    src = (TOOLS / "df_benchmark_contract.py").read_text()
    assert "_require_reserved_scope_for_fit(purpose)" in src
    # it is the FIRST thing bind does: a refusal must cost nothing
    body = src.split("def bind(contract")[1]
    assert body.index("_require_reserved_scope_for_fit") < body.index("p = []")


# ---- both directions of the proof --------------------------------------------------------------

def _uncovered_env(tmp_path):
    """An empty lease store and a generous simulated host: capacity is not the question here,
    COVERAGE is.  Nothing is allocated and no real memory is read."""
    readings = tmp_path / "readings.json"
    readings.write_text(json.dumps({"mem_available_bytes": 512 << 30, "mem_total_bytes": 1024 << 30,
                                    "slice_memory_max": None, "slice_memory_current": 0,
                                    "pressure_some_avg10": 0.0, "alive": {}, "cgroup_current": {},
                                    "cgroup_peak": {}}))
    return {**os.environ,
            "CRISPDM_ADMISSION_DIR": str(tmp_path / "empty_store"),
            "CRISPDM_ADMISSION_RESOURCES_JSON": str(readings),
            "CRISPDM_ADMISSION_MODULE": str(TOOLS / "crispdm_admission.py")}


BIND_PROBE = r"""
import importlib.util, sys
from pathlib import Path
tools = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("df_benchmark_contract", tools / "df_benchmark_contract.py")
B = importlib.util.module_from_spec(spec); sys.modules["df_benchmark_contract"] = B
spec.loader.exec_module(B)
B._require_reserved_scope_for_fit("a probe")
print("COVERED")
"""


def test_RR02_a_fit_without_a_covering_reservation_is_refused_and_told_why(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(BIND_PROBE)
    r = subprocess.run([sys.executable, str(probe), str(TOOLS)], env=_uncovered_env(tmp_path),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == GUARD_EXIT, r.stdout + r.stderr
    assert "outside a reserved scope" in r.stderr
    assert "nothing was started" in r.stderr
    assert "COVERED" not in r.stdout


def test_RR02_a_fit_inside_a_covering_reservation_passes(tmp_path):
    env = _uncovered_env(tmp_path)
    A = _load("crispdm_admission")
    store = A.Store(env["CRISPDM_ADMISSION_DIR"])
    res = A.FileResources(env["CRISPDM_ADMISSION_RESOURCES_JSON"])
    d = A.acquire(store, res, A.Request(name="probe-cover", cap_bytes=1 << 30, wall_seconds=600),
                  A.now_from_env())
    assert d["verdict"] == A.ADMITTED
    cg = Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split("::", 1)[1]
    A.arm(store, res, d["lease_id"], A.now_from_env(), pid=os.getpid(), cgroup=cg.lstrip("/"))

    probe = tmp_path / "probe.py"
    probe.write_text(BIND_PROBE)
    r = subprocess.run([sys.executable, str(probe), str(TOOLS)], env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "COVERED" in r.stdout


def test_RR02_the_proof_reserves_nothing_of_its_own(tmp_path):
    """These runners spawn their children in their own cgroup under their own MemoryMax.  A second
    reservation would count the same bytes twice and queue the work against itself."""
    env = _uncovered_env(tmp_path)
    probe = tmp_path / "probe.py"
    probe.write_text(BIND_PROBE)
    subprocess.run([sys.executable, str(probe), str(TOOLS)], env=env, capture_output=True,
                   text=True, timeout=180)
    leases = Path(env["CRISPDM_ADMISSION_DIR"]) / "leases"
    assert not leases.exists() or list(leases.glob("*.json")) == []


# ---- the census: what the chokepoint covers, and what it does not -------------------------------

def test_RR02_every_bind_call_site_is_a_fit_path_and_they_are_the_expected_four():
    """If a new fit path starts using bind() it is covered automatically.  If one STOPS using it,
    this test trips and the decision has to be made explicitly instead of drifting."""
    callers = sorted(p.stem for p in TOOLS.glob("df_*.py")
                     if "B.bind(contract" in p.read_text() or "B.bind(contract," in p.read_text())
    assert callers == ["df_e1_block", "df_e1_huber", "df_e1_phase1", "df_fin_runner"], callers


def test_RR02_the_one_remaining_seal_pinned_gap_is_named_and_still_pinned():
    """df_d2_unit_worker is NOT closed, and the reason is structural: every module its fit path
    touches is inside D2_CODE_FILES, so there is no unpinned place on that path for the proof.
    Its unit children do reserve, through df_isolated_runner.  This test states the gap so it
    cannot be quietly forgotten, and fails the day the path stops being fully pinned."""
    D = _load("df_d2_design")
    pinned = set(D.D2_CODE_FILES)
    for m in ("df_d2_design", "df_d2_unit_worker", "df_lab_evaluation", "df_isolated_runner",
              "df_snr", "df_operators", "df_snapshot", "df_contract"):
        assert m in pinned, f"{m} left D2_CODE_FILES: the gap can now be closed there"
    worker = (TOOLS / "df_d2_unit_worker.py").read_text()
    assert "df_admission_guard" not in worker and "df_benchmark_contract" not in worker
    # its unit children are started through the runner that DOES reserve
    assert "df_isolated_runner" in worker
    assert "crispdm_admission" in (TOOLS / "df_isolated_runner.py").read_text()


def test_RR02_the_census_of_remaining_bypasses_is_tracked_and_honest():
    """The return reports bypasses explicitly rather than implying none; the census is a tracked
    file so a later reader can check the claim against the code."""
    census = (Path(__file__).resolve().parents[1]
              / "docs/audits/evidence/RR02_MONITOR_20260926/FIT_ENTRY_CENSUS.json")
    doc = json.loads(census.read_text())
    assert doc["schema"] == "satoshi.fit_entry_census.v1"
    by_state = {}
    for row in doc["entry_points"]:
        assert row["state"] in ("GATED", "COVERED_BY_BIND", "SEAL_PINNED_OPEN", "OUT_OF_REPO",
                               "READ_ONLY_UNGATED")
        by_state.setdefault(row["state"], []).append(row["entry_point"])
    assert "df_d2_unit_worker" in " ".join(by_state.get("SEAL_PINNED_OPEN", []))
    assert by_state.get("COVERED_BY_BIND"), "the versioned integration must appear in the census"
    assert doc["bypasses_remaining"], "a census that names no bypass is not honest"
