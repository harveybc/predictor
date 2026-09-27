#!/usr/bin/env python3
"""DR01 follow-on (order 2026-09-26): the per-cell fit runners must PROVE they are covered.

Both directions, for the guard itself and for every runner it was made mandatory in:

* a **covered** run (a live reservation whose witness is this process's cgroup, or a holder pid
  above it) passes the guard and goes on to fail, or not, for its own reasons;
* a **bare** invocation is REFUSED with exit 75 and a message that names *why* -- that its fit
  children would run in its own cgroup, under a cap nobody reserved.

Nothing here allocates memory, starts a fit, or touches the coordinator's live readings: the
lease store and every resource reading come from the sandbox `tests/conftest.py` installs (DR01),
and the runners are invoked with roots that do not exist, so the guard is the only thing that can
answer before they would read anything.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
GIB = 1 << 30
# RR02: this module's subject is the ABSENCE of a covering reservation, so it opts out of the
# session-level declared coverage that tests/conftest.py writes for every other test.
pytestmark = pytest.mark.crispdm_uncovered

GUARD_EXIT = 75


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # a dataclass body resolves its own module through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def G():
    return _load("df_admission_guard")


@pytest.fixture()
def A():
    return _load("crispdm_admission")


def my_cgroup() -> str:
    return Path("/proc/self/cgroup").read_text().strip().splitlines()[0].split("::", 1)[1]


def reserve(A, *, cgroup=None, pid=None, cap=2 * GIB, name="covering"):
    """Write and arm one reservation in the sandbox store.  No memory is allocated."""
    store, res = A.Store(), A.resources_from_env()
    req = A.Request(name=name, cap_bytes=cap, wall_seconds=600, slice_name=A.DEFAULT_SLICE)
    d = A.acquire(store, res, req, A.now_from_env())
    assert d["verdict"] == A.ADMITTED, d
    A.arm(store, res, d["lease_id"], A.now_from_env(), pid=pid,
          cgroup=None if cgroup is None else str(cgroup).lstrip("/"))
    return d["lease_id"]


# ---------------------------------------------------------------- the guard itself

def test_2026_09_26_a_bare_process_is_not_covered_and_the_guard_says_so(G):
    cov = G.coverage()
    assert cov["covered"] is False and cov["code"] == G.NOT_COVERED
    assert cov["module"] and Path(cov["module"]).is_file()


def test_2026_09_26_a_reservation_over_this_cgroup_covers_this_process(G, A):
    lease = reserve(A, cgroup=my_cgroup())
    cov = G.coverage()
    assert cov["covered"] is True and cov["code"] == G.COVERED_BY_CGROUP
    assert cov["covering_lease_ids"] == [lease]


def test_2026_09_26_a_reservation_held_by_an_ancestor_pid_covers_this_process(G, A):
    """df_isolated_runner's PRLIMIT_AS fallback deliberately records NO cgroup witness (the child
    shares its parent's scope, which outlives it).  Such a reservation still covers its children,
    and the guard must not refuse a run that really is reserved."""
    lease = reserve(A, pid=os.getpid(), name="holder")
    cov = G.coverage()
    assert cov["covered"] is True and cov["code"] == G.COVERED_BY_HOLDER_PID
    assert cov["covering_lease_ids"] == [lease]


def test_2026_09_26_a_reservation_over_someone_elses_cgroup_does_not_cover_this_process(G, A):
    reserve(A, cgroup="crispdm-batch.slice/someone-elses-4242.scope")
    assert G.coverage()["covered"] is False


def test_2026_09_26_an_absent_admission_module_refuses_rather_than_admitting(G, monkeypatch):
    """An unenforceable cap is not a cap: the same failure direction df_dispatch takes."""
    monkeypatch.setattr(G, "admission_module", lambda: None)
    cov = G.coverage()
    assert cov["covered"] is False and cov["code"] == G.NO_ADMISSION_MODULE
    with pytest.raises(SystemExit) as exc:
        G.require_reserved_scope("df_x", "fit", stream=sys.stderr)
    assert exc.value.code == GUARD_EXIT


def test_2026_09_26_the_refusal_names_why_and_how_to_start_it(G):
    cov = G.coverage()
    text = G.refusal_text("df_x", "fit", cov, mem="8G", wall="4h", name="x", argv=["tools/df_x.py"])
    assert "REFUSED" in text and "will not fit outside a reserved scope" in text
    for phrase in ("OWN cgroup", "MemoryMax", "second reservation", "counted twice",
                   "crispdm-run", "nothing was started"):
        assert phrase in text, phrase


def test_2026_09_26_require_returns_the_coverage_when_it_is_covered(G, A):
    reserve(A, cgroup=my_cgroup())
    cov = G.require_reserved_scope("df_x", "fit")
    assert cov["covered"] is True


def test_2026_09_26_the_guard_reserves_nothing_of_its_own(G, A):
    """The whole point: a runner inside its parent's cgroup must NOT take a second reservation --
    that would count the same bytes twice.  The guard only reads."""
    before = A.state(A.Store(), A.resources_from_env(), A.now_from_env())
    G.coverage()
    after = A.state(A.Store(), A.resources_from_env(), A.now_from_env())
    assert [l["lease_id"] for l in before["live"]] == [l["lease_id"] for l in after["live"]]


def test_2026_09_26_the_guard_never_kills_raises_a_ceiling_or_drops_caches():
    """A source-level control, the same one DR01 put on the admission path."""
    text = (TOOLS / "df_admission_guard.py").read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    for forbidden in ("kill(", "SIGKILL", "drop_caches", "swapoff", "systemd-run", "MemoryMax=",
                      "oomd", "systemctl stop"):
        assert forbidden not in code, forbidden


# ------------------------------------------------------- the runners, both directions

# (tool, argv-after-the-script, the word the gated action is named by)
GATED = [
    ("df_d2_r4_replay", ["--design", "{r}/design.json", "--subset", "{r}/subset.json",
                         "--out", "{r}/out", "--reserve", "{r}/reserve", "--replay"], "replay"),
    ("df_e1_close", ["--root", "{r}/absent"], "replay"),
    ("df_mod_e0_close", ["--root", "{r}/absent"], "replay"),
    ("df_sota_repro", ["execute", "--root", "{r}/absent"], "run execute"),
]

# the same tools on a path that fits nothing and replays nothing: never gated
UNGATED = [
    ("df_d2_r4_replay", ["--design", "{r}/design.json", "--subset", "{r}/subset.json",
                         "--out", "{r}/out", "--compare"]),
    ("df_e1_close", ["--root", "{r}/absent", "--no-replay"]),
    ("df_mod_e0_close", ["--root", "{r}/absent", "--no-replay", "--no-live"]),
    ("df_sota_repro", ["report", "--root", "{r}/absent"]),
]


def run_tool(tool: str, argv: list, root: Path):
    return subprocess.run([sys.executable, str(TOOLS / f"{tool}.py"),
                           *[a.format(r=root) for a in argv]],
                          capture_output=True, text=True, timeout=300, env=dict(os.environ))


@pytest.mark.parametrize("tool,argv,action", GATED, ids=[t for t, _, _ in GATED])
def test_2026_09_26_a_bare_fit_runner_is_refused_and_the_refusal_names_why(tool, argv, action, tmp_path):
    r = run_tool(tool, argv, tmp_path)
    assert r.returncode == GUARD_EXIT, (r.returncode, r.stdout[-600:], r.stderr[-1500:])
    assert f"REFUSED: {tool} will not {action} outside a reserved scope" in r.stderr
    for phrase in ("OWN cgroup", "MemoryMax", "second reservation", "nothing was started"):
        assert phrase in r.stderr, phrase


@pytest.mark.parametrize("tool,argv,action", GATED, ids=[t for t, _, _ in GATED])
def test_2026_09_26_a_covered_fit_runner_passes_the_guard(tool, argv, action, tmp_path, A):
    reserve(A, cgroup=my_cgroup(), name=f"cov-{tool}")
    r = run_tool(tool, argv, tmp_path)
    assert r.returncode != GUARD_EXIT, (r.returncode, r.stderr[-1500:])
    assert "outside a reserved scope" not in r.stderr   # it failed for its own reasons, not the guard


@pytest.mark.parametrize("tool,argv", UNGATED, ids=[t for t, _ in UNGATED])
def test_2026_09_26_a_path_that_fits_nothing_is_not_gated(tool, argv, tmp_path):
    r = run_tool(tool, argv, tmp_path)
    assert "outside a reserved scope" not in r.stderr, r.stderr[-800:]
    assert r.returncode != GUARD_EXIT


# ------------------------------------------------------- the four that a seal pins

SEAL_PINNED = {
    # tool -> (the gate that recomputes its digest, where the pinned digests live)
    "df_e1_block": ("df_e1_block.code_drift / replay_identity.replay_code_sha256",
                    "design['source_code']['df_e1_block.py'] in every e1_block root"),
    "df_e1_huber": ("df_e1_huber.validate", "design['source_code']['df_e1_huber.py']"),
    "df_fin_runner": ("df_fin_runner.validate", "design['source_code']['df_fin_runner.py']"),
    "df_d2_unit_worker": ("df_d2_design.lab_code_sha256 (D2_CODE_FILES), compared on resume",
                          "RUN_MANIFEST.json code_sha256_at_creation and every D2 terminal"),
}


def test_2026_09_26_the_seal_pinned_runners_are_still_pinned_and_were_not_edited():
    """The guard was NOT added to these four: each is pinned by a gate that recomputes its own
    bytes, so an edit would refuse every design already sealed against them.  This test fails the
    day one of them stops being pinned -- that is when the guard can be added.
    """
    block = (TOOLS / "df_e1_block.py").read_text()
    assert "design[\"source_code\"]" in block and "sha_file(Path(__file__).resolve())" in block
    for tool in ("df_e1_huber", "df_fin_runner"):
        assert "scientific source changed" in (TOOLS / f"{tool}.py").read_text()
        assert f"\"{tool}.py\"" in (TOOLS / f"{tool}.py").read_text()
    d2 = (TOOLS / "df_d2_design.py").read_text()
    assert "\"df_d2_unit_worker\"" in d2 and "def lab_code_sha256" in d2
    for tool in SEAL_PINNED:
        assert "df_admission_guard" not in (TOOLS / f"{tool}.py").read_text(), \
            f"{tool} is seal-pinned: adding the guard needs the seal superseded first"
