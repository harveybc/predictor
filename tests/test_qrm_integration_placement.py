#!/usr/bin/env python3
"""QRM integration: the six failure modes the dispatch names, tests FIRST.

The integrated runner replaces an empty `CUDA_VISIBLE_DEVICES` with an explicit CPU/GPU placement
contract, set for the CHILD ONLY and before TensorFlow is imported, and verified on three
independent facts -- the physical device UUID through the driver, the framework's own device
registration, and where an op actually lands.  Visibility proves none of them: the telemetry lane
measured a state where the driver held the card, the UUID matched and TensorFlow registered ZERO
devices, which would have been sealed as a GPU pilot by a visibility check.

Six failure modes, one class each, none of which may be answered with a downgrade:

  1. MISSING_CUDA_LIBRARIES                   the interpreter carries no usable CUDA library set
  2. DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED   the driver's UUID is not the declared one
  3. GPU_REQUEST_FELL_BACK_TO_CPU             registration or placement failed: REFUSED, never CPU
  4. ATTEMPT_IDENTITY_MISMATCH                a stale record from an earlier attempt
  5. ENVELOPE_MISSING / RECORD_SCHEMA_*       a document that declares no envelope, or the wrong one
  6. DELIVERY_*                               the bytes the child consumed are not the bytes
                                              delivered to that unit

Everything here runs in this process on doubles for the framework and the driver: a test that needed
a real GPU could not assert the refusals, which are exactly the states where no GPU is present.  The
real-device evidence is the instrument's own selfcheck, retained separately.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PC = _load("df_placement_contract")
CS = _load("df_cell_scope")


# ---------------------------------------------------------------- doubles, named for what they fake

class _FakeTensor:
    def __init__(self, device):
        self.device = device


class _FakeTF:
    """Only the four attributes the verifier touches.  A double, and it says so."""

    def __init__(self, gpus=(), landing="/job:localhost/replica:0/task:0/device:GPU:0"):
        self._gpus, self._landing = list(gpus), landing
        outer = self

        class _config:
            class experimental:
                @staticmethod
                def get_memory_info(dev):
                    return {"current": 1024, "peak": 4096}

                @staticmethod
                def reset_memory_stats(dev):
                    return None

            @staticmethod
            def list_physical_devices(kind):
                return list(outer._gpus) if kind == "GPU" else ["CPU:0"]

        self.config = _config
        self.float32 = "float32"

    def device(self, name):
        class _ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *a):
                return False

        return _ctx()

    def zeros(self, shape, dtype=None):
        return _FakeTensor(self._landing)


def _driver(uuid, count=1, status="MEASURED"):
    def fn(index=0):
        return {"status": status, "count": count, "uuid": uuid,
                "source": "libcuda.cuDeviceGetUuid (DOUBLE)", "index": index}
    return fn


REAL_UUID = "GPU-11111111-2222-3333-4444-555555555555"
OTHER_UUID = "GPU-99999999-8888-7777-6666-555555555555"


def _wheelset(purelib, *, generation=0):
    """Write one CUDA generation's sonames into a fake site-packages.

    `generation` indexes each family's alternatives, newest first, so 0 builds the newest
    generation the contract knows and 1 builds the one before it.  A family with only one soname
    gets that one whatever is asked for.
    """
    d = purelib / "nvidia"
    out = []
    for pkg, fams in PC.CUDA_PACKAGE_LIBRARIES.items():
        lib = d / pkg / "lib"
        lib.mkdir(parents=True, exist_ok=True)
        for names in fams.values():
            (lib / names[min(generation, len(names) - 1)]).write_bytes(b"\x7fELF not a library")
        out.append(str(lib))
    return sorted(out)


@pytest.fixture
def libdirs(tmp_path):
    """A complete CUDA library set on disk, so library discovery is not the thing under test.

    Returns (purelib, dirs): the site-packages root the contract is asked about, and the
    `nvidia/*/lib` directories inside it.
    """
    purelib = tmp_path / "site-packages"
    return purelib, _wheelset(purelib)


# ============================================================ 1. missing libraries

NO_LOADER = staticmethod(lambda soname: False)     # a loader that finds nothing, as a double


def test_a_gpu_placement_with_no_cuda_libraries_is_refused_by_name(tmp_path):
    """Failure mode 1.  No wheel carries the soname AND the loader cannot find it either.

    The refusal happens while BUILDING the child's environment, so the child is never started at
    all -- which is the only point at which this is cheap to detect.  The double loader is what
    makes the case reproducible: on a host whose loader does find the libraries, this state simply
    does not exist, and a test that pretended otherwise would be testing the host.
    """
    empty = tmp_path / "site-packages"
    empty.mkdir()
    found = PC.cuda_library_dirs(purelib=empty, loader=lambda n: False)
    assert found["status"] == PC.UNKNOWN
    assert found["dirs"] == []
    assert {m["family"] for m in found["missing"]} == set(PC.required_families())

    with pytest.raises(PC.PlacementRefusal) as e:
        PC.child_placement_env({}, placement="GPU", device_uuid=REAL_UUID, purelib=empty,
                               loader=lambda n: False)
    assert e.value.code == "MISSING_CUDA_LIBRARIES"
    assert "cudart" in e.value.detail


def test_a_partial_library_set_no_route_resolves_is_refused_and_names_what_is_absent(libdirs):
    """One soname that neither the wheels nor the loader supply must refuse, not warn.

    This is the shape of telemetry state B: every library that could be loaded loaded, and
    TensorFlow still registered nothing.  A set that is almost complete is a refusal, because
    "almost" is what produced a CPU pilot with a GPU label.
    """
    purelib, dirs = libdirs
    victim = purelib / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9"
    victim.unlink()
    found = PC.cuda_library_dirs(purelib=purelib, loader=lambda n: False)
    assert found["status"] == PC.UNKNOWN
    assert [m["family"] for m in found["missing"]] == ["cudnn"]
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.child_placement_env({}, placement="GPU", device_uuid=REAL_UUID, purelib=purelib,
                               loader=lambda n: False)
    assert e.value.code == "MISSING_CUDA_LIBRARIES"


def test_the_contract_is_child_only_and_never_mutates_this_process(libdirs):
    """The contract is passed to a child; it is never exported into the parent's own environment.

    A parent that set CUDA_VISIBLE_DEVICES on itself would change the placement of every later
    sibling in the same process, which is how an empty override became global in the first place.
    """
    purelib, dirs = libdirs
    before = dict(os.environ)
    base = {"PATH": "/usr/bin"}
    env, decl = PC.child_placement_env(base, placement="GPU", device_uuid=REAL_UUID,
                                       purelib=purelib)
    assert dict(os.environ) == before, "the parent's environment was mutated"
    assert base == {"PATH": "/usr/bin"}, "the base mapping was mutated in place"
    assert env["CUDA_VISIBLE_DEVICES"] == REAL_UUID
    assert env[PC.PLACEMENT_ENV] == "GPU"
    assert env[PC.UUID_ENV] == REAL_UUID
    assert all(d in env["LD_LIBRARY_PATH"].split(":") for d in dirs)
    assert decl["scope"] == PC.CHILD_ONLY


def test_a_cpu_placement_declares_cpu_explicitly_rather_than_leaving_an_empty_override():
    """CPU is a DECLARED placement, not the absence of one.

    `CUDA_VISIBLE_DEVICES=""` alone is indistinguishable from a GPU run whose pinning was lost.
    The declaration is what makes the two different, and a later GPU claim over a CPU declaration
    is refused rather than believed.
    """
    env, decl = PC.child_placement_env({}, placement="CPU")
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env[PC.PLACEMENT_ENV] == "CPU"
    assert PC.UUID_ENV not in env
    assert decl["placement"] == "CPU"
    assert decl["establishes_a_gpu_pilot"] is False
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.child_placement_env({}, placement="CPU", device_uuid=REAL_UUID)
    assert e.value.code == "DECLARED_DEVICE_WITH_CPU_PLACEMENT"


def test_a_gpu_placement_without_a_declared_device_is_refused(libdirs):
    purelib, _ = libdirs
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.child_placement_env({}, placement="GPU", device_uuid=None, purelib=purelib)
    assert e.value.code == "GPU_PLACEMENT_WITHOUT_A_DECLARED_DEVICE"


def test_an_undeclared_placement_is_refused_rather_than_defaulted():
    """No default.  A default placement is how a GPU request becomes a CPU run silently."""
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.child_placement_env({}, placement=None)
    assert e.value.code == "PLACEMENT_NOT_DECLARED"
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.declared_placement({})
    assert e.value.code == "PLACEMENT_NOT_DECLARED"


# ============================================================ 2. wrong UUID

def test_a_driver_uuid_that_is_not_the_declared_one_is_refused_and_is_not_a_gpu_pilot():
    """Failure mode 2.  The pinning resolved to a device that is not the one declared."""
    decl = {"placement": "GPU", "device_uuid": REAL_UUID,
            "env": {"CUDA_VISIBLE_DEVICES": REAL_UUID}}
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.verify_or_refuse(decl, tf=_FakeTF(gpus=["GPU:0"]), driver=_driver(OTHER_UUID))
    assert e.value.code == "GPU_REQUEST_FELL_BACK_TO_CPU"
    assert "DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED" in e.value.evidence["refusals"]
    assert e.value.evidence["establishes_a_gpu_pilot"] is False


def test_two_visible_devices_are_ambiguous_and_refused_even_when_the_uuid_matches():
    """A figure from a process that can see two devices is not attributable to one of them."""
    decl = {"placement": "GPU", "device_uuid": REAL_UUID,
            "env": {"CUDA_VISIBLE_DEVICES": f"{REAL_UUID},{OTHER_UUID}"}}
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.verify_or_refuse(decl, tf=_FakeTF(gpus=["GPU:0"]), driver=_driver(REAL_UUID, count=2))
    assert "AMBIGUOUS_PINNING" in e.value.evidence["refusals"]


# ============================================================ 3. CPU fallback

def test_a_matching_uuid_with_zero_registered_devices_is_refused_not_downgraded():
    """Failure mode 3, and the state the telemetry lane actually measured.

    The driver has the card, the UUID matches the declaration, and TensorFlow registers ZERO
    devices.  Visibility alone would have sealed this as a GPU pilot.  The three-fact verifier
    refuses it, and the refusal is an exception -- not a record that says CPU and carries on.
    """
    decl = {"placement": "GPU", "device_uuid": REAL_UUID,
            "env": {"CUDA_VISIBLE_DEVICES": REAL_UUID}}
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.verify_or_refuse(decl, tf=_FakeTF(gpus=[]), driver=_driver(REAL_UUID))
    assert e.value.code == "GPU_REQUEST_FELL_BACK_TO_CPU"
    assert "FRAMEWORK_REGISTERED_NO_DEVICE" in e.value.evidence["refusals"]


def test_a_registered_device_whose_op_lands_on_the_cpu_is_refused():
    """Registration is not execution.  An op that lands on the CPU is a CPU run."""
    decl = {"placement": "GPU", "device_uuid": REAL_UUID,
            "env": {"CUDA_VISIBLE_DEVICES": REAL_UUID}}
    cpu_landing = "/job:localhost/replica:0/task:0/device:CPU:0"
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.verify_or_refuse(decl, tf=_FakeTF(gpus=["GPU:0"], landing=cpu_landing),
                            driver=_driver(REAL_UUID))
    assert "PLACEMENT_NOT_ON_DECLARED_DEVICE" in e.value.evidence["refusals"]


def test_all_three_facts_together_are_what_accepts_a_gpu_pilot():
    """The accepting case, so the refusals above are not vacuous."""
    decl = {"placement": "GPU", "device_uuid": REAL_UUID,
            "env": {"CUDA_VISIBLE_DEVICES": REAL_UUID}}
    v = PC.verify_or_refuse(decl, tf=_FakeTF(gpus=["GPU:0"]), driver=_driver(REAL_UUID))
    assert v["establishes_a_gpu_pilot"] is True
    assert v["device_uuid_verified"] == REAL_UUID
    assert v["framework_devices"] == ["GPU:0"]
    assert "GPU:0" in v["placement_probe"]["device"]
    assert v["facts_verified"] == ["DRIVER_UUID", "FRAMEWORK_REGISTRATION", "EXECUTION_PLACEMENT"]


def test_a_cpu_declaration_verifies_as_cpu_and_never_claims_a_device():
    v = PC.verify_or_refuse({"placement": "CPU", "device_uuid": None, "env": {}}, tf=_FakeTF())
    assert v["establishes_a_gpu_pilot"] is False
    assert v["status"] == PC.MEASURED


def test_the_contract_is_enforced_before_tensorflow_is_imported():
    """The library path must be in place BEFORE the import, so it is checked against sys.modules.

    An `LD_LIBRARY_PATH` set after TensorFlow has loaded changes nothing: the dynamic loader has
    already run.  A contract applied late is therefore not a contract, and saying so afterwards
    would be narration.
    """
    env = {PC.PLACEMENT_ENV: "GPU", PC.UUID_ENV: REAL_UUID,
           "CUDA_VISIBLE_DEVICES": REAL_UUID, "LD_LIBRARY_PATH": "/x/nvidia/cuda_runtime/lib"}
    decl = PC.enforce_before_tensorflow(env=env, modules={})
    assert decl["placement"] == "GPU"
    assert decl["before_tensorflow_import"] is True
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.enforce_before_tensorflow(env=env, modules={"tensorflow": object()})
    assert e.value.code == "TENSORFLOW_ALREADY_IMPORTED"


def test_a_gpu_declaration_whose_visibility_was_lost_on_the_way_is_refused_in_the_child():
    """The child never trusts the declaration: it re-reads the environment it actually got."""
    for broken, code in (({PC.PLACEMENT_ENV: "GPU", PC.UUID_ENV: REAL_UUID,
                           "CUDA_VISIBLE_DEVICES": ""}, "NO_VISIBLE_DEVICE"),
                         ({PC.PLACEMENT_ENV: "GPU", PC.UUID_ENV: REAL_UUID,
                           "CUDA_VISIBLE_DEVICES": REAL_UUID}, "CUDA_LIBRARY_PATH_NOT_PASSED"),
                         ({PC.PLACEMENT_ENV: "GPU", "CUDA_VISIBLE_DEVICES": REAL_UUID,
                           "LD_LIBRARY_PATH": "/x/nvidia/cuda_runtime/lib"},
                          "GPU_PLACEMENT_WITHOUT_A_DECLARED_DEVICE")):
        with pytest.raises(PC.PlacementRefusal) as e:
            PC.enforce_before_tensorflow(env=broken, modules={})
        assert e.value.code == code, broken


def test_pytorch_is_kept_out_of_the_tensorflow_memory_path_entirely():
    """PyTorch's caching allocator does not account for one byte TensorFlow allocates.

    The pre-integration `df_cell_scope.gpu_memory` imported torch to report a device figure for a
    TensorFlow recipe.  The integrated path must (a) never import torch and (b) say so in the
    record, so a reader cannot mistake an absent figure for a zero one.
    """
    src = (TOOLS / "df_cell_scope.py").read_text()
    assert "import torch" not in src
    assert "torch.cuda" not in src

    saw = []

    class _Guard(dict):
        def __missing__(self, key):
            saw.append(key)
            raise KeyError(key)

    out = CS.gpu_memory(env={PC.PLACEMENT_ENV: "CPU", "CUDA_VISIBLE_DEVICES": ""})
    assert out["status"] == CS.UNKNOWN
    assert out["framework"] == "tensorflow"
    assert out["pytorch_consulted"] is False
    assert "torch" not in sys.modules or "torch" not in json.dumps(out)


def test_the_device_figure_comes_from_tensorflows_own_allocator_when_a_gpu_is_declared():
    out = CS.gpu_memory(env={PC.PLACEMENT_ENV: "GPU", PC.UUID_ENV: REAL_UUID,
                             "CUDA_VISIBLE_DEVICES": REAL_UUID}, tf=_FakeTF(gpus=["GPU:0"]))
    assert out["status"] == CS.MEASURED
    assert out["framework"] == "tensorflow"
    assert out["source"] == "tf.config.experimental.get_memory_info"
    assert out["peak_bytes"] == 4096
    assert out["pytorch_consulted"] is False
    assert "never added" in out["basis"]


# ============================================================ 4. a stale attempt

def _envelope(record):
    return CS.embed_record({"schema": "df_q2_household_governed.v1.child"}, record)


def _fresh_record(tmp_path, cell_id="u1", attempt="0000000000001-abc", stage="CELL"):
    """A record shaped exactly as the integrated producer writes one, with every field in domain."""
    now = time.time()
    return {"schema": CS.RECORD_SCHEMA, "cell_id": cell_id, "stage": stage,
            "attempt_id": attempt, "recorded_at": now, "boot_id": CS._boot_id(),
            "kernel_limit": {"bytes": 4 * 1024 ** 3, "status": CS.MEASURED,
                             "source": "memory.max", "unlimited": False},
            "declared_cap_bytes": 4 * 1024 ** 3,
            "host_ram": {
                "cgroup_peak": {"bytes": 1 << 20, "status": CS.MEASURED, "basis": CS.CGROUP_BASIS,
                                "cgroup": "x.slice/y.scope"}},
            "scope": {"cgroup": "x.slice/y.scope", "unit": "y.scope", "is_scope": True,
                      "inode": 4242},
            "reservation": {"confirmed": True, "lease_id": "L1", "cgroup": "x.slice/y.scope",
                            "cap_bytes": 4 * 1024 ** 3, "source": "DOUBLE"}}


def test_a_record_from_an_earlier_attempt_of_the_same_cell_is_refused(tmp_path):
    """Failure mode 4.  The token is minted before the child exists, so a stale file cannot echo it.

    This is the case the pre-repair gate fell open on: a `MEASURED` peak in a file that happened to
    be at the expected path, from a run that was not this one.
    """
    now = time.time()
    stale = _fresh_record(tmp_path, attempt="0000000000001-stale")
    v = CS.verify_fresh_attempt(_envelope(stale), cell_id="u1",
                                attempt_id="0000000000002-thisrun", stage="CELL",
                                started_at=now - 1, finished_at=now + 1,
                                declared_cap_bytes=4 * 1024 ** 3, verify_reservation=False)
    assert v["accepted"] is False
    assert "ATTEMPT_IDENTITY_MISMATCH" in v["refused_by"]


def test_a_record_with_no_attempt_token_at_all_is_refused_on_absence(tmp_path):
    now = time.time()
    rec = _fresh_record(tmp_path)
    rec.pop("attempt_id")
    v = CS.verify_fresh_attempt(_envelope(rec), cell_id="u1", attempt_id="0000000000002-thisrun",
                                stage="CELL", started_at=now - 1, finished_at=now + 1,
                                declared_cap_bytes=4 * 1024 ** 3, verify_reservation=False)
    assert "ATTEMPT_IDENTITY_MISSING" in v["refused_by"]


def test_the_matching_attempt_of_the_same_cell_is_accepted(tmp_path):
    """So the refusals above are not a gate that refuses everything."""
    now = time.time()
    rec = _fresh_record(tmp_path, attempt="0000000000002-thisrun")
    rec["recorded_at"] = now
    v = CS.verify_fresh_attempt(_envelope(rec), cell_id="u1", attempt_id="0000000000002-thisrun",
                                stage="CELL", started_at=now - 1, finished_at=now + 1,
                                declared_cap_bytes=4 * 1024 ** 3, verify_reservation=False)
    assert v["refused_by"] == []
    assert v["accepted"] is True


# ============================================================ 5. a wrong envelope

def test_a_document_that_declares_no_envelope_is_refused_even_when_it_carries_the_peak(tmp_path):
    """Failure mode 5.  No field is accepted because it shares a name with the right one."""
    now = time.time()
    rec = _fresh_record(tmp_path)
    naked = {"schema": "something.else", "host_ram": rec["host_ram"], "cell_scope": rec}
    v = CS.verify_fresh_attempt(naked, cell_id="u1", attempt_id="0000000000001-abc", stage="CELL",
                                started_at=now - 1, finished_at=now + 1,
                                declared_cap_bytes=4 * 1024 ** 3, verify_reservation=False)
    assert "ENVELOPE_MISSING" in v["refused_by"]
    assert v["accepted"] is False


def test_an_envelope_that_declares_the_superseded_record_version_is_refused_by_version(tmp_path):
    now = time.time()
    rec = _fresh_record(tmp_path)
    rec["schema"] = "df_cell_scope_record.v1"
    doc = _envelope(rec)
    doc["cell_scope_envelope"]["record_schema"] = "df_cell_scope_record.v1"
    v = CS.verify_fresh_attempt(doc, cell_id="u1", attempt_id="0000000000001-abc", stage="CELL",
                                started_at=now - 1, finished_at=now + 1,
                                declared_cap_bytes=4 * 1024 ** 3, verify_reservation=False)
    assert any(r.startswith("RECORD_SCHEMA") for r in v["refused_by"]), v["refused_by"]


def test_an_envelope_pointing_at_a_place_where_no_record_sits_is_refused(tmp_path):
    now = time.time()
    doc = _envelope(_fresh_record(tmp_path))
    doc["cell_scope_envelope"]["record_at"] = "somewhere_else"
    v = CS.verify_fresh_attempt(doc, cell_id="u1", attempt_id="0000000000001-abc", stage="CELL",
                                started_at=now - 1, finished_at=now + 1,
                                declared_cap_bytes=4 * 1024 ** 3, verify_reservation=False)
    assert v["accepted"] is False
    assert v["refused_by"], "an envelope that points nowhere must refuse by name"


# ============================================================ 6. a failed delivery

def test_a_child_whose_consumed_bytes_are_not_the_delivered_bytes_refuses(tmp_path):
    """Failure mode 6.  The digest is taken of the bytes the child ACTUALLY opened.

    A delivery record is a claim by the deliverer.  The child re-digests the file it read, and a
    disagreement refuses the unit rather than costing it.
    """
    Q = _load("df_q2_household_governed")
    panel = tmp_path / "panel.parquet"
    panel.write_bytes(b"not the delivered bytes")
    v = Q.verify_consumed_bytes(panel, delivered={"sha256": "0" * 64, "bytes": 23},
                                expected_sha256="0" * 64)
    assert v["ok"] is False
    assert v["refused_by"] == "CONSUMED_BYTES_ARE_NOT_THE_DELIVERED_BYTES"
    assert v["sha256_reverified_in_child"] == Q.sha_file(panel)


def test_a_delivery_of_bytes_that_are_not_the_characterised_panel_refuses(tmp_path):
    Q = _load("df_q2_household_governed")
    panel = tmp_path / "panel.parquet"
    panel.write_bytes(b"coherent but not the panel")
    actual = Q.sha_file(panel)
    v = Q.verify_consumed_bytes(panel, delivered={"sha256": actual, "bytes": panel.stat().st_size},
                                expected_sha256="1" * 64)
    assert v["ok"] is False
    assert v["refused_by"] == "DELIVERED_PANEL_IS_NOT_THE_CHARACTERISED_PANEL"


def test_a_unit_may_not_read_a_panel_delivered_to_another_unit(tmp_path):
    """The reader is per unit.  A shared cache is not a licence to read another unit's delivery."""
    G = _load("df_e1_governed")
    root = tmp_path / "root"
    root.mkdir()
    (root / "DELIVERIES.json").write_text(json.dumps(
        {"schema": G.SCHEMA, "units": {"other": {"sha256": "0" * 64, "path": "/nowhere"}}}))
    with pytest.raises(SystemExit) as e:
        G.require_delivery(root, {"design_sha256": "d"}, "mine")
    assert "mine" in str(e.value) or "REFUSED" in str(e.value)


def test_the_matching_delivery_is_accepted_and_carries_the_childs_own_digest(tmp_path):
    Q = _load("df_q2_household_governed")
    panel = tmp_path / "panel.parquet"
    panel.write_bytes(b"the delivered bytes")
    actual = Q.sha_file(panel)
    v = Q.verify_consumed_bytes(panel, delivered={"sha256": actual, "bytes": panel.stat().st_size},
                                expected_sha256=actual)
    assert v["ok"] is True
    assert v["refused_by"] is None
    assert v["sha256_reverified_in_child"] == actual
    assert v["bytes_on_disk"] == panel.stat().st_size


# ============================================================ the runner, wired

def test_run_units_no_longer_hands_its_children_an_empty_cuda_override():
    """The integration's whole point, asserted against the source of the runner.

    `CUDA_VISIBLE_DEVICES=""` written into `run_units` makes every cell a CPU cell whatever the
    design says, which is the `run_units:976` defect the telemetry lane named.  It is replaced by
    the declared contract, and this test fails if anyone writes the literal back.
    """
    def code(name):
        """Executable lines only: a comment that QUOTES the defect is how it is documented."""
        return "\n".join(l for l in (TOOLS / name).read_text().splitlines()
                          if not l.lstrip().startswith("#"))

    src = code("df_e1_block.py")
    assert '"CUDA_VISIBLE_DEVICES": ""' not in src
    assert "df_placement_contract" in src
    # and specifically in the runner's own body, which is where the defect lived
    body = src.split("def run_units(", 1)[1].split("\ndef ", 1)[0]
    assert "CUDA_VISIBLE_DEVICES" not in body
    assert "child_placement_env" in body
    hh = code("df_q2_household_governed.py")
    assert '"CUDA_VISIBLE_DEVICES": ""' not in hh
    assert "df_placement_contract" in hh
    assert "supervise(" in hh


def test_the_household_child_is_launched_through_the_supervisor_and_not_a_bare_subprocess():
    """A bare `subprocess.run` takes no scope, no reservation and no lease.

    The retained producer-to-supervisor run excluded governed delivery, terminal reporting and the
    GPU; the household route excluded the supervisor.  The integration is the one path that has
    both, so the absence of a bare subprocess launch here is load-bearing.
    """
    hh = (TOOLS / "df_q2_household_governed.py").read_text()
    body = hh.split("def run_unit(", 1)[1].split("\ndef ", 1)[0]
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
    assert "subprocess.run(" not in body
    assert "supervise(" in body
    assert "fresh_attempt" in body
    assert "lease_id" in body


def test_the_integrated_runner_declares_both_placements_and_refuses_a_third():
    for placement in ("CPU", "GPU"):
        assert placement in PC.PLACEMENTS
    with pytest.raises(PC.PlacementRefusal) as e:
        PC.child_placement_env({}, placement="TPU")
    assert e.value.code == "PLACEMENT_NOT_DECLARED"


def test_the_bitwise_replay_stays_on_a_declared_cpu_placement():
    """The one path that is pinned to the CPU on purpose, and says why rather than leaving a hole.

    `replay_cell` reloads a SEALED cell's weights and asserts its predictions to 1e-6.  Every retained
    cell was produced on the CPU, and GPU arithmetic -- reduction order, TF32 in matmuls, cuDNN
    kernel choice -- would break a faithful replay for a reason that has nothing to do with the
    cell.  So it is a DECLARED CPU placement, not an inherited empty override, and it does not
    follow the run's placement.
    """
    src = (TOOLS / "df_e1_block.py").read_text()
    body = src.split("def replay_cell(", 1)[1].split("\ndef ", 1)[0]
    assert 'placement="CPU"' in body
    assert "child_placement_env" in body
    assert '"CUDA_VISIBLE_DEVICES": ""' not in "\n".join(
        l for l in body.splitlines() if not l.lstrip().startswith("#"))


def test_an_incomplete_wheel_set_whose_libraries_the_loader_finds_is_NOT_refused(tmp_path):
    """The false negative I shipped, and a live measurement refuted, pinned as a test.

    The first version of this contract refused a GPU placement whenever the interpreter's
    `nvidia/*/lib` inventory was incomplete.  Measured against the coordinator's real environment
    that gate refused a placement that WORKS: six of the nine sonames are absent from its wheels
    and TensorFlow 2.18 there registers `/physical_device:GPU:0` regardless, because the dynamic
    loader finds them elsewhere.  Loadability is the question; an inventory is not.  A refusal that
    stops a working GPU run is the same class of error as a downgrade that hides a broken one --
    both substitute an inventory for an outcome.
    """
    empty = tmp_path / "site-packages"
    empty.mkdir()
    found = PC.cuda_library_dirs(purelib=empty, loader=lambda n: True)
    assert found["status"] == PC.MEASURED
    assert found["missing"] == []
    assert found["wheel_inventory"] == {}
    assert all(r["route"] == "DYNAMIC_LOADER_DEFAULT_SEARCH" for r in found["resolved"].values())
    assert set(found["resolved"]) == set(PC.required_families())
    env, decl = PC.child_placement_env({}, placement="GPU", device_uuid=REAL_UUID,
                                       purelib=empty, loader=lambda n: True)
    assert env["CUDA_VISIBLE_DEVICES"] == REAL_UUID
    assert decl["placement"] == "GPU"


def test_each_soname_records_which_route_resolved_it(libdirs):
    """Which route answered matters when an environment changes underneath a pilot.

    A soname carried by the interpreter's own wheel keeps working because the contract puts that
    directory on the child's path; one found only by the loader's default search disappears if the
    host's system libraries change, and then the honest outcome is a refusal rather than a CPU
    pilot wearing a GPU label.  The record says which, per library, so that is checkable later.
    """
    purelib, _ = libdirs
    found = PC.cuda_library_dirs(purelib=purelib, loader=lambda n: False)
    assert found["status"] == PC.MEASURED
    assert set(found["resolved"]) == set(PC.required_families())
    assert all(r["route"] == "INTERPRETER_WHEEL" for r in found["resolved"].values())
    _, decl = PC.child_placement_env({}, placement="GPU", device_uuid=REAL_UUID,
                                     purelib=purelib, loader=lambda n: False)
    assert set(decl["cuda_libraries"]["routes"]) == set(PC.required_families())
    assert set(decl["cuda_libraries"]["sonames"]) == set(PC.required_families())


# ============================================================ the stops, executed

def test_the_cpu_stop_actually_kills_a_child_that_spends_its_budget():
    """CPU ENFORCEMENT, not a table row: a real child spins and the KERNEL ends it.

    `RLIMIT_CPU` raises SIGXCPU at the soft limit and SIGKILL at the hard one, so the stop does
    not depend on the child cooperating, on a watchdog thread being scheduled, or on a wall limit
    that bounds nothing when the host is busy.  The assertion is on the exit status of a process
    that really ran, which is the only thing that distinguishes an enforced budget from a declared
    one.
    """
    import subprocess
    spin = (
        "import resource\n"
        "resource.setrlimit(resource.RLIMIT_CPU, (1, 2))\n"
        "x = 0\n"
        "while True:\n"
        "    x += 1\n"
        "print('SURVIVED')\n")
    proc = subprocess.run([sys.executable, "-c", spin], capture_output=True, text=True, timeout=90)
    assert proc.returncode != 0, "a child that spent its CPU budget exited cleanly"
    assert "SURVIVED" not in proc.stdout
    # -24 is SIGXCPU, -9 is SIGKILL at the hard limit; both are the kernel ending it
    assert proc.returncode in (-24, -9, 152, 137), proc.returncode


def test_a_child_entering_a_stage_with_its_budget_already_spent_does_not_execute_its_body():
    """The stage stop checks the budget AT ENTRY, so a spent budget never starts new work."""
    T = _load("df_tf_device_telemetry")
    ran = []
    with pytest.raises(BaseException) as e:
        with T.stage_stop("spent", cpu_seconds=0.0):
            ran.append("body")
    assert ran == [], "the stage body executed although its CPU budget was spent"
    assert "spent" in str(e.value)


@pytest.mark.skipif(not CS.launcher_available(),
                    reason="the deployed crispdm-run is absent on this host")
def test_the_wall_stop_is_enforced_by_the_launcher_outside_the_child():
    """WALL ENFORCEMENT: the supervisor ends a child that ignores its wall limit.

    The stop is outside the child and does not trust it, which is the whole point: a child that
    hangs, that blocks uninterruptibly or that has lost its watchdog thread is still ended.  It is
    a 3-second sleep against a 2-second limit, so it costs nothing and proves the mechanism.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "w.log"
        sup = CS.supervise(cell_id="wallstop", argv=[sys.executable, "-c",
                                                     "import time;time.sleep(120);print('SURVIVED')"],
                           cap_bytes=512 * 1024 ** 2, wall_seconds=2,
                           supervisor_dir=Path(tmp) / "sup", log_path=log,
                           record_path=Path(tmp) / "absent.json")
        printed = log.read_text() if log.is_file() else ""
    term = sup["termination"]
    assert term["status"] != "COMPLETED", term
    assert "SURVIVED" not in printed, "the child outlived its wall stop"
    assert term["wall_seconds"] < 60, term["wall_seconds"]
    # and a child that was stopped leaves no costable record: absence refuses
    assert sup["usable_for_costing"] is False


def test_either_cuda_generation_satisfies_the_contract_and_the_record_says_which(tmp_path):
    """The second limitation a live measurement found, and the reason families replaced sonames.

    The first version of this table named only the CUDA 12 sonames.  The admitted worker's
    TensorFlow 2.21 environment carries BOTH generations -- that is the cu12/cu13 mix the telemetry
    lane named as the reason TensorFlow's own probe came back empty -- so a CUDA 13 build would
    have been refused for carrying exactly the libraries it is supposed to carry.  A family is now
    satisfied by any of its known sonames, and the record names which one answered, because two
    generations on one path is the state that produced the empty probe.
    """
    for generation in (0, 1):
        purelib = tmp_path / f"sp{generation}"
        _wheelset(purelib, generation=generation)
        found = PC.cuda_library_dirs(purelib=purelib, loader=lambda n: False)
        assert found["status"] == PC.MEASURED, found["missing"]
        assert set(found["resolved"]) == set(PC.required_families())
        for family, r in found["resolved"].items():
            assert r["soname"] in PC.family_sonames(family)
        assert found["generations_seen"]


def test_a_family_present_in_no_generation_is_the_refusal(tmp_path):
    """The refusal is per FAMILY, and it names every soname that would have satisfied it."""
    purelib = tmp_path / "sp"
    _wheelset(purelib)
    for name in PC.family_sonames("cublas"):
        f = purelib / "nvidia" / "cublas" / "lib" / name
        if f.is_file():
            f.unlink()
    found = PC.cuda_library_dirs(purelib=purelib, loader=lambda n: False)
    assert found["status"] == PC.UNKNOWN
    assert [m["family"] for m in found["missing"]] == ["cublas"]
    assert set(found["missing"][0]["any_of"]) == set(PC.family_sonames("cublas"))
