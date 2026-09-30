"""QRM02/F3 tests: the allocator that measures the model that actually trains, and stops that stop.

Written BEFORE the module, red first.  Every test here is a finding of the dictamen
`docs/audits/work_plan/MUSASHI_AUDIT_23B2EFA3_2026_09_29.md` turned into an executable refusal:

  F3a  the recipe is TensorFlow/Keras, so the framework figure must come from TensorFlow's OWN
       allocator API.  PyTorch may not be imported to measure it, not even to ask.
  F3b  an unavailable API is UNKNOWN.  Never 0, never synthesized from host RAM or a parameter count.
  F3c  the device actually used must be DECLARED and VERIFIED.  An empty CUDA_VISIBLE_DEVICES with a
       GPU placement declared is the `run_units:976` defect and is refused by name.
  F3d  framework allocator, process resident set, host cgroup and whole-device telemetry are FOUR
       scopes.  They are never added, and the module refuses to add them.
  F3e  a statistics check performed after an allocation cannot promise to abort before it.  Only a
       predeclared allocator limit can, and the two are labelled differently.
  F3f  CPU, wall and every stage need an EXECUTABLE stop.  A row in a table is not a stop, so each
       stop is proved by a child process that actually dies.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import df_tf_device_telemetry as T                                            # noqa: E402

SOURCE = (HERE / "df_tf_device_telemetry.py").read_text()

UUID_A = "GPU-11111111-2222-3333-4444-555555555555"
UUID_B = "GPU-99999999-8888-7777-6666-555555555555"


# --- F3a: the framework under measurement is TensorFlow, and PyTorch is not imported to ask -------

def test_the_module_never_imports_pytorch():
    assert not re.search(r"^\s*(import torch|from torch)\b", SOURCE, re.M), \
        "F3a: importing PyTorch to measure TensorFlow is exactly the defect"


def test_using_the_module_does_not_pull_pytorch_into_the_process():
    before = "torch" in sys.modules
    T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=123, current=45))
    T.scope_record(cell_id="c", stage="S", framework=T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=1, current=1)))
    assert ("torch" in sys.modules) == before


def test_the_framework_figure_names_tensorflows_own_api():
    got = T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=777, current=111))
    assert got["framework"] == "tensorflow"
    assert got["api"] == "tf.config.experimental.get_memory_info"
    assert got["status"] == T.MEASURED
    assert got["peak_bytes"] == 777 and got["current_bytes"] == 111
    assert got["scope"] == T.FRAMEWORK_SCOPE


def test_the_peak_reset_names_tensorflows_own_reset_api():
    fake = _FakeTF(peak=500, current=10)
    got = T.reset_tf_allocator_peak("GPU:0", tf=fake)
    assert got["api"] == "tf.config.experimental.reset_memory_stats"
    assert got["status"] == T.MEASURED and got["reset"] is True
    assert fake.resets == ["GPU:0"]


# --- F3b: an unavailable API is UNKNOWN ------------------------------------------------------------

def test_an_unavailable_allocator_api_is_unknown_and_not_zero():
    got = T.tf_allocator_memory("CPU:0", tf=_FakeTF(raises=ValueError("Allocator stats not available for device 'CPU:0'")))
    assert got["status"] == T.UNKNOWN
    assert got["peak_bytes"] is None and got["current_bytes"] is None
    assert "not available" in got["why"]


def test_an_unavailable_reset_is_unknown_and_does_not_claim_a_reset_basis():
    got = T.reset_tf_allocator_peak("CPU:0", tf=_FakeTF(raises=ValueError("Cannot reset memory stats for device 'CPU:0'")))
    assert got["status"] == T.UNKNOWN and got["reset"] is False


def test_a_missing_peak_is_unknown_in_the_record_never_zero():
    rec = T.scope_record(cell_id="c", stage="S",
                         framework=T.tf_allocator_memory("CPU:0", tf=_FakeTF(raises=ValueError("Allocator stats not available"))))
    assert rec["framework_allocator"]["status"] == T.UNKNOWN
    assert rec["framework_allocator"]["peak_bytes"] is None
    assert rec["framework_allocator"]["peak_bytes"] != 0


def test_no_framework_figure_is_ever_synthesized_from_another_scope():
    rec = T.scope_record(cell_id="c", stage="S",
                         framework=T.tf_allocator_memory("CPU:0", tf=_FakeTF(raises=ValueError("nope"))),
                         host_cgroup={"bytes": 9 << 30, "status": T.MEASURED},
                         process_rss={"bytes": 3 << 30, "status": T.MEASURED})
    assert rec["framework_allocator"]["peak_bytes"] is None, \
        "F3b: a host figure may never stand in for the framework figure"


# --- F3c: declare and verify the device actually used ----------------------------------------------

def test_an_empty_cuda_visible_devices_with_a_gpu_declared_is_refused_by_name():
    got = T.verify_declared_device(declared_uuid=UUID_A, placement="GPU",
                                   env={"CUDA_VISIBLE_DEVICES": ""}, tf=_FakeTF(gpus=[]), driver=_fake_driver([]))
    assert got["status"] == T.REFUSED
    assert "NO_VISIBLE_DEVICE" in got["refusals"], got
    assert got["establishes_a_gpu_pilot"] is False


def test_an_unset_cuda_visible_devices_with_a_gpu_declared_is_refused():
    got = T.verify_declared_device(declared_uuid=UUID_A, placement="GPU",
                                   env={}, tf=_FakeTF(gpus=[]), driver=_fake_driver([]))
    assert got["status"] == T.REFUSED and "NO_VISIBLE_DEVICE" in got["refusals"]


def test_more_than_one_visible_device_is_refused_as_ambiguous_pinning():
    got = T.verify_declared_device(declared_uuid=UUID_A, placement="GPU",
                                   env={"CUDA_VISIBLE_DEVICES": f"{UUID_A},{UUID_B}"},
                                   tf=_FakeTF(gpus=["/physical_device:GPU:0", "/physical_device:GPU:1"]),
                                   driver=_fake_driver([UUID_A, UUID_B]))
    assert got["status"] == T.REFUSED and "AMBIGUOUS_PINNING" in got["refusals"]


def test_a_driver_uuid_that_is_not_the_declared_one_is_refused():
    got = T.verify_declared_device(declared_uuid=UUID_A, placement="GPU",
                                   env={"CUDA_VISIBLE_DEVICES": UUID_A},
                                   tf=_FakeTF(gpus=["/physical_device:GPU:0"]), driver=_fake_driver([UUID_B]))
    assert got["status"] == T.REFUSED and "DECLARED_DEVICE_IS_NOT_THE_DEVICE_USED" in got["refusals"]


def test_a_declared_gpu_that_tensorflow_cannot_register_is_refused_not_downgraded_silently():
    got = T.verify_declared_device(declared_uuid=UUID_A, placement="GPU",
                                   env={"CUDA_VISIBLE_DEVICES": UUID_A},
                                   tf=_FakeTF(gpus=[]), driver=_fake_driver([UUID_A]))
    assert got["status"] == T.REFUSED
    assert "FRAMEWORK_REGISTERED_NO_DEVICE" in got["refusals"]


def test_a_verified_gpu_records_the_device_it_verified_and_how():
    got = T.verify_declared_device(declared_uuid=UUID_A, placement="GPU",
                                   env={"CUDA_VISIBLE_DEVICES": UUID_A},
                                   tf=_FakeTF(gpus=["/physical_device:GPU:0"], placed_on="/job:localhost/replica:0/task:0/device:GPU:0"),
                                   driver=_fake_driver([UUID_A]))
    assert got["status"] == T.MEASURED and got["establishes_a_gpu_pilot"] is True
    assert got["device_uuid_verified"] == UUID_A
    assert got["uuid_source"] == "libcuda.cuDeviceGetUuid"
    assert got["placement_probe"]["device"].endswith("GPU:0")


def test_a_cpu_placement_is_a_cpu_pilot_and_says_so_without_inventing_a_device():
    got = T.verify_declared_device(declared_uuid=None, placement="CPU",
                                   env={"CUDA_VISIBLE_DEVICES": ""}, tf=_FakeTF(gpus=[]), driver=_fake_driver([]))
    assert got["status"] == T.MEASURED
    assert got["establishes_a_gpu_pilot"] is False
    assert got["device_uuid_verified"] is None
    assert "CPU" in got["placement"]


# --- F3d: four scopes, never merged ----------------------------------------------------------------

def test_the_record_keeps_four_named_scopes_each_with_its_own_basis():
    rec = T.scope_record(cell_id="c", stage="S",
                         framework=T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=100, current=10)),
                         host_cgroup={"bytes": 200, "status": T.MEASURED},
                         process_rss={"bytes": 300, "status": T.MEASURED},
                         whole_device={"bytes": 400, "status": T.MEASURED})
    keys = ("framework_allocator", "host_cgroup", "process_rss", "whole_device")
    assert all(k in rec for k in keys)
    assert len({rec[k]["scope"] for k in keys}) == 4
    assert all("never added" in rec[k]["scope_rule"].lower() for k in keys)


def test_adding_two_scopes_is_refused_in_code():
    rec = T.scope_record(cell_id="c", stage="S",
                         framework=T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=100, current=10)),
                         host_cgroup={"bytes": 200, "status": T.MEASURED})
    with pytest.raises(T.ScopeRefusal):
        T.merge_scopes(rec, ["framework_allocator", "host_cgroup"])


def test_no_field_in_the_record_equals_the_sum_of_two_scopes():
    a, b, c = 1_000, 20_000, 300_000                     # no two of these sum to a third or to each other
    rec = T.scope_record(cell_id="c", stage="S",
                         framework=T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=a, current=1)),
                         host_cgroup={"bytes": b, "status": T.MEASURED},
                         process_rss={"bytes": c, "status": T.MEASURED})
    values = set(_numbers(rec))
    assert a in values and b in values and c in values
    for s in (a + b, a + c, b + c, a + b + c):
        assert s not in values, f"a field equal to {s} would be two scopes added"


# --- F3e: what is enforceable, and what is only observed -------------------------------------------

def test_a_predeclared_allocator_limit_is_the_only_thing_that_can_fail_an_allocation():
    fake = _FakeTF(gpus=["/physical_device:GPU:0"])
    got = T.install_device_envelope(12 << 30, tf=fake)
    assert got["status"] == T.MEASURED
    assert got["enforcement"] == T.ENFORCED_AT_ALLOCATION
    assert got["api"] == "tf.config.set_logical_device_configuration"
    assert got["limit_mib"] == 12288
    assert fake.logical_limits == [12288]


def test_an_envelope_asked_for_after_the_device_is_initialized_is_refused_not_ignored():
    fake = _FakeTF(gpus=["/physical_device:GPU:0"], logical_raises=RuntimeError("Virtual devices cannot be modified after being initialized"))
    got = T.install_device_envelope(12 << 30, tf=fake)
    assert got["status"] == T.REFUSED
    assert got["enforcement"] == T.NOT_ENFORCED
    assert "initialized" in got["why"]


def test_the_after_the_fact_statistics_check_never_claims_to_abort_before():
    got = T.envelope_observation(T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=(13 << 30), current=1)), envelope_bytes=12 << 30)
    assert got["enforcement"] == T.OBSERVED_AFTER_THE_FACT
    assert got["exceeded"] is True
    assert got["does_not_prevent"] is True
    assert "after" in got["reading"].lower()


def test_an_unknown_peak_cannot_pass_an_envelope_observation():
    got = T.envelope_observation(T.tf_allocator_memory("GPU:0", tf=_FakeTF(raises=ValueError("nope"))), envelope_bytes=12 << 30)
    assert got["exceeded"] is None and got["status"] == T.UNKNOWN


def test_the_source_never_promises_an_abort_before_from_a_statistic():
    """An abort-before-an-allocation phrase may appear in exactly two places: beside the mechanism
    that really does it, or inside the REFUTED_CLAIM marker that quotes the design's own error."""
    for bad in re.finditer(r"(abort|raises?|stops?)[^.\n]{0,60}before[^.\n]{0,60}(allocat)", SOURCE, re.I):
        window = SOURCE[max(0, bad.start() - 700):bad.end() + 700]
        assert (T.ENFORCED_AT_ALLOCATION in window) or ("REFUTED_CLAIM" in window), \
            f"F3e: an abort-before promise must sit with the ENFORCED mechanism or be marked a " \
            f"refuted claim, never float free beside a statistic: {bad.group(0)!r}"


def test_the_refuted_claim_is_quoted_and_marked_as_refuted():
    assert "REFUTED_CLAIM" in SOURCE
    assert T.REFUTED_ABORT_BEFORE_CLAIM
    claim = T.REFUTED_ABORT_BEFORE_CLAIM["claim"].lower()
    assert "before allocating" in claim, "the refuted claim must be the design's own words"
    assert T.REFUTED_ABORT_BEFORE_CLAIM["status"] == "REFUTED"
    assert T.ENFORCED_AT_ALLOCATION in T.REFUTED_ABORT_BEFORE_CLAIM["what_is_enforceable"]


# --- F3f: every stop is executable, and a child proves it ------------------------------------------

def _child(body: str, timeout: float = 60.0) -> tuple:
    src = textwrap.dedent(f"""
        import sys, time, os
        sys.path.insert(0, {str(HERE)!r})
        import df_tf_device_telemetry as T
        {textwrap.indent(textwrap.dedent(body), ' ' * 8).strip()}
    """)
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout)
    return p, time.monotonic() - t0


def test_the_cpu_stop_is_enforced_by_the_kernel_and_a_child_actually_dies():
    p, wall = _child("""
        T.install_cpu_stop(1)
        x = 0
        while True:
            x += 1
    """)
    assert p.returncode != 0, p.stdout + p.stderr
    assert wall < 40
    assert "CPU_STOP" in (p.stdout + p.stderr) or p.returncode < 0, (p.returncode, p.stdout, p.stderr)


def test_the_cpu_stop_names_rlimit_cpu_as_its_mechanism():
    p, _ = _child("""
        d = T.install_cpu_stop(600)
        print(d["mechanism"], d["enforced_by"], d["hard_stop"])
    """)
    assert p.returncode == 0, p.stderr
    assert "RLIMIT_CPU" in p.stdout and "kernel" in p.stdout.lower()


def test_the_wall_stop_fires_and_a_sleeping_child_actually_dies():
    p, wall = _child("""
        T.install_wall_stop(1)
        time.sleep(120)
        print("SURVIVED")
    """)
    assert p.returncode != 0 and "SURVIVED" not in p.stdout
    assert wall < 30
    assert "WALL_STOP" in (p.stdout + p.stderr) or p.returncode < 0


def test_a_stage_stop_fires_inside_the_stage_and_names_the_stage():
    p, wall = _child("""
        try:
            with T.stage_stop("warmup", wall_seconds=1):
                time.sleep(120)
        except BaseException as e:
            print("STOPPED", type(e).__name__, e)
            raise SystemExit(3)
        print("SURVIVED")
    """)
    assert p.returncode != 0 and "SURVIVED" not in p.stdout
    assert "warmup" in (p.stdout + p.stderr)
    assert wall < 30


def test_a_stage_that_finishes_inside_its_stop_returns_its_own_measurements():
    p, _ = _child("""
        with T.stage_stop("build", wall_seconds=30, cpu_seconds=30) as st:
            time.sleep(0.05)
        print("OK", st["stage"], st["stopped"], st["wall_seconds"] > 0)
    """)
    assert p.returncode == 0, p.stderr
    assert "OK build False True" in p.stdout


def test_a_stage_cpu_budget_already_spent_stops_the_stage_at_entry():
    p, _ = _child("""
        t = time.process_time()
        while time.process_time() - t < 0.3:
            pass
        try:
            with T.stage_stop("steady", cpu_seconds=0.01):
                print("ENTERED")
        except BaseException as e:
            print("STOPPED_AT_ENTRY", type(e).__name__)
            raise SystemExit(4)
    """)
    assert p.returncode == 4, (p.returncode, p.stdout, p.stderr)
    assert "ENTERED" not in p.stdout


def test_the_stops_declaration_lists_every_stop_with_a_mechanism_and_none_is_only_a_table_row():
    d = T.stops_declaration(cpu_seconds=2400, wall_seconds=1800, stages=["build", "warmup", "steady"],
                            host_envelope_bytes=12 << 30, device_envelope_bytes=12 << 30)
    for row in d["stops"]:
        assert row["mechanism"], row
        assert row["enforced_by"] in ("kernel", "framework_allocator", "supervisor", "process"), row
        assert row["executable"] is True, row
    assert {r["name"] for r in d["stops"]} >= {"cpu", "wall", "host_envelope", "device_envelope",
                                               "stage:build", "stage:warmup", "stage:steady"}


# --- helpers ---------------------------------------------------------------------------------------

def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, int) and not isinstance(obj, bool):
        yield obj


def _fake_driver(uuids):
    def driver(index=0):
        if not uuids:
            return {"status": T.UNKNOWN, "count": 0, "uuid": None,
                    "why": "the CUDA driver exposes no device to this process"}
        return {"status": T.MEASURED, "count": len(uuids), "uuid": uuids[index],
                "source": "libcuda.cuDeviceGetUuid"}
    return driver


class _FakeTF:
    """A stand-in for the TensorFlow module: only the four calls this instrument makes."""

    def __init__(self, *, peak=None, current=None, raises=None, gpus=(), placed_on="/device:CPU:0",
                 logical_raises=None):
        self._peak, self._current, self._raises = peak, current, raises
        self._gpus, self._placed_on, self._logical_raises = list(gpus), placed_on, logical_raises
        self.resets, self.logical_limits = [], []
        outer = self

        class _LDC:
            def __init__(self, memory_limit=None):
                self.memory_limit = memory_limit

        class _Experimental:
            LogicalDeviceConfiguration = _LDC

            @staticmethod
            def get_memory_info(device):
                if outer._raises:
                    raise outer._raises
                return {"current": outer._current, "peak": outer._peak}

            @staticmethod
            def reset_memory_stats(device):
                if outer._raises:
                    raise outer._raises
                outer.resets.append(device)

            @staticmethod
            def get_device_details(dev):
                return {"device_name": "FAKE", "compute_capability": (8, 9)}

        class _Config:
            experimental = _Experimental
            LogicalDeviceConfiguration = _LDC

            @staticmethod
            def list_physical_devices(kind="GPU"):
                return [_Dev(n) for n in outer._gpus] if kind == "GPU" else [_Dev("/physical_device:CPU:0")]

            @staticmethod
            def set_logical_device_configuration(dev, cfgs):
                if outer._logical_raises:
                    raise outer._logical_raises
                outer.logical_limits.extend(int(c.memory_limit) for c in cfgs)

        class _Dev:
            def __init__(self, name):
                self.name = name
                self.device_type = "GPU" if "GPU" in name else "CPU"

            def __repr__(self):
                return f"PhysicalDevice(name={self.name!r})"

        class _Tensor:
            device = placed_on

        class _Ctx:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, *a):
                return False

        self.config = _Config
        self.device = lambda name: _Ctx()
        self.zeros = lambda shape, dtype=None: _Tensor()
        self.float32 = "float32"


# --- found by running the instrument on two hosts: two TensorFlow builds, two answers -------------

def test_a_cpu_allocator_figure_is_host_ram_and_carries_a_different_scope():
    """TF 2.18 raises for CPU:0; TF 2.21 answers with its HOST allocator's statistics. Both were
    observed on real hosts. When it answers, the figure is host RAM held by TensorFlow -- a third
    thing beside the cgroup watermark and the device allocator -- and it says so."""
    got = T.tf_allocator_memory("CPU:0", tf=_FakeTF(peak=67108864, current=0))
    assert got["status"] == T.MEASURED
    assert got["memory"] == "HOST"
    assert got["scope"] == T.FRAMEWORK_SCOPE_HOST
    assert got["scope"] != T.FRAMEWORK_SCOPE


def test_a_device_envelope_is_never_compared_with_a_host_allocator_figure():
    """The defect this test exists for was real: the worker's TensorFlow registered no GPU, the
    self-check fell back to CPU:0, and the observation compared a HOST figure with a DEVICE envelope
    and reported `exceeded: false`. A pass across two scopes is not a pass."""
    host_figure = T.tf_allocator_memory("CPU:0", tf=_FakeTF(peak=67108864, current=0))
    got = T.envelope_observation(host_figure, envelope_bytes=12 << 30, expected_device="GPU:0")
    assert got["status"] == T.REFUSED
    assert got["exceeded"] is None
    assert "host RAM" in got["why"]


def test_a_device_envelope_observation_on_the_declared_device_still_works():
    got = T.envelope_observation(T.tf_allocator_memory("GPU:0", tf=_FakeTF(peak=1 << 20, current=1)),
                                 envelope_bytes=12 << 30, expected_device="GPU:0")
    assert got["status"] == T.MEASURED and got["exceeded"] is False


def test_the_record_keeps_a_host_framework_figure_under_its_own_scope():
    rec = T.scope_record(cell_id="c", stage="S",
                         framework=T.tf_allocator_memory("CPU:0", tf=_FakeTF(peak=5, current=1)))
    assert rec["framework_allocator"]["scope"] == T.FRAMEWORK_SCOPE_HOST
