#!/usr/bin/env python3
"""The explicit CPU/GPU placement contract that replaces an empty CUDA override.

Two lanes met here.  QRM01 (`tools/df_cell_scope.py`, `2fee2fc7`) gave every cell its own scope,
its own reservation and a gate that certifies the attempt it claims to certify.  QRM02/F3
(`tools/df_tf_device_telemetry.py`, `8cbe99a3`) measured the TensorFlow model with TensorFlow's own
allocator and proved that the runner's `CUDA_VISIBLE_DEVICES=""` makes every child a CPU child.
This module is the join: the runner declares a placement, passes a **child-only** contract, and the
child verifies it **before TensorFlow is imported** and then on three independent facts.

**Why three facts and not one.**  Visibility proves nothing.  The telemetry lane measured, on a real
device, a state where the driver held the card, `cuDeviceGetUuid` returned exactly the declared
UUID, and TensorFlow registered **zero** GPUs (`Cannot dlopen some GPU libraries … Skipping
registering GPU devices`).  A visibility check would have sealed that as a GPU pilot.  So the
contract verifies

  1. **the physical device**, by UUID, through the CUDA driver -- not a framework;
  2. **the framework's registration** of that device, through TensorFlow's own device list;
  3. **execution placement**, by where an op actually lands.

and a GPU request that satisfies fewer than three is **refused**, never downgraded to a CPU run
wearing a GPU label.  `PlacementRefusal` is an exception for exactly that reason: a return value
inviting a caller to continue is how the downgrade happened.

**Why a new process, and why before the import.**  On this glibc the dynamic loader reads
`LD_LIBRARY_PATH` when a process starts.  Assigning it in the running process does not change the
search `dlopen` uses, even when the assignment is made before TensorFlow is imported.  The
supervisor builds a new mapping and starts a child with that mapping.  It does not write the
mapping into its own `os.environ`, import TensorFlow, or dlopen a CUDA library.  The child keeps
the mapping it was started with and does the check.  `enforce_before_tensorflow` still refuses
when `tensorflow` is already in `sys.modules`: a contract examined after that import is narration.

**Why child-only.**  The defect this replaces was a process-wide override.  Nothing here writes to
`os.environ`, and nothing mutates the mapping it is given: the contract is a NEW mapping handed to
one child, so two siblings can hold two different placements in one parent.

**PyTorch is not in this path.**  Not imported, not consulted, not named as a source.  Its caching
allocator does not account for one byte TensorFlow allocates, and the integrated `gpu_memory`
answers from `tf.config.experimental.get_memory_info` or answers `UNKNOWN`.

An unavailable fact is `UNKNOWN`.  Never 0, never inferred from another scope.
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

SCHEMA = "df_placement_contract.v1"

MEASURED = "MEASURED"
UNKNOWN = "UNKNOWN"
REFUSED = "REFUSED"

CPU = "CPU"
GPU = "GPU"
PLACEMENTS = (CPU, GPU)

PLACEMENT_ENV = "CRISPDM_PLACEMENT"
UUID_ENV = "CRISPDM_DEVICE_UUID"

CHILD_ONLY = ("CHILD_ONLY: this mapping is handed to one child process. It is never exported into "
              "this process's os.environ, because a process-wide override is the defect being "
              "replaced -- it silently re-placed every later sibling in the same parent")

THREE_FACTS = ("DRIVER_UUID", "FRAMEWORK_REGISTRATION", "EXECUTION_PLACEMENT")

# The library FAMILIES TensorFlow's GPU build dlopens, by the wheel that ships each, with the
# alternative sonames of each CUDA generation.  It is a DECLARED set rather than a discovered one:
# "whatever is on the path" cannot tell a search-path failure (telemetry state B, every library
# present and TensorFlow registering nothing) from a genuinely absent dependency, and the two need
# different answers.
#
# **Generations, and a limitation this fixes.**  A hardcoded soname generation is a wrong question.
# The first version of this table named only the CUDA 12 sonames; the admitted worker's TensorFlow
# 2.21 environment carries BOTH generations (that is the `cu12`/`cu13` mix the telemetry lane
# named), and a CUDA 13 build would have been refused for carrying the libraries it is supposed to
# carry.  So a family is satisfied by ANY of its known sonames, and the record says WHICH -- which
# is the fact a later reader needs, because two generations on one path is exactly how TensorFlow's
# own probe came back empty.
CUDA_PACKAGE_LIBRARIES = {
    "cuda_runtime": {"cudart": ("libcudart.so.13", "libcudart.so.12")},
    "cublas": {"cublas": ("libcublas.so.13", "libcublas.so.12"),
               "cublasLt": ("libcublasLt.so.13", "libcublasLt.so.12")},
    "cufft": {"cufft": ("libcufft.so.12", "libcufft.so.11")},
    "curand": {"curand": ("libcurand.so.10",)},
    "cusolver": {"cusolver": ("libcusolver.so.12", "libcusolver.so.11")},
    "cusparse": {"cusparse": ("libcusparse.so.12",)},
    "cudnn": {"cudnn": ("libcudnn.so.9",)},
    "nvjitlink": {"nvJitLink": ("libnvJitLink.so.13", "libnvJitLink.so.12")},
}


def required_families() -> list:
    """The family names, which are what a placement actually requires."""
    return sorted(f for fams in CUDA_PACKAGE_LIBRARIES.values() for f in fams)


def family_sonames(family: str) -> tuple:
    for fams in CUDA_PACKAGE_LIBRARIES.values():
        if family in fams:
            return fams[family]
    return ()


def required_libraries() -> list:
    """Every soname that could satisfy a family, for tests and for reporting."""
    return sorted(n for fams in CUDA_PACKAGE_LIBRARIES.values()
                  for names in fams.values() for n in names)


class PlacementRefusal(RuntimeError):
    """A placement that cannot be established.  Raised, never returned, so nothing continues on it."""

    def __init__(self, code: str, detail: str, evidence: dict | None = None):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail, self.evidence = code, detail, evidence or {}


# --- the library contract -------------------------------------------------------------------------

def cuda_library_dirs(*, purelib=None, executable: str | None = None, loader=None) -> dict:
    """Can this interpreter LOAD the CUDA libraries TensorFlow dlopens, and from where.

    **A correction I made against my own first version of this function, on a live measurement.**
    It first refused a GPU placement whenever the interpreter's `nvidia/*/lib` wheel inventory was
    incomplete.  Run against the coordinator's real environment that gate refused a placement that
    **works**: six of the nine sonames are absent from its wheels, and TensorFlow 2.18 there
    registers `/physical_device:GPU:0` regardless, because the loader finds them elsewhere.  A
    refusal that stops a working GPU run is the same class of error as a downgrade that hides a
    broken one -- both substitute an inventory for an outcome.

    So the question asked here is the one TensorFlow itself asks: **is each required soname
    loadable**, either

      * from a `nvidia/*/lib` directory of THIS interpreter -- which the contract then puts on the
        child's `LD_LIBRARY_PATH`, so the child will find it; or
      * by the dynamic loader's own default search, verified by actually loading it here -- which
        the child inherits.

    A soname that is neither is genuinely absent, and THAT is `MISSING_CUDA_LIBRARIES`.  The wheel
    inventory is still reported, because which of the two routes answered matters when the
    environment changes underneath a pilot; but it no longer decides anything on its own.

    Not a global filesystem search either: the telemetry lane's worker carried both `cu12` and
    `cu13` wheel sets, so "a file called libcublas exists somewhere on this host" is not the
    question.
    """
    if purelib is None:
        if executable and executable != sys.executable:
            purelib = _purelib_of(executable)
        else:
            purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return {"status": UNKNOWN, "dirs": [], "found": {}, "missing": required_libraries(),
                "purelib": None, "why": "this interpreter reports no site-packages directory"}
    # Two layouts, both measured on real environments: the per-package one
    # (`nvidia/<pkg>/lib/lib<x>.so.12`) and the flat generation one (`nvidia/cu13/lib/…`), which is
    # how the admitted worker ships CUDA 13 beside CUDA 12.  The newest generation is searched
    # first, per family, so an environment carrying both resolves to the newer one -- which is what
    # its TensorFlow wants and what the mixed path made it fail to find.
    root = Path(purelib) / "nvidia"
    flat = [root / d / "lib" for d in ("cu13", "cu12") if (root / d / "lib").is_dir()]
    dirs, in_wheels = [], {}
    for pkg, fams in sorted(CUDA_PACKAGE_LIBRARIES.items()):
        for lib in flat + [root / pkg / "lib"]:
            present = [n for names in fams.values() for n in names if (lib / n).is_file()]
            for n in present:
                in_wheels.setdefault(n, str(lib / n))
            if present and str(lib) not in dirs:
                dirs.append(str(lib))

    load = loader if loader is not None else _loadable
    resolved, missing = {}, []
    for family in required_families():
        chosen = None
        for n in family_sonames(family):
            if n in in_wheels:
                chosen = {"route": "INTERPRETER_WHEEL", "soname": n, "path": in_wheels[n],
                          "why": ("the contract puts this directory on the child's "
                                  "LD_LIBRARY_PATH, so this wheel keeps answering even if the "
                                  "host's system libraries change")}
                break
        if chosen is None:
            for n in family_sonames(family):
                if load(n):
                    chosen = {"route": "DYNAMIC_LOADER_DEFAULT_SEARCH", "soname": n, "path": None,
                              "why": ("loaded here by soname, so the child's loader finds it by "
                                      "the same default search; this is an OUTCOME, not an "
                                      "inventory -- and it stops answering if the host changes")}
                    break
        if chosen is None:
            missing.append({"family": family, "any_of": list(family_sonames(family))})
        else:
            resolved[family] = chosen
    out = {"dirs": dirs, "wheel_inventory": in_wheels, "resolved": resolved, "missing": missing,
           "purelib": str(purelib),
           "generations_seen": sorted({n.rsplit(".so.", 1)[1].split(".")[0]
                                       for n in in_wheels} | {
                                          r["soname"].rsplit(".so.", 1)[1].split(".")[0]
                                          for r in resolved.values()}),
           "wheel_inventory_is_not_the_verdict": (
               "the inventory is reported because which route and which CUDA generation answered "
               "matters when an environment changes under a pilot, but an incomplete wheel set "
               "alone refuses nothing: a live measurement showed TensorFlow registering a device "
               "with six of nine families absent from its wheels")}
    if missing:
        names = ", ".join(m["family"] for m in missing)
        return {**out, "status": UNKNOWN,
                "why": ("this interpreter cannot load " + names + " (in any known CUDA "
                        "generation) from its own wheels or by the loader's default search; a GPU "
                        "placement is refused rather than started as a CPU one")}
    return {**out, "status": MEASURED,
            "basis": ("every library family TensorFlow dlopens is loadable by THIS interpreter, "
                      "each by a named route and a named soname; the three facts verified in the "
                      "child are what then accept a GPU pilot, and this check only removes a "
                      "failure that is cheaper to find before the child starts")}


def _loadable(soname: str) -> bool:
    """Ask the dynamic loader, in this process, rather than believing a file listing."""
    import ctypes                                                            # noqa: PLC0415
    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False


def _purelib_of(executable: str):
    import subprocess                                                        # noqa: PLC0415
    try:
        out = subprocess.run([executable, "-c",
                              "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True, timeout=30)
    except Exception:                                                        # noqa: BLE001
        return None
    return out.stdout.strip() or None


# --- the contract the runner passes ---------------------------------------------------------------

def child_placement_env(base, *, placement: str | None, device_uuid: str | None = None,
                        purelib=None, executable: str | None = None, loader=None) -> tuple:
    """Build ONE child's environment for a DECLARED placement.  Returns (env, declaration).

    Refusals, each by its own name and each before the child is started -- which is the only point
    at which a placement defect is cheap:

      PLACEMENT_NOT_DECLARED                 no placement, or one that is neither CPU nor GPU.
                                             There is deliberately no default: a default placement
                                             is how a GPU request becomes a CPU run in silence
      DECLARED_DEVICE_WITH_CPU_PLACEMENT     a device named for a CPU run
      GPU_PLACEMENT_WITHOUT_A_DECLARED_DEVICE a GPU run with no device to attribute figures to
      MISSING_CUDA_LIBRARIES                 this interpreter cannot load the GPU libraries
    """
    declared = (placement or "").strip().upper()
    if declared not in PLACEMENTS:
        raise PlacementRefusal(
            "PLACEMENT_NOT_DECLARED",
            f"placement {placement!r} is not one of {PLACEMENTS}. It is declared per run, never "
            f"defaulted: a default is how a GPU request becomes a CPU run without anyone noticing")

    env = dict(base if base is not None else {})
    if declared == CPU:
        if device_uuid:
            raise PlacementRefusal(
                "DECLARED_DEVICE_WITH_CPU_PLACEMENT",
                f"a device ({device_uuid}) was named for a CPU placement; a CPU run has no device "
                f"and a record that carries one invites a GPU reading of a CPU figure")
        env["CUDA_VISIBLE_DEVICES"] = ""
        env[PLACEMENT_ENV] = CPU
        env.pop(UUID_ENV, None)
        decl = {"schema": SCHEMA, "placement": CPU, "device_uuid": None,
                "establishes_a_gpu_pilot": False, "scope": CHILD_ONLY,
                "cuda_libraries": {"status": "NOT_REQUIRED", "why": "a CPU placement loads none"},
                "env": {"CUDA_VISIBLE_DEVICES": "", PLACEMENT_ENV: CPU},
                "reading": ("a DECLARED CPU placement. The empty CUDA_VISIBLE_DEVICES is the "
                            "consequence of the declaration, not a leftover override: the "
                            "declaration is what lets the child refuse a later GPU claim")}
        return env, decl

    if not device_uuid:
        raise PlacementRefusal(
            "GPU_PLACEMENT_WITHOUT_A_DECLARED_DEVICE",
            "a GPU placement names no device UUID, so no figure it produces could be attributed "
            "to a device; declare the device or declare CPU")
    libs = cuda_library_dirs(purelib=purelib, executable=executable, loader=loader)
    if libs["status"] != MEASURED:
        raise PlacementRefusal("MISSING_CUDA_LIBRARIES", libs["why"], {"libraries": libs})

    existing = [p for p in str(env.get("LD_LIBRARY_PATH") or "").split(":") if p]
    env["LD_LIBRARY_PATH"] = ":".join(libs["dirs"] + [p for p in existing if p not in libs["dirs"]])
    env["CUDA_VISIBLE_DEVICES"] = str(device_uuid)
    env[PLACEMENT_ENV] = GPU
    env[UUID_ENV] = str(device_uuid)
    decl = {"schema": SCHEMA, "placement": GPU, "device_uuid": str(device_uuid),
            "establishes_a_gpu_pilot": None, "scope": CHILD_ONLY,
            "cuda_libraries": {"status": MEASURED, "dirs": libs["dirs"],
                               "routes": {f: r["route"] for f, r in libs["resolved"].items()},
                               "sonames": {f: r["soname"] for f, r in libs["resolved"].items()},
                               "generations_seen": libs["generations_seen"],
                               "wheel_inventory_count": len(libs["wheel_inventory"])},
            "env": {"CUDA_VISIBLE_DEVICES": str(device_uuid), PLACEMENT_ENV: GPU,
                    UUID_ENV: str(device_uuid), "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]},
            "reading": ("a DECLARED GPU placement with exactly one device pinned by UUID and that "
                        "interpreter's own CUDA library path. establishes_a_gpu_pilot is null "
                        "until the CHILD verifies all three facts -- the declaration is a request, "
                        "never the evidence")}
    return env, decl


def declared_placement(env=None) -> dict:
    """What the child was actually told, read from the environment it actually got."""
    env = os.environ if env is None else env
    declared = (env.get(PLACEMENT_ENV) or "").strip().upper()
    if declared not in PLACEMENTS:
        raise PlacementRefusal(
            "PLACEMENT_NOT_DECLARED",
            f"{PLACEMENT_ENV} is {env.get(PLACEMENT_ENV)!r}; a child never guesses its placement, "
            f"because guessing CPU is precisely the silent downgrade")
    return {"schema": SCHEMA, "placement": declared,
            "device_uuid": (env.get(UUID_ENV) or None) if declared == GPU else None,
            "visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "ld_library_path": env.get("LD_LIBRARY_PATH")}


# --- the child's first call, before TensorFlow ----------------------------------------------------

def enforce_before_tensorflow(env=None, modules=None) -> dict:
    """Check the contract the child received, while the check can still matter.

    The dynamic loader reads `LD_LIBRARY_PATH` when TensorFlow's extension modules load.  A
    contract examined after that import is narration, so an already-imported TensorFlow is a
    refusal rather than a warning.
    """
    modules = sys.modules if modules is None else modules
    if "tensorflow" in modules:
        raise PlacementRefusal(
            "TENSORFLOW_ALREADY_IMPORTED",
            "TensorFlow is already imported in this process, so the dynamic loader has already "
            "run and a library path set now changes nothing; the placement contract must be "
            "enforced before the import, not described after it")
    env = os.environ if env is None else env
    decl = declared_placement(env)
    if decl["placement"] == CPU:
        decl.update({"before_tensorflow_import": True, "establishes_a_gpu_pilot": False,
                     "reading": "a declared CPU placement; no device library is required"})
        return decl
    if not decl["device_uuid"]:
        raise PlacementRefusal(
            "GPU_PLACEMENT_WITHOUT_A_DECLARED_DEVICE",
            f"a GPU placement reached this child with no {UUID_ENV}, so it has nothing to verify "
            f"its device against")
    vis = decl["visible_devices"]
    if vis is None or str(vis).strip() == "":
        raise PlacementRefusal(
            "NO_VISIBLE_DEVICE",
            "a GPU placement reached this child with CUDA_VISIBLE_DEVICES empty or unset. This is "
            "the run_units:976 state: the driver is never asked, so the run is a CPU run and is "
            "refused rather than relabelled")
    if len([e for e in str(vis).split(",") if e.strip()]) != 1:
        raise PlacementRefusal(
            "AMBIGUOUS_PINNING",
            f"CUDA_VISIBLE_DEVICES exposes {vis!r}: more than one device, so no figure could be "
            f"attributed to the declared one")
    path = [p for p in str(decl["ld_library_path"] or "").split(":") if p]
    if not any("nvidia" in p for p in path):
        raise PlacementRefusal(
            "CUDA_LIBRARY_PATH_NOT_PASSED",
            "a GPU placement reached this child with no CUDA library directory on LD_LIBRARY_PATH. "
            "The telemetry lane measured exactly this state: the driver holds the device, the UUID "
            "matches and TensorFlow registers zero GPUs, because its probe never finds the "
            "libraries. Refused before the import, where it is still fixable")
    decl.update({"before_tensorflow_import": True, "establishes_a_gpu_pilot": None,
                 "cuda_library_dirs_on_path": [p for p in path if "nvidia" in p],
                 "reading": ("the contract arrived intact; the three facts are verified after the "
                             "import by verify_or_refuse, and only they accept a GPU pilot")})
    return decl


# --- the three facts ------------------------------------------------------------------------------

def verify_or_refuse(declaration: dict, *, tf=None, driver=None, telemetry=None) -> dict:
    """Verify the declared placement on all three facts, and REFUSE a GPU request that falls short.

    The verification itself is QRM02/F3's `verify_declared_device`, which reads the UUID through
    `libcuda`, lists TensorFlow's registered devices and probes where a one-element tensor lands.
    What this adds is the **disposition**: a GPU declaration whose verification is `REFUSED` raises
    `GPU_REQUEST_FELL_BACK_TO_CPU` and carries the underlying refusals as evidence.  Nothing is
    returned that a caller could mistake for permission to continue on the CPU.
    """
    T = telemetry if telemetry is not None else _telemetry()
    placement = (declaration.get("placement") or "").strip().upper()
    if placement not in PLACEMENTS:
        raise PlacementRefusal("PLACEMENT_NOT_DECLARED",
                               f"nothing to verify: placement is {placement!r}")
    env = declaration.get("env")
    if not env:
        env = {"CUDA_VISIBLE_DEVICES": declaration.get("visible_devices")}
    kwargs = {"declared_uuid": declaration.get("device_uuid"), "placement": placement, "env": env}
    if driver is not None:
        kwargs["driver"] = driver
    if placement == GPU or tf is not None:
        kwargs["tf"] = tf if tf is not None else _tf()
    v = dict(T.verify_declared_device(**kwargs))
    v["schema"] = SCHEMA
    v["facts_declared"] = list(THREE_FACTS)

    if placement == CPU:
        v["facts_verified"] = []
        v["establishes_a_gpu_pilot"] = False
        if v.get("status") == REFUSED:
            raise PlacementRefusal("CPU_PLACEMENT_REFUSED", "; ".join(v.get("refusals") or []), v)
        return v

    if v.get("status") != MEASURED or not v.get("establishes_a_gpu_pilot"):
        raise PlacementRefusal(
            "GPU_REQUEST_FELL_BACK_TO_CPU",
            ("a GPU placement was declared and fewer than three facts were verified, so this run "
             "would have been a CPU run reported as a GPU one. Refusals: " +
             ", ".join(v.get("refusals") or ["<none recorded>"]) +
             ". No downgrade is available: a GPU request that cannot be established fails"),
            {"refusals": list(v.get("refusals") or []), "establishes_a_gpu_pilot": False,
             "framework_devices": v.get("framework_devices"),
             "device_uuid_verified": v.get("device_uuid_verified"),
             "placement_probe": v.get("placement_probe")})
    v["facts_verified"] = list(THREE_FACTS)
    return v


def _telemetry():
    import importlib.util                                                    # noqa: PLC0415
    name = "df_tf_device_telemetry"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tf():
    import tensorflow as tf                                                  # noqa: PLC0415
    return tf


def pytorch_consulted() -> bool:
    """False, permanently, and asserted rather than promised.

    PyTorch's caching allocator does not account for one byte TensorFlow allocates.  Nothing in
    this path imports it, so nothing in this path can report its numbers for a TensorFlow model.
    """
    return False


def _do_not_dlopen(_soname: str) -> bool:
    """The CLI supervisor resolves wheel files only. Confirming a soname here would dlopen it."""
    return False


def _mapped(needles: tuple) -> bool:
    try:
        text = Path("/proc/self/maps").read_text(errors="replace")
    except OSError:
        return False
    return any(n in text for n in needles)


_CUDA_MAPPED = (
    "libcuda.so", "libcudart.so", "libcublas.so", "libcublasLt.so", "libcudnn.so",
    "libcufft.so", "libcurand.so", "libcusolver.so", "libcusparse.so", "libnvJitLink.so",
)


def _emit(payload: dict, code: int) -> int:
    import json                                                              # noqa: PLC0415
    print(json.dumps(payload, indent=1, default=str))
    return code


def _prepared_child_env(*, placement: str, device_uuid: str | None,
                        library_dir: str | None) -> tuple:
    """The mapping a new process is started with. This process's os.environ is not written."""
    prepared, decl = child_placement_env(
        {}, placement=placement, device_uuid=device_uuid, loader=_do_not_dlopen)
    child = dict(os.environ)
    child.update(prepared)
    if library_dir:
        extra = str(library_dir)
        prior = [p for p in str(child.get("LD_LIBRARY_PATH") or "").split(":") if p]
        child["LD_LIBRARY_PATH"] = ":".join([extra] + [p for p in prior if p != extra])
    return child, decl


def start_prepared_child(argv, env, *, timeout=None):
    """Exec argv with env as its startup environment.

    On this glibc, putting LD_LIBRARY_PATH into the current process does not change dlopen's
    search. The mapping has to be present when the new process starts. Nothing here imports
    TensorFlow or loads a library.
    """
    import subprocess                                                        # noqa: PLC0415
    if env is os.environ:
        raise PlacementRefusal(
            "ENV_IS_THE_SUPERVISOR",
            "the prepared mapping is this process's os.environ; the child needs its own")
    if "tensorflow" in sys.modules:
        raise PlacementRefusal(
            "SUPERVISOR_IMPORTED_TENSORFLOW",
            "the supervisor already imported TensorFlow, so it is not a supervisor")
    before = (os.environ.get("LD_LIBRARY_PATH"), os.environ.get("CUDA_VISIBLE_DEVICES"))
    kwargs = {"capture_output": True, "text": True}
    if timeout is not None:
        kwargs["timeout"] = timeout
    proc = subprocess.run(list(argv), env=dict(env), **kwargs)
    after = (os.environ.get("LD_LIBRARY_PATH"), os.environ.get("CUDA_VISIBLE_DEVICES"))
    if after != before:
        raise PlacementRefusal(
            "SUPERVISOR_ENVIRONMENT_MUTATED",
            "starting the child wrote the prepared environment back onto the supervisor")
    return proc


def _supervise_result(decl, proc, *, soname: str | None, cuda_before: bool) -> int:
    imported = "tensorflow" in sys.modules
    cuda_after = _mapped(_CUDA_MAPPED)
    dlopened = cuda_after and not cuda_before
    mapped = bool(soname) and _mapped((soname,))
    child_failed = proc.returncode != 0
    body = {
        "schema": SCHEMA,
        "status": MEASURED if not (child_failed or imported or dlopened or mapped) else REFUSED,
        "checked_in": "child",
        "supervisor_imported_tensorflow": imported,
        "supervisor_dlopened_cuda": dlopened,
        "supervisor_cuda_mapped_before": cuda_before,
        "supervisor_mapped_soname": mapped,
        "child_returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "declaration_placement": decl.get("placement"),
    }
    if child_failed or imported or dlopened or mapped:
        if child_failed:
            code, detail = ("CHILD_CHECK_REFUSED",
                            "the child failed; the supervisor does not report that as success")
        elif imported:
            code, detail = ("SUPERVISOR_IMPORTED_TENSORFLOW",
                            "the supervisor imported TensorFlow; the check belongs to the child")
        elif dlopened:
            code, detail = ("SUPERVISOR_DLOPENED_CUDA",
                            "a CUDA library was mapped in the supervisor after the child started")
        else:
            code, detail = ("SUPERVISOR_LOADED_THE_LIBRARY",
                            "the supervisor's address space contains the soname; that is not a child check")
        body["code"] = code
        body["detail"] = detail
        return _emit(body, 3)
    return _emit(body, 0)


def _launch(child_env, decl, argv, *, timeout, soname) -> int:
    import subprocess                                                        # noqa: PLC0415
    cuda_before = _mapped(_CUDA_MAPPED)
    try:
        proc = start_prepared_child(argv, child_env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        return _emit({"schema": SCHEMA, "status": REFUSED, "code": "CHILD_CHECK_REFUSED",
                      "detail": "the child exceeded the supervisor's wait; not a success",
                      "checked_in": "child", "stdout": out or "", "stderr": err or "",
                      "declaration_placement": decl.get("placement")}, 3)
    except PlacementRefusal as e:
        return _emit({"schema": SCHEMA, "status": REFUSED, "code": e.code, "detail": e.detail,
                      "evidence": e.evidence, "checked_in": "child"}, 3)
    return _supervise_result(decl, proc, soname=soname, cuda_before=cuda_before)


def _verify_in_this_process(cpu_seconds: int | None) -> int:
    """The child. The spend limit is installed here, before TensorFlow, and only on this process."""
    import resource                                                          # noqa: PLC0415
    if cpu_seconds is not None:
        limit = int(cpu_seconds)
        if limit <= 0:
            return _emit({"status": REFUSED, "code": "CPU_LIMIT_NOT_POSITIVE",
                          "detail": f"{cpu_seconds!r} is not a positive CPU spend limit",
                          "checked_in": "child"}, 2)
        resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))
    try:
        checked = enforce_before_tensorflow()
        verification = verify_or_refuse(checked)
    except PlacementRefusal as e:
        return _emit({"status": REFUSED, "code": e.code, "detail": e.detail,
                      "evidence": e.evidence, "checked_in": "child"}, 3)
    return _emit({"status": verification.get("status", MEASURED), "checked_in": "child",
                  "verification": verification}, 0)


def _load_in_this_process(soname: str, symbol: str) -> int:
    """Load by soname using the environment this process was started with. Do not assign one."""
    import ctypes                                                            # noqa: PLC0415
    try:
        fn = getattr(ctypes.CDLL(soname), symbol)
        fn.restype = ctypes.c_int
        print(int(fn()))
    except (OSError, AttributeError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    import argparse                                                          # noqa: PLC0415
    import json                                                              # noqa: PLC0415
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--placement", choices=["CPU", "GPU", "cpu", "gpu"], required=True)
    p.add_argument("--device-uuid", default=None)
    p.add_argument("--verify", action="store_true",
                   help="prepare the child environment and verify the three facts in a new "
                        "process. This process does not import TensorFlow and does not dlopen")
    p.add_argument("--cpu-seconds", type=int, default=None,
                   help="RLIMIT_CPU the verify child installs on itself before the check. "
                        "A spend limit for that process, not a measured footprint")
    p.add_argument("--load-dir", default=None,
                   help="directory placed on the child process's startup LD_LIBRARY_PATH. "
                        "This process does not load it")
    p.add_argument("--load-soname", default=None,
                   help="soname the child loads. A child failure is a refusal, not a success")
    p.add_argument("--load-symbol", default="probe")
    p.add_argument("--verify-child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--load-child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.verify_child and a.load_child:
        return _emit({"status": REFUSED, "code": "ONE_CHECK",
                      "detail": "a child does one check"}, 2)
    if a.verify_child:
        return _verify_in_this_process(a.cpu_seconds)
    if a.load_child:
        if not a.load_soname:
            return _emit({"status": REFUSED, "code": "NO_SONAME",
                          "detail": "the child was given no soname"}, 2)
        return _load_in_this_process(a.load_soname, a.load_symbol)
    if a.verify and (a.load_soname or a.load_dir):
        return _emit({"status": REFUSED, "code": "VERIFY_AND_LOAD",
                      "detail": "--verify and --load-soname are different checks; one child does one"}, 2)
    if a.load_dir and not a.load_soname:
        return _emit({"status": REFUSED, "code": "NO_SONAME",
                      "detail": "a library directory needs the soname the child loads"}, 2)
    if a.load_soname and not a.load_dir:
        return _emit({"status": REFUSED, "code": "NO_LIBRARY_DIR",
                      "detail": "the soname needs its directory on the child's startup environment"}, 2)
    try:
        _child_env, decl = _prepared_child_env(
            placement=a.placement, device_uuid=a.device_uuid,
            library_dir=a.load_dir if a.load_soname else None)
    except PlacementRefusal as e:
        return _emit({"status": REFUSED, "code": e.code, "detail": e.detail,
                      "evidence": e.evidence}, 2)
    if not a.verify and not a.load_soname:
        print(json.dumps({"declaration": decl}, indent=1, default=str))
        return 0
    script = str(Path(__file__).resolve())
    if a.load_soname:
        argv = [sys.executable, script, "--load-child", "--placement", a.placement,
                "--load-soname", a.load_soname, "--load-symbol", a.load_symbol]
        if a.device_uuid:
            argv += ["--device-uuid", a.device_uuid]
        return _launch(_child_env, decl, argv, timeout=30, soname=a.load_soname)
    argv = [sys.executable, script, "--verify-child", "--placement", a.placement]
    if a.device_uuid:
        argv += ["--device-uuid", a.device_uuid]
    if a.cpu_seconds is not None:
        argv += ["--cpu-seconds", str(a.cpu_seconds)]
    return _launch(_child_env, decl, argv, timeout=None, soname=None)


if __name__ == "__main__":
    sys.exit(main())
