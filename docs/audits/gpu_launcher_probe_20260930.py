#!/usr/bin/env python3
"""Loader handoff probe. Innocent C libraries only. No TensorFlow and no CUDA.

PRE reproduces the glibc fact: assigning LD_LIBRARY_PATH inside a process does
not make dlopen search that directory; a new process started with the directory
in its environment does. POST drives the public entry and the integrated
subprocess handoff. Neither phase identifies a missing GPU library.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "tools" / "df_placement_contract.py"
SONAME = "libretsu_loader_probe.so"
HELPER = "libretsu_helper.so"
NEED = "libretsu_need.so"


def scrub(text: str) -> str:
    root = str(ROOT)
    home = str(Path.home())
    if root:
        text = text.replace(root, "this worktree")
    if home and home != "/":
        text = text.replace(home, "<HOME>")
    user = os.environ.get("USER") or ""
    if len(user) >= 3:
        text = text.replace(user, "<ACCOUNT>")
    node = os.uname().nodename
    if len(node) >= 3:
        text = text.replace(node, "<HOST>")
    return text


def scrub_obj(value):
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {scrub_obj(k) if isinstance(k, str) else k: scrub_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_obj(v) for v in value]
    return value


def glibc_version() -> str | None:
    try:
        return os.confstr("CS_GNU_LIBC_VERSION")
    except (OSError, ValueError):
        return None


def cuda_mapped() -> bool:
    try:
        maps = Path("/proc/self/maps").read_text(errors="replace")
    except OSError:
        return False
    needles = ("libcuda.so", "libcudart.so", "libcublas.so", "libcudnn.so")
    return any(n in maps for n in needles)


def run_captured(argv, *, env, timeout=20, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def compile_c(source: str, output: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_RUN_PATH", None)
    cmd = ["cc", "-shared", "-fPIC", "-o", str(output), "-x", "c", "-", *(extra or [])]
    return subprocess.run(cmd, input=source, text=True, capture_output=True, timeout=20, env=env)


def retained(proc: subprocess.CompletedProcess) -> dict:
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def dynamic_tags(path: Path) -> dict:
    proc = subprocess.run(
        ["readelf", "-d", str(path)], capture_output=True, text=True, timeout=20)
    needed, rpath, runpath = [], [], []
    for line in (proc.stdout or "").splitlines():
        if "(NEEDED)" in line and "[" in line:
            needed.append(line.split("[", 1)[1].split("]", 1)[0])
        if "(RPATH)" in line and "[" in line:
            rpath.append(line.split("[", 1)[1].split("]", 1)[0])
        if "(RUNPATH)" in line and "[" in line:
            runpath.append(line.split("[", 1)[1].split("]", 1)[0])
    return {
        "returncode": proc.returncode,
        "needed": needed,
        "rpath": rpath,
        "runpath": runpath,
        "stderr": proc.stderr,
    }


def blank_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.pop("LD_LIBRARY_PATH", None)
    return env


def phase_pre() -> dict:
    """Same shape as the review probe: assign inside the child, or inherit at start."""
    assign_then_load = (
        "import ctypes,os,sys\n"
        "os.environ['LD_LIBRARY_PATH']=sys.argv[1]\n"
        "lib=ctypes.CDLL(sys.argv[2])\n"
        "print(lib.probe())\n"
    )
    with tempfile.TemporaryDirectory(prefix="retsu-loader-pre-") as directory:
        library = Path(directory) / SONAME
        compiled = compile_c("int probe(void) { return 42; }\n", library)
        env = blank_env()
        in_process = run_captured(
            [sys.executable, "-c", assign_then_load, directory, SONAME], env=env)
        env = blank_env()
        env["LD_LIBRARY_PATH"] = directory
        fresh = run_captured(
            [sys.executable, "-c", assign_then_load, directory, SONAME], env=env)
        ok = (
            compiled.returncode == 0
            and in_process.returncode != 0
            and fresh.returncode == 0
            and fresh.stdout.strip() == "42"
            and "tensorflow" not in sys.modules
            and not cuda_mapped()
        )
        return {
            "phase": "PRE",
            "ok": ok,
            "scope": (
                "glibc loader mechanism with an innocent C library. "
                "No CUDA was loaded. This does not identify the worker's missing "
                "library and does not prove a GPU repair."
            ),
            "glibc": glibc_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow_imported": "tensorflow" in sys.modules,
            "cuda_mapped_in_this_process": cuda_mapped(),
            "compile": retained(compiled),
            "in_process_assignment": retained(in_process),
            "fresh_child_startup_env": retained(fresh),
            "fresh_result": fresh.stdout.strip() if fresh.returncode == 0 else None,
        }


def public_entry(directory: str, soname: str) -> subprocess.CompletedProcess:
    argv = [
        sys.executable, str(ENTRY),
        "--placement", "CPU",
        "--load-dir", directory,
        "--load-soname", soname,
        "--load-symbol", "probe",
    ]
    if "--verify" in argv:
        raise RuntimeError("the loader probe must not request the TensorFlow check")
    return run_captured(argv, env=blank_env(), timeout=30)


def parse_entry(proc: subprocess.CompletedProcess) -> dict:
    out = retained(proc)
    try:
        out["parsed"] = json.loads(proc.stdout) if proc.stdout.strip() else None
    except json.JSONDecodeError as exc:
        out["parsed"] = None
        out["parse_error"] = str(exc)
    return out


def load_no_assign_source() -> str:
    return (
        "import ctypes,sys\n"
        "lib=ctypes.CDLL(sys.argv[1])\n"
        "fn=lib.probe\n"
        "fn.restype=ctypes.c_int\n"
        "print(int(fn()))\n"
    )


def load_module(name: str, path: Path):
    """Load a file. Dataclasses look the module up in sys.modules while the body runs."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def integrated_governed(directory: str) -> dict:
    """The call at governed_run.py: subprocess.run(cmd, cwd=REPO_ROOT, env=env)."""
    mod = load_module("governed_run_handoff", ROOT / "tools" / "governed_run.py")
    before = os.environ.get("LD_LIBRARY_PATH")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(mod.REPO_ROOT))
    env["LD_LIBRARY_PATH"] = directory
    proc = subprocess.run(
        [sys.executable, "-c", load_no_assign_source(), SONAME],
        cwd=mod.REPO_ROOT, env=env, capture_output=True, text=True, timeout=20)
    after = os.environ.get("LD_LIBRARY_PATH")
    source = (ROOT / "tools" / "governed_run.py").read_text(encoding="utf-8")
    return {
        "source_passes_env": "proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)" in source,
        "parent_ld_library_path_unchanged": after == before,
        "child": retained(proc),
        "result": proc.stdout.strip() if proc.returncode == 0 else None,
    }


def integrated_supervise(directory: str) -> dict:
    """df_e1_block hands env to df_cell_scope.supervise, which Popen's with env=."""
    mod = load_module("df_cell_scope_handoff", ROOT / "tools" / "df_cell_scope.py")
    mod._find_lease = lambda watcher, proc, deadline: (
        None, "this probe does not read the admission store")
    block = (ROOT / "tools" / "df_e1_block.py").read_text(encoding="utf-8")
    body = block.split("def run_units(", 1)[1].split("\ndef ", 1)[0]
    scope = (ROOT / "tools" / "df_cell_scope.py").read_text(encoding="utf-8")
    before = os.environ.get("LD_LIBRARY_PATH")
    slice_before = os.environ.get("CRISPDM_CELL_SCOPE_SLICE_CGROUP")
    os.environ["CRISPDM_CELL_SCOPE_SLICE_CGROUP"] = "probe-not-a-slice"
    try:
        with tempfile.TemporaryDirectory(prefix="retsu-supervise-") as tmp:
            launcher = Path(tmp) / "fake-launch"
            launcher.write_text(
                "#!/bin/sh\n"
                "while [ $# -gt 0 ]; do\n"
                "  if [ \"$1\" = \"--\" ]; then shift; break; fi\n"
                "  shift\n"
                "done\n"
                "exec \"$@\"\n",
                encoding="utf-8")
            launcher.chmod(0o755)
            child = Path(tmp) / "child.py"
            child.write_text(load_no_assign_source(), encoding="utf-8")
            log = Path(tmp) / "child.log"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["LD_LIBRARY_PATH"] = directory
            sup = mod.supervise(
                cell_id="gpu-launcher-handoff",
                argv=[sys.executable, str(child), SONAME],
                cap_bytes=64 * 1024 * 1024,
                wall_seconds=20,
                supervisor_dir=Path(tmp) / "sup",
                log_path=log,
                launcher=str(launcher),
                env=env)
            printed = log.read_text(encoding="utf-8") if log.is_file() else ""
            term = sup.get("termination") or {}
            return {
                "run_units_passes_env": "env=child_env" in body and "child_placement_env" in body,
                "supervise_popen_passes_env": "env=child_env" in scope and "subprocess.Popen" in scope,
                "replay_passes_env": "env=replay_env" in block,
                "parent_ld_library_path_unchanged": os.environ.get("LD_LIBRARY_PATH") == before,
                "termination_status": term.get("status"),
                "exit_code": term.get("exit_code"),
                "log": printed,
                "result": printed.strip() if term.get("exit_code") == 0 else None,
            }
    finally:
        if slice_before is None:
            os.environ.pop("CRISPDM_CELL_SCOPE_SLICE_CGROUP", None)
        else:
            os.environ["CRISPDM_CELL_SCOPE_SLICE_CGROUP"] = slice_before


def phase_post() -> dict:
    assign_then_load = (
        "import ctypes,os,sys\n"
        "os.environ['LD_LIBRARY_PATH']=sys.argv[1]\n"
        "lib=ctypes.CDLL(sys.argv[2])\n"
        "print(lib.probe())\n"
    )
    with tempfile.TemporaryDirectory(prefix="retsu-loader-post-") as directory:
        alone = Path(directory) / "alone"
        both = Path(directory) / "both"
        alone.mkdir()
        both.mkdir()
        direct = both / SONAME
        compiled = compile_c("int probe(void) { return 42; }\n", direct)
        helper_src = "int helper(void) { return 42; }\n"
        need_src = "int helper(void); int probe(void) { return helper(); }\n"
        helper = both / HELPER
        need = both / NEED
        compiled_helper = compile_c(
            helper_src, helper, ["-Wl,-soname,libretsu_helper.so"])
        compiled_need = compile_c(
            need_src, need,
            ["-L", str(both), "-Wl,--no-as-needed", "-lretsu_helper",
             "-Wl,-soname,libretsu_need.so"])
        tags = dynamic_tags(need)
        shutil.copy2(need, alone / NEED)
        in_process = run_captured(
            [sys.executable, "-c", assign_then_load, str(both), SONAME], env=blank_env())
        opened = public_entry(str(both), SONAME)
        missing = public_entry(str(alone), NEED)
        present = public_entry(str(both), NEED)
        declaration = run_captured(
            [sys.executable, str(ENTRY), "--placement", "CPU"], env=blank_env())
        governed = integrated_governed(str(both))
        supervised = integrated_supervise(str(both))
        opened_doc = parse_entry(opened)
        missing_doc = parse_entry(missing)
        present_doc = parse_entry(present)
        try:
            declared = json.loads(declaration.stdout)
        except json.JSONDecodeError:
            declared = None
        opened_ok = (
            opened.returncode == 0
            and (opened_doc.get("parsed") or {}).get("status") == "MEASURED"
            and (opened_doc.get("parsed") or {}).get("stdout", "").strip() == "42"
            and (opened_doc.get("parsed") or {}).get("supervisor_imported_tensorflow") is False
            and (opened_doc.get("parsed") or {}).get("supervisor_dlopened_cuda") is False
            and (opened_doc.get("parsed") or {}).get("supervisor_mapped_soname") is False
            and (opened_doc.get("parsed") or {}).get("checked_in") == "child"
        )
        missing_ok = (
            missing.returncode != 0
            and (missing_doc.get("parsed") or {}).get("status") == "REFUSED"
            and (missing_doc.get("parsed") or {}).get("stdout", "").strip() != "42"
        )
        present_ok = (
            present.returncode == 0
            and (present_doc.get("parsed") or {}).get("stdout", "").strip() == "42"
        )
        governed_ok = (
            governed["source_passes_env"]
            and governed["parent_ld_library_path_unchanged"]
            and governed["child"]["returncode"] == 0
            and governed["result"] == "42"
        )
        supervise_ok = (
            supervised["run_units_passes_env"]
            and supervised["supervise_popen_passes_env"]
            and supervised["replay_passes_env"]
            and supervised["parent_ld_library_path_unchanged"]
            and supervised["exit_code"] == 0
            and supervised["result"] == "42"
        )
        declaration_ok = (
            declaration.returncode == 0
            and isinstance(declared, dict)
            and "verification" not in declared
            and (declared.get("declaration") or {}).get("placement") == "CPU"
        )
        ok = (
            compiled.returncode == 0
            and compiled_helper.returncode == 0
            and compiled_need.returncode == 0
            and HELPER in tags["needed"]
            and not tags["rpath"]
            and not tags["runpath"]
            and in_process.returncode != 0
            and in_process.stdout.strip() != "42"
            and opened_ok
            and missing_ok
            and present_ok
            and governed_ok
            and supervise_ok
            and declaration_ok
            and "tensorflow" not in sys.modules
            and not cuda_mapped()
        )
        return {
            "phase": "POST",
            "ok": ok,
            "scope": (
                "The public entry starts a child. An in-process assignment still "
                "does not load the library. No CUDA was loaded. This does not "
                "identify the worker's missing library and does not prove a GPU repair."
            ),
            "glibc": glibc_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow_imported": "tensorflow" in sys.modules,
            "cuda_mapped_in_this_process": cuda_mapped(),
            "compile_direct": retained(compiled),
            "compile_helper": retained(compiled_helper),
            "compile_need": retained(compiled_need),
            "need_dynamic_tags": tags,
            "in_process_assignment": retained(in_process),
            "public_entry": opened_doc,
            "integrated_governed_run": governed,
            "integrated_supervise": supervised,
            "declaration_without_verify": retained(declaration),
            "transitive_dependency_missing": missing_doc,
            "transitive_dependency_present": present_doc,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wrapper-name", required=True)
    args = parser.parse_args(argv)
    seen = os.environ.get("CUDA_VISIBLE_DEVICES")
    if seen != "":
        print("CUDA_VISIBLE_DEVICES is not empty", file=sys.stderr)
        return 2
    try:
        payload = phase_pre() if args.phase == "pre" else phase_post()
    except Exception as exc:
        payload = {
            "phase": args.phase.upper(),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    payload["wrapper"] = {
        "name": args.wrapper_name,
        "memory": "2G",
        "wall": "120s",
        "cuda_visible_devices": "",
    }
    payload = scrub_obj(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    args.out.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
