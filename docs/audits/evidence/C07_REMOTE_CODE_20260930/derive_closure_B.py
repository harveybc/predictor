# Derive closure B (default PyPI, CUDA torch) from index metadata: identical to A except the torch wheel
# and torch's own exact-pinned platform dependencies (nvidia-*, triton). No wheel is downloaded.
import json, re, urllib.request
from packaging.requirements import Requirement
from packaging.markers import Marker
A = json.load(open("report_A_cpu.json"))
env = {"python_version": "3.12", "python_full_version": "3.12.13", "platform_machine": "x86_64", "platform_system": "Linux", "sys_platform": "linux", "os_name": "posix", "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""}
def pypi(name, ver):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json", timeout=30) as r: return json.load(r)
def wheel_for(name, ver, tag_pref=("cp312-cp312-manylinux_2_28_x86_64", "py3-none-manylinux_2_28_x86_64", "py3-none-manylinux2014_x86_64", "py3-none-manylinux1_x86_64", "py3-none-any", "cp312-cp312-manylinux_2_17_x86_64")):
    d = pypi(name, ver); files = d["urls"]
    for t in tag_pref:
        for u in files:
            if u["packagetype"] == "bdist_wheel" and u["filename"].endswith(t + ".whl"):
                return u, d["info"].get("requires_dist") or []
    cands = [u for u in files if u["packagetype"] == "bdist_wheel" and "x86_64" in u["filename"] and ("cp312" in u["filename"] or "py3" in u["filename"])]
    if cands: return cands[0], d["info"].get("requires_dist") or []
    raise SystemExit(f"no wheel for {name}=={ver}: {[u['filename'] for u in files]}")
base = {it["metadata"]["name"].lower().replace("_","-"): it for it in A["install"]}
assert base["torch"]["metadata"]["version"].startswith("2.8.0")
# start from A minus torch
out = {}
for n, it in base.items():
    if n == "torch": continue
    out[n] = {"name": it["metadata"]["name"], "version": it["metadata"]["version"], "source": "same as A"}
todo = [("torch", "2.8.0")]
seen = set()
while todo:
    name, ver = todo.pop()
    if name in seen: continue
    seen.add(name)
    u, reqs = wheel_for(name, ver); print("REQS", name, ver, [r for r in reqs if "extra" not in r])
    out[name] = {"name": name, "version": ver, "filename": u["filename"], "url": u["url"], "download_bytes": u["size"], "sha256": u["digests"]["sha256"], "yanked": u.get("yanked", False), "source": "derived from PyPI JSON metadata"}
    for r in reqs:
        req = Requirement(r)
        if req.marker and not req.marker.evaluate(env): continue
        dn = req.name.lower().replace("_", "-")
        if dn in out and dn != name:
            # verify A's version satisfies torch's specifier
            v = out[dn]["version"]
            if req.specifier and not req.specifier.contains(v, prereleases=True):
                out[dn]["CONFLICT_WITH"] = f"{name}=={ver} requires {r}"
            continue
        pins = [s for s in req.specifier if s.operator == "=="]
        if pins:
            todo.append((dn, pins[0].version))
        else:
            out[dn] = {"name": dn, "version": "UNPINNED_BY_TORCH_METADATA:" + str(req.specifier), "source": "needs resolution"}
# fill sizes for 'same as A' from closure_A sizes
Asz = {w["name"].lower().replace("_","-"): w for w in json.load(open("closure_A_cpu_sizes.json"))["wheels"]}
for n, d in out.items():
    if d["source"] == "same as A":
        w = Asz[n]; d.update({"filename": w["filename"], "url": w["url"], "download_bytes": w["download_bytes"], "sha256": w["sha256"], "yanked": w["yanked"]})
tot = sum(d.get("download_bytes") or 0 for d in out.values())
unres = [d for d in out.values() if "download_bytes" not in d]
res = {"method": "closure A (pip-resolved for CPython 3.12.13) with torch==2.8.0+cpu replaced by the PyPI torch==2.8.0 manylinux cp312 wheel and the exact-pinned platform dependencies declared in torch 2.8.0's Requires-Dist (markers evaluated for linux/x86_64/cp312); NOT a pip resolution, so cross-checked for specifier conflicts against A's versions",
       "python": "3.12.13", "n": len(out), "download_bytes_total": tot, "unresolved": unres, "conflicts": [d for d in out.values() if "CONFLICT_WITH" in d], "wheels": sorted(out.values(), key=lambda d: d["name"].lower())}
json.dump(res, open("closure_B_pypi_derived.json", "w"), indent=1)
print("n", len(out), "download_bytes_total", tot, "unresolved", len(unres), "conflicts", len(res["conflicts"]))
for d in sorted(out.values(), key=lambda d: -(d.get("download_bytes") or 0))[:16]: print(f'{d.get("download_bytes",0):>12} {d["name"]}=={d["version"]} {d["source"][:12]}')
