import json, sys, urllib.request, urllib.parse
rep = json.load(open(sys.argv[1]))
out = []
tot = 0
cache = {}
for it in rep["install"]:
    md = it["metadata"]; name, ver = md["name"], md["version"]
    url = it["download_info"]["url"]
    sha = it["download_info"].get("archive_info", {}).get("hashes", {}).get("sha256")
    fn = url.split("/")[-1].split("#")[0]
    size = None
    if "pythonhosted.org" in url or "pypi.org" in url:
        key = (name.lower(), ver)
        if key not in cache:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/{ver}/json", timeout=30) as r:
                cache[key] = json.load(r)
        for u in cache[key]["urls"]:
            if u["filename"] == fn:
                size = u["size"]
                if sha is None: sha = u["digests"]["sha256"]
                yanked = u.get("yanked", False)
                break
    else:
        req = urllib.request.Request(url.split("#")[0], headers={"Range": "bytes=0-0", "User-Agent": "pip/25.0.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cr = r.headers.get("Content-Range")
            size = int(cr.split("/")[-1]) if cr else int(r.headers["Content-Length"])
        yanked = None
        if sha is None and "#sha256=" in url: sha = url.split("#sha256=")[1]
    tot += size or 0
    out.append({"name": name, "version": ver, "filename": fn, "url": url.split("#")[0], "download_bytes": size, "sha256": sha, "yanked": yanked,
                "requires_python": md.get("requires_python")})
out.sort(key=lambda x: x["name"].lower())
json.dump({"python": rep["environment"]["python_full_version"], "platform": rep["environment"]["platform_machine"], "pip_version": rep.get("pip_version"), "n": len(out), "download_bytes_total": tot, "wheels": out}, open(sys.argv[2], "w"), indent=1)
print("n", len(out), "download_bytes_total", tot, "missing_size", sum(1 for x in out if x["download_bytes"] is None))
for x in sorted(out, key=lambda x: -(x["download_bytes"] or 0))[:12]: print(f'{x["download_bytes"]:>12} {x["name"]}=={x["version"]} yanked={x["yanked"]}')
