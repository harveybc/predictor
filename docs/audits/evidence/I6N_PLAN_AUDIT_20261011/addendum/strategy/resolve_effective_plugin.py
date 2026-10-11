"""I6-N audit addendum, lane strategy (finding I6N-AUD-02, owner point 6b).

Which heuristic-strategy code does each Python environment ACTUALLY resolve?

For every interpreter given on the command line and for every import context, a
child process is started with ``python -I`` (isolated: no current directory, no
PYTHONPATH, no user site).  An import context optionally prepends one repository
root to sys.path, which is what an executor started from that root (``python -m``,
PYTHONPATH=root, or a tool that does sys.path.insert(0, ROOT)) would see.

The child prints, as one JSON object:
  * the distributions declaring the entry-point groups ``heuristic_strategy.plugins``
    and ``trade_lifecycle_policy.plugins`` (name, value, distribution, version);
  * the result of loading entry point ``ls_pred_strategy`` (the name predictor's
    app/config.py names) and ``prediction_entry_exit_v1``: module __file__, sha256,
    and the file the plugin's ``should_early_close`` was compiled from, or the error;
  * importlib's resolution of app.plugins.plugin_long_short_predictions,
    app.policies.prediction_entry_exit and plugins_strategy.heuristic_strategy.

Paths are printed with the home directory replaced by placeholders so the output
can live in a public repository: $CONDA_BASE (conda base prefix), $GH (the
directory holding the sibling checkouts), ~ (any other home path).

Usage:
  python resolve_effective_plugin.py --gh-root DIR --interp LABEL=PYTHON ... \
      --context LABEL=REPO_ROOT_OR_EMPTY ...
Read-only: imports modules, never writes files, never trains, never contacts a broker.
"""
import argparse
import json
import os
import subprocess
import sys

CHILD = r'''
import hashlib, importlib, importlib.util, json, sys
from importlib import metadata
prepend = sys.argv[1]
if prepend:
    sys.path.insert(0, prepend)
def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except Exception as e:
        return "ERR:" + type(e).__name__
out = {"python": sys.version.split()[0], "prefix": sys.prefix, "groups": {}, "loads": {}, "specs": {}}
for group in ("heuristic_strategy.plugins", "trade_lifecycle_policy.plugins"):
    eps = []
    for ep in metadata.entry_points(group=group):
        d = getattr(ep, "dist", None)
        eps.append({"name": ep.name, "value": ep.value,
                    "dist": None if d is None else d.metadata["Name"],
                    "version": None if d is None else d.version,
                    "dist_path": None if d is None else str(getattr(d, "_path", ""))})
    out["groups"][group] = eps
for group, name in (("heuristic_strategy.plugins", "ls_pred_strategy"),
                    ("trade_lifecycle_policy.plugins", "prediction_entry_exit_v1")):
    eps = [ep for ep in metadata.entry_points(group=group) if ep.name == name]
    if not eps:
        out["loads"][name] = {"status": "NO_ENTRY_POINT"}
        continue
    try:
        obj = eps[0].load()
        mod = sys.modules[obj.__module__]
        rec = {"status": "LOADED", "module": obj.__module__, "file": mod.__file__, "sha256": sha(mod.__file__)}
        sec = getattr(mod, "should_early_close", None)
        rec["should_early_close_from"] = None if sec is None else sec.__code__.co_filename
        app = sys.modules.get("app")
        rec["app_package_file"] = None if app is None else getattr(app, "__file__", None)
        out["loads"][name] = rec
    except Exception as e:
        out["loads"][name] = {"status": "IMPORT_FAILED", "error": type(e).__name__ + ": " + str(e)[:200]}
for modname in ("app.plugins.plugin_long_short_predictions", "app.policies.prediction_entry_exit",
                "plugins_strategy.heuristic_strategy"):
    for k in [k for k in sys.modules if k == "app" or k.startswith("app.") or k.startswith("plugins_strategy")]:
        del sys.modules[k]
    try:
        spec = importlib.util.find_spec(modname)
        out["specs"][modname] = None if spec is None else {"origin": spec.origin, "sha256": sha(spec.origin)}
    except Exception as e:
        out["specs"][modname] = {"error": type(e).__name__ + ": " + str(e)[:200]}
print(json.dumps(out, sort_keys=True))
'''


def redact(text, home, conda_base, gh_root):
    for real, tag in ((gh_root, "$GH"), (conda_base, "$CONDA_BASE"), (home, "~")):
        if real:
            text = text.replace(real.rstrip("/"), tag)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gh-root", required=True)
    ap.add_argument("--conda-base", required=True)
    ap.add_argument("--interp", action="append", required=True, help="LABEL=PYTHON")
    ap.add_argument("--context", action="append", required=True, help="LABEL=ROOT (ROOT may be empty)")
    a = ap.parse_args()
    home = os.path.expanduser("~")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    for item in a.interp:
        ilabel, py = item.split("=", 1)
        for ctx in a.context:
            clabel, root = ctx.split("=", 1)
            # cwd is a neutral directory so nothing is picked up from the caller's cwd.
            r = subprocess.run([py, "-I", "-c", CHILD, root], capture_output=True, text=True,
                               env=env, cwd="/", timeout=300)
            payload = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            try:
                body = json.loads(payload)
            except Exception:
                body = {"child_failed": r.returncode, "stderr_tail": r.stderr[-400:]}
            rec = {"interpreter": ilabel, "context": clabel, "result": body}
            print(redact(json.dumps(rec, sort_keys=True), home, a.conda_base, a.gh_root))


if __name__ == "__main__":
    main()
