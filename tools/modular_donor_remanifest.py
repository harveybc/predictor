"""Re-manifest donors saved by the pre-c1e035d7 engine under the effective-params identity.

    python tools/modular_donor_remanifest.py DONOR.keras [DONOR.keras ...]
        [--feature-names a,b,c] [--window 24] [--sample-hours 1] [--dry-run]

Donors written by ``predictor_plugins.modular_temporal`` at 556c5f3e hash their
manifest over LITERAL params with the old built-in identity
(``modular_temporal.v1:<name>``). Since c1e035d7 identity is over EFFECTIVE params
(declared defaults resolved) and built-ins carry a version, so those donors are
refused by ``load_donor``. This tool recomputes the manifest WITHOUT touching the
weights file:

* the archive's sha256 must equal the old sidecar's ``model_sha256`` before and
  after, and the deserialized weights must hash to its ``weights_sha256``;
* feature order, window and sampling period must be re-derivable from the old
  manifest (grids recomputed from ``sample_hours``; output grids must be the exact
  right-edge partition; a core's input grid must equal its upstream branches'
  output grid) and must match any ``--feature-names/--window/--sample-hours`` given;
* only built-in components can be re-derived; any other plugin is refused;
* the old sidecar is kept as ``<stem>.manifest.pre_c1e035d7.json`` (never
  overwritten once written), the new one replaces ``<stem>.manifest.json``
  atomically, and a record with both identity hashes is appended to
  ``<stem>.provenance.json`` (other keys in that file are preserved);
* the result is verified with ``load_donor`` against the new manifest;
* running it again on a current donor changes nothing (``ALREADY_CURRENT``).

Exit status is non-zero if any donor is refused; nothing is written for a refused donor.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import shutil
import sys
from pathlib import Path

OLD_PREFIX = "modular_temporal.v1:"
BACKUP_SUFFIX = ".manifest.pre_c1e035d7.json"
TOOL = "tools/modular_donor_remanifest.py"


class RemanifestRefusal(ValueError):
    pass


def _refuse(message):
    raise RemanifestRefusal(message)


def _paths(donor):
    donor = Path(donor)
    if donor.suffix != ".keras":
        _refuse(f"{donor.name}: donor path must end in .keras")
    stem = donor.with_suffix("")
    return (donor, donor.with_suffix(".manifest.json"),
            Path(str(stem) + BACKUP_SUFFIX), donor.with_suffix(".provenance.json"))


def _new_identity(mt, role, old_identity, *, current_ok):
    if not isinstance(old_identity, dict):
        _refuse("plugin identity is not a mapping")
    group, name = old_identity.get("group"), old_identity.get("name")
    implementation = str(old_identity.get("implementation", ""))
    default_group = "modular." + role
    builtin = mt.BUILTINS.get(default_group, {}).get(name)
    old_style = implementation == OLD_PREFIX + str(name)
    current = current_ok and implementation == "predictor_plugins.modular_temporal:" + str(name)
    if group != default_group or builtin is None or not (old_style or current):
        _refuse(f"{role} plugin {group}:{name} ({implementation}) is not a re-derivable built-in")
    factory, identity = mt._resolve(role, {"plugin": name}, {role: default_group})
    if factory is not builtin:
        _refuse(f"{role} plugin {name} does not resolve to the built-in")
    return factory, identity


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        _refuse(f"{label} is not a positive finite number")
    return value


def _check_grids(manifest, input_grid_expected):
    input_grid, output_grid = manifest.get("input_grid"), manifest.get("output_grid")
    if input_grid != input_grid_expected:
        _refuse(f"{manifest.get('name')}: input grid cannot be re-derived from window/sample_hours")
    steps = len(output_grid or [])
    if not steps or len(input_grid) % steps:
        _refuse(f"{manifest.get('name')}: output grid is not an exact partition")
    block = len(input_grid) // steps
    if output_grid != input_grid[block - 1::block]:
        _refuse(f"{manifest.get('name')}: output grid is not the right-edge partition")
    shape_in, shape_out = manifest.get("input_shape"), manifest.get("output_shape")
    if (not isinstance(shape_in, list) or not isinstance(shape_out, list) or len(shape_in) != 2
            or len(shape_out) != 2 or shape_in[0] != len(input_grid) or shape_out[0] != steps):
        _refuse(f"{manifest.get('name')}: shapes disagree with the grids")


def _features(manifest, expect_names):
    names, features = manifest.get("feature_names"), manifest.get("features")
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not all(isinstance(n, str) and n for n in names)):
        _refuse("feature_names cannot be re-derived (empty, duplicated or non-string)")
    if (not isinstance(features, list) or not features or len(set(features)) != len(features)
            or any(f not in names for f in features)):
        _refuse("component features are not unique members of feature_names")
    if expect_names is not None and names != list(expect_names):
        _refuse(f"feature order {names} differs from the expected {list(expect_names)}")
    return names


def _branch(mt, old, expect, *, current_ok=False):
    keys = {"schema", "role", "plugin", "params", "features", "feature_names", "name", "sample_hours",
            "input_shape", "output_shape", "input_grid", "output_grid"}
    if not isinstance(old, dict) or set(old) != keys or old["schema"] != 1 or old["role"] != "branch":
        _refuse("branch manifest does not have the schema-1 branch layout")
    _features(old, expect.get("feature_names"))
    hours = _number(old["sample_hours"], "sample_hours")
    window = old["input_shape"][0] if isinstance(old.get("input_shape"), list) and old["input_shape"] else None
    if not isinstance(window, int) or window < 1:
        _refuse("window cannot be re-derived from input_shape")
    if expect.get("window") is not None and window != expect["window"]:
        _refuse(f"window {window} differs from the expected {expect['window']}")
    if expect.get("sample_hours") is not None and hours != expect["sample_hours"]:
        _refuse(f"sample_hours {hours} differs from the expected {expect['sample_hours']}")
    if old["input_shape"][1] != len(old["features"]):
        _refuse("branch input channels differ from its feature count")
    _check_grids(old, [(i + 1) * hours for i in range(window)])
    factory, identity = _new_identity(mt, "branch", old["plugin"], current_ok=current_ok)
    new = dict(old)
    new["plugin"] = identity
    new["params"] = mt.effective_params(factory, old["params"], {"output_steps": len(old["output_grid"])})
    return mt._copy(new)


def _core(mt, old, expect, *, current_ok=False):
    keys = {"schema", "role", "plugin", "params", "features", "feature_names", "name", "sample_hours",
            "input_shape", "output_shape", "input_grid", "output_grid", "upstream"}
    if not isinstance(old, dict) or set(old) != keys or old["schema"] != 1 or old["role"] != "core":
        _refuse("core manifest does not have the schema-1 core layout")
    names = _features(old, expect.get("feature_names"))
    if old["features"] != names:
        _refuse("core features must be the full ordered feature list")
    up = old["upstream"]
    if not isinstance(up, dict) or set(up) != {"branches", "fusion"} or not up["branches"]:
        _refuse("core upstream identity cannot be re-derived")
    branches = []
    for entry in up["branches"]:
        if not isinstance(entry, dict) or set(entry) != {"manifest", "weights_sha256"}:
            _refuse("upstream branch entry layout")
        manifest = _branch(mt, entry["manifest"], expect, current_ok=current_ok)
        if manifest["feature_names"] != names or manifest["sample_hours"] != old["sample_hours"]:
            _refuse("upstream branch feature order or sampling differs from the core's")
        branches.append({"manifest": manifest, "weights_sha256": entry["weights_sha256"]})
    grid = branches[0]["manifest"]["output_grid"]
    if any(b["manifest"]["output_grid"] != grid for b in branches) or old["input_grid"] != grid:
        _refuse("core input grid differs from its upstream branches' common output grid")
    if old["input_shape"][1] != sum(b["manifest"]["output_shape"][1] for b in branches):
        _refuse("core input channels differ from the fused branch widths")
    _check_grids(old, grid)
    fusion = up["fusion"]
    if not isinstance(fusion, dict) or set(fusion) != {"identity", "weights_sha256"}:
        _refuse("fusion upstream layout")
    old_fusion = dict(fusion["identity"])
    fusion_params = old_fusion.pop("params", None)
    if not isinstance(fusion_params, dict):
        _refuse("fusion params missing")
    old_fusion.pop("version", None) if current_ok else None
    f_factory, f_identity = _new_identity(mt, "fusion", old_fusion, current_ok=current_ok)
    f_identity["params"] = mt.effective_params(f_factory, fusion_params, {})
    factory, identity = _new_identity(mt, "core", old["plugin"], current_ok=current_ok)
    new = dict(old)
    new["plugin"] = identity
    new["params"] = mt.effective_params(factory, old["params"], {
        "input_steps": len(old["input_grid"]), "output_steps": len(old["output_grid"]),
        "output_channels": old["output_shape"][1]})
    new["upstream"] = {"branches": branches,
                       "fusion": {"identity": f_identity, "weights_sha256": fusion["weights_sha256"]}}
    return mt._copy(new)


def remanifest(donor, *, feature_names=None, window=None, sample_hours=None, dry_run=False):
    """Return a result dict; raise RemanifestRefusal before writing anything."""
    from predictor_plugins import modular_temporal as mt
    path, sidecar, backup, provenance = _paths(donor)
    if not path.is_file() or not sidecar.is_file():
        _refuse(f"{path.name}: donor archive or manifest sidecar missing")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    required = {"schema", "manifest", "manifest_sha256", "model_sha256", "weights_sha256"}
    if (not isinstance(document, dict) or not required <= set(document) <= required | {"provenance"}
            or document["schema"] != 1):
        _refuse(f"{path.name}: sidecar layout is not a schema-1 donor document")
    if mt._digest(document["manifest"]) != document["manifest_sha256"]:
        _refuse(f"{path.name}: old manifest does not match its recorded hash")
    before = mt._file_hash(path)
    if before != document["model_sha256"]:
        _refuse(f"{path.name}: archive bytes differ from the recorded model_sha256 (tampered or replaced)")
    model = mt.keras.models.load_model(path, compile=False, safe_mode=True)
    if mt.weights_hash(model) != document["weights_sha256"]:
        _refuse(f"{path.name}: weights do not hash to the recorded weights_sha256")
    expect = {"feature_names": feature_names, "window": window, "sample_hours": sample_hours}
    old = document["manifest"]
    current = "provenance" in document and isinstance(old.get("plugin"), dict) and "version" in old["plugin"]
    role = old.get("role") if isinstance(old, dict) else None
    if role == "branch":
        new = _branch(mt, old, expect, current_ok=current)
    elif role == "core":
        new = _core(mt, old, expect, current_ok=current)
    else:
        _refuse(f"{path.name}: unknown donor role {role!r}")
    mt._check_manifest_model(new, model)
    new_sha = mt._digest(new)
    result = {"donor": path.name, "role": role, "old_manifest_sha256": document["manifest_sha256"],
              "new_manifest_sha256": new_sha, "weights_sha256": document["weights_sha256"],
              "model_sha256": before}
    if new_sha == document["manifest_sha256"] and current:
        mt.load_donor(path, new)
        return {**result, "status": "ALREADY_CURRENT"}
    if dry_run:
        return {**result, "status": "WOULD_REMANIFEST"}
    keras_version = mt.keras_version()
    updated = {"schema": 1, "manifest": new, "manifest_sha256": new_sha,
               "model_sha256": document["model_sha256"], "weights_sha256": document["weights_sha256"],
               "provenance": {"keras_version": keras_version,
                              "keras_version_source": "remanifest_environment",
                              "declared_params": mt._copy(old["params"]),
                              "remanifested_from_manifest_sha256": document["manifest_sha256"]}}
    if not backup.exists():
        shutil.copy2(sidecar, backup)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(json.dumps(updated, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, sidecar)
    record = {**result, "tool": TOOL, "from_identity": "engine 556c5f3e (literal params)",
              "to_identity": "engine c1e035d7+ (effective params)", "keras_version": keras_version,
              "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    log = json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else {}
    if not isinstance(log, dict):
        _refuse(f"{provenance.name}: provenance sidecar is not a JSON object")  # nothing else written yet
    log.setdefault("remanifest_records", []).append(record)
    tmp = provenance.with_name(provenance.name + ".tmp")
    tmp.write_text(json.dumps(log, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, provenance)
    if mt._file_hash(path) != before:
        _refuse(f"{path.name}: archive changed during remanifest")
    mt.load_donor(path, new)
    return {**result, "status": "REMANIFESTED"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("donors", nargs="+")
    p.add_argument("--feature-names", type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--window", type=int)
    p.add_argument("--sample-hours", type=float)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    failed = 0
    for donor in a.donors:
        try:
            out = remanifest(donor, feature_names=a.feature_names, window=a.window,
                             sample_hours=a.sample_hours, dry_run=a.dry_run)
        except (RemanifestRefusal, ValueError, OSError) as exc:
            failed += 1
            out = {"donor": str(donor), "status": "REFUSED", "reason": str(exc)}
        print(json.dumps(out, sort_keys=True), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
