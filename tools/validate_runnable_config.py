#!/usr/bin/env python3
"""C54 (order 2026-09-12): is this configuration actually runnable?

The previous derivation called a configuration "executable" because the
`x`/`y` files it names happen to exist. That is a census of candidates,
not a statement about execution: it builds no effective configuration,
applies no defaults and no plugin parameters, resolves no entry point
and validates no target.

This builds the EFFECTIVE configuration exactly as `app/main.py` does —
defaults, then the file, then the declared plugin parameters, in the
real precedence — resolves every entry point through the one resolver,
and derives the consumed subjects with the same code the eligibility
gate uses. Then it stops.

It is an INSPECTION. It imports no TensorFlow and opens no accelerator,
constructs no plugin object, creates no file and trains nothing; the
caller runs it in a subprocess and can assert all of that from outside.
Absolute paths, traversal and files outside the authorized checkout
refuse.

    python tools/validate_runnable_config.py --config <path> \\
        --checkout <root> [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RUNNABLE = "VALIDATED_RUNNABLE"
NOT_RUNNABLE = "NOT_RUNNABLE"

#: the partition keys a supervised run declares, by side.
X_KEYS = ("x_train_file", "x_validation_file", "x_test_file")
Y_KEYS = ("y_train_file", "y_validation_file", "y_test_file")


def contained(checkout: Path, value) -> tuple[Path | None, str]:
    """Resolve a declared data path INSIDE the authorized checkout.

    An absolute path or a `..` escape is refused rather than read: a
    configuration that reaches outside the checkout is not a
    configuration of this checkout.
    """
    if not isinstance(value, str) or not value.strip():
        return None, "not a path"
    p = Path(value)
    if p.is_absolute():
        return None, "absolute path: a runnable config of this " \
                     "checkout names its data relative to it"
    resolved = (checkout / p).resolve()
    try:
        resolved.relative_to(checkout.resolve())
    except ValueError:
        return None, "escapes the authorized checkout"
    if not resolved.is_file():
        return None, "declared input is not present"
    return resolved, "ok"


def validate(config_path: Path, checkout: Path) -> dict:
    from app.config import DEFAULT_VALUES
    from app.config_handler import load_config
    from app.config_merger import merge_config
    from app.plugin_resolver import (PLUGIN_ROLES, canonical_name,
                                     declared_plugin_params, resolve)
    from eligibility.consumed import resolve_consumed_subjects

    out = {
        "config": str(config_path.relative_to(checkout))
        if config_path.is_relative_to(checkout) else config_path.name,
        "verdict": NOT_RUNNABLE,
        "reasons": [],
        "effective_config_built": False,
        "entry_points_resolved": {},
        "subjects": [], "targets": [], "sides": {},
    }

    try:
        file_config = load_config(str(config_path))
    except Exception as exc:                              # noqa: BLE001
        out["reasons"].append(f"unreadable: {type(exc).__name__}")
        return out

    # the REAL precedence: defaults -> file -> declared plugin params.
    # No CLI layer here: a config is being judged, not an invocation.
    # `merge_config` narrates every key it touches. That chatter is
    # useful in a run and ruinous in a validator whose only output is
    # one JSON line, so it is captured rather than printed.
    import contextlib
    import io

    noise = io.StringIO()
    config = dict(DEFAULT_VALUES)
    with contextlib.redirect_stdout(noise):
        config = merge_config(config, {}, {}, file_config, {}, {})

    witnesses = {}
    try:
        predictor_name = (canonical_name(config, "predictor")
                          or config.get("predictor_plugin")
                          or "default_predictor")
        for role, (key, _group, _aliases) in sorted(PLUGIN_ROLES.items()):
            name = predictor_name if role == "predictor" \
                else config.get(key)
            if not name:
                continue
            witnesses[role] = resolve(role, name)
    except SystemExit as exc:
        out["reasons"].append(f"entry point: {str(exc)[:160]}")
        return out
    out["entry_points_resolved"] = {
        r: {"name": w["entry_point_name"], "origin_id": w["origin_id"],
            "inside_checkout": w["inside_checkout"]}
        for r, w in witnesses.items()}

    # the plugin parameter layer, read declaratively — importing the
    # plugin would load TensorFlow, and this path does not execute
    for role in sorted(witnesses):
        try:
            with contextlib.redirect_stdout(noise):
                config = merge_config(
                    config, declared_plugin_params(witnesses[role]),
                    {}, file_config, {}, {})
        except SystemExit as exc:
            out["reasons"].append(
                f"{role} parameters: {str(exc)[:140]}")
            return out
    out["effective_config_built"] = True

    declared = {k: config.get(k) for k in X_KEYS + Y_KEYS
                if config.get(k)}
    if not declared:
        out["reasons"].append(
            "the effective configuration declares no x/y partitions")
        return out
    for key, value in sorted(declared.items()):
        path, why = contained(checkout, value)
        if path is None:
            out["reasons"].append(f"{key}: {why}")
    if out["reasons"]:
        return out

    try:
        with contextlib.redirect_stdout(noise):
            consumed = resolve_consumed_subjects(config,
                                                 repo_root=checkout)
    except SystemExit as exc:
        out["reasons"].append(f"subjects: {str(exc)[:200]}")
        return out

    out["subjects"] = consumed["subjects"]
    out["targets"] = sorted({s["column"] for s in consumed["subjects"]
                             if s["contract_role"] == "target"})
    out["sides"] = {
        side: sum(1 for s in consumed["subjects"] if s["side"] == side)
        for side in ("x", "y")}
    if not out["targets"]:
        out["reasons"].append(
            "the effective configuration declares no target that "
            "appears in the data; a run with no target is not runnable")
        return out
    out["digests"] = consumed["digests"]
    out["verdict"] = RUNNABLE
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkout", required=True, type=Path)
    a = ap.parse_args(argv)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    result = validate(a.config.expanduser(), a.checkout.expanduser()
                      .resolve())
    # proof, from inside, that this path executed nothing
    result["inspection_only"] = {
        "frameworks_imported": sorted(
            m for m in sys.modules
            if m.split(".")[0] in ("tensorflow", "keras", "torch",
                                   "jax")),
        "note": "the caller asserts this from OUTSIDE as well, by "
                "running this module in a subprocess",
    }
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
