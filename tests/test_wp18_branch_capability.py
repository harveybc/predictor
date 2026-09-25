# -*- coding: utf-8 -*-
"""WP18 step 5 (M5PHET work plan 2026-09-24, revision 2): which cores accept SEVERAL INPUT BRANCHES?

The pipeline WP18 describes fuses one extractor per feature GROUP and hands the fused branches to a core. So the
option list of the core decision may only contain plugins that actually accept several input branches. The work plan
forbids assuming which ones do: "an agent must first test which do, by building a two-branch input in a unit test,
and declare only those".

This is that test. It is a PROBE, not a quality measurement: it says whether a plugin can be handed two branches and
build a Keras model with two inputs. It says nothing about whether the resulting model is any good.

What is probed, per entry point declared in this repository's ``setup.py`` under ``predictor.plugins``:

1. a **two-branch attempt** in each of the three shapes a plugin could plausibly declare -- a list of shapes, a tuple
   of shapes, a mapping of branch name to shape -- with ``x_train`` carrying the matching pair of arrays;
2. a **single-branch control** with one ``(window, channels)`` shape, which tells apart "this plugin refuses branches"
   from "this plugin could not be built here at all" (no GPU, a missing dependency, a broken module).

Verdicts, and nothing in between:

``MULTI_BRANCH``        a two-branch attempt produced a Keras model with two or more inputs.
``SINGLE_BRANCH_ONLY``  every two-branch attempt failed or produced a one-input model, AND the single-branch control
                        built -- so the plugin works, it just takes one tensor.
``NOT_PROBED``          the module would not import, or the control failed too. The plugin's capability is UNKNOWN
                        and it is therefore not offered as an option either; the reason is recorded verbatim.

The probe runs on the CPU (``CUDA_VISIBLE_DEVICES=""`` is set below before TensorFlow is imported): the owner's GPU
is reserved for training jobs and nothing here needs one.

Set ``WP18_BRANCH_CAPABILITY_OUT`` to write the verdict table as ``m5phet.branch_capability.v1`` JSON; M5PHET's
``m5phet.pipeline.catalog_cores()`` reads that file and offers only the ``MULTI_BRANCH`` entries.
"""

import importlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
GROUP = "predictor.plugins"
SCHEMA = "m5phet.branch_capability.v1"

MULTI_BRANCH = "MULTI_BRANCH"
SINGLE_BRANCH_ONLY = "SINGLE_BRANCH_ONLY"
NOT_PROBED = "NOT_PROBED"
VERDICTS = (MULTI_BRANCH, SINGLE_BRANCH_ONLY, NOT_PROBED)

WINDOW = 8
BRANCH_CHANNELS = (3, 2)
#: what the plugins need in their params to build at all; a horizon list of one keeps every head small
CONFIG = {"predicted_horizons": [1], "batch_size": 4, "mc_samples": 2, "epochs": 1,
          "num_hidden_layers": 1, "hidden_units": 8, "intermediate_layers": 1, "initial_layer_size": 8}


# --- the declared option list, read from setup.py (never typed from memory) -------------------------------------------

def declared_entry_points(text, group):
    """The ``name=module:attr`` entries setup.py declares under ``group``, in the order they are written."""
    block = re.search(r"['\"]" + re.escape(group) + r"['\"]\s*:\s*\[(.*?)\]", text, re.S)
    if block is None:
        return []
    entries = []
    for raw in re.findall(r"['\"]([^'\"]+)['\"]", block.group(1)):
        name, sep, target = raw.partition("=")
        if sep and name.strip() and target.strip():
            entries.append((name.strip(), target.strip()))
    return entries


@pytest.fixture(scope="module")
def entries():
    declared = declared_entry_points((REPO / "setup.py").read_text(encoding="utf-8"), GROUP)
    assert declared, f"{REPO / 'setup.py'} declares no {GROUP} entry points"
    return declared


# --- the detector, and the control that proves the detector is not the thing failing ----------------------------------

def _input_count(built, plugin):
    """How many inputs the built model has. A plugin may return the model or leave it on ``self.model``."""
    model = built if built is not None else getattr(plugin, "model", None)
    if model is None:
        return None
    inputs = getattr(model, "inputs", None)
    if inputs is None:
        return None
    return len(inputs)


def test_the_detector_counts_two_inputs_on_a_model_that_really_has_two():
    """Positive control: a SINGLE_BRANCH_ONLY verdict below is a property of the plugin, not of this detector."""
    from tensorflow.keras.layers import Concatenate, Dense, Flatten, Input
    from tensorflow.keras.models import Model

    left, right = Input(shape=(WINDOW, BRANCH_CHANNELS[0])), Input(shape=(WINDOW, BRANCH_CHANNELS[1]))
    fused = Dense(1)(Concatenate()([Flatten()(left), Flatten()(right)]))
    assert _input_count(Model([left, right], fused), None) == 2
    single = Input(shape=(WINDOW, sum(BRANCH_CHANNELS)))
    assert _input_count(Model(single, Dense(1)(Flatten()(single))), None) == 1


# --- one plugin, probed ------------------------------------------------------------------------------------------------

def _plugin_class(target):
    module_name, _, attr = target.partition(":")
    return getattr(importlib.import_module(module_name), attr or "Plugin")


def _branch_arrays():
    rng = np.random.default_rng(20260925)
    return [rng.normal(size=(16, WINDOW, channels)).astype("float32") for channels in BRANCH_CHANNELS]


def _attempts():
    """The three shapes a plugin could plausibly declare for several branches, each with matching arrays."""
    shapes = [(WINDOW, channels) for channels in BRANCH_CHANNELS]
    arrays = _branch_arrays()
    return [("list_of_shapes", list(shapes), arrays),
            ("tuple_of_shapes", tuple(shapes), tuple(arrays)),
            ("mapping_of_shapes", {f"group_{i}": shape for i, shape in enumerate(shapes)},
             {f"group_{i}": array for i, array in enumerate(arrays)})]


def _instantiate(plugin_class):
    """Some plugins take the config in the constructor and some take none; both are tried before giving up."""
    try:
        return plugin_class()
    except TypeError:
        return plugin_class(dict(CONFIG))


def _build(plugin_class, input_shape, x_train):
    plugin = _instantiate(plugin_class)
    built = plugin.build_model(input_shape, x_train, dict(CONFIG))
    return _input_count(built, plugin)


def probe(name, target):
    """One entry point's verdict, with the reason it got it. Never raises: an unprobed plugin is a verdict too."""
    record = {"entry_point": name, "target": target, "verdict": NOT_PROBED, "inputs": None, "why": ""}
    try:
        plugin_class = _plugin_class(target)
    except BaseException as error:                                                      # noqa: BLE001
        record["why"] = f"the module could not be imported: {type(error).__name__}: {error}"[:400]
        return record

    failures = []
    for label, input_shape, x_train in _attempts():
        try:
            count = _build(plugin_class, input_shape, x_train)
        except BaseException as error:                                                  # noqa: BLE001
            failures.append(f"{label}: {type(error).__name__}: {error}"[:200])
            continue
        if count is not None and count >= 2:
            record.update(verdict=MULTI_BRANCH, inputs=count,
                          why=f"built a model with {count} inputs from a {label} of two branches")
            return record
        failures.append(f"{label}: built a model with {count} input(s), not two")

    rng = np.random.default_rng(20260925)
    control = rng.normal(size=(16, WINDOW, sum(BRANCH_CHANNELS))).astype("float32")
    try:
        count = _build(plugin_class, (WINDOW, sum(BRANCH_CHANNELS)), control)
    except BaseException as error:                                                      # noqa: BLE001
        record["why"] = (f"neither branches nor one tensor could be built here, so this plugin's branch capability "
                         f"is unknown; the one-tensor control raised {type(error).__name__}: {error}"[:400])
        return record
    if count is None:
        record["why"] = ("the one-tensor control built no Keras model (this plugin builds lazily), so its branch "
                         "capability cannot be read from a build")
        return record
    record.update(verdict=SINGLE_BRANCH_ONLY, inputs=count,
                  why="the one-tensor control built a model with "
                      f"{count} input(s) while every two-branch attempt failed: " + " | ".join(failures))
    return record


# --- the table ----------------------------------------------------------------------------------------------------------

def test_every_declared_core_gets_one_verdict_and_the_table_is_written(entries):
    import tensorflow as tf

    verdicts = {}
    for name, target in entries:
        verdicts[name] = probe(name, target)
        print(f"{name:<24} {verdicts[name]['verdict']:<20} {verdicts[name]['why'][:120]}", flush=True)

    assert sorted(verdicts) == sorted(name for name, _ in entries)
    assert all(record["verdict"] in VERDICTS for record in verdicts.values())

    document = {
        "schema": SCHEMA,
        "group": GROUP,
        "registry": "predictor/setup.py",
        "probed_on": date.today().isoformat(),
        "probe": {
            "question": "does this plugin build a Keras model with two or more inputs when handed two input branches?",
            "attempts": [label for label, _shape, _arrays in _attempts()],
            "window": WINDOW, "branch_channels": list(BRANCH_CHANNELS), "config": dict(CONFIG),
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "tensorflow": tf.__version__,
            "device": "CPU (CUDA_VISIBLE_DEVICES is empty)",
            "verdicts_declared": list(VERDICTS),
        },
        "verdicts": verdicts,
        "multi_branch": sorted(name for name, record in verdicts.items() if record["verdict"] == MULTI_BRANCH),
        "note": ("a MULTI_BRANCH verdict says the model builds, not that it is any good; a NOT_PROBED plugin is not "
                 "offered as an option, because an unknown capability is not a declared one"),
    }
    assert document["multi_branch"] == sorted(name for name, record in verdicts.items()
                                              if record["verdict"] == MULTI_BRANCH)

    out = os.environ.get("WP18_BRANCH_CAPABILITY_OUT")
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}", flush=True)
