"""Modular temporal Keras architecture.

The public imports remain stable at predictor_plugins.modular_temporal.
Implementation is separated by configuration, components, assembly,
pretraining, and donor serialization.
"""

import tensorflow as tf

from .assembly import (BudgetExceeded, ModularBundle, build_modular, canonical_config_json, config_digest,
                       measure_budget, probe_alignment)
from .artifacts import _check_manifest_model, load_donor, save_donor
from .bundle import BUNDLE_SCHEMA, load_bundle, save_bundle
from .common import _copy, _digest, _file_hash, _json, keras_version, weights_hash
from .components import (ROLES, TemporalComponent, causal_conv1d, component, effective_params,
                         forecast, sequence_concat, transformer_conv)
from .config import CONFIG_SCHEMA, _normalize, default_config, regime_summary
from .layers import FeatureSelect, PositionalEncoding, SeasonalNaiveBaseline
from .pretraining import (
    branch_autoencoder,
    build_autoencoder,
    build_decoder,
    core_autoencoder,
)
from .registry import BUILTINS, DEFAULTS, _resolve, describe_component
from . import warm

keras = tf.keras

__all__ = [
    "FeatureSelect", "ModularBundle", "PositionalEncoding", "TemporalComponent",
    "BUILTINS", "DEFAULTS", "BUNDLE_SCHEMA", "CONFIG_SCHEMA", "ROLES", "BudgetExceeded", "measure_budget",
    "canonical_config_json", "causal_conv1d", "component", "config_digest", "describe_component",
    "effective_params", "forecast", "keras_version", "load_bundle", "probe_alignment",
    "regime_summary", "save_bundle", "sequence_concat", "transformer_conv",
    "branch_autoencoder", "build_autoencoder", "build_decoder", "build_modular",
    "core_autoencoder", "default_config", "load_donor", "save_donor", "weights_hash",
]
