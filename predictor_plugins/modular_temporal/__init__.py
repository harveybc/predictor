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
from .branch_control import BranchOnlyControl, build_branch_only_control
from .common import _copy, _digest, _file_hash, _json, keras_version, weights_hash
from .components import (ROLES, TemporalComponent, causal_conv1d, causal_dense_sequence,
                         component, effective_params, forecast, sequence_concat,
                         transformer_conv)
from .dense_control import (CONFIG_SCHEMA as DENSE_CONTROL_CONFIG_SCHEMA,
                            UNORDERED_LATENT_VECTOR, DenseControlAdapter,
                            DenseLatentComponent, canonical_dense_control_json,
                            causal_window_dense, dense_control_config,
                            load_dense_control_config)
from .config import CONFIG_SCHEMA, _normalize, default_config, regime_summary
from .layers import CausalFrames, FeatureSelect, PositionalEncoding
from .matched_control import (CONFIG_SCHEMA as MATCHED_CONTROL_CONFIG_SCHEMA,
                              TEMPORAL_SEQUENCE, MatchedArm, MatchedControl,
                              build_matched_control, normalize_matched_config,
                              train_only_plumbing)
from .pretraining import (
    branch_autoencoder,
    build_autoencoder,
    build_decoder,
    core_autoencoder,
)
from .registry import BUILTINS, DEFAULTS, _resolve, describe_component

keras = tf.keras

__all__ = [
    "BranchOnlyControl", "CausalFrames", "FeatureSelect", "ModularBundle", "PositionalEncoding",
    "TemporalComponent",
    "BUILTINS", "DEFAULTS", "BUNDLE_SCHEMA", "CONFIG_SCHEMA", "ROLES", "BudgetExceeded", "measure_budget",
    "canonical_config_json", "causal_conv1d", "causal_dense_sequence", "component",
    "config_digest", "describe_component",
    "DENSE_CONTROL_CONFIG_SCHEMA", "UNORDERED_LATENT_VECTOR", "DenseControlAdapter",
    "DenseLatentComponent", "canonical_dense_control_json", "causal_window_dense",
    "dense_control_config", "load_dense_control_config",
    "MATCHED_CONTROL_CONFIG_SCHEMA", "TEMPORAL_SEQUENCE", "MatchedArm", "MatchedControl",
    "build_matched_control", "normalize_matched_config", "train_only_plumbing",
    "effective_params", "forecast", "keras_version", "load_bundle", "probe_alignment",
    "regime_summary", "save_bundle", "sequence_concat", "transformer_conv",
    "branch_autoencoder", "build_autoencoder", "build_branch_only_control", "build_decoder", "build_modular",
    "core_autoencoder", "default_config", "load_donor", "save_donor", "weights_hash",
]
