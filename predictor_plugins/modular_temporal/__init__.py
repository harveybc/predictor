"""Modular temporal Keras architecture.

The public imports remain stable at predictor_plugins.modular_temporal.
Implementation is separated by configuration, components, assembly,
pretraining, and donor serialization.
"""

from .assembly import ModularBundle, build_modular
from .artifacts import load_donor, save_donor
from .common import weights_hash
from .components import TemporalComponent
from .config import default_config
from .layers import FeatureSelect, PositionalEncoding
from .pretraining import (
    branch_autoencoder,
    build_autoencoder,
    build_decoder,
    core_autoencoder,
)
from .registry import BUILTINS, DEFAULTS

__all__ = [
    "FeatureSelect", "ModularBundle", "PositionalEncoding", "TemporalComponent",
    "BUILTINS", "DEFAULTS",
    "branch_autoencoder", "build_autoencoder", "build_decoder", "build_modular",
    "core_autoencoder", "default_config", "load_donor", "save_donor", "weights_hash",
]
