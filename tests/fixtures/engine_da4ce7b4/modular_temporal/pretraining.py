"""Reconstruction decoders and branch/core autoencoder factories."""

import tensorflow as tf

from .common import _positive_int

keras = tf.keras

def build_decoder(latent_shape, output_steps, output_channels, channels=32):
    """Build a learned decoder for temporal autoencoder reconstruction.

    Parameters
    ----------
    latent_shape : tuple[int, int]
        Encoder output shape without the batch axis.
    output_steps : int
        Number of input time steps to reconstruct.
    output_channels : int
        Number of input features to reconstruct.
    channels : int, default=32
        Hidden width of the first decoder convolution.

    Returns
    -------
    keras.Model
        Decoder mapping latent sequences to reconstructed input sequences.
    """
    steps = _positive_int(output_steps, "output_steps")
    width = _positive_int(output_channels, "output_channels")
    _positive_int(channels, "channels")
    if len(latent_shape) != 2:
        raise ValueError("latent_shape must be (steps, channels)")
    for n in latent_shape:
        _positive_int(n, "latent_shape")
    if steps % latent_shape[0]:
        raise ValueError("Decoder expansion requires exact divisibility")
    inputs = keras.Input(tuple(latent_shape))
    x = keras.layers.UpSampling1D(steps // latent_shape[0])(inputs)
    x = keras.layers.Conv1D(channels, 3, padding="same", activation="gelu")(x)
    x = keras.layers.Conv1D(width, 1, padding="same")(x)
    return keras.Model(inputs, x, name="reconstruction_decoder")


def build_autoencoder(encoder, *, channels=32):
    """Wrap an encoder in a reconstruction model while sharing its weights.

    The caller owns loss, optimizer, train/validation data and early stopping.
    Fitting this model updates the same encoder object passed to the function.

    Parameters
    ----------
    encoder : keras.Model
        Single-input, rank-three temporal encoder.
    channels : int, default=32
        Hidden width of the reconstruction decoder.

    Returns
    -------
    keras.Model
        Autoencoder with the same input and output shapes as ``encoder.input``.
    """
    if len(encoder.inputs) != 1 or len(encoder.outputs) != 1 or len(encoder.input_shape) != 3 or len(encoder.output_shape) != 3:
        raise ValueError("Autoencoder requires a single rank-three encoder")
    decoder = build_decoder(tuple(encoder.output_shape[1:]), *encoder.input_shape[1:], channels=channels)
    inputs = keras.Input(tuple(encoder.input_shape[1:]))
    return keras.Model(inputs, decoder(encoder(inputs)), name=encoder.name + "_autoencoder")


def branch_autoencoder(bundle, branch_name, *, channels=32):
    """Create a reconstruction model for one named feature branch.

    Parameters
    ----------
    bundle : ModularBundle
        Model bundle returned by :func:`build_modular`.
    branch_name : str
        Branch whose local input representation will be reconstructed.
    channels : int, default=32
        Decoder hidden width.
    """
    return build_autoencoder(bundle.branch_models[branch_name], channels=channels)


def core_autoencoder(bundle, *, channels=32):
    """Create a reconstruction model for the fused sequence entering the core.

    Parameters
    ----------
    bundle : ModularBundle
        Model bundle returned by :func:`build_modular`.
    channels : int, default=32
        Decoder hidden width.
    """
    return build_autoencoder(bundle.core_model, channels=channels)
