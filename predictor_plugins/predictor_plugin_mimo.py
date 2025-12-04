#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predictor_plugin_mimo.py

Plugin de predictor MIMO multi-horizonte para el sistema actual.

Arquitectura:
  - Encoder compartido:
      * Conv1D stack (causal) + BiLSTM + GlobalAveragePooling1D
      * produce un embedding global z_global de la ventana
  - Decoder de horizontes:
      * Embeddings de horizonte
      * Concatenación z_global ⊕ embedding_horizonte
      * Bloque de MultiHeadAttention sobre el eje horizonte
      * MLP final compartida para producir un escalar por horizonte
      * Se generan salidas con nombres "output_horizon_{h}" compatibles
        con el target plugin, preprocesador y pipeline actuales.

Esta clase respeta la interfaz de BaseBayesianKerasPredictor:
  - plugin_params
  - plugin_debug_vars
  - build_model(input_shape, x_train, config)

No modifica el pipeline ni el preprocesador existentes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Lambda,
    Bidirectional,
    LSTM,
    GlobalAveragePooling1D,
    LayerNormalization,
    Dropout,
    Add,
    Concatenate,
    MultiHeadAttention,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

from .common.base import BaseBayesianKerasPredictor
from .common.positional_encoding import positional_encoding
from .common.losses import mae_magnitude


class Plugin(BaseBayesianKerasPredictor):
    """
    Plugin MIMO multi-horizonte para tu sistema de predicción.

    - Construye un modelo único con un encoder compartido para todas las salidas.
    - Implementa un decoder de horizontes basado en embeddings de horizonte
      y un bloque de atención multi-cabeza sobre el eje horizonte.
    - Genera una salida por horizonte con nombre "output_horizon_{h}" para
      mantener compatibilidad con el target plugin, preprocesador y pipeline.
    """

    plugin_params: Dict[str, Any] = {
        # Entrenamiento
        "batch_size": 32,
        "learning_rate": 1e-3,

        # Encoder
        "encoder_conv_layers": 2,
        "encoder_base_filters": 64,
        "encoder_lstm_units": 64,

        # Decoder de horizontes
        "horizon_embedding_dim": 16,
        "horizon_attn_heads": 4,
        "horizon_attn_key_dim": 32,
        "decoder_dropout": 0.1,

        # Regularización / activación
        "activation": "relu",
        "l2_reg": 1e-6,

        # Se sobreescribe desde el JSON
        "predicted_horizons": [1],

        # Positional encoding opcional (mismo patrón que el CNN)
        "positional_encoding": False,
    }

    plugin_debug_vars: List[str] = [
        "batch_size",
        "learning_rate",
        "encoder_conv_layers",
        "encoder_base_filters",
        "encoder_lstm_units",
        "horizon_embedding_dim",
        "horizon_attn_heads",
        "horizon_attn_key_dim",
        "decoder_dropout",
        "activation",
        "l2_reg",
        "predicted_horizons",
        "positional_encoding",
    ]

    def build_model(
        self,
        input_shape: Tuple[int, int],
        x_train: Any,
        config: Dict[str, Any],
    ) -> None:
        """
        Construye y compila el modelo Keras MIMO para el predictor.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            (window_size, num_features).
        x_train : Any
            Datos de entrenamiento (no usado directamente, se mantiene por compatibilidad).
        config : Dict[str, Any]
            Configuración global; se mezcla con plugin_params.
        """
        # Mezclar parámetros de config con los del plugin (mismo patrón que cnn)
        if config:
            self.params.update(config)

        window_size, num_features = input_shape

        # Lista de horizontes, ordenada
        horizons: List[int] = list(self.params.get("predicted_horizons", []))
        if not horizons:
            raise ValueError(
                "MIMO predictor: 'predicted_horizons' vacío; "
                "debes definir al menos un horizonte en la configuración."
            )
        horizons = sorted(horizons)

        activation_name: str = self.params.get("activation", "relu")
        l2_reg_value: float = float(self.params.get("l2_reg", 1e-6))

        # ------------------------------------------------------------------ #
        # 1) Entrada                                                        #
        # ------------------------------------------------------------------ #
        inputs = Input(shape=(window_size, num_features), name="input_window")

        # Positional encoding opcional (mismo estilo que cnn)
        if self.params.get("positional_encoding", False):
            pe = positional_encoding(window_size, num_features)
            x = Lambda(
                lambda t, pe=pe: t + pe,
                name="add_positional_encoding",
            )(inputs)
        else:
            x = inputs

        # ------------------------------------------------------------------ #
        # 2) Encoder compartido: Conv1D stack + BiLSTM + Global Pool        #
        # ------------------------------------------------------------------ #
        num_conv_layers: int = int(self.params.get("encoder_conv_layers", 2))
        base_filters: int = int(self.params.get("encoder_base_filters", 64))

        for layer_idx in range(num_conv_layers):
            filters = max(8, base_filters // (2 ** layer_idx))
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=3,
                padding="causal",
                activation=activation_name,
                kernel_regularizer=l2(l2_reg_value),
                name=f"enc_conv_{layer_idx+1}",
            )(x)

        lstm_units: int = int(self.params.get("encoder_lstm_units", 64))

        x_seq = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                name="enc_lstm",
            ),
            name="enc_bilstm",
        )(x)

        z_global = GlobalAveragePooling1D(name="enc_global_avg_pool")(x_seq)

        # ------------------------------------------------------------------ #
        # 3) Tokens de horizonte (embeddings + réplica por batch)           #
        # ------------------------------------------------------------------ #
        num_horizons: int = len(horizons)
        max_horizon: int = int(max(horizons))
        horizon_emb_dim: int = int(self.params.get("horizon_embedding_dim", 16))

        # Constante con ids de horizonte, e.g. [4, 8, 12, ...]
        horizon_ids = tf.constant(horizons, dtype=tf.int32, name="horizon_ids")

        horizon_embedding_layer = tf.keras.layers.Embedding(
            input_dim=max_horizon + 1,
            output_dim=horizon_emb_dim,
            name="horizon_embedding",
        )

        # (num_horizons, horizon_emb_dim)
        horizon_embs = horizon_embedding_layer(horizon_ids)

        # (1, num_horizons, horizon_emb_dim)
        horizon_embs_expanded = Lambda(
            lambda e: tf.expand_dims(e, axis=0),
            name="expand_horizon_embs",
        )(horizon_embs)

        # CORRECCIÓN: usar z_global como input de la Lambda para obtener el batch_size
        # (batch, num_horizons, horizon_emb_dim)
        horizon_embs_tiled = Lambda(
            lambda tensors: tf.tile(
                tensors[0],
                [tf.shape(tensors[1])[0], 1, 1],
            ),
            name="tile_horizon_embs",
        )([horizon_embs_expanded, z_global])

        # Expansión de z_global a (batch, 1, latent_dim)
        z_expanded = Lambda(
            lambda z: tf.expand_dims(z, axis=1),
            name="expand_global",
        )(z_global)

        # Réplica de z_global a lo largo de los horizontes: (batch, num_horizons, latent_dim)
        z_tiled = Lambda(
            lambda z: tf.tile(z, [1, num_horizons, 1]),
            name="tile_global",
        )(z_expanded)

        # Concatena z_global y el embedding de horizonte en el eje de features
        horizon_tokens = Concatenate(
            axis=-1,
            name="concat_global_horizon",
        )([z_tiled, horizon_embs_tiled])

        # ------------------------------------------------------------------ #
        # 4) Bloque de atención multi-cabeza + FFN sobre horizonte          #
        # ------------------------------------------------------------------ #
        attn_heads: int = int(self.params.get("horizon_attn_heads", 4))
        attn_key_dim: int = int(self.params.get("horizon_attn_key_dim", 32))
        decoder_dropout: float = float(self.params.get("decoder_dropout", 0.1))

        attn_output = MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=attn_key_dim,
            name="horizon_mha",
        )(horizon_tokens, horizon_tokens)

        horizon_tokens_res = Add(name="horizon_attn_residual")(
            [horizon_tokens, attn_output]
        )

        horizon_tokens_norm = LayerNormalization(name="horizon_attn_ln")(
            horizon_tokens_res
        )

        ff_dense = Dense(
            units=horizon_tokens_norm.shape[-1],
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name="horizon_ffn_dense",
        )(horizon_tokens_norm)

        ff_dense = Dropout(decoder_dropout, name="horizon_ffn_dropout")(ff_dense)

        horizon_tokens_ffn_res = Add(name="horizon_ffn_residual")(
            [horizon_tokens_norm, ff_dense]
        )

        horizon_tokens_final = LayerNormalization(
            name="horizon_ffn_ln",
        )(horizon_tokens_ffn_res)

        # ------------------------------------------------------------------ #
        # 5) Cabeza de salida MIMO: 1 escalar por horizonte                 #
        # ------------------------------------------------------------------ #
        horizon_outputs = Dense(
            units=1,
            activation=None,
            name="horizon_output_dense",
        )(horizon_tokens_final)

        outputs: List[tf.Tensor] = []
        self.output_names: List[str] = []

        for idx, h in enumerate(horizons):
            out_i = Lambda(
                lambda t, i=idx: tf.squeeze(t[:, i, :], axis=-1),
                name=f"output_horizon_{h}",
            )(horizon_outputs)

            outputs.append(out_i)
            self.output_names.append(f"output_horizon_{h}")

        # ------------------------------------------------------------------ #
        # 6) Modelo y compilación                                           #
        # ------------------------------------------------------------------ #
        self.model = Model(
            inputs=inputs,
            outputs=outputs,
            name=f"MIMOPredictor_{len(horizons)}H",
        )

        optimizer = AdamW(
            learning_rate=float(self.params.get("learning_rate", 1e-3))
        )

        loss_dict: Dict[str, Any] = {
            name: tf.keras.losses.MeanAbsoluteError()
            for name in self.output_names
        }

        metrics_dict: Dict[str, List[Any]] = {
            name: [mae_magnitude] for name in self.output_names
        }

        self.model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            metrics=metrics_dict,
        )

        self.model.summary(line_length=140)
