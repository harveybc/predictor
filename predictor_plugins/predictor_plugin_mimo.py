#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predictor_plugin_mimo.py

Plugin de predictor MIMO multi-horizonte compatible con tu sistema actual.
"""

from __future__ import annotations  # Habilita anotaciones de tipos hacia adelante

from typing import Any, Dict, List, Tuple  # Tipos para anotaciones

import tensorflow as tf  # Import principal de TensorFlow

# Import de capas Keras necesarias para la arquitectura
from tensorflow.keras.layers import (
    Input,                     # Capa de entrada del modelo
    Dense,                     # Capa densa estándar
    Lambda,                    # Capa Lambda para operaciones personalizadas
    Bidirectional,             # Wrapper para LSTM bidireccional
    LSTM,                      # Capa LSTM
    GlobalAveragePooling1D,    # Pooling promedio global sobre eje temporal
    LayerNormalization,        # Normalización por capas
    Dropout,                   # Dropout para regularización
    Add,                       # Suma residual
    Concatenate,               # Concatenación de tensores
    MultiHeadAttention,        # Atención multi-cabeza
)
from tensorflow.keras.models import Model            # Clase base de modelos Keras
from tensorflow.keras.optimizers import AdamW        # Optimizador AdamW
from tensorflow.keras.regularizers import l2         # Regularizador L2

# Imports de utilidades del propio proyecto
from .common.base import BaseBayesianKerasPredictor  # Clase base de predictores bayesianos
from .common.positional_encoding import positional_encoding  # Codificación posicional existente
from .common.losses import mae_magnitude             # Métrica MAE específica del proyecto


class Plugin(BaseBayesianKerasPredictor):
    """
    Plugin MIMO multi-horizonte.

    - Encoder compartido: stack Conv1D + BiLSTM + GlobalAveragePooling1D.
    - Decoder de horizontes: embeddings de horizonte + MultiHeadAttention + MLP.
    - Salidas: una por horizonte, con nombre "output_horizon_{h}".
    """

    # Diccionario de parámetros del plugin (pueden sobreescribirse desde config JSON)
    plugin_params: Dict[str, Any] = {
        # Parámetros de entrenamiento
        "batch_size": 32,                 # Tamaño de batch por defecto
        "learning_rate": 1e-3,            # Tasa de aprendizaje para AdamW

        # Hiperparámetros del encoder
        "encoder_conv_layers": 2,         # Número de capas Conv1D en el encoder
        "encoder_base_filters": 64,       # Filtros de la primera capa Conv1D
        "encoder_lstm_units": 64,         # Unidades de la BiLSTM

        # Hiperparámetros del decoder de horizontes
        "horizon_embedding_dim": 16,      # Dimensión de los embeddings de horizonte
        "horizon_attn_heads": 4,          # Nº de cabezas de MultiHeadAttention
        "horizon_attn_key_dim": 32,       # Dimensión de clave en MultiHeadAttention
        "decoder_dropout": 0.1,           # Dropout en el bloque de decoder

        # Regularización y activación
        "activation": "relu",             # Nombre de la activación principal
        "l2_reg": 1e-6,                   # Coeficiente de regularización L2

        # Lista de horizontes (se sobreescribe desde el JSON de config)
        "predicted_horizons": [1],        # Placeholder, se reemplaza por la lista real

        # Positional encoding opcional (como en el plugin CNN)
        "positional_encoding": False,     # Si True, se suma encoding posicional a la entrada
    }

    # Variables de depuración que se pueden inspeccionar desde fuera
    plugin_debug_vars: List[str] = [
        "batch_size",                     # Tamaño de batch
        "learning_rate",                  # Tasa de aprendizaje
        "encoder_conv_layers",            # Nº de capas Conv1D
        "encoder_base_filters",           # Filtros base de Conv1D
        "encoder_lstm_units",             # Unidades de BiLSTM
        "horizon_embedding_dim",          # Dim embeddings de horizonte
        "horizon_attn_heads",             # Nº cabezas de atención
        "horizon_attn_key_dim",           # Dimensión de clave atención
        "decoder_dropout",                # Dropout en decoder
        "activation",                     # Activación usada
        "l2_reg",                         # Regularización L2
        "predicted_horizons",             # Lista de horizontes
        "positional_encoding",            # Flag de positional encoding
    ]

    def build_model(
        self,
        input_shape: Tuple[int, int],     # Forma de entrada: (window_size, num_features)
        x_train: Any,                     # Datos de entrenamiento (no se usan aquí)
        config: Dict[str, Any],           # Config global (contiene parámetros de plugin)
    ) -> None:
        """
        Construye y compila el modelo MIMO.

        Parameters
        ----------
        input_shape : tuple
            Tupla (window_size, num_features).
        x_train : Any
            Datos de entrenamiento (no usados aquí, se mantiene por interfaz).
        config : dict
            Configuración global; se mezcla con los parámetros del plugin.
        """
        # Mezcla parámetros de config en self.params (mismo patrón que plugin CNN)
        if config:  # Si hay config externa
            self.params.update(config)  # Actualiza diccionario interno de parámetros

        # Desempaqueta la forma de entrada
        window_size, num_features = input_shape  # Largo de ventana y nº de features

        # Obtiene la lista de horizontes desde parámetros
        horizons: List[int] = list(self.params.get("predicted_horizons", []))  # Copia lista
        if not horizons:  # Si la lista está vacía, es un error
            raise ValueError(
                "MIMO predictor: 'predicted_horizons' vacío; "
                "debes definir al menos un horizonte en la configuración."
            )
        horizons = sorted(horizons)  # Ordena los horizontes de menor a mayor

        # Obtiene nombre de activación principal
        activation_name: str = self.params.get("activation", "relu")  # Activación
        # Obtiene valor de regularización L2
        l2_reg_value: float = float(self.params.get("l2_reg", 1e-6))  # L2

        # ------------------------------------------------------------------ #
        # 1) Capa de entrada                                                 #
        # ------------------------------------------------------------------ #
        inputs = Input(                       # Define capa de entrada
            shape=(window_size, num_features),  # Forma (T, d)
            name="input_window",              # Nombre de la capa de entrada
        )

        # ------------------------------------------------------------------ #
        # 2) Positional encoding opcional                                   #
        # ------------------------------------------------------------------ #
        if self.params.get("positional_encoding", False):  # Si se activa positional encoding
            pe = positional_encoding(window_size, num_features)  # Obtiene tensor de encoding
            x = Lambda(                    # Crea capa Lambda para sumar encoding
                lambda t, pe=pe: t + pe,  # Suma entrada + encoding posicional
                name="add_positional_encoding",  # Nombre de capa Lambda
            )(inputs)                      # Aplica a inputs
        else:
            x = inputs                     # Si no se usa encoding, x = inputs

        # ------------------------------------------------------------------ #
        # 3) Encoder compartido: Conv1D stack + BiLSTM + Global Pool        #
        # ------------------------------------------------------------------ #
        num_conv_layers: int = int(self.params.get("encoder_conv_layers", 2))  # Nº capas Conv1D
        base_filters: int = int(self.params.get("encoder_base_filters", 64))   # Filtros base

        # Bucle de capas Conv1D para capturar patrones locales
        for layer_idx in range(num_conv_layers):           # Itera sobre capas Conv1D
            filters = max(8, base_filters // (2 ** layer_idx))  # Filtros decrecientes
            x = tf.keras.layers.Conv1D(                    # Crea capa Conv1D
                filters=filters,                           # Nº filtros
                kernel_size=3,                             # Tamaño de kernel
                padding="causal",                          # Padding causal (no mira futuro)
                activation=activation_name,                # Activación elegida
                kernel_regularizer=l2(l2_reg_value),       # Regularización L2
                name=f"enc_conv_{layer_idx+1}",            # Nombre con índice
            )(x)                                           # Aplica la capa a x

        lstm_units: int = int(self.params.get("encoder_lstm_units", 64))  # Unidades BiLSTM

        # Capa BiLSTM para capturar dependencias temporales hacia adelante y atrás
        x_seq = Bidirectional(               # Wrapper bidireccional
            LSTM(                            # Capa LSTM interna
                lstm_units,                  # Nº unidades
                return_sequences=True,       # Devuelve secuencia completa
                name="enc_lstm",             # Nombre de LSTM
            ),
            name="enc_bilstm",               # Nombre del wrapper bidireccional
        )(x)                                 # Aplica a secuencia x

        # Pooling promedio global sobre el eje temporal → embedding global z_global
        z_global = GlobalAveragePooling1D(   # Capa de pooling promedio global
            name="enc_global_avg_pool",      # Nombre de la capa
        )(x_seq)                             # Aplica a secuencia LSTM

        # ------------------------------------------------------------------ #
        # 4) Tokens de horizonte: embeddings + réplica por batch            #
        # ------------------------------------------------------------------ #
        num_horizons: int = len(horizons)    # Nº de horizontes
        max_horizon: int = int(max(horizons))  # Horizonte máximo
        horizon_emb_dim: int = int(self.params.get("horizon_embedding_dim", 16))  # Dim embeddings

        # Tensor constante con IDs de horizonte (e.g. [4, 8, 12, ...])
        horizon_ids = tf.constant(
            horizons,                        # Lista de horizontes
            dtype=tf.int32,                  # Tipo entero
            name="horizon_ids",              # Nombre del tensor
        )

        # Capa Embedding de horizonte: input_dim = max_horizon+1 (seguro)
        horizon_embedding_layer = tf.keras.layers.Embedding(
            input_dim=max_horizon + 1,       # Rango de posibles horizontes
            output_dim=horizon_emb_dim,      # Dimensión del embedding
            name="horizon_embedding",        # Nombre de la capa
        )

        # Aplica embedding a IDs de horizonte: forma (num_horizons, horizon_emb_dim)
        horizon_embs = horizon_embedding_layer(horizon_ids)  # Embeddings por horizonte

        # Expande dimensión batch: forma (1, num_horizons, horizon_emb_dim)
        horizon_embs_expanded = Lambda(
            lambda e: tf.expand_dims(e, axis=0),            # Añade dimensión batch=1
            name="expand_horizon_embs",                     # Nombre capa Lambda
        )(horizon_embs)                                    # Aplica a embeddings

        # Replica embeddings a lo largo del batch usando z_global para conocer batch_size
        horizon_embs_tiled = Lambda(
            lambda tensors: tf.tile(                        # tile a lo largo del batch
                tensors[0],                                 # tensor de embeddings expandido
                [tf.shape(tensors[1])[0], 1, 1],            # [batch_size, num_horizons, emb_dim]
            ),
            name="tile_horizon_embs",                       # Nombre de capa Lambda
        )([horizon_embs_expanded, z_global])                # Entradas: embeddings y z_global

        # Expande z_global a forma (batch, 1, latent_dim)
        z_expanded = Lambda(
            lambda z: tf.expand_dims(z, axis=1),            # Inserta eje horizonte
            name="expand_global",                           # Nombre de capa Lambda
        )(z_global)                                         # Aplica a z_global

        # Replica z_global para cada horizonte: (batch, num_horizons, latent_dim)
        z_tiled = Lambda(
            lambda z: tf.tile(                              # tile en eje horizonte
                z,                                          # tensor expandido
                [1, num_horizons, 1],                       # [batch, num_horizons, latent_dim]
            ),
            name="tile_global",                             # Nombre de capa Lambda
        )(z_expanded)                                       # Aplica a z_expanded

        # Concatena z_global replicado y embeddings de horizonte en el eje de features
        horizon_tokens = Concatenate(
            axis=-1,                                        # Concatena en la última dimensión
            name="concat_global_horizon",                   # Nombre de la capa
        )([z_tiled, horizon_embs_tiled])                    # Entradas: z_tiled y embeddings

        # ------------------------------------------------------------------ #
        # 5) Bloque de atención multi-cabeza + FFN sobre horizontes         #
        # ------------------------------------------------------------------ #
        attn_heads: int = int(self.params.get("horizon_attn_heads", 4))   # Nº cabezas atención
        attn_key_dim: int = int(self.params.get("horizon_attn_key_dim", 32))  # Dim clave
        decoder_dropout: float = float(self.params.get("decoder_dropout", 0.1))  # Dropout

        # MultiHeadAttention sobre el eje horizonte: Q=K=V=horizon_tokens
        attn_output = MultiHeadAttention(
            num_heads=attn_heads,                           # Nº cabezas
            key_dim=attn_key_dim,                           # Dimensión de clave
            name="horizon_mha",                             # Nombre de la capa
        )(horizon_tokens, horizon_tokens)                   # Q y K/V son horizon_tokens

        # Conexión residual: tokens originales + salida de atención
        horizon_tokens_res = Add(
            name="horizon_attn_residual",                   # Nombre de capa de suma
        )([horizon_tokens, attn_output])                    # Suma elementos

        # Normalización por capas tras atención
        horizon_tokens_norm = LayerNormalization(
            name="horizon_attn_ln",                         # Nombre de la capa
        )(horizon_tokens_res)                               # Aplica LN

        # Bloque feed-forward (MLP) sobre cada token de horizonte
        ff_dense = Dense(
            units=horizon_tokens_norm.shape[-1],            # Igual dim que tokens
            activation=activation_name,                     # Activación principal
            kernel_regularizer=l2(l2_reg_value),            # L2 para regularización
            name="horizon_ffn_dense",                       # Nombre capa densa
        )(horizon_tokens_norm)                              # Aplica a tokens normalizados

        # Dropout en el bloque FFN
        ff_dense = Dropout(
            decoder_dropout,                                # Probabilidad de dropout
            name="horizon_ffn_dropout",                     # Nombre de la capa
        )(ff_dense)                                         # Aplica a salida de densa

        # Conexión residual FFN: tokens_norm + salida_ffa
        horizon_tokens_ffn_res = Add(
            name="horizon_ffn_residual",                    # Nombre de capa de suma
        )([horizon_tokens_norm, ff_dense])                  # Suma residual

        # Normalización final tras FFN
        horizon_tokens_final = LayerNormalization(
            name="horizon_ffn_ln",                          # Nombre de LN final
        )(horizon_tokens_ffn_res)                           # Aplica a residual

        # ------------------------------------------------------------------ #
        # 6) Cabeza de salida MIMO: 1 escalar por horizonte (batch, 1)      #
        # ------------------------------------------------------------------ #
        horizon_outputs = Dense(
            units=1,                                        # Un valor por horizonte
            activation=None,                                # Sin activación (regresión)
            name="horizon_output_dense",                    # Nombre de capa final MIMO
        )(horizon_tokens_final)                             # Aplica a tokens finales

        outputs: List[tf.Tensor] = []                       # Lista de salidas por horizonte
        self.output_names: List[str] = []                   # Lista de nombres de salida

        # Genera una salida por horizonte, manteniendo forma (batch, 1)
        for idx, h in enumerate(horizons):                  # Itera sobre índices y horizontes
            out_i = Lambda(
                lambda t, i=idx: t[:, i, :],                # Slice: mantiene eje final (batch,1)
                name=f"output_horizon_{h}",                 # Nombre de capa de salida
            )(horizon_outputs)                              # Aplica a tensor MIMO

            outputs.append(out_i)                           # Añade salida a la lista
            self.output_names.append(f"output_horizon_{h}") # Registra nombre de salida

        # ------------------------------------------------------------------ #
        # 7) Construcción y compilación del modelo                          #
        # ------------------------------------------------------------------ #
        self.model = Model(                                 # Crea el modelo Keras
            inputs=inputs,                                  # Entrada única
            outputs=outputs,                                # Lista de salidas
            name=f"MIMOPredictor_{len(horizons)}H",         # Nombre del modelo
        )

        optimizer = AdamW(                                  # Crea optimizador AdamW
            learning_rate=float(self.params.get("learning_rate", 1e-3)),  # LR
        )

        # Diccionario de pérdidas: MAE por cada salida
        loss_dict: Dict[str, Any] = {
            name: tf.keras.losses.MeanAbsoluteError()       # MAE estándar
            for name in self.output_names                   # Para cada salida
        }

        # Diccionario de métricas: mae_magnitude por salida (consistencia con sistema)
        metrics_dict: Dict[str, List[Any]] = {
            name: [mae_magnitude]                           # Lista de métricas
            for name in self.output_names                   # Para cada salida
        }

        # Compila el modelo con optimizador, pérdidas y métricas
        self.model.compile(
            optimizer=optimizer,                            # Optimizador AdamW
            loss=loss_dict,                                 # Diccionario de pérdidas
            metrics=metrics_dict,                           # Diccionario de métricas
        )

        # Muestra resumen del modelo para depuración
        self.model.summary(line_length=140)                 # Imprime summary en consola
