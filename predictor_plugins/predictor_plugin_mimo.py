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
        con el preprocesador, target y pipeline actuales.

Esta clase respeta la interfaz de BaseBayesianKerasPredictor:
  - plugin_params
  - plugin_debug_vars
  - build_model(input_shape, x_train, config)

No modifica el pipeline ni el preprocesador existentes.
"""

from __future__ import annotations  # Permite anotaciones de tipos hacia adelante

# Import estándar de typing para anotaciones
from typing import Any, Dict, List, Tuple

# Import de TensorFlow y Keras
import tensorflow as tf  # Import principal de TensorFlow
from tensorflow.keras.layers import (  # Capas que vamos a utilizar
    Input,                             # Entrada del modelo
    Dense,                             # Capas densas totalmente conectadas
    Conv1D,                            # Convolución 1D para el encoder
    Bidirectional,                     # LSTM bidireccional
    LSTM,                              # LSTM unidireccional
    GlobalAveragePooling1D,            # Pooling global sobre el eje temporal
    LayerNormalization,                # Normalización para el bloque de atención
    Dropout,                           # Dropout para regularización
    Add,                               # Suma residual
    Lambda,                            # Capas Lambda para operaciones personalizadas
    Concatenate,                       # Concatenación de tensores
    MultiHeadAttention,                # Atención multi-cabeza
)
from tensorflow.keras.models import Model             # Clase base de modelos Keras
from tensorflow.keras.optimizers import AdamW         # Optimizador AdamW
from tensorflow.keras.regularizers import l2          # Regularizador L2

# Import de utilidades ya existentes en tu proyecto
from .common.base import BaseBayesianKerasPredictor   # Clase base de predictores bayesianos
from .common.positional_encoding import positional_encoding  # Codificación posicional
from .common.losses import mae_magnitude              # Métrica de MAE ya definida en tu proyecto


class Plugin(BaseBayesianKerasPredictor):
    """
    Plugin MIMO multi-horizonte para tu sistema de predicción.

    Este plugin:
      - Construye un modelo único con un encoder compartido para todas las salidas.
      - Implementa un decoder de horizontes basado en embeddings de horizonte
        y un bloque de atención multi-cabeza sobre el eje horizonte.
      - Genera una salida por horizonte con nombre "output_horizon_{h}" para
        mantener compatibilidad con el target plugin, preprocesador y pipeline.
    """

    # Parámetros por defecto del plugin (tuneables por DEAP vía config JSON)
    plugin_params: Dict[str, Any] = {
        # Parámetros de entrenamiento
        "batch_size": 32,                 # Tamaño de batch por defecto
        "learning_rate": 1e-3,            # Tasa de aprendizaje para AdamW

        # Hiperparámetros del encoder
        "encoder_conv_layers": 2,         # Número de capas Conv1D en el encoder
        "encoder_base_filters": 64,       # Número de filtros base de la primera capa Conv1D
        "encoder_lstm_units": 64,         # Unidades de la BiLSTM del encoder

        # Hiperparámetros del decoder de horizontes
        "horizon_embedding_dim": 16,      # Dimensión de los embeddings de horizonte
        "horizon_attn_heads": 4,          # Número de cabezas de MultiHeadAttention en el decoder
        "horizon_attn_key_dim": 32,       # Dimensión de la clave en MultiHeadAttention
        "decoder_dropout": 0.1,           # Dropout en el bloque de decoder

        # Regularización y activación
        "activation": "relu",             # Activación principal (e.g., "relu" o "gelu")
        "l2_reg": 1e-6,                   # Coeficiente de regularización L2

        # Listado de horizontes (se sobreescribe desde el JSON de config)
        "predicted_horizons": [1],        # Placeholder, se reemplaza con la lista real

        # Positional encoding opcional (mismo patrón que en el plugin CNN)
        "positional_encoding": False,     # Si True, se aplica encoding posicional a la entrada
    }

    # Variables de depuración que se pueden inspeccionar externamente
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

        Parámetros
        ----------
        input_shape : Tuple[int, int]
            Forma de entrada (window_size, num_features).
        x_train : Any
            Datos de entrenamiento (no usado directamente aquí, pero se mantiene
            para compatibilidad con la firma actual).
        config : Dict[str, Any]
            Diccionario de configuración global; se mezcla con plugin_params.
        """
        # Mezcla de parámetros del plugin con la configuración externa
        if config is not None:
            # Actualiza el diccionario de parámetros internos con la config
            self.params.update(config)

        # Desempaqueta la forma de la entrada (ventana temporal y nº de features)
        window_size, num_features = input_shape

        # Lista de horizontes objetivo (se asume que viene del JSON ya consistente)
        horizons: List[int] = list(self.params.get("predicted_horizons", []))

        # Validación básica: no seguimos si no hay horizontes definidos
        if not horizons:
            raise ValueError(
                "MIMO predictor: 'predicted_horizons' está vacío; "
                "debes definir al menos un horizonte en la configuración."
            )

        # Ordenamos los horizontes para asegurar un orden estable
        horizons = sorted(horizons)

        # Activación principal
        activation_name: str = self.params.get("activation", "relu")

        # Coeficiente de regularización L2
        l2_reg_value: float = float(self.params.get("l2_reg", 1e-6))

        # --- 1) Capa de entrada -------------------------------------------------
        # Entrada de forma (window_size, num_features)
        inputs = Input(
            shape=(window_size, num_features),
            name="input_window",
        )

        # --- 2) Positional Encoding opcional ------------------------------------
        # Si se activa positional_encoding, se suma un tensor fijo senoidal/cosenoidal
        if self.params.get("positional_encoding", False):
            # Se obtiene el tensor de encoding posicional del helper existente
            pe_tensor = positional_encoding(window_size, num_features)
            # Se suma vía una capa Lambda para que quede en el grafo de Keras
            x = Lambda(
                lambda t, pe=pe_tensor: t + pe,
                name="add_positional_encoding",
            )(inputs)
        else:
            # Sin positional encoding, se usa la entrada directamente
            x = inputs

        # --- 3) Encoder compartido (Conv1D stack + BiLSTM + Global Pool) -------
        # Número de capas conv en el encoder
        num_conv_layers: int = int(self.params.get("encoder_conv_layers", 2))
        # Filtros base de la primera capa
        base_filters: int = int(self.params.get("encoder_base_filters", 64))

        # Bucle de capas Conv1D (causal) para capturar patrones locales/multi-escala
        for layer_idx in range(num_conv_layers):
            # Filtros decrecientes por capa para controlar capacidad y coste
            filters = max(8, base_filters // (2 ** layer_idx))
            # Capa Conv1D con padding causal, activación seleccionada y L2
            x = Conv1D(
                filters=filters,
                kernel_size=3,
                padding="causal",
                activation=activation_name,
                kernel_regularizer=l2(l2_reg_value),
                name=f"enc_conv_{layer_idx+1}",
            )(x)

        # Unidades de la BiLSTM
        lstm_units: int = int(self.params.get("encoder_lstm_units", 64))

        # Capa BiLSTM para capturar dependencias temporales en ambas direcciones
        x_seq = Bidirectional(
            LSTM(
                lstm_units,
                return_sequences=True,
                name="enc_lstm",
            ),
            name="enc_bilstm",
        )(x)

        # Pooling global promedio sobre el eje temporal para obtener z_global
        z_global = GlobalAveragePooling1D(name="enc_global_avg_pool")(x_seq)

        # --- 4) Construcción de tokens de horizonte -----------------------------
        # Número total de horizontes
        num_horizons: int = len(horizons)

        # Calculamos el máximo horizonte para definir el rango del embedding
        max_horizon: int = int(max(horizons))

        # Dimensión de los embeddings de horizonte
        horizon_emb_dim: int = int(self.params.get("horizon_embedding_dim", 16))

        # Constante con los ids de horizonte (e.g. [4, 8, 12, 16, 20, 24, ...])
        horizon_ids = tf.constant(horizons, dtype=tf.int32, name="horizon_ids")

        # Capa de embeddings de horizonte (input_dim = max_horizon+1 por seguridad)
        horizon_embedding_layer = tf.keras.layers.Embedding(
            input_dim=max_horizon + 1,
            output_dim=horizon_emb_dim,
            name="horizon_embedding",
        )

        # Embeddings de horizonte de forma (num_horizons, horizon_emb_dim)
        horizon_embs = horizon_embedding_layer(horizon_ids)

        # Expande a (1, num_horizons, horizon_emb_dim) para poder replicar por batch
        horizon_embs_expanded = Lambda(
            lambda e: tf.expand_dims(e, axis=0),
            name="expand_horizon_embs",
        )(horizon_embs)

        # Replica los embeddings de horizonte a lo largo del batch
        horizon_embs_tiled = Lambda(
            lambda e: tf.tile(
                e,
                [tf.shape(z_global)[0], 1, 1],
            ),
            name="tile_horizon_embs",
        )(horizon_embs_expanded)

        # Expande z_global para tener dimensión de horizonte: (batch, 1, latent_dim)
        z_expanded = Lambda(
            lambda z: tf.expand_dims(z, axis=1),
            name="expand_global",
        )(z_global)

        # Replica z_global a lo largo de todos los horizontes: (batch, num_horizons, latent_dim)
        z_tiled = Lambda(
            lambda z: tf.tile(z, [1, num_horizons, 1]),
            name="tile_global",
        )(z_expanded)

        # Concatena z_global y el embedding de horizonte en el eje de features
        # Resultado: (batch, num_horizons, latent_dim + horizon_emb_dim)
        horizon_tokens = Concatenate(
            axis=-1,
            name="concat_global_horizon",
        )([z_tiled, horizon_embs_tiled])

        # --- 5) Bloque de atención multi-cabeza + feed-forward -----------------
        # Hiperparámetros del bloque de atención
        attn_heads: int = int(self.params.get("horizon_attn_heads", 4))
        attn_key_dim: int = int(self.params.get("horizon_attn_key_dim", 32))
        decoder_dropout: float = float(self.params.get("decoder_dropout", 0.1))

        # Capa de MultiHeadAttention sobre el eje horizonte (query=key=value=horizon_tokens)
        attn_output = MultiHeadAttention(
            num_heads=attn_heads,
            key_dim=attn_key_dim,
            name="horizon_mha",
        )(horizon_tokens, horizon_tokens)

        # Residual: suma de la salida de atención con los tokens originales
        horizon_tokens_res = Add(name="horizon_attn_residual")(
            [horizon_tokens, attn_output]
        )

        # Normalización por capas tras la atención
        horizon_tokens_norm = LayerNormalization(name="horizon_attn_ln")(
            horizon_tokens_res
        )

        # Bloque feed-forward (MLP) aplicado sobre cada token de horizonte
        ff_dense = Dense(
            units=horizon_tokens_norm.shape[-1],
            activation=activation_name,
            kernel_regularizer=l2(l2_reg_value),
            name="horizon_ffn_dense",
        )(horizon_tokens_norm)

        # Dropout en el bloque feed-forward
        ff_dense = Dropout(decoder_dropout, name="horizon_ffn_dropout")(ff_dense)

        # Residual nuevamente (FFN + residual)
        horizon_tokens_ffn_res = Add(name="horizon_ffn_residual")(
            [horizon_tokens_norm, ff_dense]
        )

        # Normalización final tras el bloque FFN
        horizon_tokens_final = LayerNormalization(
            name="horizon_ffn_ln",
        )(horizon_tokens_ffn_res)

        # --- 6) Cabeza de salida MIMO (una salida escalar por horizonte) -------
        # Capa densa final que produce 1 valor por horizonte (log-retorno esperado)
        # Salida: (batch, num_horizons, 1)
        horizon_outputs = Dense(
            units=1,
            activation=None,
            name="horizon_output_dense",
        )(horizon_tokens_final)

        # Ahora separamos cada horizonte en una salida independiente nombrada
        outputs: List[tf.Tensor] = []
        self.output_names: List[str] = []

        for idx, h in enumerate(horizons):
            # Lambda para extraer la componente correspondiente al horizonte idx
            # Resultado: (batch,) para cada horizonte
            out_i = Lambda(
                lambda t, i=idx: tf.squeeze(t[:, i, :], axis=-1),
                name=f"output_horizon_{h}",
            )(horizon_outputs)

            # Añadimos a la lista de salidas del modelo
            outputs.append(out_i)

            # Registramos el nombre de la salida para pérdidas y métricas
            self.output_names.append(f"output_horizon_{h}")

        # --- 7) Construcción y compilación del modelo -------------------------
        # Crea el modelo Keras con entrada única y lista de salidas
        self.model = Model(
            inputs=inputs,
            outputs=outputs,
            name=f"MIMOPredictor_{len(horizons)}H",
        )

        # Optimizador AdamW con tasa de aprendizaje configurable
        optimizer = AdamW(
            learning_rate=float(self.params.get("learning_rate", 1e-3))
        )

        # Definición de la función de pérdida por salida
        # Usamos MAE estándar por horizonte (estable y fácil de interpretar)
        loss_dict: Dict[str, Any] = {}
        for name in self.output_names:
            # Asignamos la misma función de pérdida MAE para cada salida
            loss_dict[name] = tf.keras.losses.MeanAbsoluteError()

        # Métricas: reutilizamos mae_magnitude para ser consistentes con el sistema actual
        metrics_dict: Dict[str, List[Any]] = {
            name: [mae_magnitude] for name in self.output_names
        }

        # Compilamos el modelo con el optimizador, pérdidas y métricas definidas
        self.model.compile(
            optimizer=optimizer,
            loss=loss_dict,
            metrics=metrics_dict,
        )

        # Imprime un resumen del modelo para depuración
        self.model.summary(line_length=140)
