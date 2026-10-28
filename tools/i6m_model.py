"""General clock-length adapter; the sealed compact factory remains unchanged.

Branches preserve chronology through fusion, positional encoding, full residual
Transformer blocks and the three causal channel-reduction stages. Only the
forecast head selects the last position. Capacity is held fixed across clocks.
"""

def build_model(samples, n_features, spacing_hours):
    """Build the 34,921-parameter default for 20 features and 3..241 samples."""
    import tensorflow as tf
    from predictor_plugins.modular_temporal.components import causal_conv1d
    from tools.i6e_masked_model import ClockView

    if (type(samples) is not int or not 3 <= samples <= 241
            or type(n_features) is not int or n_features < 1
            or type(spacing_hours) is not int or spacing_hours < 1):
        raise ValueError("invalid temporal model dimensions")
    K = tf.keras
    inp = K.Input((samples, 2*n_features+1), name="clock_input")
    mask = ClockView("mask", name="market_presence")(inp)
    clean = ClockView("clean", name="ignore_placeholders")(inp)
    grid = tuple((i-samples+1)*spacing_hours for i in range(samples))
    parts = []
    for index in range(n_features):
        feature = ClockView("feature", index=index, name=f"feature_{index}")(clean)
        branch = causal_conv1d(input_shape=(samples, 2), time_grid=grid,
                              output_steps=samples, name=f"branch_{index}",
                              params=dict(channels=16, kernel_size=3))
        parts.append(ClockView("gate", name=f"branch_mask_{index}")([branch.model(feature), mask]))
    fused = K.layers.Concatenate(name="fusion")(parts) if len(parts) > 1 else parts[0]
    x = ClockView("position", spacing_hours=float(spacing_hours), name="physical_position")(fused)
    x = K.layers.Conv1D(32, 1, name="core_projection")(x)
    x = ClockView("gate", name="core_entry_mask")([x, mask])
    attention_mask = ClockView("attention", name="causal_market_attention")(mask)
    for block in range(2):
        attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0., name=f"attention_{block}")(x, x, attention_mask=attention_mask)
        x = K.layers.LayerNormalization(name=f"attention_norm_{block}")(K.layers.Add(name=f"attention_add_{block}")([x, attention]))
        ff = K.layers.Dense(64, activation="gelu", name=f"ff_expand_{block}")(x)
        ff = K.layers.Dense(32, name=f"ff_project_{block}")(ff)
        x = K.layers.LayerNormalization(name=f"ff_norm_{block}")(K.layers.Add(name=f"ff_add_{block}")([x, ff]))
        x = ClockView("gate", name=f"transformer_mask_{block}")([x, mask])
    for stage, channels in enumerate((24, 16, 8)):
        residual = K.layers.Conv1D(channels, 1, name=f"core_skip_{stage}")(x)
        update = K.layers.Conv1D(channels, 3, padding="causal", activation="gelu", name=f"core_conv_{stage}")(x)
        x = K.layers.LayerNormalization(name=f"core_norm_{stage}")(K.layers.Add(name=f"core_add_{stage}")([residual, update]))
        x = ClockView("gate", name=f"core_mask_{stage}")([x, mask])
    output = K.layers.Dense(1, name="forecast_head")(ClockView("last", name="head_input")(x))
    return K.Model(inp, output, name="I6M_masked_temporal")
