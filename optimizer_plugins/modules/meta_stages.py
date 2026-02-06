"""
Meta-Optimization Stage Definitions for Level 2 → Level 3 Training

Defines the 8-stage incremental parameter deployment for hierarchical optimization.
Each stage unlocks a specific group of parameters while others remain at base values.
"""

# 8-Stage Incremental Parameter Deployment
META_OPTIMIZATION_STAGES = {
    1: {
        "name": "Base Architecture",
        "description": "Core model structure and capacity",
        "parameters": [
            "encoder_conv_layers",
            "encoder_base_filters", 
            "encoder_lstm_units",
            "intermediate_layers"
        ]
    },
    2: {
        "name": "Geometry & Scaling",
        "description": "Layer sizing and scaling patterns",
        "parameters": [
            "initial_layer_size",
            "layer_size_divisor"
        ]
    },
    3: {
        "name": "Attention & Temporality",
        "description": "Attention mechanisms and temporal encoding",
        "parameters": [
            "window_size",
            "positional_encoding",
            "horizon_attn_heads",
            "horizon_attn_key_dim",
            "horizon_embedding_dim"
        ]
    },
    4: {
        "name": "Training Dynamics",
        "description": "Learning rate, activation, batch configuration",
        "parameters": [
            "learning_rate",
            "activation",
            "batch_size"
        ]
    },
    5: {
        "name": "Regularization & Stability",
        "description": "Overfitting prevention and training stability",
        "parameters": [
            "l2_reg",
            "decoder_dropout",
            "min_delta"
        ]
    },
    6: {
        "name": "Stochasticity/Bayesian",
        "description": "Uncertainty quantification and Bayesian components",
        "parameters": [
            "kl_weight",
            "kl_anneal_epochs",
            "mc_samples"
        ]
    },
    7: {
        "name": "Signal Pipelines",
        "description": "Feature engineering and decomposition methods",
        "parameters": [
            "use_stl",
            "stl_period",
            "use_wavelets",
            "use_multi_tapper",
            "use_log1p_features"
        ]
    },
    8: {
        "name": "Strategy & Horizons",
        "description": "Prediction horizons and evaluation strategies",
        "parameters": [
            "predicted_horizons",
            "max_steps_train",
            "max_steps_val",
            "max_steps_test",
            "use_predicted_decompositions",
            "use_real_decompositions",
            "use_ideal_predictions"
        ]
    }
}


def get_active_parameters_for_stage(stage):
    """
    Get cumulative list of parameters active up to and including the specified stage.
    
    Args:
        stage (int): Current stage (1-8)
        
    Returns:
        list: All parameter names active in this stage (cumulative)
    """
    active_params = []
    for s in range(1, stage + 1):
        if s in META_OPTIMIZATION_STAGES:
            active_params.extend(META_OPTIMIZATION_STAGES[s]["parameters"])
    return active_params


def get_all_meta_parameters():
    """Get complete ordered list of all 27 meta-optimization parameters."""
    all_params = []
    for stage in sorted(META_OPTIMIZATION_STAGES.keys()):
        all_params.extend(META_OPTIMIZATION_STAGES[stage]["parameters"])
    return all_params


def get_stage_info(stage):
    """Get name and description for a stage."""
    if stage in META_OPTIMIZATION_STAGES:
        return META_OPTIMIZATION_STAGES[stage]["name"], META_OPTIMIZATION_STAGES[stage]["description"]
    return "Unknown", ""


def get_total_stages():
    """Get total number of stages."""
    return len(META_OPTIMIZATION_STAGES)


def get_new_parameters_in_stage(stage):
    """Get only the NEW parameters introduced in this specific stage."""
    if stage in META_OPTIMIZATION_STAGES:
        return META_OPTIMIZATION_STAGES[stage]["parameters"]
    return []
