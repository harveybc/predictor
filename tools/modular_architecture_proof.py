"""Real-Keras proof of the hourly default modular architecture (no fitting).

    CUDA_VISIBLE_DEVICES= python tools/modular_architecture_proof.py --features 3 7 --out <dir>

For each declared feature count F it builds ``default_config`` with real Keras,
writes ``shapes_F{F}.json`` (every layer of every component with its output
shape, plus the time grids) and ``graph_F{F}.dot`` (``keras.utils.model_to_dot``
with shapes, nested components expanded). Rendering DOT to PNG needs only the
graphviz ``dot`` binary (``dot -Tpng graph_F3.dot -o graph_F3.png``), so the
PNG can be produced on a host without TensorFlow. FeatureSelect nodes are
labelled "column routing (tf.gather)": they are fixed channel routing, not a
learned selector.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def layer_rows(model, prefix=""):
    rows = []
    for layer in model.layers:
        shape = getattr(layer, "output", None)
        shape = None if shape is None else (
            [list(t.shape) for t in shape] if isinstance(shape, (list, tuple)) else list(shape.shape))
        rows.append({"layer": prefix + layer.name, "class": type(layer).__name__, "output_shape": shape,
                     "weights": int(sum(w.numpy().size for w in layer.weights))})
        if hasattr(layer, "layers") and layer.layers and type(layer).__name__ in ("Functional", "Model"):
            rows.extend(layer_rows(layer, prefix + layer.name + "/"))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, nargs="+", default=[3, 7])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import keras
    from keras.src.utils import model_visualization as mv
    from predictor_plugins import modular_temporal as mt
    mv.check_graphviz = lambda: None        # DOT text only; rendering happens with the dot binary
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.features:
        names = [f"feature_{i:03d}" for i in range(f)]
        b = mt.build_modular(mt.default_config(names))
        report = {"F": f, "keras_version": keras.__version__,
                  "input": list(b.forecast_model.input.shape),
                  "branch_outputs": {n: list(m.output.shape) for n, m in b.branch_models.items()},
                  "fused": list(b.fusion_model.output.shape),
                  "latent": list(b.encoder_model.output.shape),
                  "forecast": list(b.forecast_model.output.shape),
                  "branch_time_grid": list(b.branch_time_grid), "core_time_grid": list(b.core_time_grid),
                  "core_layers": layer_rows(b.core_model),
                  "forecast_model_layers": layer_rows(b.forecast_model),
                  "parameters": int(b.forecast_model.count_params())}
        (out / f"shapes_F{f}.json").write_text(json.dumps(report, indent=1) + "\n")
        dot = keras.utils.model_to_dot(b.forecast_model, show_shapes=True, show_layer_names=True,
                                       expand_nested=True, dpi=96)
        text = dot.to_string().replace("FeatureSelect", "column routing (tf.gather)")
        (out / f"graph_F{f}.dot").write_text(text)
        print(json.dumps({k: report[k] for k in ("F", "input", "fused", "latent", "forecast", "parameters")}))
        for row in report["core_layers"]:
            print(f"  core {row['layer']:<36} {row['class']:<22} {row['output_shape']}")


if __name__ == "__main__":
    main()
