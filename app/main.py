#!/usr/bin/env python3
"""
main.py

Punto de entrada de la aplicación de predicción de EUR/USD. Este script orquesta:
    - La carga y fusión de configuraciones (CLI, archivos locales y remotos).
    - La inicialización de los plugins: Predictor, Optimizer, Pipeline y Preprocessor.
    - La selección entre ejecutar la optimización de hiperparámetros o entrenar y evaluar directamente.
    - El guardado de la configuración resultante de forma local y/o remota.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Quiet mode: suppress verbose output when PREDICTOR_QUIET=1 or --quiet flag
# Only allows ERROR/WARN/final-metric lines through. Progress bars are killed
# by setting verbose=0 on model.fit (handled in common/base.py).
# ---------------------------------------------------------------------------
import builtins
_original_print = builtins.print

def _quiet_print(*args, **kwargs):
    """Filtered print that only passes through important messages."""
    if args:
        msg = str(args[0])
        # Always allow errors, warnings, and final metrics
        _pass = any(kw in msg.upper() for kw in [
            'ERROR', 'WARN', 'EXCEPTION', 'TRACEBACK', 'FATAL',
            'FINAL', 'BEST VAL', 'TEST MAE', 'VAL MAE', 'RESULT',
            'IMPROVEMENT', 'VERDICT', 'SUMMARY',
        ])
        if _pass:
            _original_print(*args, **kwargs)
        return
    _original_print(*args, **kwargs)

if os.environ.get('PREDICTOR_QUIET', '0') == '1' or '--quiet' in sys.argv:
    builtins.print = _quiet_print
import json
import pandas as pd
from typing import Any, Dict
from pathlib import Path

from app.config_handler import (
    load_config,
    save_config,
    remote_load_config,
    remote_save_config,
    remote_log
)
from app.cli import parse_args
from app.config import DEFAULT_VALUES
from app.plugin_loader import load_plugin
from config_merger import merge_config, process_unknown_args

# Se asume que los siguientes plugins se cargan desde sus respectivos namespaces:
# - predictor.plugins
# - optimizer.plugins
# - pipeline.plugins
# - preprocessor.plugins


def _repo_root() -> Path:
    # main.py -> app/ -> <repo_root>
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(p: Any) -> str | None:
    if not p:
        return None
    try:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((_repo_root() / pp).resolve())
    except Exception:
        return str(p)


def _ensure_csv_header(path: str, header_line: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header_line.rstrip("\n") + "\n")
            f.flush()
            os.fsync(f.fileno())


def _configure_tensorflow_memory() -> None:
    """Import TensorFlow and make GPU allocation predictable.

    C34: this is EXECUTION — it loads a framework and enumerates
    devices — so it is called only on the execution side of the gate,
    never during SUBMIT_ONLY.
    """
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    try:
        import tensorflow as tf  # noqa: WPS433 (runtime import intentional)

        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                # If TF already initialized somewhere, we can't change
                # this. Keep going; the env var above still helps.
                pass
        if gpus:
            print("TensorFlow GPU memory growth configured for "
                  f"{len(gpus)} GPU(s).")
    except Exception as e:
        print(f"INFO: TensorFlow memory configuration skipped: {e}")


def _validate_logging_config(config: Dict[str, Any]) -> None:
    """Fail-fast-ish validation so remote runs never silently produce 'no logs'."""
    mem_log = _resolve_repo_path(config.get("memory_log_file"))
    opt_log = _resolve_repo_path(config.get("optimizer_resource_log_file"))
    if mem_log:
        config["memory_log_file"] = mem_log
    if opt_log:
        config["optimizer_resource_log_file"] = opt_log

    print(
        "[LOGGING_CONFIG] "
        f"memory_log_file={config.get('memory_log_file')} "
        f"optimizer_resource_log_file={config.get('optimizer_resource_log_file')} "
        f"memory_log_gpu={config.get('memory_log_gpu')} memory_log_gc={config.get('memory_log_gc')} "
        f"max_rss_gb={config.get('max_rss_gb')} max_rss_mb={config.get('max_rss_mb')}"
    )

    # Ensure files are writable and have headers (so tail/grep always works).
    try:
        if mem_log:
            _ensure_csv_header(
                mem_log,
                "ts,epoch,tag,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,gc0,gc1,gc2",
            )
    except Exception as e:
        print(f"[LOGGING_CONFIG] WARN: cannot initialize memory_log_file: {e}")

    try:
        if opt_log:
            _ensure_csv_header(
                opt_log,
                "ts,stage,generation,candidate,VmRSS_kB,VmHWM_kB,gpu_current_B,gpu_peak_B,extra",
            )
    except Exception as e:
        print(f"[LOGGING_CONFIG] WARN: cannot initialize optimizer_resource_log_file: {e}")

def main():
    """C22: wrap the WHOLE run so exactly one durable terminal is
    emitted on every exit path — complete, failed, inconclusive,
    refused or quarantined.

    The previous emitter fired only after the pipeline returned,
    so an exception, a refusal or an inconclusive outcome left the
    cube with no record that the run had happened at all.
    """
    from olap.terminal import terminal_run

    shared: Dict[str, Any] = {}

    def _body():
        return _run_main(shared)

    # C13: the campaign is the EXPERIMENT. Two runs of one
    # experiment share it; two experiments never do.
    def _campaign_key():
        cfg = shared.get("config", {})
        name = cfg.get("load_config") or "predictor_run"
        return f"predictor::{Path(str(name)).stem}"

    campaign_key = "predictor_run"
    try:
        return terminal_run(
            _body, campaign_key=_campaign_key,
            producer="predictor",
            # read at FINISH time: the run builds its config
            # part-way through, and a snapshot taken now would
            # report UNAVAILABLE for everything it did
            config=lambda: shared.get("config", {}),
            # C36: lazy for the same reason the config is. The run
            # only knows where it writes once its configuration is
            # merged, so reading this now would freeze None and the
            # operational-gap file would have nowhere to land — in
            # exactly the situation the file exists for.
            results_dir=lambda: shared.get("results_dir"))
    finally:
        pass


def _run_main(shared: Dict[str, Any]):
    """
    Orquesta la ejecución completa del sistema, incluyendo la optimización (si se configura)
    y la ejecución del pipeline completo (preprocesamiento, entrenamiento, predicción y evaluación).
    """
    print("Parsing initial arguments...")
    args, unknown_args = parse_args()
    cli_args: Dict[str, Any] = vars(args)

    # C34 (order 2026-09-11): importing TensorFlow and enumerating
    # devices is EXECUTION. It used to happen here, before the gate, so
    # a phase the report called "nothing was executed" had already
    # loaded a framework and touched the accelerator. It now runs after
    # the gate, and never at all in SUBMIT_ONLY.

    print("Loading default configuration...")
    config: Dict[str, Any] = DEFAULT_VALUES.copy()

    file_config: Dict[str, Any] = {}
    # Carga remota de configuración si se solicita
    if args.remote_load_config:
        try:
            file_config = remote_load_config(args.remote_load_config, args.username, args.password)
            print(f"Loaded remote config: {file_config}")
        except Exception as e:
            print(f"Failed to load remote configuration: {e}")
            sys.exit(1)

    # Carga local de configuración si se solicita
    if args.load_config:
        try:
            file_config = load_config(args.load_config)
            print(f"Loaded local config: {file_config}")
        except Exception as e:
            print(f"Failed to load local configuration: {e}")
            sys.exit(1)

    # Primera fusión de la configuración (sin parámetros específicos de plugins)
    print("Merging configuration with CLI arguments and unknown args (first pass, no plugin params)...")
    unknown_args_dict = process_unknown_args(unknown_args)
    config = merge_config(config, {}, {}, file_config, cli_args, unknown_args_dict)

    # Selección del plugins
    if not cli_args.get('predictor_plugin'):
        cli_args['predictor_plugin'] = config.get('predictor_plugin', 'default_predictor')
    # C31: ONE resolver names the predictor, and it refuses BEFORE any
    # model object exists if the legacy `plugin` key disagrees with the
    # canonical `predictor_plugin`. That disagreement used to train one
    # architecture while the eligibility identity recorded another.
    from app.plugin_resolver import canonical_name
    plugin_name = (canonical_name(config, "predictor")
                   or 'default_predictor')
    config['predictor_plugin'] = plugin_name
    
    
    # --- RESOLUCIÓN DE PLUGINS (sin construir nada) ---
    # C34: SUBMIT_ONLY is a phase of INSPECTION. It may read the
    # config, the headers, the bytes and a component's declared
    # metadata; it may not build the components. Every plugin's
    # `plugin_params` is a CLASS attribute, so the contract those
    # defaults belong to can be merged without instantiating anything
    # — which is what the five constructors and their set_params calls
    # used to do before the gate was ever asked.
    from app.plugin_resolver import declared_plugin_params, resolve
    PLUGIN_ORDER = (
        ("predictor", 'predictor_plugin', 'default_predictor'),
        ("optimizer", 'optimizer_plugin', 'default_optimizer'),
        ("pipeline", 'pipeline_plugin', 'default_pipeline'),
        ("target", 'target_plugin', 'default_target'),
        ("preprocessor", 'preprocessor_plugin', 'default_preprocessor'),
    )
    witnesses: Dict[str, Any] = {}
    for role, key, default in PLUGIN_ORDER:
        name = plugin_name if role == "predictor" \
            else config.get(key, default)
        print(f"Resolving {role} plugin: {name}")
        try:
            witnesses[role] = resolve(role, name)
        except SystemExit:
            raise
        except Exception as e:
            print(f"Failed to resolve {role} plugin '{name}': {e}")
            sys.exit(1)

    # fusión de configuración con los parámetros DECLARADOS por cada
    # plugin — leídos del código fuente con `ast`, sin importar el
    # módulo. Importarlo cargaría TensorFlow, y eso es ejecución.
    print("Merging configuration with CLI arguments and unknown args "
          "(second pass, with declared plugin params)...")
    for role, _key, _default in PLUGIN_ORDER:
        config = merge_config(config,
                              declared_plugin_params(witnesses[role]),
                              {}, file_config, cli_args,
                              unknown_args_dict)


    # --- ELIGIBILITY GATE (order C1) ---
    # Asked BEFORE the optimizer, before any pipeline, before any
    # window is built and before any model is fitted or loaded.
    # The previous version stood after the optimizer and still
    # called itself the single choke point; it was not.
    from eligibility.integration import (
        STATUS_SUBMITTED, assert_contract_unchanged,
        assert_optimizer_result_is_hyperparameters_only,
        describe, gate_run)

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    # the terminal wrapper needs the LIVE config object, so it can
    # report the eligibility stamp and results file of whatever
    # outcome occurs
    shared["config"] = config
    shared["results_dir"] = Path(
        str(config.get("results_file", "."))).parent
    eligibility_stamp = gate_run(
        config, repo_root=_REPO_ROOT,
        consumer="predictor.main")
    print(describe(eligibility_stamp))

    # C17 phase one stops here: the submission is persisted and
    # nothing downstream observes or alters the bytes under
    # review.
    if eligibility_stamp.get("eligibility_status") == \
            STATUS_SUBMITTED:
        print("SUBMIT_ONLY complete — submission "
              f"{eligibility_stamp['submission_sha256'][:12]} "
              f"written to {eligibility_stamp['submission_file']}"
              "; nothing was executed. Hand it to the reviewer, "
              "then re-run with execution_purpose="
              "EXECUTE_REVIEWED and "
              "eligibility_submission_sha256 set.")
        return

    # ==================================================================
    # Everything below this line EXECUTES. Nothing above it may.
    # ==================================================================
    _configure_tensorflow_memory()

    # Validate logging destinations — this CREATES files, so it is on
    # the execution side of the line.
    _validate_logging_config(config)

    # --- CONSTRUCCIÓN DE PLUGINS ---
    # The classes are loaded from the SAME witnesses the identity
    # bound, so what runs is what was reviewed.
    from app.plugin_resolver import load_from_witness
    try:
        classes = {role: load_from_witness(witnesses[role])
                   for role, _k, _d in PLUGIN_ORDER}
        predictor_plugin = classes["predictor"](config)
        predictor_plugin.set_params(**config)
        optimizer_plugin = classes["optimizer"]()
        optimizer_plugin.set_params(**config)
        pipeline_plugin = classes["pipeline"]()
        pipeline_plugin.set_params(**config)
        target_plugin = classes["target"]()
        target_plugin.set_params(**config)
        preprocessor_plugin = classes["preprocessor"]()
        preprocessor_plugin.set_params(**config)
    except Exception as e:
        print(f"Failed to initialize plugins: {e}")
        sys.exit(1)

    # --- DECISIÓN DE EJECUCIÓN ---
    if config.get('use_optimizer', False) and not config.get('load_model', False):
        print("Running hyperparameter optimization with Optimizer Plugin...")
        try:
            optimal_params = optimizer_plugin.optimize(predictor_plugin, preprocessor_plugin, config)
            # C19: an optimizer proposes hyperparameters. It may
            # not change the data, partitions, target, plugins or
            # authority that were approved before it ran.
            optimal_params = \
                assert_optimizer_result_is_hyperparameters_only(
                    optimal_params, consumer="predictor.main")
            optimizer_output_file = config.get("optimizer_output_file", "optimizer_output.json")
            with open(optimizer_output_file, "w") as f:
                json.dump(optimal_params, f, indent=4)
            print(f"Optimized parameters saved to {optimizer_output_file}.")
            config.update(optimal_params)
        except Exception as e:
            print(f"Hyperparameter optimization failed: {e}")
            sys.exit(1)
    else:
        if not config.get('use_optimizer', False):
            print("Skipping hyperparameter optimization.")
        print("Running prediction pipeline...")

    # C19: re-derive the whole identity and require equality with
    # what the gate approved, immediately before anything is
    # consumed. Approval before a change does not cover what
    # would run now.
    assert_contract_unchanged(config, repo_root=_REPO_ROOT,
                              consumer="predictor.main")

    # Pipeline Plugin orchestrates preprocessing, training (or model loading), evaluation
    pipeline_plugin.run_prediction_pipeline(
        config,
        predictor_plugin,
        preprocessor_plugin,
        target_plugin
    )
        
    # Guardado de la configuración local y remota
    if config.get('save_config'):
        try:
            save_config(config, config['save_config'])
            print(f"Configuration saved to {config['save_config']}.")
        except Exception as e:
            print(f"Failed to save configuration locally: {e}")

    if config.get('remote_save_config'):
        print(f"Remote saving configuration to {config['remote_save_config']}")
        try:
            remote_save_config(config, config['remote_save_config'], config.get('username'), config.get('password'))
            print("Remote configuration saved.")
        except Exception as e:
            print(f"Failed to save configuration remotely: {e}")

if __name__ == "__main__":
    main()
