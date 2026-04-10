from setuptools import setup, find_packages

setup(
    name='predictor',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'predictor=app.main:main'
        ],
        # Plugins para el Predictor
        'predictor.plugins': [
            'default_predictor=predictor_plugins.predictor_plugin_ann:Plugin',
            'ann=predictor_plugins.predictor_plugin_ann:Plugin',
            'n_beats=predictor_plugins.predictor_plugin_n_beats:Plugin',
            'cnn=predictor_plugins.predictor_plugin_cnn:Plugin',
            'lstm=predictor_plugins.predictor_plugin_lstm:Plugin',
            'transformer=predictor_plugins.predictor_plugin_transformer:Plugin',
            'mimo=predictor_plugins.predictor_plugin_mimo:Plugin',
            'tft=predictor_plugins.predictor_plugin_tft:Plugin',
            'tcn=predictor_plugins.predictor_plugin_tcn:Plugin',
            'base=predictor_plugin.predictor_plugin_base:Plugin',
            # Binary classification plugins
            'binary_ann=predictor_plugins.binary.predictor_plugin_binary_ann:Plugin',
            'binary_cnn=predictor_plugins.binary.predictor_plugin_binary_cnn:Plugin',
            'binary_lstm=predictor_plugins.binary.predictor_plugin_binary_lstm:Plugin',
            'binary_transformer=predictor_plugins.binary.predictor_plugin_binary_transformer:Plugin',
            'binary_n_beats=predictor_plugins.binary.predictor_plugin_binary_n_beats:Plugin',
            'binary_tft=predictor_plugins.binary.predictor_plugin_binary_tft:Plugin',
            'binary_tcn=predictor_plugins.binary.predictor_plugin_binary_tcn:Plugin',
            'binary_mimo=predictor_plugins.binary.predictor_plugin_binary_mimo:Plugin',
            'binary_logistic=predictor_plugins.binary.predictor_plugin_binary_logistic:Plugin',
        ],
        # Plugins para la Optimización (por defecto, basado en DEAP)
        'optimizer.plugins': [
            'default_optimizer=optimizer_plugins.default_optimizer:Plugin',
            'neat_optimizer=optimizer_plugins.neat_optimizer:Plugin'
        ],
        # Plugins para el Pipeline (orquestación del flujo completo)
        'pipeline.plugins': [
            'default_pipeline=pipeline_plugins.default_pipeline:PipelinePlugin',
            'stl_pipeline=pipeline_plugins.stl_pipeline:STLPipelinePlugin',
            'binary_pipeline=pipeline_plugins.binary_pipeline:BinaryPipelinePlugin'
        ],
        # Plugins para el Preprocesamiento (incluye process_data, ventanas deslizantes y STL)
        'preprocessor.plugins': [
            'default_preprocessor=preprocessor_plugins.default_preprocessor:PreprocessorPlugin',
            'stl_preprocessor=preprocessor_plugins.stl_preprocessor:PreprocessorPlugin'
        ],
        'target.plugins': [
            'default_target=target_plugins.default_target:TargetPlugin',
            'stl_target=target_plugins.stl_target:TargetPlugin',
            'binary_target=target_plugins.binary_target:TargetPlugin',
        ],
        # Target plugins
        'target.plugins': [
            'default_target=target_plugins.default_target:TargetPlugin',
            'stl_target=target_plugins.stl_target:TargetPlugin',
            'binary_target=target_plugins.binary_target:TargetPlugin',
        ]
    },
    install_requires=[
        'build'
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description=(
        'A timeseries prediction system that supports dynamic loading of plugins for prediction, '
        'optimization, pipeline orchestration, and data pre-processing.'
    )
)
