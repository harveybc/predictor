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
            'base=predictor_plugin.predictor_plugin_base:Plugin'
        ],
        # Plugins para la Optimización (por defecto, basado en DEAP)
        'optimizer.plugins': [
            'default_optimizer=optimizer_plugins.default_optimizer:Plugin'
        ],
        # Plugins para el Pipeline (orquestación del flujo completo)
        'pipeline.plugins': [
            'default_pipeline=pipeline_plugins.default_pipeline:PipelinePlugin',
            'stl_pipeline=pipeline_plugins.stl_pipeline:PipelinePlugin'
        ],
        # Plugins para el Preprocesamiento (incluye process_data, ventanas deslizantes y STL)
        'preprocessor.plugins': [
            'default_preprocessor=preprocessor_plugins.default_preprocessor:PreprocessorPlugin',
            'stl_preprocessor=preprocessor_plugins.stl:PreprocessorPlugin'
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
