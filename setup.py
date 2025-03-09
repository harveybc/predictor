from setuptools import setup, find_packages

setup(
    name='predictor',
    version='0.1.0',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'predictor=app.main:main'
        ],
        'predictor.plugins': [
            'default=app.plugins.predictor_plugin_ann:Plugin',
            'ann=app.plugins.predictor_plugin_ann:Plugin',
            'cnn=app.plugins.predictor_plugin_cnn:Plugin',
            'cnn_mmd=app.plugins.predictor_plugin_cnn_mmd:Plugin',
            'lstm=app.plugins.predictor_plugin_lstm:Plugin',
            'transformer=app.plugins.predictor_plugin_transformer:Plugin',
            'transformer_mmd=app.plugins.predictor_plugin_transformer_mmd:Plugin',
            'base=app.plugins.predictor_plugin_base:Plugin'
        ]
    },
    install_requires=[
        'build'           
    ],
    author='Harvey Bastidas',
    author_email='your.email@example.com',
    description='A timeseries prediction system that supports dynamic loading of predictor plugins for processing time series data.'
)
