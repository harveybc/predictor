from setuptools import find_packages, setup

setup(
    name="olap-lake",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"web_plugins": ["templates/*.html", "static/css/*.css"]},
    entry_points={
        "console_scripts": ["olap-lake=app.main:main"],
        "olaplake.pipeline": [
            "default_pipeline=pipeline_plugins.default_pipeline:Plugin",
        ],
        "olaplake.web": ["default_web=web_plugins.default_web:Plugin"],
        "olaplake.query": ["sql_query=query_plugins.sql_query:Plugin"],
    },
    install_requires=["flask>=3.0", "sqlalchemy>=2.0"],
)
