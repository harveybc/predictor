from setuptools import find_packages, setup

setup(
    name="predictor-olap-store",
    version="0.1.0",
    description="The predictor OLAP cube as a backend of a data-warehouse host",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={"datawarehouse.backends": ["predictor_olap=predictor_olap_store.provider:backend"]},
    install_requires=["sqlalchemy>=2.0"],
    python_requires=">=3.10",
)
