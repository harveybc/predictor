from setuptools import find_packages, setup

setup(
    name="predictor-duckdb-store",
    version="0.1.0",
    description="The predictor OLAP cube on DuckDB, as a backend of a data-warehouse host",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "datawarehouse.backends": [
            "predictor_duckdb=predictor_duckdb_store.provider:backend"
        ]
    },
    # The governance semantics live in predictor-olap-store and are REUSED, not reimplemented:
    # a second copy of the temporal rules would agree until the day it did not.
    install_requires=["sqlalchemy>=2.0", "duckdb>=1.0", "duckdb-engine>=0.13",
                      "predictor-olap-store>=0.1.0", "pandas>=2.0"],
    python_requires=">=3.10",
)
