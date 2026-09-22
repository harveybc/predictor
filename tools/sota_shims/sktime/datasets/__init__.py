def load_from_tsfile_to_dataframe(*args, **kwargs):
    raise RuntimeError("sota_shims: sktime.datasets.load_from_tsfile_to_dataframe was CALLED on the reproduced path; the shim only "
                       "satisfies an import the long-term-forecasting path never exercises. Install sktime and re-run.")
