"""RP95 operational shim: the author code imports `sktime.datasets.load_from_tsfile_to_dataframe` at module level for its UEA
classification loader, which the long-term-forecasting path never calls. The shim satisfies the import and REFUSES any call, so
it cannot have a mathematical effect on the reproduced path (a call would fail loudly, never approximate)."""
