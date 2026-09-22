"""RP95 operational shim for `patoolib`, imported by the author's M4 dataset module (short-term forecasting only)."""


def extract_archive(*args, **kwargs):
    raise RuntimeError("sota_shims: patoolib.extract_archive was CALLED on the reproduced path; install patool and re-run.")
