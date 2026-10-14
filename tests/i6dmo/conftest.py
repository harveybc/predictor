"""Import the model module (which forces the determinism env) BEFORE TensorFlow.

Production spawns ``python -m tools.i6dmo.train`` with the flags in the
interpreter's environment from the start; pytest would otherwise import
TensorFlow during collection and every in-process record would carry
``tf_imported_before_flags=true`` (typed refusal). Importing here keeps the
in-process test path faithful to the production init order.
"""

import tools.i6dmo.model  # noqa: F401  (sets static-algorithm env, then imports TF itself)
