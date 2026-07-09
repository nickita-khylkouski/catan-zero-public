"""Compatibility shim for old Catanatron/NetworkX/checkpoint imports."""

import numpy as _np
import sys as _sys

if not hasattr(_np, "int"):
    _np.int = int
if not hasattr(_np, "float"):
    _np.float = float
if not hasattr(_np, "bool"):
    _np.bool = bool
if not hasattr(_np, "object"):
    _np.object = object

# Some checkpoints were saved with NumPy 2.x and pickle references such as
# ``numpy._core.multiarray``. Older eval hosts may only expose ``numpy.core``.
try:
    import numpy._core  # noqa: F401
except ModuleNotFoundError:
    import numpy.core as _np_core
    import numpy.core.multiarray as _np_multiarray
    import numpy.core.numeric as _np_numeric

    _sys.modules.setdefault("numpy._core", _np_core)
    _sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
    _sys.modules.setdefault("numpy._core.numeric", _np_numeric)
