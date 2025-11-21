import importlib.metadata
import warnings

from pytorch_ndonnx._patch import monkey_patches
from pytorch_ndonnx._tensor import Tensor

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError as e:  # pragma: no cover
    warnings.warn(f"Could not determine version of {__name__}\n{e!s}", stacklevel=2)
    __version__ = "unknown"

__all__ = ["Tensor", "monkey_patches"]
