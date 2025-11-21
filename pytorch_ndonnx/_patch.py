import contextlib
from types import EllipsisType

import torch

from pytorch_ndonnx._tensor import Tensor


@contextlib.contextmanager
def monkey_patches():
    """Apply monkey patches to force compatibility with [pytorch_ndonnx.Tensor][]
    objects.

    Many models can be converted without these patches, but some models do require them, unfortunately.

    Currently, this applies the following patches:
    - `transformers.utils.is_tensor` recognizes `pytorch_ndonnx.Tensor` objects as tensors
    - `torch.jit.is_tracing` always returns `True`
    - `torch.Tensor.__getitem__` is enabled to consume `pytorch_ndonnx.Tensor` objects
    """
    with _patch_is_tensor(), _patch___getitem__(), _patch_pytorch_jit_is_tracing():
        yield


@contextlib.contextmanager
def _patch_is_tensor():
    """Patch 'transformers' to recognize 'pytorch_ndonnx.Tensor' as a "tensor"."""
    try:
        import transformers
    except ImportError:
        return
    orig = transformers.utils.is_tensor  # type: ignore

    def is_tensor(x) -> bool:
        if isinstance(x, Tensor):
            return True
        return orig(x)

    try:
        transformers.utils.is_tensor = is_tensor
        yield
    finally:
        transformers.utils.is_tensor = orig


@contextlib.contextmanager
def _patch_pytorch_jit_is_tracing():
    """Make `torch.jit.is_tracing` always return true."""
    orig = torch.jit.is_tracing

    try:
        torch.jit.is_tracing = lambda: True
        yield
    finally:
        torch.jit.is_tracing = orig


@contextlib.contextmanager
def _patch___getitem__():
    """Patch 'torch.Tensor.__getitem__' to work with 'Tensor' indices."""

    orig = torch.Tensor.__getitem__

    def is_otensor_like(
        item: int | Tensor | EllipsisType | torch.Tensor | slice,
    ) -> bool:
        if isinstance(item, Tensor):
            return True
        if isinstance(item, slice):
            return (
                isinstance(item.start, Tensor)
                or isinstance(item.stop, Tensor)
                or isinstance(item.step, Tensor)
            )
        return False

    def __getitem__(self, key):  # noqa: N807
        if (
            is_otensor_like(key)
            or isinstance(key, tuple)
            and any(is_otensor_like(el) for el in key)
        ):
            return Tensor(self)[key]
        return orig(self, key)

    try:
        torch.Tensor.__getitem__ = __getitem__  # type: ignore[method-assign]
        yield
    finally:
        torch.Tensor.__getitem__ = orig  # type: ignore[method-assign]
