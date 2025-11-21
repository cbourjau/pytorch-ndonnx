from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from types import EllipsisType
from typing import Any, Literal

import ndonnx as ndx
import numpy as np
import spox.opset.ai.onnx.v21 as op
import torch
from spox import Var
from typing_extensions import Self, TypeIs

HANDLED_FUNCTIONS = {}


def _implements(torch_function):
    """Register a torch function override for Tensor."""

    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


class Tensor:
    """Ndonnx-backed tensor that ducktypes as [torch.Tensor][].

    `Tensor` objects may also be used in the torch functions listed [here](pytorch_functions.md).
    """

    _inner: ndx.Array

    def __init__(self, value: torch.Tensor | np.ndarray | ndx.Array | Tensor | Var, /):
        if isinstance(value, Tensor):
            self._inner = value._inner.copy()
            return
        elif isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()
        self._inner = ndx.asarray(value)

    def __repr__(self) -> str:
        return f"Array(shape={self._inner.shape}, dtype={self._inner.dtype})"

    def argmax(
        self,
        dim: None | int = None,
        keepdim: bool = False,
        *,
        out: None | Tensor = None,
    ) -> Tensor:
        """See [torch.Tensor.argmax][] for details."""
        return argmax(self, dim, keepdim, out=out)

    def argmin(
        self,
        dim: None | int = None,
        keepdim: bool = False,
        *,
        out: None | Tensor = None,
    ) -> Tensor:
        """See [torch.Tensor.argmin][] for details."""
        return argmin(self, dim, keepdim, out=out)

    def chunk(self, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
        """See [torch.Tensor.chunk][] for details."""
        return chunk(self, chunks=chunks, dim=dim)

    def contiguous(self, memory_format=torch.contiguous_format) -> Tensor:
        """See [torch.Tensor.contiguous][] for details."""
        return self

    def dim(self) -> int:
        """See [torch.Tensor.dim][] for details."""
        return self.ndim()

    @property
    def device(self) -> None:
        """The device on which the data is allocated.

        This is always `None` since the concept does not apply to ONNX export
        """
        return None

    @property
    def dtype(self) -> torch.dtype:
        """See `torch.Tensor.dtype for details."""
        inverse = {v: k for k, v in _dtype_map.items()}
        return inverse[self._inner.dtype]

    def expand(self, *sizes: int | torch.Size | Tensor) -> Tensor:
        """See [torch.Tensor.expand][] for details."""
        sizes_ = []
        for el in sizes:
            if isinstance(el, torch.Size):
                el = el.numel()
            if isinstance(el, int):
                sizes_.append(ndx.asarray(el))
            else:
                sizes_.append(el._inner)

        return Tensor(ndx.broadcast_to(self._inner, ndx.stack(sizes_)))

    def masked_fill(self, mask: Tensor, value: float):
        """See [torch.Tensor.masked_fill][] for details."""
        return Tensor(ndx.where(mask._inner, value, self._inner))

    def mean(
        self,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """See [torch.Tensor.mean][] for details."""
        x = self._inner
        if dtype is not None:
            x = x.astype(_dtype_map[dtype])
        arr = ndx.mean(x, axis=dim, keepdims=keepdim)
        return Tensor(arr)

    def ndim(self) -> int:
        """Return the rank of this tensor."""
        return self._inner.ndim

    def to(self, dtype: torch.dtype) -> Tensor:
        """See [torch.Tensor.to][] for details."""
        return Tensor(self._inner.astype(_dtype_map[dtype]))

    def unwrap_numpy(self) -> np.ndarray:
        """Return the result of constant value propagation or raise a `ValueError`.

        This function is primarily useful for debugging and testing. It only returns a
        value if the value does not depend on any graph inputs. However, if that is the
        case in production code, such a value should be stored as a NumPy array or
        pytorch Tensor in the first place.
        """
        return self._inner.unwrap_numpy()

    def to_ndonnx(self) -> ndx.Array:
        """Return the inner ndonnx array."""
        return self._inner

    def view(self, *shape: int | torch.Size) -> Self:
        """See [torch.Tensor.view][] for details."""
        shape_: tuple[int, ...] | ndx.Array
        if any(isinstance(el, torch.Size) for el in shape):
            raise NotImplementedError
        elif any(isinstance(el, Tensor) for el in shape):
            shape_ = ndx.concat(
                [
                    el._inner[None] if isinstance(el, Tensor) else ndx.asarray([el])
                    for el in shape
                ]
            )
        else:
            shape_ = shape  # type: ignore
        arr = ndx.reshape(self._inner, shape_)  # type: ignore
        return type(self)(arr)

    @property
    def shape(self) -> int | Tensor | tuple[int | Tensor, ...]:
        """The shape of the tensor.

        Contrary to `pytorch.Tensor.shape`, this function may return
        `Tensor` objects for dimensions with a dynamic length.
        """
        return self.size(None)

    def size(self, dim: int | None = None) -> int | Tensor | tuple[int | Tensor, ...]:
        """The size of the tensor.

        Contrary to `pytorch.Tensor.size`, this function may return
        `Tensor` objects for dimensions with a dynamic length.
        """
        if dim is None:
            size = self._inner.shape
            if any(el is None for el in size):
                out = []
                for static, dynamic in zip(size, self._inner.dynamic_shape):
                    out.append(Tensor(dynamic) if static is None else static)
                return tuple(out)
            return size  # type: ignore
        size_int = self._inner.shape[dim]
        if size_int is None:
            return Tensor(self._inner.dynamic_shape[dim])
        return size_int

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        """See [torch.Tensor.transpose][] for details."""
        return transpose(self, dim0, dim1)

    def __add__(self, other: Tensor | torch.Tensor | int | float) -> Tensor:
        return add(self, other)

    def __radd__(self, other: Tensor | torch.Tensor | int | float) -> Tensor:
        return add(other, self)

    def __sub__(self, other: Tensor | torch.Tensor | int | float) -> Tensor:
        return sub(self, other)

    def __rsub__(self, other: Tensor | torch.Tensor | int | float) -> Tensor:
        return sub(other, self)

    def __bool__(self) -> bool:
        raise NotImplementedError

    def __eq__(self, value: object, /) -> bool:
        raise NotImplementedError

    def __floordiv__(self, other: Tensor | torch.Tensor | int | float) -> Tensor:
        if isinstance(other, torch.Tensor | Tensor):
            other_: ndx.Array | int | float = Tensor(other)._inner
        else:
            other_ = other
        return Tensor(self._inner // other_)

    def __getitem__(
        self, key: int | tuple[int | slice | EllipsisType | None, ...]
    ) -> Tensor:
        def normalize_slice(s: slice) -> slice:
            start = s.start._inner if isinstance(s.start, Tensor) else s.start
            stop = s.stop._inner if isinstance(s.stop, Tensor) else s.stop
            step = s.step._inner if isinstance(s.step, Tensor) else s.step

            return slice(start, stop, step)

        def normalize_element(
            el: int | slice | EllipsisType | None,
        ) -> int | slice | EllipsisType | None:
            if isinstance(el, int | EllipsisType | None):
                return el
            return normalize_slice(el)

        if isinstance(key, int):
            key = (key,)

        key = tuple(normalize_element(el) for el in key)

        def is_tuple_key(key) -> TypeIs[tuple[int | slice | None]]:
            return isinstance(key, tuple) and all(
                isinstance(el, int | slice | None) for el in key
            )

        if is_tuple_key(key):
            # Pytorch allows specifying fewer dimensions than the
            # array has. Thus we pad with an ellipsis if there is
            # None, yet.
            if not any(el is ... for el in key):
                key = key + (...,)
            return Tensor(self._inner[key])
        raise NotImplementedError

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in HANDLED_FUNCTIONS or not all(
            issubclass(t, (torch.Tensor | Tensor)) for t in types
        ):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def unwrap(x: torch.Tensor | Tensor | int | float) -> ndx.Array | int | float:
    if isinstance(x, torch.Tensor):
        x = Tensor(x)
    if isinstance(x, Tensor):
        return x._inner
    return x


@_implements(torch.add)
def add(
    input: torch.Tensor | Tensor | int | float,
    other: torch.Tensor | Tensor | int | float,
    *,
    alpha: int | float = 1,
    out: Tensor | None = None,
) -> Tensor:
    x1 = unwrap(input)
    x2 = unwrap(other)

    if alpha != 1:
        x2 *= alpha

    res = ndx.asarray(x1 + x2)
    if out is None:
        return Tensor(res)
    out._inner = res
    return out


@_implements(torch.sub)
def sub(
    input: torch.Tensor | Tensor | int | float,
    other: torch.Tensor | Tensor | int | float,
    *,
    alpha: int | float = 1,
    out: Tensor | None = None,
) -> Tensor:
    x1 = unwrap(input)
    x2 = unwrap(other)

    if alpha != 1:
        x2 *= alpha

    res = ndx.asarray(x1 - x2)
    if out is None:
        return Tensor(res)
    out._inner = res
    return out


@_implements(torch.nn.functional.adaptive_avg_pool2d)
def adaptive_avg_pool2d(
    input: Tensor, output_size: int | None | tuple[int | None, int | None]
) -> Tensor:
    x = input._inner
    x_shape = x.dynamic_shape

    s0: int | ndx.Array | None
    s1: int | ndx.Array | None

    if isinstance(output_size, int):
        s0, s1 = (output_size, output_size)
    elif output_size is None:
        return Tensor(x.copy())
    elif isinstance(output_size, tuple):
        s0, s1 = output_size

    cat_list = [
        x_shape[:-2],
    ]
    if s0 is None and s1 is None:
        return Tensor(x.copy())
    if s0 is not None:
        s0 = ndx.asarray(s0)
        cat_list += [x_shape[-2][None] // s0, s0[None]]
    else:
        cat_list += [x_shape[-2]]
    if s1 is not None:
        s1 = ndx.asarray(s1)
        cat_list += [x_shape[-1][None] // s1, s1[None]]
    else:
        cat_list += [x_shape[-1]]

    if s0 is None:
        reduce_axis: tuple[int, ...] = (-3,)
    elif s1 is None:
        reduce_axis = (-2,)
    else:
        reduce_axis = (-4, -2)

    chunked_shape = ndx.concat(cat_list)
    reshaped_x = ndx.asarray(op.reshape(x.unwrap_spox(), chunked_shape.unwrap_spox()))
    result = ndx.mean(reshaped_x, axis=reduce_axis)

    return Tensor(result)


@_implements(torch.argmax)
def argmax(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
    *,
    out: None | Tensor = None,
) -> Tensor:
    return _argmaxmin(input, dim, keepdim, out=out, op_name="argmax")


@_implements(torch.argmin)
def argmin(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
    *,
    out: None | Tensor = None,
) -> Tensor:
    return _argmaxmin(input, dim, keepdim, out=out, op_name="argmin")


def _argmaxmin(
    input: Tensor,
    dim: None | int = None,
    keepdim: bool = False,
    *,
    out: None | Tensor = None,
    op_name: Literal["argmax", "argmin"],
) -> Tensor:
    if input.ndim() == 0 and (dim is None or dim in [0, -1]):
        # Special case not covered by array-api?
        return Tensor(ndx.asarray(0, dtype=ndx.int64))

    if op_name == "argmax":
        op = ndx.argmax
    elif op_name == "argmin":
        op = ndx.argmin
    else:
        ValueError
    res = op(input._inner, axis=dim, keepdims=keepdim)
    if out is not None:
        out._inner = res
    return Tensor(res)


@_implements(torch.nn.functional.batch_norm)
def batch_norm(
    input: Tensor,
    running_mean: Tensor | torch.Tensor | None,
    running_var: Tensor | torch.Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    if running_mean is None or running_var is None:
        raise NotImplementedError

    input_mean = Tensor(running_mean)._inner
    input_var = Tensor(running_var)._inner

    if training:
        raise ValueError("'training' mode is not supported")

    x = input._inner
    scale = (
        ndx.asarray(1.0, dtype=ndx.float32) if weight is None else Tensor(weight)._inner
    )
    B = ndx.asarray(0.0, dtype=ndx.float32) if bias is None else Tensor(bias)._inner

    if len(input._inner.shape) == 4:
        expand_idx: tuple[Any, ...] = (..., None, None)
    elif len(input._inner.shape) == 3:
        expand_idx = (..., None)
    elif len(input._inner.shape) in (1, 2):
        expand_idx = (...,)
    else:
        raise NotImplementedError

    arr = (
        (x - input_mean[expand_idx]) / ndx.sqrt(input_var[expand_idx] + eps)
    ) * scale[expand_idx] + B[expand_idx]
    return Tensor(arr)


@_implements(torch.cat)
def cat(
    tensors: Sequence[Tensor], dim: int = 0, *, out: Tensor | None = None
) -> Tensor:
    res = ndx.concat([el._inner for el in tensors], axis=dim)
    if out is not None:
        out._inner = res
        return out
    return Tensor(res)


@_implements(torch.chunk)
def chunk(input: Tensor, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
    x = input._inner
    if dim < 0:
        dim = x.ndim + dim
    len_dim = x.dynamic_shape[dim] // chunks

    x_shape = x.dynamic_shape
    shape = ndx.concat(
        [x_shape[:dim], ndx.asarray([chunks]), len_dim[None], x_shape[dim + 1 :]]
    )
    x = ndx.reshape(x, shape)
    key_prefix = [slice(None) for _ in range(dim)]
    res = [x[tuple(key_prefix + [i, ...])] for i in range(chunks)]
    return tuple(Tensor(arr) for arr in res)


@_implements(torch.conv2d)
def conv2d(
    input: Tensor,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | str = 0,
    dilation=1,
    groups=1,
) -> Tensor:
    X = input._inner
    if X.ndim == 3:
        X = X[None, ...]

    W = ndx.asarray(weight.detach().numpy()).unwrap_spox()
    B = None if bias is None else ndx.asarray(bias.detach().numpy()).unwrap_spox()
    stride = _normalize_conv2d_stride(stride)
    padding_, auto_pad = _parse_padding(padding)
    pads = None if padding_ is None else _normalize_padding_2d(padding_)
    dilations = _normalized_dilations_2d(dilation)

    var = op.conv(
        X.unwrap_spox(),
        W,
        B,
        strides=stride,
        pads=pads,
        group=groups,
        dilations=dilations,
        auto_pad=auto_pad,
    )
    arr = ndx.asarray(var)
    # Strip away forth dimension if we added it previously
    if input._inner.ndim == 3:
        arr = ndx.squeeze(arr, axis=0)
    return Tensor(arr)


@_implements(torch.conv_transpose2d)
def conv_transpose2d(
    input: Tensor,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter | None = None,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] | str = 0,
    output_padding=0,
    groups=1,
    dilation=1,
) -> Tensor:
    X = input._inner
    if X.ndim == 3:
        X = X[None, ...]

    W = ndx.asarray(weight.detach().numpy())
    B = None if bias is None else ndx.asarray(bias.detach().numpy()).unwrap_spox()

    stride = _normalize_conv2d_stride(stride)
    padding_, auto_pad = _parse_padding(padding)
    pads = None if padding_ is None else _normalize_padding_2d(padding_)
    dilations = _normalized_dilations_2d(dilation)
    output_padding = (
        (output_padding, output_padding)
        if isinstance(output_padding, int)
        else output_padding
    )

    var = op.conv_transpose(
        X.unwrap_spox(),
        W.unwrap_spox(),
        B,
        strides=stride,
        pads=pads,
        group=groups,
        dilations=dilations,
        auto_pad=auto_pad,
        output_padding=output_padding,
    )
    arr = ndx.asarray(var)
    # Strip away forth dimension if we added it previously
    if input._inner.ndim == 3:
        arr = ndx.squeeze(arr, axis=0)
    return Tensor(arr)


@_implements(torch.flatten)
def flatten(input: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    start_dim = _normalize_axis(start_dim, input.ndim())
    end_dim = _normalize_axis(end_dim, input.ndim())
    if input.ndim() == 1:
        # Explicitly returning the input object as described in the docs
        return input
    x = input._inner
    if start_dim == 0 and end_dim == input.ndim() - 1:
        return Tensor(ndx.reshape(x, (-1,)))
    x_shape = x.dynamic_shape
    shape = ndx.concat(
        [
            x_shape[:start_dim],
            ndx.asarray([-1], dtype=ndx.int64),
            x_shape[end_dim + 1 :],
        ]
    )
    return Tensor(op.reshape(x.unwrap_spox(), shape.unwrap_spox()))


@_implements(torch.nn.functional.gelu)
def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    var = op.gelu(input._inner.unwrap_spox(), approximate=approximate)
    return Tensor(var)


@_implements(torch.nn.functional.hardtanh)
def hardtanh(
    input: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    x = input._inner
    arr = ndx.clip(x, min_val, max_val)
    if inplace:
        input._inner = arr
        return input
    return Tensor(arr)


@_implements(torch.nn.functional.layer_norm)
def layer_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    axis = -len(normalized_shape)
    if weight is None:
        weight_ = ndx.ones(tuple(normalized_shape), dtype=ndx.float32).unwrap_spox()
    else:
        weight_ = Tensor(weight)._inner.unwrap_spox()
    y, _mean, _std = op.layer_normalization(
        input._inner.unwrap_spox(),
        weight_,
        None if bias is None else Tensor(bias)._inner.unwrap_spox(),
        axis=axis,
        epsilon=eps,
        stash_type=1,  # float?
    )

    return Tensor(y)


@_implements(torch.pixel_shuffle)
def pixel_shuffle(input: Tensor, upscale_factor: int) -> Tensor:
    x = input._inner
    if x.ndim < 3:
        raise ValueError(
            f"'pixel_shuffle' expects input of at least rank-3, but got `{x.ndim}`"
        )
    x_shape = x.dynamic_shape
    leading = x_shape[:-3]
    c_out = x_shape[-3:-2] // upscale_factor**2
    hw = x_shape[-2:]
    hw_out = hw * upscale_factor

    out_shape = ndx.concat([leading, c_out, hw_out])
    tmp_shape = ndx.concat(
        [
            ndx.asarray([-1], dtype=ndx.int64),
            ndx.asarray([upscale_factor], dtype=ndx.int64),
            ndx.asarray([upscale_factor], dtype=ndx.int64),
            hw,
        ]
    )
    x = ndx.reshape(x, tmp_shape)
    #  (...,C*r2, H, W) -> (M, R, R, H, W) -> (M, H, R, W, R)
    #  shapes:             (0, 1, 2, 3, 4) -> (0, 3, 1, 4, 2)

    x = ndx.permute_dims(x, (0, 3, 1, 4, 2))
    x = ndx.reshape(x, out_shape)
    return Tensor(x)


@_implements(torch.sigmoid)
def sigmoid(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    arr = ndx.asarray(op.sigmoid(input._inner.unwrap_spox()))

    if out is not None:
        out._inner = arr
    return Tensor(arr)


@_implements(torch.tanh)
def tanh(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    arr = ndx.tanh(input._inner)

    if out is not None:
        out._inner = arr
    return Tensor(arr)


@_implements(torch.transpose)
def transpose(input: Tensor, dim0: int, dim1: int) -> Tensor:
    if dim0 == dim1:
        return Tensor(input)
    axes = list(range(input.ndim()))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

    arr = ndx.permute_dims(input._inner, axes=tuple(axes))
    return Tensor(arr)


def _parse_padding(
    padding: int | tuple[int, int] | str, /
) -> tuple[int | tuple[int, int] | None, str]:
    auto_pad = "NOTSET"
    if padding == "same":
        auto_pad = "SAME_UPPER"
        return None, auto_pad  # type: ignore
    elif padding == "valid":
        auto_pad = "VALID"
        return None, auto_pad  # type: ignore
    if isinstance(padding, str):
        raise ValueError(f"unexpected 'padding' value: `{padding}`")

    return padding, auto_pad


@_implements(torch.nn.functional.avg_pool2d)
def nn__functional__avg_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
):
    if not stride:
        # OpInfo tests also pass in `()`...
        stride = kernel_size
    kernel_shape = _normalized_kernel_shape_2d(kernel_size)
    pads = _normalize_padding_2d(padding)
    strides = None if stride is None else _normalize_conv2d_stride(stride)

    # Validate that the kernel, strides and padding line up. Otherwise
    # we run into UB.
    if strides is not None and strides != (1, 1):

        def validate_dim(
            x_len: int | None,
            pad_start: int,
            pad_end: int,
            kernel_len: int,
            stride: int,
        ):
            if x_len is None:
                raise ValueError(
                    "pooling dimensions must be known statically to avoid undefined behavior"
                )
            if (pad_start + x_len + pad_end - kernel_len) % stride:
                raise ValueError(
                    f"pooling dimensions of length {x_len=} with total padding {pad_start + pad_end}, {kernel_len=} and {stride=} do not align"
                )

        *_, h, w = input._inner.shape[-2:]
        validate_dim(h, pads[0], pads[2], kernel_shape[0], strides[0])
        validate_dim(w, pads[1], pads[3], kernel_shape[1], strides[1])

    var = op.average_pool(
        input._inner.unwrap_spox(),
        ceil_mode=ceil_mode,
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
        count_include_pad=count_include_pad,
    )
    arr = ndx.asarray(var)
    if divisor_override is not None:
        if not count_include_pad:
            fix_counts: ndx.Array | int = ndx.asarray(
                op.average_pool(
                    ndx.ones_like(input._inner).unwrap_spox(),
                    ceil_mode=ceil_mode,
                    kernel_shape=kernel_shape,
                    pads=pads,
                    strides=strides,
                    count_include_pad=True,
                )
            )
        else:
            fix_counts = 1
        arr = arr * (math.prod(kernel_shape) * fix_counts / divisor_override)
    return Tensor(arr)


@_implements(torch.nn.functional.dropout2d)
def nn__functional__dropout2d(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    return input


@_implements(torch.nn.functional.max_pool2d)
@_implements(torch.max_pool2d)
def nn__functional__max_pool2d(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    x = input._inner
    if x.ndim == 3:
        x = x[None, ...]
    if not stride:
        stride = kernel_size
    max_var, indices_var = op.max_pool(
        x.unwrap_spox(),
        ceil_mode=ceil_mode,
        kernel_shape=_normalized_kernel_shape_2d(kernel_size),
        pads=_normalize_padding_2d(padding),
        dilations=_normalized_dilations_2d(dilation),
        strides=None if stride is None else _normalize_conv2d_stride(stride),
    )

    max_arr = ndx.asarray(max_var)
    indices_arr = ndx.asarray(indices_var)

    shape = x.dynamic_shape
    N, C, H, W = shape[0], shape[1], shape[2], shape[3]
    offsets = (
        ndx.cumulative_sum(ndx.broadcast_to(ndx.asarray(H * W), (N * C)[None, ...]))
        - H * W
    )
    indices_arr = (
        indices_arr
        - ndx.reshape(offsets, indices_arr.dynamic_shape[:2])[..., None, None]
    )
    if input._inner.ndim == 3:
        max_arr = ndx.squeeze(max_arr, axis=0)
        indices_arr = ndx.squeeze(indices_arr, axis=0)
    if return_indices:
        return Tensor(max_arr), Tensor(indices_arr)
    return Tensor(ndx.asarray(max_arr))


@_implements(torch.nn.functional.max_pool2d_with_indices)
def nn__functional__max_pool2d_with_indices(
    input: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    ceil_mode: bool = False,
    return_indices: bool = True,
) -> tuple[Tensor, Tensor]:
    res = nn__functional__max_pool2d(
        input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )
    if isinstance(res, tuple) and len(res) == 2:
        return res
    raise ValueError(f"expected tuple-result with two elements found `{res}`")


@_implements(torch.nn.functional.prelu)
def nn__functional__prelu(input: Tensor, weight: torch.Tensor) -> Tensor:
    # prelu may be used in cases where inputs are either (N, C, H, W)
    # or (C, H, W), but it may also be us in any other case with no
    # restriction on the input dimensions and semantics. The former
    # case is not strictly NumPy broadcastable: the input is NCHW
    # while the weights are C.
    #
    # The following is inspired by the pytorch reference implementation.
    x = input._inner
    w = Tensor(weight)._inner

    if x.ndim == 0:
        w = w[0] if w.ndim == 1 else w
    else:
        w = broadcast_in_dim(x, w, () if w.ndim == 0 else (0 if x.ndim == 1 else 1,))

    return Tensor(ndx.where(x >= 0, x, w * x))


def _sqrt(x: Tensor) -> Tensor:
    return Tensor(ndx.sqrt(x._inner))


def _fill(*size: ndx.Array, value: int | float, dtype: torch.dtype) -> ndx.Array:
    shape = ndx.concat([el[None] for el in size])
    var = op.constant_of_shape(
        shape.unwrap_spox(), value=torch.asarray([value], dtype=dtype).detach().numpy()
    )
    return ndx.asarray(var)


def _zeros(*size: ndx.Array, dtype: torch.dtype) -> ndx.Array:
    return _fill(*size, value=0.0, dtype=dtype)


def _ones(*size: ndx.Array, dtype: torch.dtype) -> ndx.Array:
    return _fill(*size, value=1, dtype=dtype)


@_implements(torch.softmax)
def softmax(
    input: Tensor,
    dim: int | None,
    *,
    dtype: None | torch.dtype = None,
) -> Tensor:
    arr = input._inner
    if dtype is not None:
        arr = arr.astype(_dtype_map[dtype])
    exp_ = ndx.exp(arr)
    if arr.ndim == 0 and dim in [0, -1]:
        # Seen in opinfo tests
        dim = None
    res = exp_ / ndx.sum(exp_, axis=dim, keepdims=True)
    return Tensor(res)


@_implements(torch.sqrt)
def sqrt(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    res = ndx.sqrt(input._inner)
    if out is None:
        return Tensor(res)
    out._inner = res
    return out


@_implements(torch.nn.functional.scaled_dot_product_attention)
def nn__functional__scaled_dot_product_attention(
    query: Tensor,
    key: Tensor | torch.Tensor,
    value: Tensor | torch.Tensor,
    attn_mask: Tensor | torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> Tensor:
    query_shape = query._inner.dynamic_shape
    key_shape = Tensor(key)._inner.dynamic_shape
    L, S = query_shape[-2], key_shape[-2]
    # TODO: Check sqrt behavior of ndx/np for npx.sqrt(8)
    scale_factor = (
        1 / ndx.sqrt(query_shape[-1].astype(_dtype_map[query.dtype]))
        if scale is None
        else scale
    )
    attn_bias = _zeros(L, S, dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = ndx.tril(_ones(L, S, dtype=torch.bool), k=0)
        attn_bias = ndx.where(
            temp_mask, attn_bias, ndx.asarray(float("-inf"), dtype=ndx.float32)
        )
        attn_bias = attn_bias.astype(_dtype_map[query.dtype])

    if attn_mask is not None:
        attn_mask_ = Tensor(attn_mask)._inner
        if attn_mask.dtype == torch.bool:
            attn_bias = ndx.where(
                ~attn_mask_, attn_bias, ndx.asarray(float("-inf"), dtype=ndx.float32)
            )
        else:
            attn_bias = attn_mask_ + attn_bias

    if enable_gqa:
        raise NotImplementedError
        # key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        # value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    def softmax(x: ndx.Array, dim: int) -> ndx.Array:
        # Softmax with shift
        maxes = ndx.max(x, axis=dim, keepdims=True)
        shifted_exp = ndx.exp(x - maxes)
        return shifted_exp / shifted_exp.sum(axis=dim, keepdims=True)

    attn_weight = (query._inner * scale_factor) @ Tensor(key.transpose(-2, -1))._inner
    attn_weight += attn_bias
    attn_weight = softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return Tensor(attn_weight @ Tensor(value)._inner)


def broadcast_in_dim(
    input: ndx.Array, a: ndx.Array, broadcast_dimensions: tuple[int, ...]
) -> ndx.Array:
    shape: tuple[int | None, ...] = input.shape
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = ndx.expand_dims(v, axis=idx)

    v, _ = ndx.broadcast_arrays(v, input)

    return v


@_implements(torch.relu)
@_implements(torch.nn.functional.relu)
def nn__functional__relu(input: Tensor, inplace: bool = False) -> Tensor:
    arr = ndx.maximum(input._inner, ndx.zeros_like(input._inner))
    if inplace:
        input._inner = arr
        return input
    return Tensor(arr)


@_implements(torch.nn.functional.linear)
def nn__functional__linear(
    input: Tensor, weight: torch.nn.Parameter, bias: torch.nn.Parameter | None = None
) -> Tensor:
    weight_ = Tensor(weight.T)._inner
    res = input._inner @ weight_
    if bias is not None:
        res += Tensor(bias)._inner
    return Tensor(res)


@_implements(torch.nn.functional.dropout)
def nn__functional__dropout(
    input: Tensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> Tensor:
    return input


@_implements(torch.nn.functional.embedding)
def torch__nn__functional__embedding(
    input: Tensor,
    weight: Tensor | torch.Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    if padding_idx or scale_grad_by_freq:
        raise ValueError("unsupporting training parameters set")
    arr = ndx.take(Tensor(weight)._inner, input._inner)
    return Tensor(arr)


@_implements(torch.nn.functional.log_softmax)
def nn__functional__log_softmax(
    input: Tensor,
    dim: int | None = None,
    _stacklevel=3,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if dim is None:
        raise NotImplementedError

    x = input._inner
    if dtype is not None:
        x = x.astype(_dtype_map[dtype])

    # Squeeze in a leading dimension if we are dealing with a scalar (seen in opinfo tests)
    if x.ndim == 0:
        x = x[None]
    arr = ndx.asarray(op.log_softmax(input=x.unwrap_spox(), axis=dim))
    if input._inner.ndim == 0:
        arr = arr[0, ...]
    return Tensor(arr)


@_implements(torch.nn.functional.leaky_relu)
def nn__functional__leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
) -> Tensor:
    x = input._inner
    res = ndx.where(x >= 0, x, negative_slope * x)
    if inplace:
        input._inner = res
        return input

    return Tensor(res)


def _normalize_axis(axis: int, rank: int) -> int:
    """Construct an `axis` array for reduction operations such as `mean` and normalize
    it to positive values."""
    if axis >= 0:
        return axis
    return axis + rank


def _normalized_dilations_2d(val: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(val, int):
        return (val, val)
    return val


def _normalized_kernel_shape_2d(val: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(val, int):
        return (val, val)
    return val


def _normalize_padding_2d(padding: int | tuple[int, int]) -> tuple[int, int, int, int]:
    if isinstance(padding, tuple) and len(padding) == 1:
        # Appears in OpInfo test cases...
        (padding,) = padding
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    return (padding[0], padding[1], padding[0], padding[1])


def _normalize_conv2d_stride(stride: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(stride, int):
        return (stride, stride)
    return stride


_dtype_map: dict[torch.dtype, ndx.DType] = {
    torch.float16: ndx.float16,
    torch.float32: ndx.float32,
    torch.float64: ndx.float64,
    # ints
    torch.int16: ndx.int16,
    torch.int16: ndx.int16,
    torch.int32: ndx.int32,
    torch.int64: ndx.int64,
    # uints
    torch.uint16: ndx.uint16,
    torch.uint16: ndx.uint16,
    torch.uint32: ndx.uint32,
    torch.uint64: ndx.uint64,
    # others
    torch.bool: ndx.bool,
}
