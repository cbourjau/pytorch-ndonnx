import inspect
from collections.abc import Callable
from math import prod
from typing import Any

import numpy as np
import pytest
import torch
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import OpInfo

from pytorch_ndonnx import Tensor
from pytorch_ndonnx._tensor import HANDLED_FUNCTIONS

OP_DB = {op.name: op for op in op_db}


def unpack_opinfo(op: OpInfo) -> list[tuple[torch.Tensor, Any, dict[str, Any]]]:
    out = []
    for item in op.sample_inputs_func(
        op, device="cpu", dtype=torch.float32, requires_grad=False
    ):
        out.append((item.input, item.args, item.kwargs))
    return out


def default_test(op: Callable, fun_name: str, **test_kwargs):
    @pytest.mark.parametrize("input, args, kwargs", unpack_opinfo(OP_DB[fun_name]))
    def fun(input: torch.Tensor, args, kwargs):
        expected = op(input, *args, **kwargs)
        candidate = op(Tensor(input), *args, **kwargs).unwrap_numpy()  # type: ignore

        np.testing.assert_allclose(candidate, expected, **test_kwargs)

    return fun


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.adaptive_avg_pool2d"])
)
def test_adaptive_avg_pool2d(input: torch.Tensor, args, kwargs):
    osize, *_ = args + tuple(kwargs.values())
    if osize is None:
        pass
    elif isinstance(osize, int):
        if (input.shape[-2] % osize) or (input.shape[-1] % osize):
            pytest.skip("adaptive pool only supports equal-sized segments")
    else:
        s0, s1 = osize
        if s0 is not None and input.shape[-2] % s0:
            pytest.skip("adaptive pool only supports equal-sized segments")
        if s1 is not None and input.shape[-1] % s1:
            pytest.skip("adaptive pool only supports equal-sized segments")

    op = torch.nn.functional.adaptive_avg_pool2d
    expected = op(input, *args, **kwargs)
    candidate = op(Tensor(input), *args, **kwargs).unwrap_numpy()  # type: ignore

    np.testing.assert_allclose(candidate, expected, rtol=1e-6)


def test_adaptive_avg_pool2d_extra():
    # adaptive pool tests are either trivial or are skipped because
    # they divide a dim in uneven chunks.
    input = torch.arange(0, 4, dtype=torch.float32).reshape((1, 1, 2, 2))
    args = [(1, 1)]
    op = torch.nn.functional.adaptive_avg_pool2d

    op = torch.nn.functional.adaptive_avg_pool2d
    expected = op(input, *args)
    candidate = op(Tensor(input), *args).unwrap_numpy()  # type: ignore

    np.testing.assert_allclose(candidate, expected, rtol=1e-6)


test_add = default_test(torch.add, "add", rtol=1e-6)
test_sub = default_test(torch.sub, "sub", rtol=1e-6)


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.avg_pool2d"])
)
def test_avg_pool2d(input: torch.Tensor, args, kwargs):
    op = torch.nn.functional.avg_pool2d

    arguments = (
        inspect.signature(HANDLED_FUNCTIONS[op])
        .bind(Tensor(input), *args, **kwargs)
        .arguments
    )
    arguments.pop("input")

    bad_args_dicts = [
        {"kernel_size": (4, 4), "stride": (2, 3), "padding": 1},
        {"kernel_size": (6, 6), "stride": (3, 3), "padding": (2, 3)},
    ]
    if any(
        {k: v for k, v in arguments.items() if k in item} == item  # type: ignore
        for item in bad_args_dicts
    ):  # type: ignore
        with pytest.raises(ValueError):
            op(Tensor(input), *args, **kwargs)  # type: ignore
        return

    expected = op(input, **arguments)
    candidate = op(Tensor(input), **arguments).unwrap_numpy()  # type: ignore

    np.testing.assert_allclose(candidate, expected, rtol=1e-6)


test_argmax = default_test(torch.argmax, "argmax")
test_argmin = default_test(torch.argmax, "argmin")


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.batch_norm"])
)
def test_batch_norm(input: torch.Tensor, args, kwargs):
    op = torch.nn.functional.batch_norm

    if kwargs.get("training"):
        pytest.skip("'training' mode is not supported")

    expected = op(input, *args, **kwargs)
    candidate = op(Tensor(input), *args, **kwargs).unwrap_numpy()  # type: ignore

    np.testing.assert_allclose(candidate, expected, rtol=1e-5)


@pytest.mark.parametrize("input, args, kwargs", unpack_opinfo(OP_DB["cat"]))
def test_cat(input: list[torch.Tensor], args, kwargs):
    if any([t.shape == torch.Size([0]) for t in input]):
        pytest.skip(
            reason="cannot reliably filter out zero-sized tensors from concatenation"
        )
    op = torch.cat
    expected = op(input, *args, **kwargs)
    candidate = op([Tensor(el) for el in input], *args, **kwargs).unwrap_numpy()  # type: ignore

    np.testing.assert_allclose(candidate, expected)


@pytest.mark.parametrize("input, args, kwargs", unpack_opinfo(OP_DB["chunk"]))
def test_chunk(input: torch.Tensor, args, kwargs):
    args = list(args) + list(kwargs.get("chunks", [])) + list(kwargs.get("dim", []))
    if len(args) == 1:
        (chunks,) = args
        dim = 0
    else:
        chunks, dim = args
    if input.shape[dim] % chunks:
        pytest.skip("Only fixed-size chunks are supported")
    op = torch.chunk
    expected = op(input, *args, **kwargs)
    candidate = op(Tensor(input), *args, **kwargs)  # type: ignore

    for exp, cand in zip(expected, candidate, strict=True):
        np.testing.assert_allclose(cand.unwrap_numpy(), exp)  # type: ignore


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.conv2d"])
)
def test_conv2d(input: torch.Tensor, args, kwargs):
    if kwargs.get("padding") == "same" and kwargs.get("dilation", 1) != 1:
        pytest.skip(reason="dilations other than 1 are not supported with auto_pad")
    op = torch.nn.functional.conv2d
    expected = op(input, *args, **kwargs).numpy()
    candidate = op(Tensor(input), *args, **kwargs).unwrap_numpy()  # type: ignore

    np.testing.assert_almost_equal(candidate, expected, decimal=4)


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.conv_transpose2d"])
)
def test_conv_transpose2d(input: torch.Tensor, args, kwargs):
    op = torch.nn.functional.conv_transpose2d
    if kwargs.get("output_padding") == (2, 3):
        # Maybe related to https://github.com/microsoft/onnxruntime/issues/14208
        pytest.xfail("a concerning mystery why this is failing")
    expected = op(input, *args, **kwargs).numpy()
    candidate = op(Tensor(input), *args, **kwargs).unwrap_numpy()  # type: ignore

    np.testing.assert_almost_equal(candidate, expected, decimal=4)


# OpInfo tests seems broken
# @pytest.mark.parametrize(
#     "input, args, kwargs", _unpack_opinfo(OP_DB["nn.functional.embedding"])
# )
def test_embedding():
    # Example from docs
    op = torch.nn.Embedding(10, 3)
    input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    expected = op(input).detach().numpy()
    candidate = op(Tensor(input)).unwrap_numpy()  # type: ignore

    np.testing.assert_array_equal(candidate, expected)

    # example with padding_idx
    op = torch.nn.Embedding(10, 3, padding_idx=0)
    input = torch.LongTensor([[0, 2, 0, 5]])
    expected = op(input).detach().numpy()
    candidate = op(Tensor(input)).unwrap_numpy()  # type: ignore

    np.testing.assert_array_equal(candidate, expected)


test_gelu = default_test(torch.nn.functional.gelu, "nn.functional.gelu", rtol=1e-4)
test_flatten = default_test(torch.flatten, "flatten")
test_hardtanh = default_test(torch.nn.functional.hardtanh, "nn.functional.hardtanh")
test_leaky_relu = default_test(
    torch.nn.functional.leaky_relu, "nn.functional.leaky_relu"
)
test_linear = default_test(
    torch.nn.functional.linear, "nn.functional.linear", rtol=1e-5
)
test_log_softmax = default_test(torch.nn.functional.log_softmax, "log_softmax")


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.layer_norm"])
)
def test_layer_norm(input: torch.Tensor, args, kwargs):
    op = torch.nn.functional.layer_norm

    if input.ndim <= 1 or input.size(0) == 0:
        pytest.skip(reason="onnxruntime fails to propagate this case")
    expected = op(input, *args, **kwargs)
    candidate = op(Tensor(input), *args, **kwargs)  # type: ignore

    np.testing.assert_allclose(candidate.unwrap_numpy(), expected, rtol=1e-6)  # type: ignore


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.max_pool2d"])
)
def test_max_pool2d(input: torch.Tensor, args, kwargs):
    op = torch.nn.functional.max_pool2d
    expected = op(input, *args, **kwargs)
    candidate = op(Tensor(input), *args, **kwargs)  # type: ignore

    if kwargs.get("return_indices"):
        np.testing.assert_allclose(candidate[0].unwrap_numpy(), expected[0], rtol=1e-6)
        np.testing.assert_array_equal(candidate[1].unwrap_numpy(), expected[1])
    else:
        np.testing.assert_allclose(candidate.unwrap_numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "input, args, kwargs", unpack_opinfo(OP_DB["nn.functional.pixel_shuffle"])
)
def test_pixel_shuffle(input: torch.Tensor, args, kwargs):
    torch.utils.backcompat.broadcast_warning.enabled = True
    op = torch.nn.functional.pixel_shuffle

    if prod(input.shape) == 0:
        pytest.skip(reason="'pixel_shuffle' is not implemented for zero-sized input.")

    expected = op(input, *args, **kwargs)
    candidate = op(Tensor(input), *args, **kwargs).unwrap_numpy()  # type: ignore

    np.testing.assert_allclose(candidate, expected)


test_prelu = default_test(torch.nn.functional.prelu, "nn.functional.prelu")
test_relu = default_test(torch.nn.functional.relu, "nn.functional.relu")


@pytest.mark.parametrize(
    "input, args, kwargs",
    unpack_opinfo(OP_DB["nn.functional.scaled_dot_product_attention"]),
)
def test_scaled_dot_product_attention(input: torch.Tensor, args, kwargs):
    if kwargs.get("dropout_p", 0.0) > 0.0:
        pytest.skip("'dropout_p' is not supported'")
    op = torch.nn.functional.scaled_dot_product_attention
    expected = op(input, *args, **kwargs).numpy()
    candidate = op(Tensor(input), *args, **kwargs)._inner.unwrap_numpy()  # type: ignore

    np.testing.assert_almost_equal(candidate, expected, decimal=4)


test_softmax = default_test(torch.softmax, "softmax")
test_sqrt = default_test(torch.sqrt, "sqrt", rtol=1e-6)
test_tanh = default_test(torch.tanh, "tanh", rtol=1e-6)
test_transpose = default_test(torch.transpose, "transpose")
