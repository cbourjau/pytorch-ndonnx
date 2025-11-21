from collections.abc import Callable
from contextlib import contextmanager
from time import perf_counter
from typing import ParamSpec, TypeVar

import ndonnx as ndx
import onnx
import onnxruntime as ort
import spox
import spox._future
import torch

from pytorch_ndonnx._tensor import Tensor, _dtype_map

T = TypeVar("T")
P = ParamSpec("P")


def build_model(fun: Callable, input: torch.Tensor, *args, **kwargs) -> onnx.ModelProto:
    x = ndx.argument(shape=tuple(input.shape), dtype=_dtype_map[input.dtype])
    with disable_spox_value_prop():
        y: Tensor = fun(Tensor(x), *args, **kwargs)  # type: ignore

    return ndx.build({"x": x}, {"y": y._inner})


def build_session(model: onnx.ModelProto) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), opts)


def jit(fun: Callable[P, T]) -> Callable[P, T]:
    def inner(input: torch.Tensor, *args, **kwargs):
        model_proto = build_model(fun, input, *args, **kwargs)
        session = build_session(model_proto)
        (out,) = session.run(None, {"x": input.detach().numpy()})
        return torch.asarray(out)

    return inner  # type: ignore


@contextmanager
def disable_spox_value_prop():
    previous_backend = spox._value_prop._VALUE_PROP_BACKEND
    spox._future.set_value_prop_backend(spox._future.ValuePropBackend.NONE)
    yield
    spox._future.set_value_prop_backend(previous_backend)


@contextmanager
def timeit(msg: str):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print(f"{msg}: {(t1 - t0):.3}")
