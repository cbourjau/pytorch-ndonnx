import platform
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa
from torchvision.models import (
    alexnet,
    densenet121,
    googlenet,
    inception_v3,
    mnasnet1_0,
    mobilenet_v2,
    resnet50,
    shufflenet_v2_x1_0,
)

# from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
# from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
# from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18
from pytorch_ndonnx import Tensor
from pytorch_ndonnx._jit import jit

torch.set_num_interop_threads(1)
torch.set_num_threads(1)

BATCH_SIZE = 2


@contextmanager
def timeit(msg: str):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print(f"{msg}: {(t1 - t0):.3}s")


def _predict_candidate_expected(
    model, x: torch.Tensor, onnx_jit: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    model.train(False)

    if onnx_jit:
        with timeit("ONNX jit'ed:"):
            candidate = jit(model)(x).detach().numpy()
    else:
        with timeit("ONNX eager:"):
            candidate = model(Tensor(x.numpy()))._inner.unwrap_numpy()

    # Pytorch compile is not available on Windows
    if platform.system() == "Windows":
        with timeit("pytorch eager:"):
            expected = model(x).detach().numpy()
    else:
        with timeit("pytorch eager"):
            model(x).detach().numpy()
        with timeit("pytorch compile:"):
            model.compile(fullgraph=True)
            _jit_warmup = model(x).detach().numpy()
        with timeit("pytorch inference compiled:"):
            expected = model(x).detach().numpy()

    return candidate, expected


def test_alexnet():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(*_predict_candidate_expected(alexnet(), x))


def test_densenet():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(
        *_predict_candidate_expected(densenet121(), x), decimal=5
    )


def test_googlenet():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(
        *_predict_candidate_expected(googlenet(init_weights=True), x),
    )


@pytest.mark.skip(reason="Flaky")
def test_inception_v3():
    x = torch.full((BATCH_SIZE, 3, 299, 299), 0.5)
    np.testing.assert_allclose(
        *_predict_candidate_expected(inception_v3(init_weights=True), x), rtol=1e-3
    )


def test_mnasnet():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(
        *_predict_candidate_expected(mnasnet1_0(), x), decimal=5
    )


def test_mobilenet_v2():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(
        *_predict_candidate_expected(mobilenet_v2(), x), decimal=5
    )


def test_resnet50():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(
        *_predict_candidate_expected(resnet50(), x), decimal=4
    )


def test_shufflenet_v2_x1_0():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    np.testing.assert_array_almost_equal(
        *_predict_candidate_expected(shufflenet_v2_x1_0(), x), decimal=3
    )
