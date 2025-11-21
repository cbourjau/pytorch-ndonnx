from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F  # noqa

from benchmarks.models import (
    MNIST,
    EmbeddingNetwork1,
    EmbeddingNetwork2,
    SqueezeNet,
    SRResNet,
    SuperResolutionNet,
    dcgan,
)
from pytorch_ndonnx import Tensor
from pytorch_ndonnx._jit import jit

BATCH_SIZE = 2


def test_dcgan_netd():
    model = dcgan.NetD(1)
    model.apply(dcgan.weights_init)
    model.train(False)

    x = torch.empty(dcgan.bsz, 3, dcgan.imgsz, dcgan.imgsz).normal_(0, 1)

    expected = model(x).detach().numpy()
    np.testing.assert_allclose(
        model(Tensor(x.numpy()))._inner.unwrap_numpy(), expected, rtol=1e-6
    )


def test_dcgan_netg():
    model = dcgan.NetG(1)
    model.apply(dcgan.weights_init)
    model.train(False)

    x = torch.empty(dcgan.bsz, dcgan.nz, 1, 1).normal_(0, 1)

    expected = model(x).detach().numpy()
    candidate = model(Tensor(x.numpy()))._inner.unwrap_numpy()
    np.testing.assert_array_almost_equal(candidate, expected, decimal=8)


def test_embedding_sequential_1():
    x = torch.randint(0, 10, (BATCH_SIZE, 3))
    model = EmbeddingNetwork1()
    model.train(False)

    expected = model(x).detach().numpy()
    candidate = model(Tensor(x.numpy()))._inner.unwrap_numpy()
    np.testing.assert_array_almost_equal(candidate, expected, decimal=6)


def test_embedding_sequential_2():
    x = torch.randint(0, 10, (BATCH_SIZE, 3))
    model = EmbeddingNetwork2()
    model.train(False)

    expected = model(x).detach().numpy()
    candidate = model(Tensor(x.numpy()))._inner.unwrap_numpy()
    np.testing.assert_array_almost_equal(candidate, expected, decimal=7)


def test_mnist():
    x = torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0)
    model = MNIST()
    model.train(False)

    expected = model(x).detach().numpy()
    np.testing.assert_allclose(
        model(Tensor(x.numpy()))._inner.unwrap_numpy(), expected, rtol=1e-6
    )


def test_squeezenet():
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    model = SqueezeNet(version=1.1)
    model.train(False)

    expected = model(x).detach().numpy()
    np.testing.assert_array_almost_equal(
        model(Tensor(x.numpy()))._inner.unwrap_numpy(), expected, decimal=6
    )


def test_srresnet():
    torch.utils.backcompat.broadcast_warning.enabled = True
    x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
    model = SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)
    model.train(False)

    # jitting saves a lot of time compared to eager computation
    with timeit("ONNX jit and predict"):
        wrapped = jit(model)
        candidate = wrapped(x).detach().numpy()

    # Pytorch compilation is VERY slow
    with timeit("Pytorch eager"):
        expected = model(x).detach().numpy()
    # TODO: Revisit this bad precision
    np.testing.assert_array_almost_equal(candidate, expected, decimal=2)


def test_super_resolution():
    x = torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0)
    model = SuperResolutionNet(upscale_factor=3)
    model.train(False)

    expected = model(x).detach().numpy()
    candidate = model(Tensor(x.numpy()))._inner.unwrap_numpy()
    # TODO: Revisit this bad precision
    np.testing.assert_array_almost_equal(candidate, expected)


@contextmanager
def timeit(msg: str):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print(f"{msg}: {(t1 - t0):.3}")
