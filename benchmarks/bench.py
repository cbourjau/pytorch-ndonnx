# Benchmark for each model:
# - Conversion time
# - Jit time (defector Conversion + session creation)
# - Torch compile
# - For different batch sizes:
#   - Constant propagation time
#   - Pure onnxruntime duration
#   - Torch compile execute

import platform
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from io import StringIO
from pathlib import Path
from time import perf_counter

import onnx
import torch
import torch.nn.functional as F  # noqa
from models import (  # type: ignore
    MNIST,
    SqueezeNet,
    SuperResolutionNet,
    dcgan,
)
from torchvision.models import (
    alexnet,
    densenet121,
    googlenet,
    mnasnet1_0,
    mobilenet_v2,
    resnet50,
    shufflenet_v2_x1_0,
)

from pytorch_ndonnx import Tensor
from pytorch_ndonnx._jit import build_model, build_session, jit

torch.set_num_interop_threads(1)
torch.set_num_threads(1)


@contextmanager
def timeit(out: dict[str, float], key: str, allow_failure: bool = False):
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    out[f"{key}"] = t1 - t0


def build_model_pytorch_onnx(model, x: torch.Tensor) -> onnx.ModelProto:
    # Create a temporary directory
    tmpdir = tempfile.mkdtemp()
    model_path = Path(tmpdir) / "model.onnx"
    torch.onnx.export(
        deepcopy(model),
        (x,),
        f=model_path,
        input_names=["x"],
        output_names=["y"],
        do_constant_folding=True,
        opset_version=21,
        dynamo=True,
    )
    model_proto = onnx.load_model(model_path)
    return model_proto


def bench_model(model, x: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    model.train(False)

    with timeit(out, "ONNX build"):
        model_proto = build_model(model, x)
    print("model size: ", len(model_proto.SerializeToString()) / 1e6)

    with timeit(out, "ONNX session creation"):
        session = build_session(model_proto)
    with timeit(out, "ONNX infer from session"):
        session.run(None, {"x": x.detach().numpy()})
    with timeit(out, "ONNX jit"):
        jit(model)(x).detach().numpy()
    with timeit(out, "ONNX eager"):
        model(Tensor(x))._inner.unwrap_numpy()

    with timeit(out, "pytorch eager:"):
        model(x).detach().numpy()

    # Pytorch compile is not available on Windows
    if platform.system() != "Windows":
        try:
            with timeit(out, "pytorch compile"):
                model.compile(fullgraph=True)
                _jit_warmup = model(x).detach().numpy()
            with timeit(out, "pytorch inference compiled"):
                model(x).detach().numpy()
        except Exception:
            # Pytorch compilation is crashy at times
            out["pytorch compile"] = float("nan")
            out["pytorch inference compiled"] = float("nan")

    try:
        with timeit(out, "pytorch onnx export"):
            mp = build_model_pytorch_onnx(model, x)
    except Exception:
        out["pytorch onnx export"] = float("nan")
    else:
        print("pytorch model size: ", len(mp.SerializeToString()) / 1e6)

    return out


def bench_model_for_batch_sizes(
    model, shape: tuple[int, ...]
) -> dict[int, dict[str, float]]:
    out = {}
    for batch_size in [
        2,
        10,
    ]:
        model = deepcopy(model)  # deepcopy to avoid compiled version being reused
        x = torch.randn(batch_size, *shape).fill_(1.0)
        out[batch_size] = bench_model(model, x)

    return out


def build_markdown_tables(data: dict[str, dict[int, dict[str, float]]]) -> str:
    buf = StringIO()

    for model, batches in data.items():
        row_labels = sorted({k for b in batches.values() for k in b.keys()})
        batch_sizes = sorted(batches.keys())

        buf.write(f"## {model}\n\n")
        buf.write("| Batch size | " + " | ".join(str(b) for b in batch_sizes) + " |\n")
        buf.write("|" + " --- |" * (1 + len(batch_sizes)) + "\n")

        for row in row_labels:
            vals = []
            for b in batch_sizes:
                v = batches[b].get(row, "")
                if isinstance(v, float):
                    v = f"{v:.4f}"
                vals.append(str(v))
            buf.write("| " + row + " | " + " | ".join(vals) + " |\n")
        buf.write("\n")

    return buf.getvalue()


def main():
    out = {}

    out["alexnet"] = bench_model_for_batch_sizes(alexnet(), (3, 224, 224))
    out["densenet"] = bench_model_for_batch_sizes(densenet121(), (3, 224, 224))
    out["googlenet"] = bench_model_for_batch_sizes(googlenet(), (3, 224, 224))
    out["mnasnet"] = bench_model_for_batch_sizes(mnasnet1_0(), (3, 224, 224))
    out["mobilenet_v2"] = bench_model_for_batch_sizes(mobilenet_v2(), (3, 224, 224))
    out["resnet50"] = bench_model_for_batch_sizes(resnet50(), (3, 224, 224))
    out["shufflenet_v2_x1_0"] = bench_model_for_batch_sizes(
        shufflenet_v2_x1_0(), (3, 224, 224)
    )

    out["MNIST"] = bench_model_for_batch_sizes(MNIST(), (1, 28, 28))
    out["SqueezeNet"] = bench_model_for_batch_sizes(
        SqueezeNet(version=1.1), (3, 224, 224)
    )
    # out["SRResNet"] = bench_model_for_batch_sizes(SRResNet(rescale_factor=4, n_filters=64, n_blocks=8), (3, 224, 224))
    out["SuperResolutionNet"] = bench_model_for_batch_sizes(
        SuperResolutionNet(upscale_factor=3), (1, 224, 224)
    )
    out["dcgan"] = bench_model_for_batch_sizes(
        dcgan.NetD(1), (3, dcgan.imgsz, dcgan.imgsz)
    )

    res = build_markdown_tables(out)

    with open("bench.md", "w") as f:
        f.write(res)


main()
