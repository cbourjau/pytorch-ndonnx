# pytorch-ndonnx

[![CI](https://github.com/cbourjau/pytorch-ndonnx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cbourjau/pytorch-ndonnx/actions/workflows/ci.yml)
[![Documentation](https://app.readthedocs.org/projects/pytorch-ndonnx/badge/?version=latest)](https://pytorch-ndonnx.readthedocs.io/en/latest/)

`pytorch-ndonnx` is a library that can be used to convert pytorch models to ONNX.
It is based on [ndonnx](https://github.com/Quantco/ndonnx).

Conversion is achieved by simply ducktyping existing code using a `pytorch_ndonnx.Tensor` object.

## Installation

### PyPI

```bash
pip install pytorch-ndonnx
```

### conda-forge

Coming soon

### Development

You can install the package in development mode using:

```bash
git clone https://github.com/cbourjau/pytorch-ndonnx
cd pytorch-ndonnx
pixi run pre-commit-install
pixi run postinstall
pixi run test
```

## Example

```python
import ndonnx as ndx
import onnx
import pytorch_ndonnx as tx
import torch.nn as nn
import torch.nn.functional as F


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)

    def forward(self, x):
        return F.relu(F.max_pool2d(self.conv1(x), 2))

# 'arguments' are inputs to a graph
arr = ndx.argument(shape=("N", 1, 28, 28), dtype=ndx.float32)
# Wrap the resulting ndonnx array in a pytorch_ndonnx.Tensor
tensor = tx.Tensor(arr)
# Use 'tensor' to simply ducktype the existing code
result = MyModule()(tensor)

# Create the onnx graph in between the specified arguments and resulting arrays.
model_proto = ndx.build({"x": arr}, {"y": result.to_ndonnx()})
# Write the model to disk
onnx.save_model(model_proto, "example.onnx")
```
