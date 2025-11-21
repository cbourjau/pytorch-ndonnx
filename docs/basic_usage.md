# Basic usage

`pytorch_ndonnx` is build around the idea of [duck-typing](https://docs.python.org/3/glossary.html#term-duck-typing).
Existing code that is written to be called with `pytorch.Tensor` objects is instead called using `pytorch_ndonnx.Tensor` objects.
The latter is a thin wrapper around a [ndonnx.Array][] which in turn keeps track of any operation applied to it.
Ultimately, the entire computational graph can be reconstructed from a given output.
See the ndonnx documentation for further details.

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

# 'arguments' are input to a graph
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

## Calling pytorch functions

Pytorch provides an extension protocol that allows for calling its various free functions with custom tensor objects.
`pytorch_ndonnx` provides support for the functions listed [here](pytorch_functions.md).
