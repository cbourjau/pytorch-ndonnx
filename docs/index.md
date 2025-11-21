# pytorch-ndonnx

Pytorch-to-ONNX converters based on ndonnx.
Conversion is achieved by simply ducktypign existing code using a `pytorch_ndonnx.Tensor` object.

See [here](./tensor.md) for API documentation.

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
