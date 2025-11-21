from pathlib import Path

import torch

from pytorch_ndonnx._tensor import HANDLED_FUNCTIONS

undocumented = [torch.nn.functional.max_pool2d_with_indices]


def get_torch_name(fun) -> str:
    # Try to find the function in torch.nn.functional and fall back to top-level namespace
    if hasattr(torch.nn.functional, fun.__name__):
        return f"torch.nn.functional.{fun.__name__.replace('__', '.')}"
    if hasattr(torch, fun.__name__):
        return f"torch.{fun.__name__.replace('__', '.')}"
    raise NotImplementedError(f"`{fun}` not found in 'torch' nor 'torch.nn.functional'")


names = set()
for fun in HANDLED_FUNCTIONS:
    name = get_torch_name(fun)
    names.add((name, fun in undocumented))

out = """# Supported functions in the `torch` names space

The following pytorch functions can be called with [pytorch_ndonnx.Tensor][] objects.


"""
for name, undoc in sorted(names):
    if undoc:
        out += f"- `{name}` (undocumented upstream)\n"
    else:
        out += f"- [{name}][]\n"


with open(Path(__file__).parent.parent / "docs/overloaded.md", "w") as f:
    f.write(out)
torch.max_pool2d
