from inspect import signature

import pytest
from torch.overrides import (
    get_overridable_functions,
    get_testing_overrides,
    resolve_name,
)

from pytorch_ndonnx._tensor import HANDLED_FUNCTIONS

_DUMMYS = get_testing_overrides()
_OVERRIDABLE_MODULES = get_overridable_functions()
_OVERRIDABLE_FUNS = set(sum((list(el) for el in _OVERRIDABLE_MODULES.values()), []))
_BUGGY_UPSTREAM_SIGNATURES = [
    "torch.max_pool2d",
    "torch.argmax",
    "torch.argmin",
    "torch.add",
    "torch.sub",
    "torch.nn.functional.scaled_dot_product_attention",
]


@pytest.mark.parametrize(
    "tfun_override", HANDLED_FUNCTIONS.items(), ids=lambda item: resolve_name(item[0])
)
def test_signatures(tfun_override):
    tfun, override = tfun_override
    name = resolve_name(tfun)
    if name in _BUGGY_UPSTREAM_SIGNATURES:
        pytest.skip("signature of '{name}' is buggy upstream")

    assert tfun in _OVERRIDABLE_FUNS

    try:
        expected_sig = signature(tfun)
        # Sometimes there are signatures, but they are useless...
        if ("args", "kwargs") == tuple(signature(tfun).parameters):
            expected_sig = signature(_DUMMYS[tfun])
    except ValueError:
        # dummy signatures are sometimes wrong; use as fallback
        expected_sig = signature(_DUMMYS[tfun])

    override_sig = signature(override)

    assert list(expected_sig.parameters.keys()) == list(override_sig.parameters.keys())
    for dummy_parame, override_param in zip(
        expected_sig.parameters, override_sig.parameters
    ):
        assert dummy_parame == override_param
