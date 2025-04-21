# -*- coding: utf-8 -*-

import pytest
import torch

from fla.modules.l2norm import l2_norm
from fla.utils import device


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("D", [50, 64, 128, 2048])
def test_l2norm(B: int, H: int, T: int, D: int):
    torch.manual_seed(42)
    x = torch.randn(B, T, H, D).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref_y = torch.nn.functional.normalize(x, dim=-1, p=2)
    tri_y = l2_norm(x)
    ref_dx = torch.autograd.grad(ref_y.sum(), x)[0]
    tri_dx = torch.autograd.grad(tri_y.sum(), x)[0]

    torch.testing.assert_close(ref_y, tri_y, rtol=1e-3, atol=3e-3)
    torch.testing.assert_close(ref_dx, tri_dx, rtol=1e-3, atol=3e-3)
