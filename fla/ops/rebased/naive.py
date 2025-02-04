# -*- coding: utf-8 -*-

import torch

from fla.ops.rebased.parallel import parallel_rebased


def naive_parallel_rebased(q, k, v, use_scale=True, use_norm=True):
    if use_scale:
        q = q * (q.shape[-1] ** -0.5)
    attn = q @ k.transpose(-2, -1)
    attn = (attn ** 2)
    attn.masked_fill_(~torch.tril(torch.ones(
        q.shape[-2], q.shape[-2], dtype=torch.bool, device=q.device)), 0)
    o = attn @ v
    if use_norm:
        z = attn.sum(-1)
        return o / (z[..., None] + 1e-6)
    else:
        return o


if __name__ == "__main__":
    from fla.utils import device
    B = 4
    H = 4
    L = 128
    # D = 15
    dtype = torch.float32
    q = (torch.randn(B, H, L, 16).to(device).to(dtype)).requires_grad_(True)
    k = (torch.randn(B, H, L, 16).to(device).to(dtype)).requires_grad_(True)
    v = torch.randn(B, H, L, 128).to(device).to(dtype).requires_grad_(True)

    do = torch.randn_like(v).to(device)
    ref = naive_parallel_rebased(q, k, v, True, True)
    ref.backward(do, retain_graph=True)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_rebased(q, k, v, 1e-6, True, True)
    tri.backward(do, retain_graph=True)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    print((ref-tri).abs().max())
    print((ref_dq-tri_dq).abs().max())
    print((ref_dk-tri_dk).abs().max())
    print((ref_dv-tri_dv).abs().max())
