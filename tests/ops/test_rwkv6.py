# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F

from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6, native_recurrent_rwkv6
from fla.ops.rwkv6.recurrent_naive import (naive_recurrent_rwkv6,
                                           naive_recurrent_rwkv6_bwd)
from fla.utils import device
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("D", [100])
@pytest.mark.parametrize("dtype", [torch.float])
def test_recurrent_naive(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)

    q = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device=device).to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device=device)).to(dtype).requires_grad_(True)
    u = torch.randn(B, H, D, device=device).to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=dtype, requires_grad=True)

    o, _ = naive_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    dh, h.grad = h.grad.clone(), None

    dq2, dk2, dv2, dw2, du2, dh2 = naive_recurrent_rwkv6_bwd(q, k, v, w, u, o, do, initial_state=h)

    assert dq.allclose(dq2, atol=1e-3)
    assert dk.allclose(dk2, atol=1e-3)
    assert dv.allclose(dv2, atol=1e-3)
    assert dw.allclose(dw2, atol=1e-3)
    assert du.allclose(du2, atol=1e-3)
    assert dh.allclose(dh2, atol=1e-3)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("use_h", [False, True])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    use_h: bool
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-1

    q = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device=device).to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device=device)).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device=device).to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=dtype, requires_grad=True)

    ref_o, _ = native_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None, output_final_state=use_h)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None, output_final_state=use_h)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert ref_o.allclose(tri_o, atol=atol)
    assert ref_dq.allclose(tri_dq, atol=atol)
    assert ref_dk.allclose(tri_dk, atol=atol)
    assert ref_dv.allclose(tri_dv, atol=atol)
    assert ref_dw.allclose(tri_dw, atol=atol)
    assert ref_du.allclose(tri_du, atol=atol)
    if use_h:
        assert ref_dh.allclose(tri_dh, atol=atol)


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [512, 1024])
@pytest.mark.parametrize("D", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("use_h", [False, True])
def test_chunk_with_initial_h(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    use_h: bool
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-2

    q = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    k = torch.randn(B, H, T, D, device=device).to(dtype).requires_grad_(True)
    v = torch.randn(B, H, T, 2*D, device=device).to(dtype).requires_grad_(True)
    w = F.logsigmoid(torch.randn(B, H, T, D, device=device)).to(dtype).requires_grad_(True)
    u = torch.randn(H, D, device=device).to(dtype).requires_grad_(True)
    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, D, 2*D, device=device, dtype=dtype, requires_grad=True)

    ref_o, _ = fused_recurrent_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None, output_final_state=use_h)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = chunk_rwkv6(q, k, v, w, u, scale=1.0, initial_state=h if use_h else None, output_final_state=use_h)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert get_err_ratio(ref_o, tri_o) < atol
    assert get_err_ratio(ref_dq, tri_dq) < atol
    assert get_err_ratio(ref_dk, tri_dk) < atol
    assert get_err_ratio(ref_dv, tri_dv) < atol
    assert get_err_ratio(ref_dw, tri_dw) < atol
    assert get_err_ratio(ref_du, tri_du) < atol
    if use_h:
        assert get_err_ratio(ref_dh, tri_dh) < atol

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def val(x):
    return x.detach().float().cpu().numpy()

def LOSS(y):
    return ((y * y) - torch.tanh(y)).sum()

def RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, h):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, state = chunk_rwkv6(r, k, v, w, u=u, scale=1, initial_state=h, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), state

def RUN_FLA_FUSED(B, T, C, H, r, k, v, w, u, h):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, state = fused_recurrent_rwkv6(r, k, v, w, u=u, scale=1, initial_state=h, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), state

def RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, r, k, v, w, u, h):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, state = naive_recurrent_rwkv6(r, k, v, w, u=u, scale=1, initial_state=h, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), state


def RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r, k, v, w, u, h):
    r = r.view(B,T,H,-1).transpose(1,2)
    k = k.view(B,T,H,-1).transpose(1,2)
    v = v.view(B,T,H,-1).transpose(1,2)
    w = -torch.exp(w.view(B,T,H,-1).transpose(1,2))
    o, state = native_recurrent_rwkv6(r, k, v, w, u=u, scale=1, initial_state=h, output_final_state=True)
    return o.transpose(1,2).reshape(B,T,C), state

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("C", [4096])
@pytest.mark.parametrize("HEAD_SIZE", [64])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_chunk_error_ratio(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype
):
    atol = 2e-2 if dtype == torch.float else 1e-2
    H = C // HEAD_SIZE

    set_seed(42)
    with torch.no_grad():
        r = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        w = torch.empty(B, T, C, device=device).uniform_(-8, -6).to(dtype=dtype)
        u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype)
        initial_state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=device).to(dtype=dtype)

    def clear_grad():
        r.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        w.requires_grad_()
        u.requires_grad_()
        initial_state.requires_grad_()
        if r.grad is not None: r.grad.data.zero_()
        if k.grad is not None: k.grad.data.zero_()
        if v.grad is not None: v.grad.data.zero_()
        if w.grad is not None: w.grad.data.zero_()
        if u.grad is not None: u.grad.data.zero_()
        if initial_state.grad is not None: initial_state.grad.data.zero_()

    clear_grad()
    y32, _ = RUN_FLA_FUSED(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float(), initial_state.float())
    LOSS(y32).backward()
    gr = r.grad.data.clone()
    gk = k.grad.data.clone()
    gv = v.grad.data.clone()
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()
    gh = initial_state.grad.data.clone()
    clear_grad()

    yF16, _ = RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, initial_state)
    LOSS(yF16).backward()
    gr_chunk = r.grad.data.clone()
    gk_chunk = k.grad.data.clone()
    gv_chunk = v.grad.data.clone()
    gw_chunk = w.grad.data.clone()
    gu_chunk = u.grad.data.clone()
    gh_chunk = initial_state.grad.data.clone()
    clear_grad()

    assert get_err_ratio(yF16, y32) < atol, f"output, {get_err_ratio(yF16, y32)}"
    assert get_err_ratio(gr_chunk, gr) < atol, f"r, {get_err_ratio(gr_chunk, gr)}"
    assert get_err_ratio(gk_chunk, gk) < atol, f"k, {get_err_ratio(gk_chunk, gk)}"
    assert get_err_ratio(gv_chunk, gv) < atol, f"v, {get_err_ratio(gv_chunk, gv)}"
    assert get_err_ratio(gw_chunk, gw) < atol, f"w, {get_err_ratio(gw_chunk, gw)}"
    assert get_err_ratio(gu_chunk, gu) < atol, f"u, {get_err_ratio(gu_chunk, gu)}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}"



@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("C", [4096])
@pytest.mark.parametrize("HEAD_SIZE", [64])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_chunk_error_ratio_multi_state(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype
):
    atol = 2e-2 if dtype == torch.float else 1e-2
    H = C // HEAD_SIZE
    set_seed(42)
    with torch.no_grad():
        r = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        w = torch.empty(B, T, C, device=device).uniform_(-8, -6).to(dtype=dtype)
        u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype)
        initial_state = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=device).to(dtype=dtype)
        r1 = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        k1 = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        v1 = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        w1 = torch.empty(B, T, C, device=device).uniform_(-8, -6).to(dtype=dtype)
        u1 = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype)

    def clear_grad():
        r.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        w.requires_grad_()
        u.requires_grad_()
        r1.requires_grad_()
        k1.requires_grad_()
        v1.requires_grad_()
        w1.requires_grad_()
        u1.requires_grad_()
        initial_state.requires_grad_()
        if r.grad is not None: r.grad.data.zero_()
        if k.grad is not None: k.grad.data.zero_()
        if v.grad is not None: v.grad.data.zero_()
        if w.grad is not None: w.grad.data.zero_()
        if u.grad is not None: u.grad.data.zero_()
        if r1.grad is not None: r1.grad.data.zero_()
        if k1.grad is not None: k1.grad.data.zero_()
        if v1.grad is not None: v1.grad.data.zero_()
        if w1.grad is not None: w1.grad.data.zero_()
        if u1.grad is not None: u1.grad.data.zero_()
        if initial_state.grad is not None: initial_state.grad.data.zero_()

    # Check reuse the first state
    clear_grad()
    y32, state = RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float(), initial_state.float())
    y32_1, _ = RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, r1.float(), k1.float(), v1.float(), w1.float(), u1.float(), state.float())
    loss = LOSS(y32_1) + LOSS(y32/100)
    loss.backward()
    gr = r.grad.data.clone()
    gk = k.grad.data.clone()
    gv = v.grad.data.clone()
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()
    gr1 = r1.grad.data.clone()
    gk1 = k1.grad.data.clone()
    gv1 = v1.grad.data.clone()
    gw1 = w1.grad.data.clone()
    gu1 = u1.grad.data.clone()
    gh = initial_state.grad.data.clone()


    clear_grad()
    yF16, state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float(), initial_state.float())
    yF16_1, _ = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r1.float(), k1.float(), v1.float(), w1.float(), u1.float(), state.float())
    loss = LOSS(yF16_1) + LOSS(yF16/100)
    loss.backward()
    gr_chunk = r.grad.data.clone()
    gk_chunk = k.grad.data.clone()
    gv_chunk = v.grad.data.clone()
    gw_chunk = w.grad.data.clone()
    gu_chunk = u.grad.data.clone()
    gr_chunk1 = r1.grad.data.clone()
    gk_chunk1 = k1.grad.data.clone()
    gv_chunk1 = v1.grad.data.clone()
    gw_chunk1 = w1.grad.data.clone()
    gu_chunk1 = u1.grad.data.clone()
    gh_chunk = initial_state.grad.data.clone()
    clear_grad()

    assert get_err_ratio(yF16_1, y32_1) < atol, f"output, {get_err_ratio(yF16_1, y32_1)}"
    assert get_err_ratio(gr_chunk, gr) < atol, f"r, {get_err_ratio(gr_chunk, gr)}"
    assert get_err_ratio(gk_chunk, gk) < atol, f"k, {get_err_ratio(gk_chunk, gk)}"
    assert get_err_ratio(gv_chunk, gv) < atol, f"v, {get_err_ratio(gv_chunk, gv)}"
    # assert get_err_ratio(gw_chunk, gw) < atol # This will fail because of the log space
    assert get_err_ratio(gu_chunk, gu) < atol, f"u, {get_err_ratio(gu_chunk, gu)}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}"
    assert get_err_ratio(gr_chunk1, gr1) < atol, f"r1, {get_err_ratio(gr_chunk1, gr1)}"
    assert get_err_ratio(gk_chunk1, gk1) < atol, f"k1, {get_err_ratio(gk_chunk1, gk1)}"
    assert get_err_ratio(gv_chunk1, gv1) < atol, f"v1, {get_err_ratio(gv_chunk1, gv1)}"
    assert get_err_ratio(gw_chunk1, gw1) < atol, f"w1, {get_err_ratio(gw_chunk1, gw1)}"
    assert get_err_ratio(gu_chunk1, gu1) < atol, f"u1, {get_err_ratio(gu_chunk1, gu1)}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}"

    clear_grad()
    y32, state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float(), initial_state.float())
    y32_1, _ = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r1.float(), k1.float(), v1.float(), w1.float(), u1.float(), state.float())
    loss = LOSS(y32_1) + LOSS(y32/100)
    loss.backward()
    gr = r.grad.data.clone()
    gk = k.grad.data.clone()
    gv = v.grad.data.clone()
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()
    gr1 = r1.grad.data.clone()
    gk1 = k1.grad.data.clone()
    gv1 = v1.grad.data.clone()
    gw1 = w1.grad.data.clone()
    gu1 = u1.grad.data.clone()
    gh = initial_state.grad.data.clone()


    clear_grad()
    yF16, state = RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, initial_state)
    yF16_1, _ = RUN_FLA_CHUNK(B, T, C, H, r1, k1, v1, w1, u1, state)
    loss = LOSS(yF16_1) + LOSS(yF16/100)
    loss.backward()
    gr_chunk = r.grad.data.clone()
    gk_chunk = k.grad.data.clone()
    gv_chunk = v.grad.data.clone()
    gw_chunk = w.grad.data.clone()
    gu_chunk = u.grad.data.clone()
    gr_chunk1 = r1.grad.data.clone()
    gk_chunk1 = k1.grad.data.clone()
    gv_chunk1 = v1.grad.data.clone()
    gw_chunk1 = w1.grad.data.clone()
    gu_chunk1 = u1.grad.data.clone()
    gh_chunk = initial_state.grad.data.clone()
    clear_grad()

    assert get_err_ratio(yF16_1, y32_1) < atol, f"output, {get_err_ratio(yF16_1, y32_1)}"
    assert get_err_ratio(gr_chunk, gr) < atol, f"r, {get_err_ratio(gr_chunk, gr)}"
    assert get_err_ratio(gk_chunk, gk) < atol, f"k, {get_err_ratio(gk_chunk, gk)}"
    assert get_err_ratio(gv_chunk, gv) < atol, f"v, {get_err_ratio(gv_chunk, gv)}"
    assert get_err_ratio(gw_chunk, gw) < atol, f"w, {get_err_ratio(gw_chunk, gw)}"
    assert get_err_ratio(gu_chunk, gu) < atol, f"u, {get_err_ratio(gu_chunk, gu)}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}"
    assert get_err_ratio(gr_chunk1, gr1) < atol, f"r1, {get_err_ratio(gr_chunk1, gr1)}"
    assert get_err_ratio(gk_chunk1, gk1) < atol, f"k1, {get_err_ratio(gk_chunk1, gk1)}"
    assert get_err_ratio(gv_chunk1, gv1) < atol, f"v1, {get_err_ratio(gv_chunk1, gv1)}"
    assert get_err_ratio(gw_chunk1, gw1) < atol, f"w1, {get_err_ratio(gw_chunk1, gw1)}"
    assert get_err_ratio(gu_chunk1, gu1) < atol, f"u1, {get_err_ratio(gu_chunk1, gu1)}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}"



    clear_grad()
    y32, state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r.float(), k.float(), v.float(), w.float(), u.float(), initial_state.float())
    y32_1, _ = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r1.float(), k1.float(), v1.float(), w1.float(), u1.float(), state.float())
    loss = LOSS(y32_1) + LOSS(y32/100)
    loss.backward()
    gr = r.grad.data.clone()
    gk = k.grad.data.clone()
    gv = v.grad.data.clone()
    gw = w.grad.data.clone()
    gu = u.grad.data.clone()
    gr1 = r1.grad.data.clone()
    gk1 = k1.grad.data.clone()
    gv1 = v1.grad.data.clone()
    gw1 = w1.grad.data.clone()
    gu1 = u1.grad.data.clone()
    gh = initial_state.grad.data.clone()

    clear_grad()
    yF16, state = RUN_FLA_FUSED(B, T, C, H, r, k, v, w, u, initial_state)
    yF16_1, _ = RUN_FLA_FUSED(B, T, C, H, r1, k1, v1, w1, u1, state)
    loss = LOSS(yF16_1) + LOSS(yF16/100)
    loss.backward()
    gr_chunk = r.grad.data.clone()
    gk_chunk = k.grad.data.clone()
    gv_chunk = v.grad.data.clone()
    gw_chunk = w.grad.data.clone()
    gu_chunk = u.grad.data.clone()
    gr_chunk1 = r1.grad.data.clone()
    gk_chunk1 = k1.grad.data.clone()
    gv_chunk1 = v1.grad.data.clone()
    gw_chunk1 = w1.grad.data.clone()
    gu_chunk1 = u1.grad.data.clone()
    gh_chunk = initial_state.grad.data.clone()
    clear_grad()

    assert get_err_ratio(yF16_1, y32_1) < atol, f"output, {get_err_ratio(yF16_1, y32_1)}"
    assert get_err_ratio(gr_chunk, gr) < atol, f"r, {get_err_ratio(gr_chunk, gr)}"
    assert get_err_ratio(gk_chunk, gk) < atol, f"k, {get_err_ratio(gk_chunk, gk)}"
    assert get_err_ratio(gv_chunk, gv) < atol, f"v, {get_err_ratio(gv_chunk, gv)}"
    assert get_err_ratio(gw_chunk, gw) < atol, f"w, {get_err_ratio(gw_chunk, gw)}"
    assert get_err_ratio(gu_chunk, gu) < atol, f"u, {get_err_ratio(gu_chunk, gu)}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}"
    assert get_err_ratio(gr_chunk1, gr1) < atol, f"r1, {get_err_ratio(gr_chunk1, gr1)}"
    assert get_err_ratio(gk_chunk1, gk1) < atol, f"k1, {get_err_ratio(gk_chunk1, gk1)}"
    assert get_err_ratio(gv_chunk1, gv1) < atol, f"v1, {get_err_ratio(gv_chunk1, gv1)}"
    assert get_err_ratio(gw_chunk1, gw1) < atol, f"w1, {get_err_ratio(gw_chunk1, gw1)}"
    assert get_err_ratio(gu_chunk1, gu1) < atol, f"u1, {get_err_ratio(gu_chunk1, gu1)}"

@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [512])
@pytest.mark.parametrize("C", [4096])
@pytest.mark.parametrize("HEAD_SIZE", [64])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
def test_multi_state_backworad_with_native(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype
):
    atol = 1e-3 if dtype == torch.float else 1e-2
    import torch.nn as nn
    H = C // HEAD_SIZE
    with torch.no_grad():
        u = torch.empty(H, HEAD_SIZE, device=device, dtype=dtype).uniform_(-1, 1).requires_grad_(True)
        image_feature = torch.randn(B, T, C, device=device, dtype=dtype).requires_grad_(True)
        text_emb = torch.randn(B, T, C, device=device, dtype=dtype).requires_grad_(True)

    def clear_image_grad():
        image_feature.requires_grad_()
        u.requires_grad_()
        text_emb.requires_grad_()
        if image_feature.grad is not None: image_feature.grad.data.zero_()
        if u.grad is not None: u.grad.data.zero_()
        if text_emb.grad is not None: text_emb.grad.data.zero_()
        if proj_layer.weight.grad is not None: proj_layer.weight.grad.data.zero_()
    proj_layer = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    linear_r = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    linear_w = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    linear_k = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    linear_v = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
    linear_r.requires_grad_(False)
    linear_w.requires_grad_(False)
    linear_k.requires_grad_(False)
    linear_v.requires_grad_(False)
    img = proj_layer(image_feature)
    r_img = linear_r(img)
    w_img = linear_w(img)
    k_img = linear_k(img)
    v_img = linear_v(img)
    y_img, img_state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, HEAD_SIZE, r_img.float(), k_img.float(), v_img.float(), w_img.float(), u.float(), h=None)

    r_text = linear_r(text_emb)
    w_text = linear_w(text_emb)
    k_text = linear_k(text_emb)
    v_text = linear_v(text_emb)
    y_text, text_state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r_text.float(), k_text.float(), v_text.float(), w_text.float(), u.float(), h=img_state.float())

    LOSS(y_text).backward()
    gproj = proj_layer.weight.grad.data.clone()
    clear_image_grad()
    img = proj_layer(image_feature)
    r_img = linear_r(img)
    w_img = linear_w(img)
    k_img = linear_k(img)
    v_img = linear_v(img)
    y_img, img_state = RUN_FLA_FUSED(B, T, C, HEAD_SIZE, r_img, k_img, v_img, w_img, u, h=None)


    r_text = linear_r(text_emb)
    w_text = linear_w(text_emb)
    k_text = linear_k(text_emb)
    v_text = linear_v(text_emb)
    y_text, text_state = RUN_FLA_FUSED(B, T, C, H, r_text, k_text, v_text, w_text, u, h=img_state)

    LOSS(y_text).backward()
    gproj1 = proj_layer.weight.grad.data.clone()
    assert get_err_ratio(gproj, gproj1) < atol, f"proj, {get_err_ratio(gproj, gproj1)}"
    has_non_zero = torch.any(gproj1 != 0).item()
    assert has_non_zero, "gproj1 is all zeros!"

    clear_image_grad()
    img = proj_layer(image_feature)
    r_img = linear_r(img)
    w_img = linear_w(img)
    k_img = linear_k(img)
    v_img = linear_v(img)
    y_img, img_state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, HEAD_SIZE, r_img.float(), k_img.float(), v_img.float(), w_img.float(), u.float(), h=None)

    r_text = linear_r(text_emb)
    w_text = linear_w(text_emb)
    k_text = linear_k(text_emb)
    v_text = linear_v(text_emb)
    y_text, text_state = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r_text.float(), k_text.float(), v_text.float(), w_text.float(), u.float(), h=img_state.float())

    LOSS(y_text).backward()
    gproj = proj_layer.weight.grad.data.clone()
    clear_image_grad()
    img = proj_layer(image_feature)
    r_img = linear_r(img)
    w_img = linear_w(img)
    k_img = linear_k(img)
    v_img = linear_v(img)
    y_img, img_state = RUN_FLA_CHUNK(B, T, C, HEAD_SIZE, r_img, k_img, v_img, w_img, u, h=None)


    r_text = linear_r(text_emb)
    w_text = linear_w(text_emb)
    k_text = linear_k(text_emb)
    v_text = linear_v(text_emb)
    y_text, text_state = RUN_FLA_CHUNK(B, T, C, H, r_text, k_text, v_text, w_text, u, h=img_state)

    LOSS(y_text).backward()
    assert get_err_ratio(gproj, gproj1) < atol, f"proj, {get_err_ratio(gproj, gproj1)}"
    has_non_zero = torch.any(gproj1 != 0).item()
    assert has_non_zero, "gproj1 is all zeros!"


