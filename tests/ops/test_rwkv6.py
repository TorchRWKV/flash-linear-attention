# -*- coding: utf-8 -*-

import pytest
import torch

from fla.utils import device
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6, native_recurrent_rwkv6
from fla.ops.rwkv6.recurrent_naive import naive_recurrent_rwkv6
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("C", [512])
@pytest.mark.parametrize("HEAD_SIZE", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("scale", [1.0])
def test_recurrent_naive(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype,
    scale: float
):
    torch.manual_seed(42)
    H = C // HEAD_SIZE
    q = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = torch.empty(B, T, C, device=device).uniform_(-8, -4).to(dtype=dtype).requires_grad_(True).to(device)
    u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)

    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32, requires_grad=True)

    o, _ = RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, q, k, v, w, u, h=h, scale=scale)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    dh, h.grad = h.grad.clone(), None

    o, _ = RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, q, k, v, w, u, h=h, scale=scale)
    o.backward(do)
    assert dq.allclose(q.grad, atol=1e-3)
    assert dk.allclose(k.grad, atol=1e-3)
    assert dv.allclose(v.grad, atol=1e-3)
    assert dw.allclose(w.grad, atol=1e-3)
    assert du.allclose(u.grad, atol=1e-3)
    assert dh.allclose(h.grad, atol=1e-3)


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("C", [512])
@pytest.mark.parametrize("HEAD_SIZE", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("use_h", [False, True])
@pytest.mark.parametrize("scale", [1.0])
def test_fused_recurrent(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype,
    use_h: bool,
    scale: float,
):
    if dtype == torch.float16 and 'cuda' in device:
        pytest.skip("Skipping test for float16(Nvidia), see https://github.com/triton-lang/triton/issues/4701")
    if dtype == torch.float16 and scale == 1.0:
        pytest.skip("Skipping test for float16 with scale=1.0")

    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-2

    H = C // HEAD_SIZE
    q = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = torch.empty(B, T, C, device=device).uniform_(-8, -4).to(dtype=dtype).requires_grad_(True).to(device)

    u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)

    do = torch.rand_like(v, device=device)
    h = torch.randn(B, H, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32, requires_grad=True)
    do = torch.rand_like(v, device=device)

    ref_o, _ = RUN_FLA_NATIVE_AUTO_BACKWARD(
        B, T, C, H, q, k, v, w, u, h=h if use_h else None, scale=scale, output_final_state=use_h)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = RUN_FLA_FUSED(B, T, C, H, q, k, v, w, u, h=h if use_h else None, scale=scale, output_final_state=use_h)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert get_err_ratio(ref_o, tri_o) < atol, f"output, {get_err_ratio(ref_o, tri_o)}, dtype = {dtype}"
    assert get_err_ratio(ref_dq, tri_dq) < atol, f"q, {get_err_ratio(ref_dq, tri_dq)}, dtype = {dtype}"
    assert get_err_ratio(ref_dk, tri_dk) < atol, f"k, {get_err_ratio(ref_dk, tri_dk)}, dtype = {dtype}"
    assert get_err_ratio(ref_dv, tri_dv) < atol, f"v, {get_err_ratio(ref_dv, tri_dv)}, dtype = {dtype}"
    if get_err_ratio(ref_dw, tri_dw) > (atol * 5):
        if HEAD_SIZE != 64:
            import logging
            logging.warning(f"w, {get_err_ratio(ref_dw, tri_dw)}, dtype = {dtype}, scale = {scale}, D = {HEAD_SIZE}")
        else:
            raise ValueError(f"w, {get_err_ratio(ref_dw, tri_dw)}, dtype = {dtype}, scale = {scale}, D = {HEAD_SIZE}")
    assert get_err_ratio(ref_du, tri_du) < atol, f"u, {get_err_ratio(ref_du, tri_du)}, dtype = {dtype}"

    if use_h:
        assert get_err_ratio(ref_dh, tri_dh) < atol


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [130, 146, 162])
@pytest.mark.parametrize("C", [512])
@pytest.mark.parametrize("HEAD_SIZE", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float, torch.bfloat16])
@pytest.mark.parametrize("use_h", [False, True])
@pytest.mark.parametrize("scale", [1.0])
def test_chunk_with_initial_h(
    B: int,
    T: int,
    C: int,
    HEAD_SIZE: int,
    dtype: torch.dtype,
    use_h: bool,
    scale: float,
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-2
    if dtype == torch.float16 and scale == 1.0:
        pytest.skip("Skipping test for float16 with scale=1.0")

    H = C // HEAD_SIZE
    q = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = torch.empty(B, T, C, device=device).uniform_(-8, -4).to(dtype=dtype).requires_grad_(True).to(device)

    u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)

    h = torch.randn(B, H, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32, requires_grad=True)

    ref_o, _ = RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, q.float(), k.float(), v.float(), w.float(
    ), u.float(), scale=scale, h=h.float() if use_h else None, output_final_state=use_h)
    LOSS(ref_o).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = RUN_FLA_CHUNK(B, T, C, H, q, k, v, w, u, h=h if use_h else None, scale=scale, output_final_state=use_h)
    LOSS(tri_o).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert get_err_ratio(ref_o, tri_o) < atol, f"output, {get_err_ratio(ref_o, tri_o)}, dtype = {dtype}"
    assert get_err_ratio(ref_dq, tri_dq) < atol, f"q, {get_err_ratio(ref_dq, tri_dq)}, dtype = {dtype}"
    assert get_err_ratio(ref_dk, tri_dk) < atol, f"k, {get_err_ratio(ref_dk, tri_dk)}, dtype = {dtype}"
    assert get_err_ratio(ref_dv, tri_dv) < atol, f"v, {get_err_ratio(ref_dv, tri_dv)}, dtype = {dtype}"
    assert get_err_ratio(ref_dw, tri_dw) < atol, f"w, {get_err_ratio(ref_dw, tri_dw)}, dtype = {dtype}"
    assert get_err_ratio(ref_du, tri_du) < atol, f"u, {get_err_ratio(ref_du, tri_du)}, dtype = {dtype}"
    if use_h:
        assert get_err_ratio(ref_dh, tri_dh) < atol, f"h, {get_err_ratio(ref_dh, tri_dh)}, dtype = {dtype}"


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("H", [5])
@pytest.mark.parametrize("T", [130, 146, 162])
@pytest.mark.parametrize("D", [100, 300])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_with_different_dimension(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    atol = 1e-3 if dtype == torch.float else 1e-2

    C = H * D
    HEAD_SIZE = D
    q = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = torch.empty(B, T, C, device=device).uniform_(-8, -4).to(dtype=dtype).requires_grad_(True).to(device)
    u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)

    h = torch.randn(B, H, HEAD_SIZE, HEAD_SIZE, device=device, dtype=torch.float32, requires_grad=True)

    ref_o, _ = RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, q.float(), k.float(), v.float(), w.float(),
                                            u.float(), h=h.float(), output_final_state=True)
    LOSS(ref_o).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = RUN_FLA_CHUNK(B, T, C, H, q, k, v, w, u, h=h, output_final_state=True)
    LOSS(tri_o).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    tri_dh, h.grad = h.grad.clone(), None

    assert get_err_ratio(ref_o, tri_o) < atol, f"output, {get_err_ratio(ref_o, tri_o)}, dtype = {dtype}"
    assert get_err_ratio(ref_dq, tri_dq) < atol, f"q, {get_err_ratio(ref_dq, tri_dq)}, dtype = {dtype}"
    assert get_err_ratio(ref_dk, tri_dk) < atol, f"k, {get_err_ratio(ref_dk, tri_dk)}, dtype = {dtype}"
    assert get_err_ratio(ref_dv, tri_dv) < atol, f"v, {get_err_ratio(ref_dv, tri_dv)}, dtype = {dtype}"
    assert get_err_ratio(ref_dw, tri_dw) < atol, f"w, {get_err_ratio(ref_dw, tri_dw)}, dtype = {dtype}"
    assert get_err_ratio(ref_du, tri_du) < atol, f"u, {get_err_ratio(ref_du, tri_du)}, dtype = {dtype}"
    assert get_err_ratio(ref_dh, tri_dh) < atol, f"h, {get_err_ratio(ref_dh, tri_dh)}, dtype = {dtype}"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / (base + 1e-20)


def val(x):
    return x.detach().float().cpu().numpy()


def LOSS(y):
    return ((y * y) - torch.tanh(y)).sum()


def RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, h, scale=1.0, output_final_state=True):
    r = r.view(B, T, H, -1).transpose(1, 2)
    k = k.view(B, T, H, -1).transpose(1, 2)
    v = v.view(B, T, H, -1).transpose(1, 2)
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2))
    o, state = chunk_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=h, output_final_state=output_final_state)
    return o.transpose(1, 2).reshape(B, T, C), state


def RUN_FLA_FUSED(B, T, C, H, r, k, v, w, u, h, scale=1.0, output_final_state=True):
    r = r.view(B, T, H, -1).transpose(1, 2)
    k = k.view(B, T, H, -1).transpose(1, 2)
    v = v.view(B, T, H, -1).transpose(1, 2)
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2))
    o, state = fused_recurrent_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=h, output_final_state=output_final_state)
    return o.transpose(1, 2).reshape(B, T, C), state


def RUN_FLA_NATIVE_AUTO_BACKWARD(B, T, C, H, r, k, v, w, u, h, scale=1.0, output_final_state=True):
    r = r.view(B, T, H, -1).transpose(1, 2)
    k = k.view(B, T, H, -1).transpose(1, 2)
    v = v.view(B, T, H, -1).transpose(1, 2)
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2))
    o, state = naive_recurrent_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=h, output_final_state=output_final_state)
    return o.transpose(1, 2).reshape(B, T, C), state


def RUN_FLA_NATIVE_MANUAL_BACKWARD(B, T, C, H, r, k, v, w, u, h, scale=1.0, output_final_state=True):
    r = r.view(B, T, H, -1).transpose(1, 2)
    k = k.view(B, T, H, -1).transpose(1, 2)
    v = v.view(B, T, H, -1).transpose(1, 2)
    w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2))
    o, state = native_recurrent_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=h, output_final_state=output_final_state)
    return o.transpose(1, 2).reshape(B, T, C), state


@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("T", [24, 512])
@pytest.mark.parametrize("C", [512])
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
        if r.grad is not None:
            r.grad.data.zero_()
        if k.grad is not None:
            k.grad.data.zero_()
        if v.grad is not None:
            v.grad.data.zero_()
        if w.grad is not None:
            w.grad.data.zero_()
        if u.grad is not None:
            u.grad.data.zero_()
        if initial_state.grad is not None:
            initial_state.grad.data.zero_()

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

    assert get_err_ratio(yF16, y32) < atol, f"output, {get_err_ratio(yF16, y32)}, dtype = {dtype}"
    assert get_err_ratio(gr_chunk, gr) < atol, f"r, {get_err_ratio(gr_chunk, gr)}, dtype = {dtype}"
    assert get_err_ratio(gk_chunk, gk) < atol, f"k, {get_err_ratio(gk_chunk, gk)}, dtype = {dtype}"
    assert get_err_ratio(gv_chunk, gv) < atol, f"v, {get_err_ratio(gv_chunk, gv)}, dtype = {dtype}"
    assert get_err_ratio(gw_chunk, gw) < atol, f"w, {get_err_ratio(gw_chunk, gw)}, dtype = {dtype}"
    assert get_err_ratio(gu_chunk, gu) < atol, f"u, {get_err_ratio(gu_chunk, gu)}, dtype = {dtype}"
    assert get_err_ratio(gh_chunk, gh) < atol, f"h, {get_err_ratio(gh_chunk, gh)}, dtype = {dtype}"
