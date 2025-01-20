# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange
from fla.utils import device
from fla.ops.rwkv7.recurrent_naive import naive_recurrent_rwkv7, native_recurrent_rwkv7
from fla.ops.rwkv7.chunk import chunk_rwkv7
from fla.ops.rwkv7.fused_recurrent import fused_recurrent_rwkv7
from fla.ops.generalized_delta_rule.dplr.chunk import chunk_dplr_delta_rule
from fla.utils import device
import os

def get_abs_err(x, y):
    return (x - y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


def assert_close(prefix, ref, tri, ratio):
    msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
    assert get_err_ratio(ref, tri) < ratio, msg


torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [1024])
@pytest.mark.parametrize("C", [512])
@pytest.mark.parametrize("HEAD_SIZE", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("scale", [1.0])
def test_forward(
    B: int, T: int, C: int, HEAD_SIZE: int, dtype: torch.dtype, scale: float
):
    torch.manual_seed(42)
    H = C // HEAD_SIZE
    L = T
    D = HEAD_SIZE
    r = (
        torch.empty(B, H, L, D, device=device)
        .uniform_(-1, 1)
        .to(dtype=dtype)
        .requires_grad_(True)
    )
    k = (
        torch.empty(B, H, L, D, device=device)
        .uniform_(-1, 1)
        .to(dtype=dtype)
        .requires_grad_(True)
    )
    v = (
        torch.empty(B, H, L, D, device=device)
        .uniform_(-1, 1)
        .to(dtype=dtype)
        .requires_grad_(True)
    )

    # 时间衰减参数w，保持在负数范围
    w = (
        torch.empty(B, H, L, D, device=device)
        .uniform_(-8, -6)
        .to(dtype=dtype)
        .requires_grad_(True)
    )
    h0 = torch.randn(B, H, D, D, device=device, dtype=torch.float32)

    # 生成归一化的kk
    kk = torch.empty(B, H, L, D, device=device).uniform_(-1, 1)
    # 对 kk l2norm
    kk = torch.nn.functional.normalize(kk, p=2, dim=-1).to(dtype=dtype)

    # 生成a参数（对应-kk）和b参数（对应kk*a）
    a = -kk.clone().requires_grad_(True)  # -kk
    b = (kk * a).requires_grad_(True)  # kk*a

    with torch.no_grad():
        ref_o, ref_state, _ = naive_recurrent_rwkv7(
            r,
            k,
            v,
            w,
            a,
            b,
            scale=scale,
            initial_state=h0.transpose(-1, -2),
            output_final_state=True,
        )
        result, state = fused_recurrent_rwkv7(
            r, k, v, w, a, b, initial_state=h0, scale=scale, head_first=True, training=False, use_log_w=False
        )
        result2, state2 = chunk_rwkv7(
            r=r,
            k=k,
            v=v,
            w=w,
            a=a,
            b=b,
            initial_state=h0,
            scale=scale,
            use_log_w=False,
            head_first=True
        )

        ref_o = ref_o.to(dtype=torch.float32).to(device)
        result = result.to(dtype=torch.float32).to(device)
        result2 = result2.to(dtype=torch.float32).to(device)
        ref_state = ref_state.to(dtype=torch.float32).to(device)
        state = state.to(dtype=torch.float32).to(device)
        state2 = state2.to(dtype=torch.float32).to(device)
        tol = 3e-1
        diff = torch.abs(result - ref_o)
        diff_state = torch.abs(state - ref_state.transpose(-1, -2))
        assert diff.max().item() < tol
        assert diff_state.max().item() < tol

        tol = 5e-1
        diff2 = torch.abs(result2 - ref_o)
        diff_state2 = torch.abs(state2 - ref_state.transpose(-1, -2))
        assert diff2.max().item() < tol
        assert diff_state2.max().item() < tol



def chunk_dplr_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    scale: float = None,
    chunk_size: int = 64,
    head_first: bool = True,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        b = b.transpose(1, 2)
        gk = gk.transpose(1, 2)
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT

    q, k, v, a, b, gk = map(lambda x: F.pad(x, (0, 0, 0, pad_len)).to(torch.float32), [q, k, v, a, b, gk])
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    q = q * scale

    S = k.new_zeros(B, H, DK, DV)
    if initial_state is not None:
        S += initial_state

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, a, b, gk = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, a, b, gk])
    gk_cumsum = gk.cumsum(-2)
    A_ab = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)
    A_qk = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)
    A_ak = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)
    A_qb = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)

    for i in range(chunk_size):
        a_i = a[:, :, :, i, None]
        q_i = q[:, :, :, i, None]
        gk_i = gk_cumsum[:, :, :, i, None]
        mask = (torch.arange(chunk_size) <= i).to(q.device)
        attn_i = (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
        A_qb[:, :, :, i, :] = (q_i * b * attn_i).sum(-1).clone()
        mask = (torch.arange(chunk_size) < i).to(q.device)
        # shift by one.
        attn_i = (gk_i - gk[:,:,:,i,None] - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_ab[:, :, :, i, :] = (a_i * b * attn_i).sum(-1).clone()
        A_ak[:, :, :, i, :] = (a_i * k * attn_i).sum(-1).clone()

    A_ab = A_ab
    for i in range(1, chunk_size):
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)

    A_ab = A_ab + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    u = A_ab @ (A_ak @ v)
    w = A_ab @ ((gk_cumsum-gk).exp() * a)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, L // chunk_size):
        q_i, k_i, v_i, u_i, w_i, b_i = q[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], b[:, :, i]
        v2_i = u_i + w_i @ S
        o_1 = A_qk[:, :, i] @ v_i
        o_2 = A_qb[:, :, i] @ v2_i
        o_3 = (q_i * gk_cumsum[:, :, i].exp()) @ S
        o[:, :, i] = o_1 + o_2 + o_3
        decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
        S = S*gk_cumsum[:, :, i, -1, :, None].exp() + (k_i * decay).transpose(-1, -2) @ v_i + (b_i * decay).transpose(-1, -2) @ v2_i

    S = None if output_final_state is False else S
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    if not head_first:
        o = o.transpose(1, 2)
    return o, S


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("gate_logit_normalizer", [1, 0.1, 10])
@pytest.mark.parametrize("T", [256, 300])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("D", [256, 100])
@pytest.mark.parametrize("scale", [0.25])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
    head_first: bool,
):
    if head_first:
        q = torch.randn(B, H, T, D, dtype=dtype)
        k = torch.randn(B, H, T, D, dtype=dtype)
        v = torch.randn(B, H, T, D, dtype=dtype)
        a = torch.rand(B, H, T, D, dtype=dtype)
        gk = (torch.randn(B, H, T, D, dtype=torch.float)) 
    else:
        q = torch.randn(B, T, H, D, dtype=dtype)
        k = torch.randn(B, T, H, D, dtype=dtype)
        v = torch.randn(B, T, H, D, dtype=dtype)
        a = torch.rand(B, T, H, D, dtype=dtype)
        gk = torch.randn(B, T, H, D, dtype=torch.float)

    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = -a
    gk = torch.nn.functional.logsigmoid(gk) / gate_logit_normalizer
 
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, a, b, gk, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, a, b, gk, h0))
    ref, ref_ht = chunk_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        head_first=head_first
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_da, ref_db, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad
    q.grad = k.grad = v.grad = a.grad = b.grad = gk.grad = h0.grad = None   

    tri, tri_ht = chunk_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        head_first=head_first
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_da, tri_db, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad
    q.grad = k.grad = v.grad = a.grad = b.grad = gk.grad = h0.grad = None

    assert_close("  o", ref, tri, 0.007)
    assert_close(" ht", ref_ht, tri_ht, 0.008)
    assert_close(" dq", ref_dq, tri_dq, 0.008)
    assert_close(" dk", ref_dk, tri_dk, 0.008)
    assert_close(" dv", ref_dv, tri_dv, 0.008)
    assert_close(" da", ref_da, tri_da, 0.008)
    assert_close(" db", ref_db, tri_db, 0.008)
    if gate_logit_normalizer >= 1 and ref_dg.norm() > 0.01: # otherwise it is meaningless
        assert_close(" dg", ref_dg, tri_dg, 0.008)
    assert_close(" dh0", ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [64, 128, 200, 250, 256, 300, 400, 512, 1000, 2048])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("D", [50, 100, 200])
@pytest.mark.parametrize("scale", [0.25])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_varlen(
    N: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).to(device).sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn(1, T, H, D, dtype=dtype)
    k = torch.randn(1, T, H, D, dtype=dtype)
    v = torch.randn(1, T, H, D, dtype=dtype)
    a = torch.rand(1, T, H, D, dtype=dtype)
    gk = torch.randn(1, T, H, D, dtype=torch.float)
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = -a
    gk = torch.nn.functional.logsigmoid(gk)
    h0 = torch.randn(N, H, D, D, dtype=torch.float32)
    q, k, v, a, b, gk, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, a, b, gk, h0))

    tri, tri_ht = chunk_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        offsets=offsets,
        head_first=False
    )
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_da, tri_db, tri_dg, tri_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad
    q.grad = k.grad = v.grad = a.grad = b.grad = gk.grad = h0.grad = None

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = chunk_dplr_delta_rule_ref(
            q=q[:, offsets[i]:offsets[i+1]],
            k=k[:, offsets[i]:offsets[i+1]],
            v=v[:, offsets[i]:offsets[i+1]],
            a=a[:, offsets[i]:offsets[i+1]],
            b=b[:, offsets[i]:offsets[i+1]],
            gk=gk[:, offsets[i]:offsets[i+1]],
            scale=scale,
            initial_state=h0[i, None],
            output_final_state=True,
            head_first=False
        )
        ref.append(ref_i)
        ref_ht.append(ref_ht_i)

    ref = torch.cat(ref, 1)
    ref_ht = torch.cat(ref_ht, 0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_da, ref_db, ref_dg, ref_dh0 = q.grad, k.grad, v.grad, a.grad, b.grad, gk.grad, h0.grad

    assert_close("  o", ref, tri, 0.007)
    assert_close(" ht", ref_ht, tri_ht, 0.008)
    assert_close(" dq", ref_dq, tri_dq, 0.008)
    assert_close(" dk", ref_dk, tri_dk, 0.008)
    assert_close(" dv", ref_dv, tri_dv, 0.008)
    assert_close(" da", ref_da, tri_da, 0.008)
    assert_close(" db", ref_db, tri_db, 0.008)
    assert_close(" dg", ref_dg, tri_dg, 0.008)
    assert_close(" dh0", ref_dh0, tri_dh0, 0.008)
