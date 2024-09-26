# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import check_pytorch_version, device


def naive_recurrent_rwkv7(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    manual_einsum: bool = False,
):
    """
    Naive recurrent implementation of RWKV-7 (Goose) attention mechanism.

    Args:
        q, k, v: Query, Key, and Value tensors
        w: Time decay weights
        a: Dynamic learning rate modulator, influences the in-context learning rate
        b: State update modulator, directly participates in state update calculation
        scale: Scaling factor for attention scores
        initial_state: Initial state for the recurrent computation
        output_final_state: Whether to output the final state

    Returns:
        Attention output and optionally the final state
    """
    if (torch.is_autocast_enabled(device) if check_pytorch_version('2.4') else torch.is_autocast_enabled()):
        torch_dtype = torch.get_autocast_dtype(device) if check_pytorch_version('2.4') else torch.get_autocast_gpu_dtype()
    else:
        torch_dtype = torch.float32 if q.dtype != torch.float16 else torch.float16
    orig_dtype = q.dtype
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    w = torch.exp(w)
    if manual_einsum:
        state_new = torch.zeros_like(state)
        o1 = torch.zeros(B, H, N, dtype=torch_dtype, device=q.device)
    for t in range(L):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t] * scale
        v_t = v[:, :, t] * scale
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        if not manual_einsum:
            sab = torch.einsum('bhik,bhk,bhj->bhij', state, a_t, b_t)
            state = state * w[:, :, t, None, :] + sab + torch.einsum('bhj,bhi->bhij', k_t, v_t)
            o[:, :, t] = torch.einsum('bhj,bhij->bhi', q_t, state)
        else:
            # o1 = torch.zeros(B, H, N, dtype=torch_dtype, device=q.device)
            # for i in range(N):
            #     temp = torch.sum(state[:, :, i, :] * a_t[:, :, :], dim=-1)
            #     for j in range(N):
            #         # Update new_state
            #         state_new[:, :, i, j] = state[:, :, i, j] * w_t[:, :, j] \
            #             + temp * b_t[:, :, j] + k_t[:, :, j] * v_t[:, :, i]
            #         # Compute o1
            #         o1[:, :, i] += q_t[:, :, j] * state[:, :, i, j]

            # o[:, :, t] = o1
            # state = state_new
            # Also equivalent to the following code
            # Calculate sab
            sa = torch.bmm(state.view(B*H, V, V), a_t.view(B*H, V, 1))
            sab = torch.bmm(sa.view(B*H, V, 1), b_t.view(B*H, 1, V))

            # Update state
            kv = torch.bmm(k_t.view(B*H, V, 1), v_t.view(B*H, 1, V))
            state = state * w[:, :, t, None, :] + sab.view(B, H, V, V) + kv.view(B, H, V, V)
            # Calculate o
            o_t = torch.bmm(q_t.unsqueeze(2).transpose(2, 3).reshape(B*H, 1, V),
                            state.transpose(2, 3).reshape(B*H, V, V)).reshape(B, H, 1, V).squeeze(2)
            o[:, :, t] = o_t.view(B, H, V)

    ht = state if output_final_state else None
    return o.to(orig_dtype), ht


@torch.no_grad()
def naive_recurrent_rwkv7_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    doutput: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None
):
    if dtype is None:
        torch_dtype = torch.float32 if q.dtype != torch.float16 else torch.float16
    else:
        torch_dtype = dtype
    q, k, v, w, a, b, doutput = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b, doutput))
    if dh_t is not None:
        dh_t = dh_t.to(dtype=torch_dtype)
    B, H, L, N, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, a, b = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b))
    # q, k, v, a, b, w,
    # shape: (B, H, L, D), (B, H, L, D), (B, H, T, V), (B, H, L, D), (B, H, L, D), (B, H, L, D)
    state = torch.zeros(B, H, N, V, dtype=torch_dtype, device=q.device)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    dw = torch.zeros_like(w)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    dstate = torch.zeros_like(state)

    if dh_t is not None:
        dstate += dh_t

    if scale == -1.0:
        scale = N ** -0.5

    if initial_state is not None:
        state += initial_state.to(dtype=torch_dtype)

    w = torch.exp(w)

    # 从前向后计算所有时间步的状态
    states = [torch.zeros((B, H, D, D), dtype=torch_dtype, device=device) if initial_state is None else initial_state]
    for t in range(L):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t] * scale
        v_t = v[:, :, t] * scale
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        temp = torch.bmm(state.view(B*H, V, V), a_t.view(B*H, V, 1))
        sab = torch.bmm(temp.view(B*H, V, 1), b_t.view(B*H, 1, V))

        # Update state
        kv = torch.bmm(k_t.view(B*H, V, 1), v_t.view(B*H, 1, V))
        state = state * w[:, :, t, None, :] + sab.view(B, H, V, V) + kv.view(B, H, V, V)
        states.append(state)

    dstate = torch.zeros_like(state, dtype=torch_dtype) if dh_t is None else dh_t

    for t in range(L-1, -1, -1):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t] * scale
        v_t = v[:, :, t] * scale
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        w_t = w[:, :, t]
        state = states[t+1]
        prev_state = states[t]

        # Gradient of output
        # torch.einsum('bhi,bhij->bhj', doutput[:, :, t], state)
        dq[:, :, t] += torch.bmm(doutput[:, :, t].view(B*H, 1, V), state.view(B*H, V, V)).view(B,H,V) * scale

        # torch.einsum('bhi,bhj->bhij', doutput[:, :, t], q_t)
        dstate += torch.mul(doutput[:, :, t].unsqueeze(3), q_t.unsqueeze(2))


        # Gradient of state update
        dw[:, :, t] += torch.sum(dstate * prev_state, dim=(-2)) * w_t

        # Gradient of sab
        # torch.einsum('bhij,bhik,bhj->bhk', dstate, prev_state, b_t)
        temp = torch.bmm(dstate.view(B*H, V, V).permute(0, 2, 1), prev_state.view(B*H, V, V)).view(B*H, V, V)
        da[:, :, t] += torch.bmm(temp.permute(0, 2, 1), b_t.view(B*H, V, 1)).view(B, H, V)
        
        # torch.einsum('bhij,bhik,bhk->bhj', dstate, prev_state, a_t)
        db[:, :, t] += torch.bmm(temp, a_t.view(B*H, V, 1)).view(B, H, V)

        # Gradient of k_t * v_t
        # torch.einsum('bhij,bhi->bhj', dstate, v_t)
        dk[:, :, t] += torch.bmm(dstate.view(B*H, V, V).permute(0, 2, 1), v_t.view(B*H, V, 1)).view(B, H, V) * scale
        # torch.einsum('bhij,bhj->bhi', dstate, k_t)
        dv[:, :, t] += torch.bmm(dstate.view(B*H, V, V), k_t.view(B*H, V, 1)).view(B, H, V) * scale

        # Gradient for previous state
        # torch.einsum('bhij,bhk,bhj->bhik', dstate, a_t, b_t)
        mul_result = dstate.unsqueeze(3) * a_t.unsqueeze(2).unsqueeze(-1) # [B, H, V, 1, V] * [B, H, 1, V, 1] = [B, H, V, 1, V]
        dprev_state = torch.bmm(mul_result.view(B*H, V*V, V), b_t.view(B*H, V, 1)).view(B, H, V, V)
        
        dprev_state += dstate *  w[:, :, t, None, :]

        # Update dstate for next iteration
        dstate = dprev_state

    return dq, dk, dv, dw, da, db, dstate


class NativeRecurrentRWKV7Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, a, b, scale, initial_state, training: bool = True, dtype: Optional[torch.dtype] = None):
        o, ht = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=scale, initial_state=None, manual_einsum=False)
        if initial_state is not None:
            initial_state = initial_state.clone()
        if training:
            ctx.save_for_backward(q, k, v, w, a, b, o, initial_state)
            ctx.scale = scale
            ctx.dtype = dtype
        return o, ht

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, a, b, o, initial_state = ctx.saved_tensors
        dq, dk, dv, dw, da, db, dh = naive_recurrent_rwkv7_bwd(q, k, v, w, a, b, do, dht, initial_state, ctx.scale, dtype=ctx.dtype)
        dh = dh if initial_state is not None else None
        return dq, dk, dv, dw, da, db, None, dh, None, None


def native_recurrent_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    training: bool = True,
    dtype: Optional[torch.dtype] = None,
    causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `(B, H, T, K)`. Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            data-dependent decays of shape `(B, H, T, K)` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `(H, K)` or `(B, H, K)` for each head.
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
    """
    if scale == -1.0:
        scale = r.shape[-1] ** -0.5
    o, h_t = NativeRecurrentRWKV7Function.apply(r, k, v, w, a, b, scale, initial_state, training, dtype)
    final_state = h_t if output_final_state else None
    return o, final_state


if __name__ == "__main__":
    from fla.utils import get_available_device
    device = 'cpu'
    B = 4
    H = 64
    L = 32
    D = 64
    dtype = torch.float64
    require_grad = True
    torch.manual_seed(42)

    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-20)
    q = (torch.randn(B, H, L, D).to(device).to(dtype)).fill_(torch.rand(1).item()).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).to(device).to(dtype)).fill_(torch.rand(1).item()).requires_grad_(require_grad)
    v = torch.randn(B, H, L, D).to(device).to(dtype).fill_(torch.rand(1).item()).requires_grad_(require_grad)
    w = (torch.randn(B, H, L, D).uniform_(0.95, 0.9997).to(device).to(dtype)).requires_grad_(require_grad)

    # a: 基于 torch.sigmoid(...) * 2.0 的计算，范围在 [0, 2]
    a = torch.rand(B, H, L, D, device=device, dtype=dtype).clamp(0, 2).requires_grad_(require_grad)

    # b: 在模型中是 kk*a，其中 kk 是归一化的，所以范围可能在 [-2, 2]
    b = torch.randn(B, H, L, D, device=device, dtype=dtype).clamp(-2, 2).requires_grad_(require_grad)

    do = torch.rand_like(v).to(device).fill_(torch.rand(1).item())
    h = torch.zeros(B, H, D, D, device=device, dtype=torch.float32).fill_(torch.rand(1).item()).requires_grad_(require_grad)

    with torch.no_grad():
        ref_o, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=None, manual_einsum=False)
        ref_o1, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=None, manual_einsum=True)
        assert get_err_ratio(ref_o, ref_o1) < 1e-6
    ref_o, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=None, manual_einsum=False)
    ref_o.backward(do)
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_da, a.grad = a.grad.clone(), None
    ref_db, b.grad = b.grad.clone(), None

    # ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = native_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=None, dtype=dtype)
    tri_o.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_da, a.grad = a.grad.clone(), None
    tri_db, b.grad = b.grad.clone(), None

    atol = 1e-2

    assert get_err_ratio(ref_o, tri_o) < atol
    print(get_err_ratio(ref_dq, tri_dq))  # pass
    print(get_err_ratio(ref_dk, tri_dk))
    print(get_err_ratio(ref_dv, tri_dv))
    print(get_err_ratio(ref_dw, tri_dw))
    print(get_err_ratio(ref_da, tri_da))
    print(get_err_ratio(ref_db, tri_db))
    assert get_err_ratio(ref_dq, tri_dq) < atol
    assert get_err_ratio(ref_dk, tri_dk) < atol
    assert get_err_ratio(ref_dv, tri_dv) < atol
    assert get_err_ratio(ref_dw, tri_dw) < atol
    assert get_err_ratio(ref_da, tri_da) < atol
    assert get_err_ratio(ref_db, tri_db) < atol

