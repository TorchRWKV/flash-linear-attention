# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import device


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
    state_ckpt: bool = False,
    state_ckpt_interval: int = 16
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
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
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

    state_cache = []

    for t in range(L):
        q_t = q[:, :, t] * scale
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        if t % state_ckpt_interval == 0 and state_ckpt:
            state_cache.append(state.detach())

        # from bo's code
        sab = torch.einsum('bhik,bhk,bhj->bhij', state, a_t, b_t)
        state = state * torch.exp(-torch.exp(w[:, :, t, None, :])) + sab + torch.einsum('bhj,bhi->bhij', k_t, v_t)
        o[:, :, t] = torch.einsum('bhj,bhij->bhi', q_t, state)

    if not output_final_state:
        ht = None
    elif initial_state is not None:
        ht = state.to(initial_state.dtype)
    else:
        ht = state.to(orig_dtype)

    state_cache = torch.stack(state_cache) if state_ckpt else None
    return o.to(orig_dtype), ht, state_cache if state_ckpt else None


def naive_recurrent_rwkv7_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,  # Dynamic learning rate modulator
    b: torch.Tensor,  # State update modulator
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
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
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
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

    for t in range(L):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t] * scale
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = torch.exp(-torch.exp(w[bi, hi, t]))

                # 计算sa
                sa = (a_t[:, None] * state[bi, hi]).sum(dim=0)

                # 2. 更新state和计算输出 (与kernel一致)
                state[bi, hi] = (state[bi, hi] * w_t[:, None] +  # [N,V] * [N,1]
                                 k_t[:, None] * v_t[None, :] +     # [N,1] * [1,V]
                                 sa * b_t[:, None])                # [V] * [N,1]

                # 计算输出(矩阵乘法)
                y = (state[bi, hi] * q_t[:, None]).sum(dim=0)

                o[bi, hi, t] = y

    ht = state if output_final_state else None
    return o.to(orig_dtype), ht,  None


@torch.no_grad()
def naive_recurrent_rwkv7_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    doutput: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    state_cache: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    state_ckpt_interval: int = 16
):
    torch_dtype = q.dtype if q.dtype in [torch.float64, torch.float] else torch.float
    q, k, v, w, a, b, doutput, state_cache = (x.to(dtype=torch_dtype) for x in (q, k, v, w, a, b, doutput, state_cache))
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

    state += state_cache[0]

    def reconstruct_state_cache(state_t, last_ckpt, length):
        states = [state_t]
        for t in range(last_ckpt, length):
            k_t, v_t, a_t, b_t, w_t = k[:, :, t], v[:, :, t], a[:, :, t], b[:, :, t], torch.exp(-torch.exp(w[:, :, t]))
            sa = torch.matmul(states[-1], a_t.unsqueeze(-1))
            sab = torch.matmul(sa, b_t.unsqueeze(-2))
            kv = torch.matmul(v_t.unsqueeze(-1), k_t.unsqueeze(-2))
            new_state = states[-1] * w_t[:, :, None, :] + sab + kv
            states.append(new_state)
        return states

    # Rebuild all states for the last chunk
    last_ckpt = (L - 1) // state_ckpt_interval * state_ckpt_interval
    states = reconstruct_state_cache(state_cache[-1], last_ckpt, L)

    for t in range(L-1, -1, -1):
        chunk_index = t // state_ckpt_interval
        local_index = t % state_ckpt_interval

        if local_index == state_ckpt_interval - 1 and t != L-1:
            # Rebuild states for the next chunk
            states = reconstruct_state_cache(state_cache[chunk_index], chunk_index, chunk_index + state_ckpt_interval)

        state = states[local_index + 1]
        prev_state = states[local_index]

        q_t = q[:, :, t] * scale
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        a_t = a[:, :, t]
        b_t = b[:, :, t]
        w_t_temp = -torch.exp(w[:, :, t])
        w_t = torch.exp(w_t_temp)

        # Gradient of output
        dq[:, :, t] += torch.matmul(doutput[:, :, t].unsqueeze(-2), state).squeeze(-2) * scale
        # torch.einsum('bhi,bhj->bhij', doutput[:, :, t], q_t)
        dstate += torch.mul(doutput[:, :, t].unsqueeze(3), q_t.unsqueeze(2))

        # Gradient of state update
        dw[:, :, t] += torch.sum(dstate * prev_state, dim=(-2)) * w_t * w_t_temp

        # Gradient of sab
        # torch.einsum('bhij,bhik,bhj->bhk', dstate, prev_state, b_t)
        # temp = torch.bmm(dstate.view(B*H, V, V).permute(0, 2, 1), prev_state.view(B*H, V, V)).view(B*H, V, V)
        temp = torch.matmul(dstate.permute(0, 1, 3, 2), prev_state)
        da[:, :, t] += torch.matmul(temp.permute(0, 1, 3, 2), b_t.unsqueeze(-1)).squeeze(-1)

        # torch.einsum('bhij,bhik,bhk->bhj', dstate, prev_state, a_t)
        db[:, :, t] += torch.matmul(temp, a_t.unsqueeze(-1)).squeeze(-1)

        # Gradient of k_t * v_t
        # torch.einsum('bhij,bhi->bhj', dstate, v_t)
        # dk[:, :, t] += torch.bmm(dstate.view(B*H, V, V).permute(0, 2, 1), v_t.view(B*H, V, 1)).view(B, H, V)
        dk[:, :, t] += torch.matmul(dstate.permute(0, 1, 3, 2), v_t.unsqueeze(-1)).squeeze(-1)
        # torch.einsum('bhij,bhj->bhi', dstate, k_t)
        # dv[:, :, t] += torch.bmm(dstate.view(B*H, V, V), k_t.view(B*H, V, 1)).view(B, H, V)
        dv[:, :, t] += torch.matmul(dstate, k_t.unsqueeze(-1)).squeeze(-1)

        # Gradient for previous state
        # torch.einsum('bhij,bhk,bhj->bhik', dstate, a_t, b_t)
        # [B, H, V, 1, V] * [B, H, 1, V, 1] = [B, H, V, 1, V]
        mul_result = (dstate.unsqueeze(3) * a_t.unsqueeze(2).unsqueeze(-1)).view(B, H, V*V, V)
        # dprev_state = torch.bmm(mul_result.view(B*H, V*V, V), b_t.view(B*H, V, 1)).view(B, H, V, V)
        dprev_state = torch.matmul(mul_result, b_t.unsqueeze(-1)).view(B, H, V, V)

        dprev_state += dstate * w_t[..., None, :]

        # Update dstate for next iteration
        dstate = dprev_state

    return dq, dk, dv, dw, da, db, dstate


class NativeRecurrentRWKV7Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, a, b, scale, initial_state,
                training: bool = True, dtype: Optional[torch.dtype] = None,
                state_ckpt_interval: int = 16):
        o, ht, state_cache = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=scale, initial_state=initial_state,
                                                   state_ckpt=True, state_ckpt_interval=state_ckpt_interval)
        if training:
            ctx.save_for_backward(q, k, v, w, a, b, state_cache)
            ctx.scale = scale
            ctx.dtype = dtype
            ctx.ckpt_interval = state_ckpt_interval
            ctx.use_initial_state = initial_state is not None
        return o, ht

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, a, b, state_cache = ctx.saved_tensors
        dq, dk, dv, dw, da, db, dh = naive_recurrent_rwkv7_bwd(
            q, k, v, w, a, b, do, dht, state_cache, ctx.scale, dtype=ctx.dtype)
        dh = dh if ctx.use_initial_state else None
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
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = True,
    use_log_w: bool = False,
    training: bool = True,
    dtype=torch.float32
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
    assert cu_seqlens is None
    assert head_first is True
    assert use_log_w is False
    if scale == -1.0:
        scale = r.shape[-1] ** -0.5
    o, h_t = NativeRecurrentRWKV7Function.apply(r, k, v, w, a, b, scale, initial_state, training, dtype)
    final_state = h_t if output_final_state else None
    return o, final_state


if __name__ == "__main__":
    B = 4
    H = 4
    L = 20
    D = 64
    dtype = torch.float
    require_grad = True
    torch.manual_seed(44)

    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-20)
    q = torch.empty(B, H, L, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, H, L, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.randn(B, H, L, D).to(device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    w = (torch.randn(B, H, L, D).uniform_(-8, -6).to(device).to(dtype)).requires_grad_(require_grad)

    # a: 基于 torch.sigmoid(...) * 2.0 的计算，范围在 [0, 2]
    a = torch.rand(B, H, L, D, device=device, dtype=dtype).clamp(0, 0.1).requires_grad_(require_grad)

    # b: 在模型中是 kk*a，其中 kk 是归一化的，所以范围可能在 [-2, 2]
    b = torch.randn(B, H, L, D, device=device, dtype=dtype).clamp(-0.2, 0.2).requires_grad_(require_grad)

    do = torch.rand_like(v).to(device).fill_(torch.rand(1).item())
    h = torch.zeros(B, H, D, D, device=device, dtype=torch.float32).requires_grad_(require_grad)
    h1 = h.clone().detach().requires_grad_(True)
    with torch.no_grad():
        ref_o, _, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=h)
        ref_o1, _, _ = naive_recurrent_rwkv7_2(q, k, v, w, a, b, scale=1.0, initial_state=h)

        torch.testing.assert_close(ref_o1, ref_o, atol=1e-4, rtol=1e-4)
        # assert torch.allclose(ref_o, ref_o1, atol=1e-11), print(ref_o - ref_o1)
        print("Forward pass test passed")

    ref_o, _, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=None)
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
    torch.testing.assert_close(ref_dq, tri_dq, rtol=1e-4, atol=1e-4)
    print(get_err_ratio(ref_dk, tri_dk))
    torch.testing.assert_close(ref_dk, tri_dk, rtol=1e-4, atol=1e-4)
    print(get_err_ratio(ref_dv, tri_dv))
    torch.testing.assert_close(ref_dv, tri_dv, rtol=1e-4, atol=1e-4)
    print(get_err_ratio(ref_dw, tri_dw))
    torch.testing.assert_close(ref_dw, tri_dw, rtol=1e-4, atol=1e-4)
    print(get_err_ratio(ref_da, tri_da))
    torch.testing.assert_close(ref_da, tri_da, rtol=1e-4, atol=1e-4)
    print(get_err_ratio(ref_db, tri_db))
    torch.testing.assert_close(ref_db, tri_db, rtol=1e-4, atol=1e-4)
    assert get_err_ratio(ref_dq, tri_dq) < atol
    assert get_err_ratio(ref_dk, tri_dk) < atol
    assert get_err_ratio(ref_dv, tri_dv) < atol
    assert get_err_ratio(ref_dw, tri_dw) < atol
    assert get_err_ratio(ref_da, tri_da) < atol
    assert get_err_ratio(ref_db, tri_db) < atol
