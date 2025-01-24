# -*- coding: utf-8 -*-
# Copyright (c) 2024, Zhiyuan Li


from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.generalized_delta_rule import fused_recurrent_dplr_delta_rule
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import device, set_torch_device


@triton.jit
def fused_rwkv7_kernel(
    q_ptr, k_ptr, v_ptr,
    w_ptr, a_ptr, b_ptr,
    state_ptr, output_ptr,
    state_output_ptr,
    scale: tl.constexpr,
    N: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    pid = tl.program_id(0)

    b_idx = pid // H
    h_idx = pid % H

    xindex = tl.arange(0, BLOCK)
    xmask = xindex < N

    state = tl.zeros([BLOCK, V], dtype=tl.float32)

    if USE_INITIAL_STATE:
        state_offset = (b_idx * H + h_idx) * N * V
        state += tl.load(state_ptr + state_offset + (xindex[:, None] * V + tl.arange(0, V)[None, :]),
                         mask=xmask[:, None]).to(tl.float32)

    for t in range(L):
        t_offset = (b_idx * H * L + h_idx * L + t) * N

        # Step 1: sa
        a = tl.load(a_ptr + t_offset + xindex, mask=xmask).to(tl.float32)
        sa = tl.sum(a[:, None] * state, axis=0)

        # Step 2: update state
        w = tl.load(w_ptr + t_offset + xindex, mask=xmask).to(tl.float32)
        k = tl.load(k_ptr + t_offset + xindex, mask=xmask).to(tl.float32)
        v = tl.load(v_ptr + (b_idx * H * L + h_idx * L + t) * V + tl.arange(0, V)).to(tl.float32)
        b = tl.load(b_ptr + t_offset + xindex, mask=xmask).to(tl.float32)

        w = tl.exp(-tl.exp(w))
        state = (state * w[:, None] +
                 k[:, None] * v[None, :] +
                 sa[None, :] * b[:, None])

        # Step 3
        q = tl.load(q_ptr + t_offset + xindex, mask=xmask).to(tl.float32) * scale
        output = tl.sum(state * q[:, None], axis=0)

        out_offset = (b_idx * H * L + h_idx * L + t) * V
        tl.store(output_ptr + out_offset + tl.arange(0, V), output.to(output_ptr.dtype.element_ty))

    if STORE_FINAL_STATE:
        state_offset = (b_idx * H + h_idx) * N * V
        tl.store(state_output_ptr + state_offset + (xindex[:, None] * V + tl.arange(0, V)[None, :]),
                 state.to(state_output_ptr.dtype.element_ty), mask=xmask[:, None])


class FusedRecurrentRWKV7Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, a, b, scale, initial_state, output_final_state: bool = True):
        """
        Args:
            args: List containing:
                q: (B, H, L, N) Query tensor
                k: (B, H, L, N) Key tensor
                v: (B, H, L, V) Value tensor
                w: (B, H, L, N) Time decay weights
                a: (B, H, L, N) Dynamic learning rate modulator
                b: (B, H, L, N) State update modulator
                state: (B, H, N, V) Current state
        """

        B, H, L, K = q.shape
        V = v.shape[-1]

        output = torch.empty_like(v)

        if initial_state is not None:
            final_state = torch.empty_like(initial_state)
        elif output_final_state:
            final_state = q.new_empty(B, H, K, V)
        else:
            final_state = None

        fused_rwkv7_kernel[(B * H,)](
            q, k, v, w, a, b,
            initial_state, output, final_state,
            scale,
            K, V, L, H,
            BLOCK=K,
            USE_INITIAL_STATE=True if initial_state is not None else False,
            STORE_FINAL_STATE=output_final_state,
        )

        return output, final_state if output_final_state else None

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Fused wkv7 backward function is not implemented. "
            "Please use chunk_rwkv7 for training!"
        )


def fused_recurrent_rwkv7(
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
    head_first: bool = False,
    use_log_w: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        scale (float):
            scale of the attention.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
        use_log_w (bool):
            if use_log_w == false, will apply w = -torch.exp(w)
    """
    if scale == -1.0:
        scale = r.shape[-1] ** -0.5
    set_torch_device(r)
    if not use_log_w:
        assert use_log_w is False
        assert head_first is True
        return FusedRecurrentRWKV7Function.apply(r, k, v, w, a, b, scale, initial_state, output_final_state)
    else:
        log_w = -torch.exp(w)
        return fused_recurrent_dplr_delta_rule(
            q=r,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=log_w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            head_first=head_first
        )


if __name__ == '__main__':
    from fla.ops.rwkv7.recurrent_naive import naive_recurrent_rwkv7
    B = 4
    H = 64
    L = 4096
    D = 64
    dtype = torch.bfloat16
    require_grad = True
    torch.manual_seed(44)

    def get_err_ratio(x, y):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-20)
    q = torch.empty(B, H, L, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    k = torch.empty(B, H, L, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)
    v = torch.empty(B, H, L, D, device=device).uniform_(-1, 1).to(dtype=dtype).requires_grad_(True)

    # 时间衰减参数w，保持在负数范围
    w = torch.empty(B, H, L, D, device=device).uniform_(-8, -6).to(dtype=dtype).requires_grad_(True)

    # 生成归一化的kk
    kk = torch.empty(B, H, L, D, device=device).uniform_(-1, 1)
    kk = torch.nn.functional.normalize(kk, dim=-1).to(dtype=dtype)

    # 生成a参数（对应-kk）和b参数（对应kk*a）
    a = -kk.clone().requires_grad_(True)  # -kk
    a_scale = torch.empty(B, H, L, D, device=device).uniform_(0, 0.1).to(dtype=dtype)
    b = (kk * a_scale).requires_grad_(True)  # kk*a

    do = torch.rand_like(v).to(device).fill_(torch.rand(1).item())
    h = torch.rand(B, H, D, D, device=device, dtype=torch.float32).requires_grad_(require_grad)

    with torch.no_grad():
        q, k, v, w, a, b, h = (x.to(dtype=torch.float64).to('cpu') for x in (q, k, v, w, a, b, h))
        ref_o, ref_state, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=h)
        q, k, v, w, a, b, h = (x.to(dtype=dtype).to(device) for x in (q, k, v, w, a, b, h))
        result, state = fused_recurrent_rwkv7(q, k, v, w, a, b, initial_state=h.transpose(-1, -2),
                                              use_log_w=False, head_first=True)

        ref_o = ref_o.to(dtype=torch.float32).to(device)
        result = result.to(dtype=torch.float32).to(device)
        ref_state = ref_state.to(dtype=torch.float32).to(device)
        state = state.to(dtype=torch.float32).to(device)
        tol = 1e-3 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(result, ref_o, atol=tol, rtol=tol)
        diff = torch.abs(result - ref_o)
        diff_state = torch.abs(state - ref_state.transpose(-1, -2))
        print("Max error:", diff.max().item(), diff_state.max().item())
        print("Mean error:", diff.mean().item(), diff_state.mean().item())
        print("Forward pass test passed", (ref_o - result).abs().max().item())
    print('test passed')
