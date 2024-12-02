# -*- coding: utf-8 -*-
# Copyright (c) 2024, Zhiyuan Li



from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_cumsum
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import device


@triton.jit
def fused_rwkv7_kernel(
    q_ptr, k_ptr, v_ptr, w_ptr, a_ptr, b_ptr, 
    state_ptr, output_ptr,
    scale: tl.constexpr,
    N:tl.constexpr, 
    V:tl.constexpr, 
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
        tl.store(state_ptr + state_offset + (xindex[:, None] * V + tl.arange(0, V)[None, :]), 
                state.to(state_ptr.dtype.element_ty), mask=xmask[:, None])

class FusedRecurrentRWKV7Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, a, b, scale, state, output_final_state:bool=True, training: bool = True):
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

        B, H, L, N = q.shape
        V = v.shape[-1]
        output = torch.empty_like(v)
        
        fused_rwkv7_kernel[(B * H ,)](
            q, k, v, w, a, b, state, output,
            scale,
            N, V, L, H,
            BLOCK=N,
            USE_INITIAL_STATE=True if state is not None else False,
            STORE_FINAL_STATE=output_final_state,
        )

        ctx.save_for_backward(q, k, v, w, a, b, state)
        return output, state if output_final_state else None
    
    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        pass





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
    training: bool = True,
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
    o, h_t = FusedRecurrentRWKV7Function.apply(r, k, v, w, a, b, scale, initial_state, output_final_state, training)
    return o, h_t


if __name__ == '__main__':
    from fla.utils import device
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
        ref_o, _, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=h)
        q, k, v, w, a, b, h = (x.to(dtype=dtype).to(device) for x in (q, k, v, w, a, b, h))
        result, _ = fused_recurrent_rwkv7(q, k, v, w, a, b, initial_state=h.transpose(-1, -2))

        ref_o = ref_o.to(dtype=torch.float32).to(device)
        result = result.to(dtype=torch.float32).to(device)
        tol = 1e-3 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(result, ref_o, atol=tol, rtol=tol)
        diff = torch.abs(result - ref_o)
        print("Max error:", diff.max().item())
        print("Mean error:", diff.mean().item())
        print("Forward pass test passed", (ref_o - result).abs().max().item())
    print('test passed')

