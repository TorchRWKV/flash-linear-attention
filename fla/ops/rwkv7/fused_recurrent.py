# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import rearrange
import triton
import triton.language as tl

from fla.ops.generalized_delta_rule import fused_recurrent_dplr_delta_rule
from fla.ops.utils.op import exp
from fla.utils import input_guard, use_cuda_graph


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [16, 32, 64]
        for num_warps in [2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=['BK'],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_rwkv7_fwd_kernel(
    r,
    w,
    k,
    v,
    kk,
    a,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    REVERSE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_r = r + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_w = w + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_k = k + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_v = v + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v
    p_a = a + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k
    p_kk = kk + (bos + ((T-1) if REVERSE else 0)) * H*K + i_h * K + o_k

    p_o = o + (bos + ((T-1) if REVERSE else 0)) * H*V + i_h * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[None, :] & mask_v[:, None]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[None, :] * V + o_v[:, None]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    if IS_DECODE:
        b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
        b_act_a = -b_kk
        b_b = b_kk * b_a

        tmp = tl.sum(b_h * b_act_a[None, :], axis=1)
        b_h = exp(b_w)[None, :] * b_h + (tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
        b_o = tl.sum(b_h * b_r[None, :], axis=1)

        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    else:
        for _ in range(0, T):
            b_r = tl.load(p_r, mask=mask_k, other=0).to(tl.float32) * scale
            b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
            b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_kk = tl.load(p_kk, mask=mask_k, other=0).to(tl.float32)
            b_act_a = -b_kk
            b_b = b_kk * b_a

            tmp = tl.sum(b_h * b_act_a[None, :], axis=1)
            b_h = exp(b_w)[None, :] * b_h + (tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
            b_o = tl.sum(b_h * b_r[None, :], axis=1)

            tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            p_r += (-1 if REVERSE else 1) * H*K
            p_w += (-1 if REVERSE else 1) * H*K
            p_k += (-1 if REVERSE else 1) * H*K
            p_v += (-1 if REVERSE else 1) * H*V
            p_a += (-1 if REVERSE else 1) * H*K
            p_kk += (-1 if REVERSE else 1) * H*K
            p_o += (-1 if REVERSE else 1) * H*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[None, :] * V + o_v[:, None]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@input_guard
def fused_recurrent_rwkv7_fwd(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    IS_DECODE = (T == 1)

    h0 = initial_state
    if not output_final_state:
        ht = None
    else:
        ht = r.new_empty(N, H, K, V, dtype=torch.float32)
    o = torch.empty_like(v)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)
    fused_recurrent_rwkv7_fwd_kernel[grid](
        r,
        w,
        k,
        v,
        kk,
        a,
        o,
        h0,
        ht,
        cu_seqlens,
        scale,
        T=T,
        B=B,
        H=H,
        K=K,
        V=V,
        BK=BK,
        REVERSE=reverse,
        IS_DECODE=IS_DECODE
    )
    return o, ht
from fla.utils import input_guard, use_cuda_graph
from fla.ops.utils.op import exp


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4, 8, 16]
        for num_stages in [2, 3, 4]
    ],
    key=["BK"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=['T'])
def fused_rwkv7_kernel(
    q_ptr, k_ptr, v_ptr,
    w_ptr, a_ptr, b_ptr,
    state_ptr, output_ptr,
    state_output_ptr,
    K: tl.constexpr,
    V: tl.constexpr,
    L,
    H: tl.constexpr,
    offsets,
    scale: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_OFFSETS: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    # indices
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    if USE_OFFSETS:
        bos, eos = tl.load(offsets + i_n).to(tl.int64), tl.load(offsets + i_n + 1).to(tl.int64)
        L = eos - bos
    else:
        bos, eos = i_n * L, i_n * L + L

    if HEAD_FIRST:
        p_q = q_ptr + i_nh * L*K + tl.arange(0, BK)
        p_k = k_ptr + i_nh * L*K + tl.arange(0, BK)
        p_w = w_ptr + i_nh * L*K + tl.arange(0, BK)
        p_a = a_ptr + i_nh * L*K + tl.arange(0, BK)
        p_b = b_ptr + i_nh * L*K + tl.arange(0, BK)
        p_o = output_ptr + i_nh * L*V + i_v * BV + tl.arange(0, BV)
        p_v = v_ptr + i_nh * L*V + i_v * BV + tl.arange(0, BV)
    else:
        p_q = q_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_k = k_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_w = w_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_a = a_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_b = b_ptr + (bos * H + i_h) * K + tl.arange(0, BK)
        p_v = v_ptr + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)
        p_o = output_ptr + (bos * H + i_h) * V + i_v * BV + tl.arange(0, BV)

    mask_k = tl.arange(0, BK) < K
    mask_v = (i_v * BV + tl.arange(0, BV)) < V
    mask_h = mask_k[None, :] & mask_v[:, None]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = state_ptr + i_nh * K * V + (tl.arange(0, BK)[None, :]) * V + ((i_v * BV + tl.arange(0, BV))[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, L):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_w = tl.load(p_w, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32) * scale
        b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
        b_b = tl.load(p_b, mask=mask_k, other=0).to(tl.float32)
        # to store
        tmp = tl.sum(b_h * b_a[None, :], axis=1)
        b_h = exp(-exp(b_w))[None, :] * b_h + (tmp[:, None] * b_b[None, :] + b_k[None, :] * b_v[:, None])
        _o = b_h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += K if HEAD_FIRST else K*H
        p_k += K if HEAD_FIRST else K*H
        p_w += K if HEAD_FIRST else K*H
        p_o += V if HEAD_FIRST else V*H
        p_v += V if HEAD_FIRST else V*H
        p_a += K if HEAD_FIRST else K*H
        p_b += K if HEAD_FIRST else K*H

    if STORE_FINAL_STATE:
        p_ht = state_output_ptr + i_nh * K * V + (tl.arange(0, BK)[None, :]) * V + ((i_v * BV + tl.arange(0, BV))[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


class FusedRecurrentRWKV7Function(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, q, k, v, w, a, b,
                scale=None,
                initial_state=None,
                output_final_state=False,
                offsets=None,
                head_first=False
                ):
        if head_first:
            B, H, L, K, V = *k.shape, v.shape[-1]
        else:
            B, L, H, K, V = *k.shape, v.shape[-1]
        N = B if offsets is None else len(offsets) - 1
        output = torch.empty_like(v)

        BK = triton.next_power_of_2(K)
        if initial_state is not None:
            final_state = torch.empty_like(initial_state)
            use_initial_state = True
        elif output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
            use_initial_state = False
        else:
            final_state = None
            use_initial_state = False

        def grid(meta): return (triton.cdiv(V, meta['BV']), N * H)

        fused_rwkv7_kernel[grid](
            q, k, v, w, a, b,
            initial_state, output, final_state,
            K, V, L, H,
            offsets=offsets,
            scale=scale,
            BK=BK,
            HEAD_FIRST=head_first,
            USE_INITIAL_STATE=use_initial_state,
            STORE_FINAL_STATE=output_final_state,
        )
        return output, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Fused wkv7 backward function is not implemented. "
            "Please use chunk_rwkv7 for training!"
        )


def fused_recurrent_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor = None,
    log_w: torch.Tensor = None,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,

) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        k (torch.Tensor):
            k of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        v (torch.Tensor):
            v of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        a (torch.Tensor):
            a of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        b (torch.Tensor):
            b of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
        w (torch.Tensor):
            decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`, kernel
            will apply log_w = -torch.exp(w)
        log_w (torch.Tensor):
            log decay of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
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
    """
    if w is not None:
        if cu_seqlens is not None:
            if r.shape[0] != 1:
                raise ValueError(f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`."
                                 f"Please flatten variable-length inputs before processing.")
            if head_first:
                raise RuntimeError("Sequences with variable lengths are not supported for head-first mode")
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(f"The number of initial states is expected to be equal to the number of input sequences, "
                                 f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.")
        if scale is None:
            scale = r.shape[-1] ** -0.5
        else:
            assert scale > 0, "scale must be positive"
        o, final_state = FusedRecurrentRWKV7Function.apply(
            r, k, v, w, a, b,
            scale,
            initial_state,
            output_final_state,
            cu_seqlens,
            head_first
        )
        return o, final_state
    elif log_w is not None:
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


def fused_mul_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    scale: Optional[float] = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    This function computes the recurrence S_t = S_t @ (I + a_t b_t^T) + v_t k_t^T in a recurrent manner.

    Args:
        r (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        w (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        v (torch.Tensor):
            a of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        kk (torch.Tensor):
            b of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        a (torch.Tensor):
            gk of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`. decay term in log space!
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: 1.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        reverse (Optional[bool]):
            If `True`, process the state passing in reverse order. Default: `False`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape `[N + 1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.
    """
    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        r, w, k, v, kk, a = map(lambda x: rearrange(x, 'b h t ... -> b t h ...'), (r, w, k, v, kk, a))
    if not head_first and r.shape[1] < r.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({r.shape[1]}) < num_heads ({r.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if r.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {r.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = r.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    o, final_state = fused_recurrent_rwkv7_fwd(
        r,
        w,
        k,
        v,
        kk,
        a,
        scale,
        initial_state,
        output_final_state,
        reverse,
        cu_seqlens,
    )
    if head_first:
        o = rearrange(o, 'b t h ... -> b h t ...')
    return o, final_state
    else:
        raise ValueError("Either `w` or `log_w` must be provided.")
