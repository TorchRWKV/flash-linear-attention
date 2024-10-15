# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang, Zhiyuan Li

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.utils import chunk_global_reversed_cumsum
from fla.utils import contiguous, device_capacity, check_pytorch_version, device, detect_tf32


@triton.autotune(
    configs=[
        triton.Config({'BS': 16}, num_warps=2, num_stages=1),
        triton.Config({'BS': 16}, num_warps=2, num_stages=2),
        triton.Config({'BS': 16}, num_warps=4, num_stages=1),
        triton.Config({'BS': 16}, num_warps=4, num_stages=2),
        triton.Config({'BS': 16}, num_warps=8, num_stages=1),
        triton.Config({'BS': 16}, num_warps=8, num_stages=2),
        triton.Config({'BS': 32}, num_warps=2, num_stages=1),
        triton.Config({'BS': 32}, num_warps=2, num_stages=2),
        triton.Config({'BS': 32}, num_warps=4, num_stages=1),
        triton.Config({'BS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BS': 32}, num_warps=8, num_stages=1),
        triton.Config({'BS': 32}, num_warps=8, num_stages=2),
        triton.Config({'BS': 64}, num_warps=2, num_stages=1),
        triton.Config({'BS': 64}, num_warps=2, num_stages=2),
        triton.Config({'BS': 64}, num_warps=4, num_stages=1),
        triton.Config({'BS': 64}, num_warps=4, num_stages=2),
        triton.Config({'BS': 64}, num_warps=8, num_stages=1),
        triton.Config({'BS': 64}, num_warps=8, num_stages=2),
    ],
    key=['S']
)
@triton.jit
def chunk_rwkv6_fwd_kernel_cum(
    s,
    o,
    o_minus_s,
    s_s_h,
    s_s_t,
    s_s_d,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    USE_TF32: tl.constexpr
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)

    p_s = tl.make_block_ptr(s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    p_o_minus_s = tl.make_block_ptr(o_minus_s + i_bh * s_s_h, (T, S), (s_s_t, s_s_d), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.dot(m_s, b_s, allow_tf32=USE_TF32)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_o_minus_s, (b_o - b_s).to(p_o_minus_s.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def post_process_grad(
    q,
    k,
    v,
    u,
    do,
    dk,
    dq,
    du,
    scale: tl.constexpr,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    H: tl.constexpr,
    T: tl.constexpr,
    BT: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    U_2D: tl.constexpr,
    TLTYPE: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_u = i_bh if not U_2D else i_bh % H
    # Note that BK = tl.next_power_of_2(K), BK = tl.next_power_of_2(V)
    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_du = tl.make_block_ptr(du + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (i_t * BT, 0), (BT, BK), (1, 0))
    p_u = tl.make_block_ptr(u + i_u * K, (K,), (1,), (0,), (BK,), (0,))

    b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale).to(TLTYPE)
    b_k = (tl.load(p_k, boundary_check=(0, 1)) * scale).to(TLTYPE)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(TLTYPE)
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(TLTYPE)
    b_u = tl.load(p_u, boundary_check=(0,)).to(TLTYPE)

    b_vdo = tl.sum(b_v * b_do, axis=1)
    b_du = b_vdo[:, None] * b_k * b_q
    b_dq = b_vdo[:, None] * b_k * b_u[None, :] * scale
    b_dk = b_vdo[:, None] * b_q * b_u[None, :] * scale

    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    tl.store(p_du, b_du.to(p_du.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['BK']
)
@triton.jit
def chunk_rwkv6_fwd_kernel_h(
    k,
    v,
    g,
    h,
    h0,
    ht,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    TLTYPE: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_h = tl.zeros([BK, BK], dtype=TLTYPE)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * K, (K, K), (K, 1), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        b_h += tl.load(p_h0, boundary_check=(0, 1))

    for i_t in range(NT):
        o_t = min(i_t * BT + BT, T)

        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (i_t * BT, i_v * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * K, (K, K), (s_h_t, s_h_d), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((o_t - 1) * K + i_k * BK,), (BK,), (0,))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = (tl.load(p_k, boundary_check=(0, 1)) * scale).to(TLTYPE)
        # [BT, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(TLTYPE)
        # [BK, BT]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        if i_t < NT - 1:
            # [BK,]
            b_gn = tl.load(p_gn, boundary_check=(0,)).to(tl.float32)
        else:
            b_gn = tl.min(b_g, axis=1).to(tl.float32)
        b_h *= tl.exp(b_gn)[:, None].to(TLTYPE)
        b_k = b_k * tl.exp(b_gn[:, None] - b_g).to(TLTYPE)
        b_h += tl.dot(b_k, b_v, allow_tf32=False).to(TLTYPE)  # must be false

    if STORE_FINAL_STATE:
        p_h = tl.make_block_ptr(ht + i_bh * K * K, (K, K), (K, 1), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))


def generate_configs(group_sizes, num_warps_options, num_stages_options):
    configs = []
    for group_size in group_sizes:
        for num_warps in num_warps_options:
            for num_stages in num_stages_options:
                configs.append(triton.Config({'GROUP_SIZE': group_size}, num_warps=num_warps, num_stages=num_stages))
    return configs


group_sizes = [64, 128, 256]
num_warps_options = [2, 4, 8]
num_stages_options = [1, 2]
configs = generate_configs(group_sizes, num_warps_options, num_stages_options)


@triton.autotune(
    configs=configs,
    key=['BK']
)
@triton.jit
def chunk_rwkv6_fwd_kernel_intra(
    q,
    k,
    g,
    gs,
    u,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    scale: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    DK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    U_2D: tl.constexpr,
    TLTYPE: tl.constexpr,
    USE_TF32: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_u = i_bh if not U_2D else i_bh % H
    # Group ID and position within group
    group_id = i_c // GROUP_SIZE
    local_id = i_c % GROUP_SIZE

    # Calculate i_t, i_i, i_j using group-based ordering
    group_start = group_id * GROUP_SIZE
    i_c_ordered = group_start + local_id

    i_t = i_c_ordered // (NC * NC)
    i_i = (i_c_ordered % (NC * NC)) // NC
    i_j = i_c_ordered % NC

    n_bh = tl.num_programs(2)

    o_k = i_k * BK + tl.arange(0, BK)
    o_q = i_t * BT + i_i * BC
    m_k = o_k < K

    if i_i > i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_A = tl.make_block_ptr(A + (i_k * n_bh + i_bh) * T * BT, (T, BT), (BT, 1),
                                (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BK,]
        b_gn = tl.load(g + i_bh * T * K + (o_q - 1) * K + o_k, mask=(m_k & (i_i > 0) & (o_q <= T)), other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)) * scale
        b_gs = tl.load(p_gs, boundary_check=(0, 1))
        b_qg = (b_q * tl.exp((b_gs - b_gn[None, :]))).to(b_q.dtype)
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1)) * scale
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = (b_k * tl.exp((b_gn[:, None] - b_gk))).to(b_k.dtype)
        # [BC, BC]
        b_A = tl.dot(b_qg, b_kg, allow_tf32=USE_TF32)
        tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))
    elif i_i == i_j:
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))
        p_q_u = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (s_k_d,), ((i_t * BT + i_j * BC) * K + i_k * BK,), (BK,), (0,))

        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)) * scale
        b_gs = tl.load(p_gs, boundary_check=(0, 1)).to(tl.float32)
        o_i = tl.arange(0, BC)
        o_g = i_bh * T * K + (i_t * BT + i_j * BC) * K + o_k
        o_A = (i_bh + i_k * n_bh) * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
        m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
        p_u = tl.make_block_ptr(u + i_u * DK, (DK,), (1,), (i_k * BK), (BK,), (0,))
        b_u = tl.load(p_u, boundary_check=(0,))
        for j in range(BC):
            # [BK,]
            b_k = tl.load(p_k, boundary_check=(0,)) * scale
            b_gk = tl.load(g + o_g + j * K, mask=(m_k & ((i_t * BT + i_j * BC + j) < T)), other=0).to(tl.float32)
            # [BC,]
            b_A = tl.sum(b_q.to(TLTYPE) * b_k[None, :] * (tl.exp(b_gs - b_gk[None, :]).to(TLTYPE)), 1)
            b_A = tl.where(o_i > j, b_A, 0.)
            # self
            b_q_u = tl.load(p_q_u, boundary_check=(0,)) * scale
            b_A_u = tl.sum(b_q_u * b_k * b_u, axis=0)
            m_u = tl.arange(0, BC) == j
            b_A = tl.where(m_u, b_A_u, b_A)
            tl.store(A + o_A + j, b_A.to(A.dtype.element_ty), mask=m_A)
            p_k = tl.advance(p_k, (K,))
            p_q_u = tl.advance(p_q_u, (K,))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['BK']
)
@triton.jit
def chunk_rwkv6_fwd_kernel_inter(
    q,
    v,
    gs,
    h,
    o,
    A,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    TLTYPE: tl.constexpr,
    USE_TF32: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_o = tl.zeros([BT, BK], dtype=TLTYPE)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * K, (K, K), (s_h_t, s_h_d), (i_k * BK, i_v * BK), (BK, BK), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1)).to(TLTYPE)
        b_q = (b_q * scale).to(b_q.dtype)
        # [BT, BK]
        b_gs = tl.load(p_gs, boundary_check=(0, 1)).to(tl.float32)
        # [BT, BK]
        b_qg = (b_q * tl.exp(b_gs)).to(b_q.dtype)
        # [BK, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1)).to(TLTYPE)
        # works but dkw, owing to divine benevolence
        # [BT, BK]
        # if i_k >= 0:
        b_o += tl.dot(b_qg, b_h, allow_tf32=USE_TF32).to(TLTYPE)
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (i_t * BT, i_v * BK), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (i_t * BT, i_v * BK), (BT, BK), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BK]
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(TLTYPE)
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(TLTYPE)
    b_o += tl.dot(b_A, b_v, allow_tf32=USE_TF32).to(TLTYPE)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
    ],
    key=['BK']
)
@triton.jit
def chunk_rwkv6_bwd_kernel_dh(
    q,
    g,
    gs,
    do,
    dh,
    dh0,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    TLTYPE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    p_dh_last = tl.make_block_ptr(dh + i_bh * s_h_h + (NT-1) * K * K, (K, K),
                                  (s_h_t, s_h_d), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
    b_dh = tl.load(p_dh_last, boundary_check=(0, 1)).to(TLTYPE)
    for i_t in range(NT - 1, -1, -1):
        o_t = min(i_t * BT + BT, T)

        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (i_t * BT, i_v * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * K, (K, K),
                                 (s_h_t, s_h_d), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((o_t - 1) * K + i_k * BK,), (BK,), (0,))

        # [BK, BT]
        b_q = (tl.load(p_q, boundary_check=(0, 1)) * scale)
        # [BT, BK]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))

        # [BK,]
        b_gn = tl.load(p_gn, boundary_check=(0,)).to(tl.float32)
        # [BK, BK]
        b_dh = b_dh * tl.exp(b_gn)[:, None].to(TLTYPE)
        # [BK, BT]
        b_gs = tl.load(p_gs, boundary_check=(0, 1)).to(tl.float32)
        b_q = b_q * tl.exp(b_gs)

        # [BK, BK]
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False).to(TLTYPE)  # must be false

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * K, (K, K), (K, 1), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ],
    key=['BK']
)
@triton.jit
def chunk_rwkv6_bwd_kernel_inter(
    k,
    v,
    h,
    g,
    gs,
    A,
    do,
    dh,
    dq,
    dk,
    dv,
    dA,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    s_h_h,
    s_h_t,
    s_h_d,
    scale: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    TLTYPE: tl.constexpr,
    USE_TF32: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    offset_k = i_t * BT
    offset_BK = i_k * BK
    o_t = min(offset_k + BT, T)

    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k, offset_BK), (BT, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k, offset_BK), (BT, BK), (1, 0))
    p_gq = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k, offset_BK), (BT, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,), ((o_t - 1) * K + offset_BK,), (BK,), (0,))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (BT, T), (1, BT), (0, offset_k), (BT, BT), (0, 1))

    # [BT, BK]
    b_k = (tl.load(p_k, boundary_check=(0, 1)) * scale).to(TLTYPE)
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_gq = tl.load(p_gq, boundary_check=(0, 1)).to(tl.float32)
    b_gn = tl.exp(((tl.load(p_gn, boundary_check=(0,))[None, :]) - b_gk).to(tl.float32))
    b_k = (b_k * b_gn).to(b_k.dtype)
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1)).to(TLTYPE)

    b_dq = tl.zeros([BT, BK], dtype=TLTYPE)
    b_dk = tl.zeros([BT, BK], dtype=TLTYPE)
    b_dA = tl.zeros([BT, BT], dtype=TLTYPE)

    for i_v in range(tl.cdiv(K, BK)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (offset_k, i_v * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * K, (K, K), (s_h_d, s_h_t),
                                (i_v * BK, offset_BK), (BK, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, K), (s_v_t, s_v_d), (offset_k, i_v * BK), (BT, BK), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * K, (K, K),
                                 (s_h_t, s_h_d), (offset_BK, i_v * BK), (BK, BK), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_v_h, (T, K),
                                 (s_v_t, s_v_d), (offset_k, i_v * BK), (BT, BK), (1, 0))

        # [BT, BK]
        b_v = tl.load(p_v, boundary_check=(0, 1)).to(TLTYPE)
        # [BK, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BK]
        b_do = tl.load(p_do, boundary_check=(0, 1)).to(TLTYPE)
        # [BK, BK]
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BT, BK]
        b_dv = tl.dot(b_k, b_dh.to(b_k.dtype), allow_tf32=USE_TF32)
        if i_k == 0:
            b_dv += tl.dot(b_A, b_do, allow_tf32=USE_TF32)
        b_do = (b_do * scale).to(b_do.dtype)

        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # [BT, BT]
        b_dA += tl.dot(b_do, tl.trans(b_v), allow_tf32=USE_TF32).to(TLTYPE)
        # [BT, BK]
        b_dq += tl.dot(b_do.to(TLTYPE), b_h.to(TLTYPE), allow_tf32=False).to(TLTYPE)  # must be false
        # [BT, BK]
        b_dk += (tl.dot(b_v.to(TLTYPE), tl.trans(b_dh).to(TLTYPE), allow_tf32=False) * scale).to(TLTYPE)  # must be false

    b_dq = b_dq * tl.exp(b_gq).to(TLTYPE)
    b_dk = b_dk * b_gn

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k, offset_BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k, offset_BK), (BT, BK), (1, 0))
    p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT, ), (BT, 1), (offset_k, 0), (BT, BT), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] > o_i[None, :]
    # [BT, BT]
    b_dA = tl.where(m_s, b_dA, 0.).to(b_k.dtype)
    if i_k == 0:
        tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_rwkv6_bwd_kernel_intra(
    q,
    k,
    g,
    gs,
    dA,
    dq,
    dk,
    s_k_h,
    s_k_t,
    s_k_d,
    scale: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    TLTYPE: tl.constexpr
):
    i_k, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_c // NC, i_c % NC
    offset_k = i_t * BT
    offset_BK = i_k * BK

    o_k = offset_BK + tl.arange(0, BK)
    o_q = offset_k + i_i * BC
    m_k = o_k < K

    p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_i * BC, offset_BK), (BC, BK), (1, 0))
    # [BK,]
    b_gn = tl.load(g + i_bh * T * K + (o_q - 1) * K + o_k, mask=(m_k & (i_i > 0) & (o_q <= T)), other=0).to(tl.float32)
    # [BC, BK]
    b_gs = tl.load(p_gs, boundary_check=(0, 1)).to(tl.float32)
    b_dq_dk_shared = tl.zeros([BC, BK], dtype=TLTYPE)

    for i_j in range(0, i_i):
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_j * BC, offset_BK), (BC, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_j * BC, offset_BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (offset_k + i_i * BC, i_j * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1)) * scale
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        b_kg = b_k * safe_exp_in_intra(b_gn[None, :] - b_gk)
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(TLTYPE)
        # [BC, BK]
        b_dq_dk_shared += tl.dot(b_dA, b_kg, allow_tf32=False).to(TLTYPE)  # must be false
    b_dq_dk_shared = b_dq_dk_shared * safe_exp_in_intra(b_gs - b_gn[None, :]).to(TLTYPE)

    o_i = tl.arange(0, BC)
    o_dA = i_bh * T * BT + (offset_k + i_i * BC + tl.arange(0, BC)) * BT + i_i * BC
    m_dA = (offset_k + i_i * BC + tl.arange(0, BC)) < T

    for j in range(0, BC):
        p_kj = tl.make_block_ptr(k + i_bh * s_k_h, (T * K,), (1,), ((offset_k + i_i * BC + j) * K + offset_BK,), (BK,), (0,))

        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, boundary_check=(0,)) * scale
        b_gkj = tl.load(g + i_bh * T * K + (o_q + j) * K + o_k, mask=(m_k & ((o_q + j) < T)), other=0)
        # [BC, BK]
        m_i = o_i[:, None] > j
        # [BC, BK]
        b_dq_dk_shared += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * safe_exp_in_intra(b_gs - b_gkj[None, :]), 0.)

    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_i * BC, offset_BK), (BC, BK), (1, 0))

    b_dq_dk_shared = b_dq_dk_shared + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq, b_dq_dk_shared.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # tl.debug_barrier()
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_i * BC, offset_BK), (BC, BK), (1, 0))
    p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_i * BC, offset_BK), (BC, BK), (1, 0))
    p_gn = tl.make_block_ptr(g + i_bh * s_k_h, (T * K,), (s_k_d,),
                             ((offset_k + i_i * BC + BC - 1) * K + offset_BK,), (BK,), (0,))
    # [BK,]
    b_gn = tl.load(p_gn, boundary_check=(0,)).to(tl.float32)
    # [BC, BK]
    b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(TLTYPE)
    b_dq_dk_shared = tl.where(True, 0., b_dq_dk_shared).to(TLTYPE)

    for i_j in range(i_i + 1, NC):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_j * BC, offset_BK), (BC, BK), (1, 0))
        p_gs = tl.make_block_ptr(gs + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_j * BC, offset_BK), (BC, BK), (1, 0))
        p_dA = tl.make_block_ptr(dA + i_bh * T * BT, (T, BT), (BT, 1), (offset_k + i_j * BC, i_i * BC), (BC, BC), (1, 0))
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_gs = tl.load(p_gs, boundary_check=(0, 1)).to(TLTYPE)
        b_qg = (b_q * safe_exp_in_intra(b_gs - b_gn[None, :]))
        # [BC, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BC, BK]
        b_dq_dk_shared += tl.dot(tl.trans(b_dA), b_qg, allow_tf32=False).to(TLTYPE) * scale

    b_dq_dk_shared = b_dq_dk_shared * safe_exp_in_intra(b_gn[None, :] - b_gk).to(TLTYPE)

    o_dA = i_bh * T * BT + (offset_k + i_i * BC) * BT + i_i * BC + tl.arange(0, BC)
    for j in range(0, BC):
        p_qj = tl.make_block_ptr(q + i_bh * s_k_h, (T * K,), (1,), ((offset_k + i_i * BC + j) * K + offset_BK,), (BK,), (0,))
        p_gqj = tl.make_block_ptr(gs + i_bh * s_k_h, (T * K,), (1,), ((offset_k + i_i * BC + j) * K + offset_BK,), (BK,), (0,))
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * BT, mask=(offset_k + i_i * BC + j < T), other=0)
        # [BK,]
        b_qj = tl.load(p_qj, boundary_check=(0,))
        b_gqj = tl.load(p_gqj, boundary_check=(0,))
        # [BC, BK]
        m_i = o_i[:, None] < j
        b_dq_dk_shared += tl.where(m_i, b_dA[:, None] * b_qj[None, :] *
                                   safe_exp_in_intra(b_gqj[None, :] - b_gk), 0.).to(TLTYPE) * scale

    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, s_k_d), (offset_k + i_i * BC, offset_BK), (BC, BK), (1, 0))
    b_dq_dk_shared = b_dq_dk_shared + tl.load(p_dk, boundary_check=(0, 1)).to(TLTYPE)
    tl.store(p_dk, b_dq_dk_shared.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def safe_exp_in_intra(x):
    # since it's only calculated in fp32, we can use magic numbers -30 and 88
    # because exp(-30) is close to 0 and exp(88) is close to inf(fp32)
    safe_temp = tl.maximum(x, -30)
    safe_temp = tl.minimum(safe_temp, 88)
    return tl.exp(safe_temp)


class ChunkRWKV6Function(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, r, k, v, g, u, scale, initial_state, output_final_state, checkpoint_level,
                u_2d: bool = False, training: bool = True, use_tf32: bool = False, BT: int = 32):
        q = r  # alias
        B, H, T, K= q.shape
        BC = 16
        BK = min(64, triton.next_power_of_2(K)) if device_capacity else min(32, triton.next_power_of_2(K))
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK = triton.cdiv(K, BK)
        BH = B * H

        if (torch.is_autocast_enabled(device) if check_pytorch_version('2.4') else torch.is_autocast_enabled()):
            torch_dtype = torch.get_autocast_dtype(device) if check_pytorch_version('2.4') else torch.get_autocast_gpu_dtype()
            q, k, v, g, u = (x.to(dtype=torch_dtype) for x in (q, k, v, g, u))
            initial_state = initial_state.to(dtype=torch_dtype) if initial_state is not None else initial_state
        else:
            torch_dtype = torch.float32 if q.dtype != torch.float16 else torch.float16

        tl_dtype = tl.float32 if q.dtype != torch.float16 else tl.float16
        g_org, g, gs, o, final_state, A, h = g, torch.empty_like(
            g, dtype=torch.float32), torch.empty_like(
            g, dtype=torch.float32), torch.empty_like(v), q.new_empty(
            B, H, K, K, dtype=torch_dtype) if output_final_state is not None else None, torch.zeros(
                NK, B, H, T, BT, dtype=torch_dtype, device=q.device), torch.zeros(
                    B, H, NT * K, K, dtype=torch_dtype, device=q.device)

        def grid(meta): return ((triton.cdiv(meta['S'], meta['BS']), NT, BH))
        # keep cummulative normalizer in fp32
        # this kernel is equivalent to
        # g_org = g_org.view(B, H, NT, BT, -1)
        # g = g_org.cumsum(-2).view(B, H, T, -1)
        # gs = g - g_org
        chunk_rwkv6_fwd_kernel_cum[grid](
            g_org, g, gs,
            g.stride(1), g.stride(2), g.stride(3),
            T=T, S=K, BT=BT,
            USE_TF32=use_tf32
        )

        # grid = (NK, NK, BH)
        chunk_rwkv6_fwd_kernel_h[(NK, NK, BH)](
            k, v, g, h, initial_state, final_state,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2), h.stride(3),
            scale=scale,
            T=T, K=K, BT=BT, BK=BK, NT=NT,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            TLTYPE=tl_dtype,
        )

        # grid = (NK, NT * NC * NC, BH)
        chunk_rwkv6_fwd_kernel_intra[(NK, NT * NC * NC, BH)](
            q, k, g, gs, u, A,
            k.stride(1), k.stride(2), k.stride(3),
            scale,
            H=H, T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC, DK=K,
            U_2D=u_2d, TLTYPE=tl_dtype, USE_TF32=use_tf32
        )
        A = A.sum(0, dtype=A.dtype)

        # grid = (NK, NT, BH)
        chunk_rwkv6_fwd_kernel_inter[(NK, NT, BH)](
            q, v, gs, h, o, A,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2), h.stride(3),
            scale,
            T=T, K=K, BT=BT, BK=BK,
            TLTYPE=tl.float32, USE_TF32=use_tf32
        )

        if checkpoint_level > 1:
            del h
            h_t, initial_state = None, None
        else:
            h_t, initial_state = h, (None if initial_state is None else initial_state.clone())
        del g, gs
        if training:
            ctx.save_for_backward(q, k, v, g_org, u, h_t, initial_state, A)
            ctx.BT, ctx.scale, ctx.checkpoint_level, ctx.u_2d, ctx.use_tf32 = BT, scale, checkpoint_level, u_2d, use_tf32
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        q, k, v, g, u, h, initial_state, A = ctx.saved_tensors
        scale, u_2d, use_tf32 = ctx.scale, ctx.u_2d, ctx.use_tf32
        dtype = q.dtype
        B, H, T, K, V = *q.shape, v.shape[-1]
        BT, BC = ctx.BT, 16
        next_pk_2 = triton.next_power_of_2(K)
        BK = min(64, next_pk_2) if device_capacity else min(32, next_pk_2)
        NT, NC = triton.cdiv(T, BT), triton.cdiv(BT, BC)
        NK = triton.cdiv(K, BK)
        BH = B * H
        num_warps = 4 if BK == 64 else 2
        num_stages = 1
        tl_dtype = tl.float32 if q.dtype != torch.float16 else tl.float16

        def fwd_inner(q, k, v, g, B, H, T, K, V, BT, BK, NT, h, h0=None, ht=None):
            NK = triton.cdiv(K, BK)
            grid = (NK, NK, BH)
            chunk_rwkv6_fwd_kernel_h[grid](
                k, v, g, h, h0, ht,
                k.stride(1), k.stride(2), k.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2), h.stride(3),
                T=T, K=K, BT=BT, BK=BK, NT=NT, h=h,
                USE_INITIAL_STATE=h0 is not None,
                STORE_FINAL_STATE=ht is not None,
            )

        # recompute cumulative log decays.
        g_org, g, gs = g, torch.empty_like(g, dtype=torch.float32), torch.empty_like(g, dtype=torch.float32)

        def grid(meta): return ((triton.cdiv(meta['S'], meta['BS']), NT, BH))
        # keep cummulative normalizer in fp32
        # this kernel is equivalent to
        # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
        chunk_rwkv6_fwd_kernel_cum[grid](
            s=g_org, o=g, o_minus_s=gs,
            s_s_h=g.stride(1), s_s_t=g.stride(2), s_s_d=g.stride(3),
            T=T, S=K, BT=BT,
            USE_TF32=use_tf32
        )
        del g_org

        # rerun the forward pass to get h if checkpoint_level >= 1
        if ctx.checkpoint_level == 1:
            h = torch.zeros(B, H, NT * K, V, dtype=torch.float32, device=q.device)
            fwd_inner(
                q=q, k=k, v=v, g=g,
                B=B, H=H, T=T, K=K, BT=BT, BK=BK, NT=NT, h=h,
                h0=initial_state,
                ht=None
            )
        dq, dk, dv, dA, dh, dh0 = torch.empty_like(q, dtype=torch.float), torch.empty_like(k, dtype=torch.float), \
            v.new_empty(NK, *v.shape), torch.empty(B, H, T, BT, dtype=torch.float, device=q.device), \
            torch.zeros(B, H, NT * K, V, dtype=torch.float, device=q.device), \
            torch.empty_like(initial_state) if initial_state is not None else None

        if dht is not None:
            dh[:, :, -K:, :] += dht.to(dh.dtype)

        # bwd_inner
        # grid = (NK, NK, BH)
        chunk_rwkv6_bwd_kernel_dh[(NK, NK, BH)](
            q, g, gs, do, dh, dh0,
            q.stride(1), q.stride(2), q.stride(3),
            do.stride(1), do.stride(2), do.stride(3),
            dh.stride(1), dh.stride(2), dh.stride(3),
            scale,
            T=T, K=K, BT=BT, BK=BK, NT=NT,
            USE_INITIAL_STATE=initial_state is not None,
            TLTYPE=tl_dtype,
        )

        # grid = (NK, NT, BH)
        chunk_rwkv6_bwd_kernel_inter[(NK, NT, BH)](
            k, v, h, g, gs, A, do, dh, dq, dk, dv, dA,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2), h.stride(3),
            scale,
            T=T, K=K, BT=BT, BK=BK,
            TLTYPE=tl_dtype, USE_TF32=use_tf32
        )
        dv = dv.sum(0, dtype=dv.dtype)
        # grid = (NK, NT * NC, BH)
        chunk_rwkv6_bwd_kernel_intra[(NK, NT * NC, BH)](
            q, k, g, gs, dA, dq, dk,
            k.stride(1), k.stride(2), k.stride(3),
            scale=scale,
            T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC,
            TLTYPE=tl.float32,
            num_warps=num_warps,
            num_stages=num_stages
        )

        du = g
        del g, gs, dA, A, dh

        # TODO: fuse?
        dg = (dq * q)[:, :, 1:] - (dk * k)[:, :, 0:-1]
        dg = torch.nn.functional.pad(dg, (0, 0, 0, 1, 0, 0, 0, 0), value=0)
        dg = chunk_global_reversed_cumsum(dg)
        # equivalent to the following pytorch code.
        # du = ((do * v).sum(-1)[..., None] * k * q * scale).sum(-2).to(u)
        # dq += ((do * v).sum(-1)[..., None] * k * scale * u[:, :, None, :])
        # dk += ((do * v).sum(-1)[..., None] * q * scale * u[:, :, None, :])

        # grid = (NT, BH)

        post_process_grad[(NT, BH)](
            q, k, v, u, do, dk, dq, du, scale,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3), H=H,
            T=T, BT=BT, K=K, BK=next_pk_2, U_2D=u_2d,
            TLTYPE=tl_dtype,
            num_warps=num_warps,
            num_stages=num_stages
        )
        du = du.sum([0, 2]) if u_2d else du.sum(2)
        dh0 = dh0.to(q) if initial_state is not None else None
        return dq.to(dtype), dk.to(dtype), dv.to(dtype), dg.to(dtype), du.to(dtype), None, \
            dh0, None, None, None, None, None, None


_detect_use_tf32 = None


def chunk_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    u: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    checkpoint_level: Optional[int] = 0,
    training: bool = True,
    use_tf32: Optional[bool] = None,
    chunk_size: int = 32
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
        checkpoint_level (Optional[int]):
            Checkpointing level; higher values will save more memories and do more recomputations during backward.
            Default: `0`:
            - Level `0`: store forward hidden states for backprop.
            - Level `1`: recompute the forward hidden states during backward.
    """
    global _detect_use_tf32
    assert checkpoint_level in [0, 1]
    scale = r.shape[-1] ** -0.5 if scale == -1.0 else scale
    u_2d = True if u.dim() == 2 else False
    if use_tf32 is None and _detect_use_tf32 is None:
        _detect_use_tf32 = detect_tf32()
    else:
        _detect_use_tf32 = use_tf32
    o, final_state = ChunkRWKV6Function.apply(r, k, v, g, u, scale, initial_state,
                                              output_final_state, checkpoint_level, u_2d, training, _detect_use_tf32, chunk_size)
    return o, final_state


if __name__ == "__main__":
    def get_err_ratio(x, y):
        x = x.to(y.device)
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-14)
    from fla.ops.rwkv6 import fused_recurrent_rwkv6, native_recurrent_rwkv6
    from fla.ops.rwkv6.recurrent_naive import naive_recurrent_rwkv6
    scale = 1.0
    use_h = True
    u_2d = False
    B = 1
    T = 1024*4
    C = 4096
    HEAD_SIZE = 64
    H = C // HEAD_SIZE
    dtype = torch.bfloat16
    from fla.utils import device
    from torch.nn import functional as F
    torch.manual_seed(142)
    atol = 1e-3 if dtype == torch.float else 1e-2

    def RUN_FLA_FUSED(B, T, C, H, r, k, v, w, u, h, scale=1.0):
        r = r.view(B, T, H, -1).transpose(1, 2)
        k = k.view(B, T, H, -1).transpose(1, 2)
        v = v.view(B, T, H, -1).transpose(1, 2)
        w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2))
        o, state = fused_recurrent_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=h, output_final_state=True)
        return o.transpose(1, 2).reshape(B, T, C), state

    def RUN_FLA_CHUNK(B, T, C, H, r, k, v, w, u, h, scale=1.0, chunk_size=32):
        r = r.view(B, T, H, -1).transpose(1, 2)
        k = k.view(B, T, H, -1).transpose(1, 2)
        v = v.view(B, T, H, -1).transpose(1, 2)
        w = -torch.exp(w.view(B, T, H, -1).transpose(1, 2))
        o, state = chunk_rwkv6(r, k, v, w, u=u, scale=scale, initial_state=h, output_final_state=True, chunk_size=chunk_size)
        return o.transpose(1, 2).reshape(B, T, C), state

    with torch.no_grad():
        q = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        k = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        v = torch.empty(B, T, C, device=device).uniform_(-1, 1).to(dtype=dtype)
        w = torch.empty(B, T, C, device=device).uniform_(-8, -6).to(dtype=dtype)
        u = torch.empty(H, HEAD_SIZE, device=device).uniform_(-1, 1).to(dtype=dtype)
        h = torch.zeros(B, H, HEAD_SIZE, HEAD_SIZE, device=device).to(dtype=dtype)
        do = torch.rand_like(v, device=device)

    def clear_grad():
        q.requires_grad_()
        k.requires_grad_()
        v.requires_grad_()
        w.requires_grad_()
        u.requires_grad_()
        h.requires_grad_()
        if q.grad is not None:
            q.grad.data.zero_()
        if k.grad is not None:
            k.grad.data.zero_()
        if v.grad is not None:
            v.grad.data.zero_()
        if w.grad is not None:
            w.grad.data.zero_()
        if u.grad is not None:
            u.grad.data.zero_()
        if h.grad is not None:
            h.grad.data.zero_()
    clear_grad()

    def LOSS(y):
        return ((y * y) - torch.tanh(y)).sum()
    ref_o, _ = RUN_FLA_FUSED(B, T, C, H, q.float(), k.float(), v.float(), w.float(), u.float(), h.float())
    LOSS(ref_o).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dw, w.grad = w.grad.clone(), None
    ref_du, u.grad = u.grad.clone(), None
    if use_h:
        ref_dh, h.grad = h.grad.clone(), None

    tri_o, _ = RUN_FLA_CHUNK(B, T, C, H, q, k, v, w, u, h)
    LOSS(tri_o).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dw, w.grad = w.grad.clone(), None
    tri_du, u.grad = u.grad.clone(), None
    if use_h:
        tri_dh, h.grad = h.grad.clone(), None

    assert get_err_ratio(ref_o, tri_o) < atol
    print("dq", get_err_ratio(ref_dq, tri_dq))  # pass
    print("dk", get_err_ratio(ref_dk, tri_dk))
    print("dv", get_err_ratio(ref_dv, tri_dv))
    print("dw", get_err_ratio(ref_dw, tri_dw))
    print("du", get_err_ratio(ref_du, tri_du))
    if use_h:
        print("dh", get_err_ratio(ref_dh, tri_dh))
    assert get_err_ratio(ref_dq, tri_dq) < atol
    assert get_err_ratio(ref_dk, tri_dk) < atol
    assert get_err_ratio(ref_dv, tri_dv) < atol
    assert get_err_ratio(ref_dw, tri_dw) < atol
    assert get_err_ratio(ref_du, tri_du) < atol
    if use_h:
        assert get_err_ratio(ref_dh, tri_dh) < atol
