import torch as th
import torch
from typing import Optional, Tuple


def naive_chunk_rwkv7(w_, q_, k_, v_, a_, b_, s0_, y_, s_, sT_):
    B, T, H, C = w_.shape
    K = 16
    assert (T % K == 0)
    dtype = w_.dtype

    for bi in range(B):
        for hi in range(H):
            state = s0_[bi, hi, :, :].float()
            for t in range(T//K):
                sw, sq, sk, sv, sa, sb = [x[bi, t*K:t*K+K, hi, :].float() for x in (w_, q_, k_, v_, a_, b_)]

                w = (-sw.exp()).exp()
                fw = th.prod(w, dim=0, keepdim=True)
                incl_pref = th.cumprod(w, dim=0)
                non_incl_pref = incl_pref/w
                inv_incl_pref = 1/incl_pref

                wq = sq * incl_pref
                wa = sa * non_incl_pref
                kwi = sk * inv_incl_pref
                bwi = sb * inv_incl_pref

                ab = (wa @ bwi.mT).tril(-1)
                ak = (wa @ kwi.mT).tril(-1)
                ab_inv = torch.eye(K, device=ab.device, dtype=ab.dtype)
                for i in range(K):
                    for j in range(i):
                        ab_inv[i] = ab_inv[i] + ab[i, j] * ab_inv[j]  # 注意这里是 + 因为原来是 -ab
                ab_u = ak @ sv + wa @ state.mT
                u = ab_inv @ ab_u
                qk = (wq @ kwi.mT).tril(0)
                qb = (wq @ bwi.mT).tril(0)
                yy = qk @ sv + qb @ u + wq @ state.mT
                y_[bi, t*K:t*K+K, hi, :] = yy.to(dtype)

                s_[bi, hi, t, :, :] = state.to(dtype)
                state = state * fw + sv.mT @ (kwi*fw) + u.mT @ (bwi*fw)
            sT_[bi, hi, :, :] = state.to(dtype)


def native_chunk_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
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
    B, H, L, D = r.shape
    y_ = torch.zeros(B, L, H, D, device=device, dtype=dtype)
    s_ = torch.zeros(B, H, L//16, D, D, device=device, dtype=dtype)
    s0_ = torch.zeros(B, H, D, D, device=device, dtype=dtype)
    sT_ = torch.zeros(B, H, D, D, device=device, dtype=dtype)
    naive_chunk_rwkv7(w_t, q_t, k_t, v_t, a_t, b_t, s0_, y_, s_, sT_)
    o = y_.permute(0, 2, 1, 3)  # [B,H,T,C]
    return o, None


if __name__ == "__main__":
    from fla.utils import device
    B = 1
    H = 1
    L = 16
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
    w = (torch.randn(B, H, L, D).uniform_(0.95, 0.9997).to(device).to(dtype)).requires_grad_(require_grad)

    # a: 基于 torch.sigmoid(...) * 2.0 的计算，范围在 [0, 2]
    a = torch.rand(B, H, L, D, device=device, dtype=dtype).clamp(0, 0.1).requires_grad_(require_grad)

    # b: 在模型中是 kk*a，其中 kk 是归一化的，所以范围可能在 [-2, 2]
    b = torch.randn(B, H, L, D, device=device, dtype=dtype).clamp(-0.2, 0.2).requires_grad_(require_grad)

    do = torch.rand_like(v).to(device).fill_(torch.rand(1).item())
    h = torch.zeros(B, H, D, D, device=device, dtype=torch.float32).fill_(torch.rand(1).item()).requires_grad_(require_grad)

    y_ = torch.zeros(B, L, H, D, device=device, dtype=dtype)
    s_ = torch.zeros(B, H, L//16, D, D, device=device, dtype=dtype)
    s0_ = torch.zeros(B, H, D, D, device=device, dtype=dtype)
    sT_ = torch.zeros(B, H, D, D, device=device, dtype=dtype)

    # 调用参考实现
    q_t = q.permute(0, 2, 1, 3)  # [B,T,H,C]
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    w_t = w.permute(0, 2, 1, 3)
    a_t = a.permute(0, 2, 1, 3)
    b_t = b.permute(0, 2, 1, 3)
    # compiled_v7 = torch.compile(naive_chunk_rwkv7)

    from fla.ops.rwkv7.recurrent_naive import naive_recurrent_rwkv7

    with torch.no_grad():
        y_ref, _ = native_chunk_rwkv7(q, k, v, w, a, b)
        w = -torch.exp(w)
        ref_o, _, _ = naive_recurrent_rwkv7(q, k, v, w, a, b, scale=1.0, initial_state=None, manual_einsum=False)
        torch.testing.assert_close(ref_o, y_ref, rtol=1e-4, atol=1e-4)
        assert torch.allclose(ref_o, y_ref, rtol=1e-4, atol=1e-4)
        print("Pass")
