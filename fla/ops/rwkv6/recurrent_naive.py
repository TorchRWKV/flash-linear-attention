# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import check_pytorch_version, device



def naive_recurrent_rwkv6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = 1.0,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    torch_dtype = q.dtype if q.dtype in [torch.float16, torch.float32, torch.float64] else torch.float32
    orig_dtype = q.dtype
    B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    q, k, v, w, u = (x.to(dtype=torch_dtype) for x in (q, k, v, w, u))
    h = torch.zeros(B, H, K, V, dtype=torch_dtype, device=q.device)
    o = torch.zeros_like(v)

    if scale == -1.0:
        scale = K ** -0.5

    if initial_state is not None:
        h += initial_state.to(dtype=torch_dtype)

    w = w.exp()


    u_expand = u[None, ..., None]

    for i in range(T):
        q_i = q[:, :, i, :] * scale
        k_i = k[:, :, i] * scale
        v_i = v[:, :, i, :]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        o_i = (h + u_expand * kv_i) * q_i[..., None]
        o[:, :, i] = o_i.sum(-2)
        h = h * w_i[..., None] + kv_i

    if not output_final_state:
        ht = None
    elif initial_state is not None:
        ht = h.to(initial_state.dtype)
    else:
        ht = h.to(orig_dtype)

    return o.to(orig_dtype), ht


def naive_recurrent_rwkv6_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dh_t: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    scale: float = 1.0,
):
    torch_type = torch.float32 if q.dtype != torch.float16 else torch.float16
    q, k, v, w, u, o, do = (x.to(dtype=torch_type) for x in (q, k, v, w, u, o, do))
    B, H, T, K, V = q.shape[0], q.shape[1], q.shape[2], q.shape[3], v.shape[-1]
    h = torch.zeros(B, H, K, V, dtype=torch_type, device=q.device)
    dq = torch.zeros_like(q)
    dq_aux = torch.zeros_like(q)

    if initial_state is not None:
        h += initial_state

    if scale == -1.0:
        scale = K ** -0.5

    w = w.exp()

    u_expand = u[None, ..., None]
    sum_dims = [0, -1]

    for i in range(T):
        k_i = k[:, :, i] * scale
        v_i = v[:, :, i]
        w_i = w[:, :, i]
        kv_i = k_i[..., None] * v_i[..., None, :]
        h_i = (h + u_expand * kv_i)
        dq_i = (do[:, :, i, None, :] * h_i).sum(-1)
        dq_aux_i = (do[:, :, i, None, :] * h).sum(-1)
        dq[:, :, i] = dq_i * scale
        dq_aux[:, :, i] = dq_aux_i
        h = h * w_i[..., None] + kv_i

    du = torch.zeros_like(u)
    dh = torch.zeros_like(h)
    if dh_t is not None:
        dh += dh_t
    dk = torch.zeros_like(k)
    dk_aux = torch.zeros_like(k)
    dv = torch.zeros_like(v)



    for i in range(T - 1, -1, -1):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i] * scale
        v_i = v[:, :, i]

        d_kv_i = do[:, :, i, None, :] * q_i[..., None]
        du += (d_kv_i * k_i[..., None] * v_i[..., None, :]).sum(sum_dims)

        dk_i = (dh * v_i[..., None, :]).sum(-1)
        dk_aux[:, :, i] = dk_i
        dk_i += (d_kv_i * u_expand * v_i[..., None, :]).sum(-1)

        dv_i = (d_kv_i * u_expand * k_i[..., None]).sum(-2)
        dv_i += (dh * k_i[..., None]).sum(-2)

        dk[:, :, i] = dk_i * scale
        dv[:, :, i] = dv_i
        dh = dh * w[:, :, i, :, None] + d_kv_i


    # dw = q * dq_aux - k * dk_aux
    dw = torch.zeros_like(w)
    for i in range(T - 2, -1, -1):
        dw[:, :, i] = dw[:, :, i + 1] + dq_aux[:, :, i + 1] * q[:, :, i + 1] * scale - dk_aux[:, :, i] * k[:, :, i] * scale

    return dq, dk, dv, dw, du, dh


class NativeRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, w, u, scale, initial_state, output_final_state: bool = False, training: bool = True):
        o, ht = naive_recurrent_rwkv6(q, k, v, w, u, scale, initial_state, output_final_state)
        if initial_state is not None:
            initial_state = initial_state.clone()
        if training:
            ctx.save_for_backward(q, k, v, w, u, o, initial_state)
            ctx.scale = scale
        return o, ht

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, do, dht):
        q, k, v, w, u, o, initial_state = ctx.saved_tensors
        dq, dk, dv, dw, du, dh = naive_recurrent_rwkv6_bwd(q, k, v, w, u, o, do, dht, initial_state, ctx.scale)
        dh = dh if initial_state is not None else None
        return dq, dk, dv, dw, du, None, dh, None, None


def native_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
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

    o, final_state = NativeRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state, training)

    return o, final_state


if __name__ == "__main__":
    from fla.utils import get_available_device
    device = get_available_device()
    B = 4
    H = 64
    L = 1024
    D = 64
    dtype = torch.float16
    require_grad = True
    torch.manual_seed(42)
    q = (torch.randn(B, H, L, D).to(device).to(dtype)).requires_grad_(require_grad)
    k = (torch.randn(B, H, L, D).to(device).to(dtype)).requires_grad_(require_grad)
    v = torch.randn(B, H, L, D).to(device).to(dtype).requires_grad_(require_grad)
    w = torch.nn.functional.logsigmoid(torch.randn(B, H, L, D)).to(device).to(dtype).requires_grad_(require_grad)
    u = (torch.randn(B, H, D).to(device).to(dtype)).requires_grad_(require_grad)
    do = torch.rand_like(v).to(device)
    h = torch.randn(B, H, D, D, device=device, dtype=torch.float32, requires_grad=True)
    o, _ = naive_recurrent_rwkv6(q, k, v, w, u, scale=-1.0, initial_state=h)
    o.backward(do)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
    dw, w.grad = w.grad.clone(), None
    du, u.grad = u.grad.clone(), None
    dh, h.grad = h.grad.clone(), None

    o2, _ = NativeRecurrentRWKV6Function.apply(q.float(), k.float(), v.float(), w.float(), u.float(), -1.0, h.float())
    o2.backward(do)

    def rmsre(x, y, eps=1e-11):
        err = (x-y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base+eps)

    def print_diff(name, grad1, grad2):
        abs_diff = (grad1 - grad2).abs()
        max_diff = abs_diff.max().item()
        rmsre_value = rmsre(grad1.float(), grad2.float())
        print(f"{name}: Max Abs Diff = {max_diff:.6f}, RMSRE = {rmsre_value:.6f}")

    print(f"o: {(o - o2).abs().max().item():.6f}")
    print_diff("q", q.grad, dq)
    print_diff("k", k.grad, dk)
    print_diff("v", v.grad, dv)
    print_diff("w", w.grad, dw)
    print_diff("u", u.grad, du)
    print_diff("h", h.grad, dh)

    all_grads1 = torch.cat([q.grad.flatten(), k.grad.flatten(), v.grad.flatten(),
                            w.grad.flatten(), u.grad.flatten(), h.grad.flatten()])
    all_grads2 = torch.cat([dq.flatten(), dk.flatten(), dv.flatten(),
                            dw.flatten(), du.flatten(), dh.flatten()])
    overall_rmsre = rmsre(all_grads1, all_grads2)
    print(f"\nOverall RMSRE: {overall_rmsre:.6f}")
