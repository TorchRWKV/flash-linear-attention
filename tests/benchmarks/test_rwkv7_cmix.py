import torch
import torch.nn as nn
import time
from typing import Tuple
import gc
from fla.ops.rwkv7.channel_mixing import (
    channel_mixing_rwkv7,
    channel_mixing_rwkv7_torch,
)
from fla.utils import device


def benchmark_forward_backward(
    fn, args: Tuple, warmup: int = 5, repeats: int = 10, need_grad: bool = True
) -> Tuple[float, float, float, float]:
    """
    测试函数的前向和反向传播性能

    Returns:
        (fwd_mean, fwd_std, bwd_mean, bwd_std)
    """
    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    fwd_times = []
    bwd_times = []

    # 预热
    for _ in range(warmup):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            out = fn(*args)
            if need_grad:
                loss = out[0].sum() + out[1].sum()
                loss.backward()

    # 正式测试
    for _ in range(repeats):
        # 前向传播计时
        torch.cuda.synchronize()
        fwd_start = time.perf_counter()

        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            out = fn(*args)

        torch.cuda.synchronize()
        fwd_end = time.perf_counter()
        fwd_times.append((fwd_end - fwd_start) * 1000)  # 转换为ms

        # 反向传播计时
        if need_grad:
            bwd_start = time.perf_counter()
            loss = out[0].sum() + out[1].sum()
            loss.backward()
            torch.cuda.synchronize()
            bwd_end = time.perf_counter()
            bwd_times.append((bwd_end - bwd_start) * 1000)

    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times) if need_grad else torch.zeros_like(fwd_times)

    return (
        fwd_times.mean().item(),
        fwd_times.std().item(),
        bwd_times.mean().item(),
        bwd_times.std().item(),
    )


def measure_memory_usage(fn, args):
    """详细测量函数执行期间的显存使用情况"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # 初始显存
    init_mem = torch.cuda.memory_allocated()

    # 测量纯前向传播(no_grad)
    with torch.no_grad():
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            _ = fn(*args)
        torch.cuda.synchronize()
        forward_only_mem = torch.cuda.max_memory_allocated()

    # 重置统计
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # 测量带梯度的完整过程
    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
        out = fn(*args)
        torch.cuda.synchronize()
        forward_with_grad_mem = torch.cuda.max_memory_allocated()

        loss = out[0].sum() + out[1].sum()
        loss.backward()
        torch.cuda.synchronize()
        total_mem = torch.cuda.max_memory_allocated()

    return {
        "initial": init_mem / 1024**2,
        "forward_only": (forward_only_mem - init_mem) / 1024**2,  # 纯前向
        "forward_with_grad": (forward_with_grad_mem - init_mem)
        / 1024**2,  # 带梯度的前向
        "total": (total_mem - init_mem) / 1024**2,  # 总峰值
    }


def main():
    # 设置参数
    batch_size = 8
    seq_len = 4096
    n_embd = 4096
    dim_ffn = n_embd * 4
    print("batch_size = ", batch_size, "seq_len = ", seq_len, "n_embd = ", n_embd)

    # 准备测试数据
    x = nn.Parameter(torch.randn(batch_size, seq_len, n_embd)).to(device).bfloat16()
    x_prev = nn.Parameter(torch.randn(batch_size, n_embd)).to(device).bfloat16()
    x_k = nn.Parameter(torch.randn(1, 1, n_embd)).to(device).bfloat16()
    K_ = nn.Parameter(torch.randn(n_embd, dim_ffn)).to(device).bfloat16()
    V_ = nn.Parameter(torch.randn(dim_ffn, n_embd)).to(device).bfloat16()

    args = (x, x_prev, x_k, K_, V_)

    # 测试PyTorch版本
    print("\n=== Testing PyTorch Implementation ===")
    torch_fwd_mean, torch_fwd_std, torch_bwd_mean, torch_bwd_std = (
        benchmark_forward_backward(channel_mixing_rwkv7_torch, args)
    )
    torch_mem = measure_memory_usage(channel_mixing_rwkv7_torch, args)

    # 测试Triton版本
    print("\n=== Testing Triton Implementation ===")
    triton_fwd_mean, triton_fwd_std, triton_bwd_mean, triton_bwd_std = (
        benchmark_forward_backward(channel_mixing_rwkv7, args)
    )
    triton_mem = measure_memory_usage(channel_mixing_rwkv7, args)

    # 打印结果
    print("\n=== Performance Comparison ===")
    print(f"{'':15} {'PyTorch':>12} {'Triton':>12} {'Speedup':>12}")
    print("-" * 50)
    print(
        f"Forward (ms): {torch_fwd_mean:>12.2f} {triton_fwd_mean:>12.2f} {torch_fwd_mean/triton_fwd_mean:>12.2f}x"
    )
    print(
        f"Backward (ms): {torch_bwd_mean:>12.2f} {triton_bwd_mean:>12.2f} {torch_bwd_mean/triton_bwd_mean:>12.2f}x"
    )

    print(
        "\n=== Memory Usage (MB) ===\n"
        f"{'':20} {'PyTorch':>12} {'Triton':>12} {'Ratio':>12}\n" + "-" * 56 + "\n"
        f"Initial: {torch_mem['initial']:>12.2f} {triton_mem['initial']:>12.2f}\n"
        f"Forward (no_grad): {torch_mem['forward_only']:>12.2f} {triton_mem['forward_only']:>12.2f} "
        f"{torch_mem['forward_only']/triton_mem['forward_only']:>12.2f}x\n"
        f"Forward (with_grad): {torch_mem['forward_with_grad']:>12.2f} {triton_mem['forward_with_grad']:>12.2f} "
        f"{torch_mem['forward_with_grad']/triton_mem['forward_with_grad']:>12.2f}x\n"
        f"Total Peak: {torch_mem['total']:>12.2f} {triton_mem['total']:>12.2f} "
        f"{torch_mem['total']/triton_mem['total']:>12.2f}x"
    )


if __name__ == "__main__":
    main()
