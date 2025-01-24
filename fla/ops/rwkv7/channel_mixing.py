import triton
import triton.language as tl
import torch
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous
from fla.utils import check_pytorch_version, set_torch_device
import logging

logger = logging.getLogger(__name__)

if not check_pytorch_version('2.4'):
    logger.warning('PyTorch < 2.4 detected - computations may be slower due to lack of optimizations')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['hidden_dim']
)
@triton.jit
def rwkv_seq_mix_kernel(
    x_ptr,              # 输入序列张量(batch, seq_len, hidden_dim)
    x_prev_ptr,         # 前一个时间步的状态(batch, seq_len, hidden_dim)
    mix_k_ptr,          # 混合系数K
    output_ptr,         # 输出张量
    batch_size: tl.constexpr,    # batch size
    token_length,   # 序列长度
    hidden_dim: tl.constexpr,     # 隐藏维度
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    block_idx = block_start + tl.arange(0, BLOCK_SIZE)[:]

    # 计算batch、序列和特征索引
    total_seq_dim = token_length * hidden_dim
    batch_idx = block_idx // total_seq_dim
    seq_and_feat = block_idx % total_seq_dim
    seq_idx = seq_and_feat // hidden_dim
    feat_idx = seq_and_feat % hidden_dim
    # 添加序列长度检查
    is_valid = (batch_idx < batch_size) & (seq_idx < token_length)  # 序列长度上限

    x_idx = batch_idx * total_seq_dim + seq_idx * hidden_dim + feat_idx

    # 加载当前值
    curr_x = tl.load(x_ptr + x_idx, mask=is_valid, other=0.0).to(tl.float32)
    k_value = tl.load(mix_k_ptr + feat_idx).to(tl.float32)

    # 处理第一个时间步
    is_first = seq_idx < 1
    prev_state_idx = batch_idx * hidden_dim + feat_idx
    prev_state = tl.load(x_prev_ptr + prev_state_idx,
                         mask=(is_first & is_valid),
                         other=0.0).to(tl.float32)

    # 加载前一个时间步的值
    prev_x_idx = x_idx - hidden_dim
    prev_x = tl.load(x_ptr + prev_x_idx,
                     mask=(~is_first & is_valid),
                     other=0.0).to(tl.float32)

    # 计算输出
    prev_value = tl.where(is_first, prev_state, prev_x)
    state_diff = prev_value - curr_x
    mixed = state_diff * k_value
    result = curr_x + mixed
    # 存储结果
    tl.store(output_ptr + x_idx, result.to(output_ptr.dtype.element_ty), mask=is_valid)


@triton.jit
def rwkv_channel_mixing_pow_and_relu(
    in_ptr,    # pointer to input/output tensor
    out_ptr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused ReLU and Power operation: x = ReLU(x)^2"""
    # 计算全局位置
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    x0 = xindex
    x = tl.load(in_ptr + (x0), None)
    # ReLU操作
    x = tl.maximum(x, 0.0).to(tl.float32)
    # square
    x = x * x
    tl.store(out_ptr + (x0), x.to(out_ptr.dtype.element_ty), None)


def torch_rwkv_mix(x: torch.Tensor, x_prev: torch.Tensor, x_k: torch.Tensor):
    """
    处理 RWKV mix 操作，自动处理有无 batch 维度的情况
    Args:
        x: 输入张量，形状可以是 (batch_size, seq_len, hidden_dim)
        x_prev: 前一状态，形状 (batch_size, hidden_dim)
        x_k: 混合系数，形状应为 (1, 1, hidden_dim)
    Returns:
        处理后的张量，维度与输入 x 相同
    """
    x_prev = x_prev.unsqueeze(1)  # (batch_size, 1, hidden_dim)
    xx = torch.cat((x_prev, x[:, :-1, :]), dim=1) - x
    k = x + xx * x_k

    return k


def torch_rwkv_relu_and_square(x: torch.Tensor):
    return torch.relu(x) ** 2


def triton_rwkv_mix(x, x_prev, x_k):
    """
    Triton 实现的 RWKV mix 操作，自动处理有无 batch 维度的情况
    Args:
        x: 输入张量，形状是 (batch_size, seq_len, hidden_dim)
        x_prev: 前一状态，形状是 (batch_size, hidden_dim)
        x_k: 混合系数，形状应为 (1, 1, hidden_dim)
        inplace: 是否进行原地操作
    """
    has_batch = x.dim() == 3

    if has_batch:
        batch_size, token_length, hidden_dim = x.shape
    else:
        token_length, hidden_dim = x.shape
        batch_size = 1
        # 添加 batch 维度
        x = x.unsqueeze(0)
        x_prev = x_prev.unsqueeze(0)

    token_length = x.shape[1]
    hidden_dim = x.shape[2]
    total_elements = batch_size * token_length * hidden_dim

    output = torch.empty_like(x)

    def grid(meta): return (
        (total_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],  # grid_0
        1,  # grid_1
        1   # grid_2
    )

    rwkv_seq_mix_kernel[grid](
        x.contiguous(),
        x_prev.contiguous(),
        x_k.squeeze(),
        output,
        batch_size=batch_size,
        token_length=token_length,
        hidden_dim=hidden_dim,
    )
    if not has_batch:
        output = output.squeeze(0)
    return output


def triton_rwkv_relu_and_square(x: torch.Tensor, inplace: bool = True):
    """
    Triton implementation of RWKV's ReLU and square operation
    Args:
        x: Input tensor
    Returns:
        Tensor after ReLU and square operations
    """
    x = x.contiguous()
    output = x if inplace else torch.empty_like(x)

    def grid(meta): return (
        (output.numel() + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],  # grid_0
        1,  # grid_1
        1   # grid_2
    )
    # 让 autotune 来决定最佳的 BLOCK 大小
    rwkv_channel_mixing_pow_and_relu[grid](
        x,
        output,
        BLOCK_SIZE=4096,
    )

    return output


@triton.jit
def relu_square_backward(
    out_ptr,  # 输出的梯度
    forward_input_ptr,  # 前向传播的输入
    BLOCK_SIZE: tl.constexpr  # block大小
):
    """ReLU(x)^2 的反向传播实现
    grad_input = grad_output * 2 * x if x > 0 else 0
    """
    # 计算线程索引
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 加载数据
    x = tl.load(forward_input_ptr + offsets).to(tl.float32)
    grad = tl.load(out_ptr + offsets).to(tl.float32)

    # 计算ReLU的mask
    x = tl.maximum(x, 0.0)

    # 计算梯度: grad * 2x if x > 0 else 0
    grad_input = grad * 2 * x

    # 存储结果
    tl.store(out_ptr + offsets, grad_input.to(out_ptr.dtype.element_ty))


@triton.jit
def rwkv_mix_backward(
    dk1_ptr0,
    xk_ptr,
    dx_ptr,
    dx_prev_ptr,
    token_length: tl.constexpr,    # 总元素数
    n_embd: tl.constexpr,
    XBLOCK: tl.constexpr
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]

    x0 = xindex % n_embd
    x1 = (xindex // n_embd) % token_length

    # 计算主要项
    dk1 = tl.load(dk1_ptr0 + xindex).to(tl.float32)
    xk = tl.load(xk_ptr + x0, eviction_policy='evict_last').to(tl.float32)
    prod = dk1 * xk

    mask = x1 < token_length-1
    prev_term = tl.where(mask,
                         tl.load(dk1_ptr0 + (n_embd + xindex), mask).to(tl.float32) *
                         tl.load(xk_ptr + x0, mask).to(tl.float32),
                         0.0)

    # 存储结果
    tl.store(dx_ptr + xindex, (dk1 - prod + prev_term).to(dx_ptr.dtype.element_ty))
    tl.store(dx_prev_ptr + xindex, prod.to(dx_prev_ptr.dtype.element_ty))


def channel_mixing_rwkv7_torch(x, x_prev, x_k, key_weight, value_weight):
    k1 = torch_rwkv_mix(x, x_prev, x_k)
    k1_K = k1 @ key_weight
    k = torch_rwkv_relu_and_square(k1_K)
    return k @ value_weight, x[-1, :]


@torch.compile(fullgraph=True)
def compute_x_k_grad(dk1, x, x_prev):
    """计算x_k的梯度的原生torch写法
    Args:
        dk1: (batch*seq_len, hidden_dim)
        x: (batch, seq_len, hidden_dim)
        x_prev: (batch, hidden_dim)
    """
    hidden_dim = x.shape[2]

    # 计算xx = cat(x_prev, x[:, :-1]) - x
    x_prev = x_prev.unsqueeze(1)  # (batch, 1, hidden_dim)
    xx = torch.cat((x_prev, x[:, :-1, :]), dim=1) - x  # (batch, seq_len, hidden_dim)

    # reshape到正确的形状
    # 计算梯度
    grad_x_k = (dk1 * xx.reshape(-1, hidden_dim)).sum(dim=0).unsqueeze(0).unsqueeze(0)    # (hidden_dim,)
    # (1, 1, hidden_dim)

    return grad_x_k


def rwkv_channel_mixing_triton_backward(grad_output, x, x_prev, x_k, key_weight, value_weight, k1, k1_K, k):

    batch_size = x.shape[0] if x.dim() == 3 else 1
    seq_len = x.shape[-2]
    n_embd = x.shape[-1]

    dV = k.transpose(-2, -1) @ grad_output

    dk = grad_output @ value_weight.transpose(-2, -1)

    BLOCK_SIZE = 4096
    grid = ((dk.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_square_backward[grid](
        dk,  # 会被原地修改
        k1_K,
        BLOCK_SIZE=4096
    )

    dK = k1.transpose(-2, -1) @ dk  # 使用局部梯度
    dk1 = dk @ key_weight.transpose(-2, -1)  # 使用局部梯度
    dk1 = dk1.view(-1, n_embd)

    dx_prev = torch.empty((batch_size, seq_len, n_embd), device=x.device, dtype=x.dtype)

    dk_reduced = compute_x_k_grad(dk1, x, x_prev)

    dx = torch.empty_like(x)
    # 最后的mix backward kernel
    grid_mix = ((batch_size * seq_len * n_embd + 4096 - 1) // 4096, 1, 1)
    rwkv_mix_backward[grid_mix](
        dk1.contiguous(),
        x_k.squeeze(),
        dx,
        dx_prev,
        seq_len,
        n_embd,
        XBLOCK=4096
    )
    # dx_prev.shape batch_size, seq_len, n_embd
    return dx, dx_prev[:, 0, :], dk_reduced, dK, dV


class Rwkv7ChannelMixing(torch.autograd.Function):
    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, x, x_prev, x_k, key_weight, value_weight, train_mode=True):
        # Step 1: 使用 triton_rwkv_mix 计算 k1
        k1 = triton_rwkv_mix(x, x_prev, x_k)

        # Step 2: 矩阵乘法 k1 @ K_
        k1_K = k1 @ key_weight

        # Step 3: 使用 triton_rwkv_relu_and_square 计算 k
        k = triton_rwkv_relu_and_square(k1_K, inplace=(not train_mode))  # 可以用inplace因为k1_K不需要保留

        # Step 4: 最后一个矩阵乘法 k @ V_
        # 返回结果和最后一个状态
        ctx.save_for_backward(x, x_prev, x_k, key_weight, value_weight, k1, k1_K, k)
        return k @ value_weight

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, dkv):
        x, x_prev, x_k, K_, V_, k1, k1_K, k = ctx.saved_tensors

        dx, dx_prev, dk_reduced, dK, dV = rwkv_channel_mixing_triton_backward(dkv, x, x_prev, x_k, K_, V_, k1, k1_K, k)
        return dx, dx_prev, dk_reduced, dK, dV, None


def channel_mixing_rwkv7(x: torch.Tensor, x_prev: torch.Tensor, x_k: torch.Tensor,
                         key_weight: torch.Tensor, value_weight: torch.Tensor,
                         train_mode: bool = True):
    """
        x: 输入张量，形状是 (batch_size, seq_len, hidden_dim)
        x_prev: 前一状态，形状是 (batch_size, hidden_dim)
        x_k: 混合系数，形状应为 (1, 1, hidden_dim)
    """
    assert x.dim() == 3
    set_torch_device(x)
    return Rwkv7ChannelMixing.apply(x, x_prev, x_k, key_weight, value_weight, train_mode), x[-1, :]
