import pytest
import torch
import torch.testing
from fla.ops.rwkv7.channel_mixing import (
    triton_rwkv_mix, 
    torch_rwkv_mix,
    triton_rwkv_relu_and_square,
    torch_rwkv_relu_and_square,
    channel_mixing_rwkv7_torch,
    channel_mixing_rwkv7
)
from fla.utils import device

@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rwkv_mix(batch_size, seq_len, hidden_dim, dtype):
    torch.manual_seed(13)
    
    x = torch.randn(batch_size, seq_len, hidden_dim, 
                    device=device, dtype=dtype)
    x_prev = torch.randn(batch_size, hidden_dim, 
                         device=device, dtype=dtype)
    x_k = torch.randn(1, 1, hidden_dim, 
                      device=device, dtype=dtype)
    
    torch_output = torch_rwkv_mix(x.to(torch.float32), x_prev.to(torch.float32), x_k.to(torch.float32))
    triton_output = triton_rwkv_mix(x, x_prev, x_k)
    rtol = 1e-5 if dtype == torch.float32 else 1e-2
    atol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(torch_output, triton_output.to(torch.float32), rtol=rtol, atol=atol)

@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
@pytest.mark.parametrize("hidden_dim", [512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("inplace", [True, False])
def test_rwkv_relu_and_square(seq_len, hidden_dim, dtype, inplace):
    torch.manual_seed(42)
    
    x = torch.randn(seq_len, hidden_dim, 
                    device=device, dtype=dtype)
    
    torch_output = torch_rwkv_relu_and_square(x)
    triton_output = triton_rwkv_relu_and_square(x, inplace=inplace)
    
    torch.testing.assert_close(torch_output, triton_output, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("seq_len", [1024, 2048])
@pytest.mark.parametrize("n_embd", [512, 1024])
@pytest.mark.parametrize("dim_ffn", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_channel_mixing_gradients(batch_size, seq_len, n_embd, dim_ffn, dtype):
    torch.manual_seed(42)
    torch._dynamo.config.cache_size_limit = 512

    x = torch.randn(batch_size, seq_len, n_embd, 
                    device=device, dtype=dtype, requires_grad=True)
    x_prev = torch.randn(batch_size, n_embd, 
                        device=device, dtype=dtype, requires_grad=True)
    x_k = torch.randn(1, 1, n_embd, 
                      device=device, dtype=dtype, requires_grad=True)
    K_ = torch.randn(n_embd, dim_ffn, 
                     device=device, dtype=dtype, requires_grad=True)
    V_ = torch.randn(dim_ffn, n_embd, 
                     device=device, dtype=dtype, requires_grad=True)

    # Clone data for second implementation
    x2 = x.clone().detach().requires_grad_(True)
    x_prev2 = x_prev.clone().detach().requires_grad_(True)
    x_k2 = x_k.clone().detach().requires_grad_(True)
    K_2 = K_.clone().detach().requires_grad_(True)
    V_2 = V_.clone().detach().requires_grad_(True)

    # First implementation
    out1, last1 = channel_mixing_rwkv7_torch(x.to(torch.float32), x_prev.to(torch.float32), x_k.to(torch.float32), K_.to(torch.float32), V_.to(torch.float32))
    loss1 = out1.mean() + last1.mean()
    loss1.backward()

    # Second implementation
    out2, last2 = channel_mixing_rwkv7(x2, x_prev2, x_k2, K_2, V_2)
    loss2 = out2.mean() + last2.mean()
    loss2.backward()

    # Test gradients
    rtol = 1e-3 if dtype == torch.float32 else 0.025
    atol = 1e-3 if dtype == torch.float32 else 0.025
    
    torch.testing.assert_close(x.grad, x2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_prev.grad, x_prev2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(x_k.grad, x_k2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(K_.grad, K_2.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(V_.grad, V_2.grad, rtol=rtol, atol=atol)