# -*- coding: utf-8 -*-

import functools

import torch


def contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def require_version(version, hint):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(ctx, *args, **kwargs):
            from transformers.utils.versions import require_version
            require_version(version, hint)
            return fn(ctx,
                      *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                      **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
        return wrapper
    return decorator


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(func, *args, **kwargs)
    return wrapper

def get_available_device():
    # 检查 CUDA
    if torch.cuda.is_available():
        return 'cuda'

    # 检查 XPU
    try:
        pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if pytorch_version > (2, 4):
            if torch.xpu.is_available():
                return 'xpu'
        else:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                return 'xpu'
    except ImportError:
        pass

    # 检查 MUSA
    try:
        import torch_musa
        if torch.musa.is_available():
            return 'musa'
    except ImportError:
        pass

    # 检查 NPU, 昇腾不支持triton
    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    # 如果没有找到可用的特殊设备，返回 'cpu'
    return 'cpu'


import torch
from packaging import version

if version.parse(torch.__version__) > version.parse('2.4'):
    from torch.amp import custom_fwd, custom_bwd
    def custom_fwd_wrapper(**kwargs):
        return custom_fwd(**kwargs)
    def custom_bwd_wrapper(**kwargs):
        return custom_bwd(**kwargs)

else:
    from torch.cuda.amp import custom_fwd,  custom_bwd
    def custom_fwd_wrapper(**decorator_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                all_kwargs = {**decorator_kwargs, **func_kwargs}
                all_kwargs.pop('device_type', None)
                return custom_fwd(func)(*args, **all_kwargs)
            return wrapper
        return decorator
    def custom_bwd_wrapper(**decorator_kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **func_kwargs):
                all_kwargs = {**decorator_kwargs, **func_kwargs}
                all_kwargs.pop('device_type', None)
                return custom_bwd(func)(*args, **all_kwargs)
            return wrapper
        return decorator



