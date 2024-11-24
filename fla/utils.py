# -*- coding: utf-8 -*-

import functools

import torch
import triton
import os
from functools import lru_cache
from packaging import version


def contiguous(fn):
    """
    Make sure all input tensors are contiguous.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def require_version(version, hint):
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.
    """
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


@lru_cache(maxsize=None)
def get_available_device():
    if torch.cuda.is_available():
        return 'cuda'

    try:
        import intel_extension_for_pytorch as ipex
    except ImportError:
        pass
    if torch.xpu.is_available():
        return 'xpu'

    try:
        import torch_musa
        if torch.musa.is_available():
            return 'musa'
    except ImportError:
        pass

    try:
        import torch_npu
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    return 'cpu'


@lru_cache(maxsize=None)
def check_compute_capacity():
    try:
        max_shared_memory = triton.runtime.driver.active.utils.get_device_properties(0)['max_shared_mem']
        if max_shared_memory < 102400:
            return False
        else:
            return True
    except BaseException as e:
        return False


@lru_cache(maxsize=None)
def check_pytorch_version(version_s: str):
    current_parts = torch.__version__.split('.')[:2]  # 只取主版本号和次版本号
    required_parts = version_s.split('.')
    current = float(f"{current_parts[0]}.{current_parts[1]}")
    required = float(f"{required_parts[0]}.{required_parts[1]}")
    return current >= required


device = 'cuda' if get_available_device() == 'cpu' else get_available_device()
device_capacity = check_compute_capacity()


if check_pytorch_version('2.4'):
    from torch.amp import custom_fwd, custom_bwd

    def autocast_custom_fwd(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return custom_fwd(device_type=device)(args[0])
        kwargs.setdefault('device_type', device)
        return custom_fwd(**kwargs)

    def autocast_custom_bwd(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return custom_bwd(device_type=device)(args[0])
        kwargs.setdefault('device_type', device)
        return custom_bwd(**kwargs)

else:
    autocast_custom_fwd = getattr(torch, f"{device.split(':')[0]}").amp.custom_fwd
    autocast_custom_bwd = getattr(torch, f"{device.split(':')[0]}").amp.custom_bwd


@lru_cache(maxsize=None)
def detect_tf32():
    env_tf32 = os.environ.get('USE_TF32', 'true').lower()

    if env_tf32 in ('1', 'true', 'yes', 'on'):
        return True
    elif env_tf32 in ('0', 'false', 'no', 'off'):
        return False

    return False
