# -*- coding: utf-8 -*-

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import os
from functools import lru_cache


def contiguous(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous.
    """
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args),
                  **{k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()})
    return wrapper


def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

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


def checkpoint(fn):
    def wrapper(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)
    return wrapper


@lru_cache(maxsize=None)
def get_available_device():
    if torch.cuda.is_available():
        return 'cuda'

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        pass
    if torch.xpu.is_available():
        return 'xpu'

    try:
        import torch_musa  # noqa: F401
        if torch.musa.is_available():
            return 'musa'
    except ImportError:
        pass

    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return 'npu'
    except ImportError:
        pass

    return 'cpu'


@lru_cache(maxsize=None)
def check_compute_capacity(max_shared_mem=102400):
    try:
        max_shared_memory = triton.runtime.driver.active.utils.get_device_properties(0)['max_shared_mem']
        if max_shared_memory < max_shared_mem:
            return False
        else:
            return True
    except BaseException:
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
device_torch_lib = getattr(torch, device)


def set_torch_device(x: torch.Tensor):
    device_torch_lib.set_device(x.device.index)


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
