# -*- coding: utf-8 -*-

from fla.utils import device_platform


def fp32_to_bf16_asm() -> str:
    ASM_DICT = {
        'nvidia': 'cvt.rna.tf32.f32 $0, $1;'
    }
    if device_platform in ASM_DICT:
        return ASM_DICT[device_platform]
    else:
        raise ValueError(f"Unsupported device platform: {device_platform}. Available platforms are: {list(ASM_DICT.keys())}")
