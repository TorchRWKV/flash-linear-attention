# -*- coding: utf-8 -*-

try:
    import triton
except ImportError:
    raise ImportError(
        """Please install triton, you can install it with `pip install triton`
Or you can install if with `pip install rwkv-fla[cuda]`, `pip install rwkv-fla[xpu]`, `pip install rwkv-fla[rocm]`
For more information, please visit your Graphics Card's official website."""
    )

from fla.layers import (ABCAttention, Attention, BasedLinearAttention,
                        DeltaNet, GatedLinearAttention, HGRN2Attention,
                        LinearAttention, MultiScaleRetention,
                        ReBasedLinearAttention)
from fla.models import (ABCForCausalLM, ABCModel, DeltaNetForCausalLM,
                        DeltaNetModel, GLAForCausalLM, GLAModel,
                        HGRN2ForCausalLM, HGRN2Model, HGRNForCausalLM,
                        HGRNModel, LinearAttentionForCausalLM,
                        LinearAttentionModel, RetNetForCausalLM, RetNetModel,
                        RWKV6ForCausalLM, RWKV6Model, TransformerForCausalLM,
                        TransformerModel)
from fla.ops import (chunk_gla, chunk_retention, fused_chunk_based,
                     fused_chunk_gla, fused_chunk_retention)

__all__ = [
    'ABCAttention',
    'Attention',
    'BasedLinearAttention',
    'DeltaNet',
    'HGRN2Attention',
    'GatedLinearAttention',
    'LinearAttention',
    'MultiScaleRetention',
    'ReBasedLinearAttention',
    'ABCForCausalLM',
    'ABCModel',
    'DeltaNetForCausalLM',
    'DeltaNetModel',
    'HGRNForCausalLM',
    'HGRNModel',
    'HGRN2ForCausalLM',
    'HGRN2Model',
    'GLAForCausalLM',
    'GLAModel',
    'LinearAttentionForCausalLM',
    'LinearAttentionModel',
    'RetNetForCausalLM',
    'RetNetModel',
    'RWKV6ForCausalLM',
    'RWKV6Model',
    'TransformerForCausalLM',
    'TransformerModel',
    'chunk_gla',
    'chunk_retention',
    'fused_chunk_based',
    'fused_chunk_gla',
    'fused_chunk_retention'
]

__version__ = '0.1'
