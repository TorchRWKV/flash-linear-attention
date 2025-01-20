# -*- coding: utf-8 -*-

try:
    import triton
except ImportError:
    raise ImportError(
        """Please install triton, you can install it with `pip install triton`
Or you can install if with `pip install rwkv-fla[cuda] --upgrade`, `pip install rwkv-fla[xpu] --upgrade`,
 `pip install rwkv-fla[rocm]--upgrade`
For more information, please visit your Graphics Card's official website."""
    )

from fla.layers import (ABCAttention, Attention, BasedLinearAttention,
                        BitAttention, DeltaNet, GatedDeltaNet,
                        GatedLinearAttention, GatedSlotAttention,
                        HGRN2Attention, HGRNAttention, LinearAttention,
                        MultiScaleRetention, ReBasedLinearAttention,
                        RWKV6Attention, RWKV7Attention)
from fla.models import (ABCForCausalLM, ABCModel, BitNetForCausalLM,
                        BitNetModel, DeltaNetForCausalLM, DeltaNetModel,
                        GatedDeltaNetForCausalLM, GatedDeltaNetModel,
                        GLAForCausalLM, GLAModel, GSAForCausalLM, GSAModel,
                        HGRN2ForCausalLM, HGRN2Model, HGRNForCausalLM,
                        LinearAttentionForCausalLM, LinearAttentionModel,
                        RetNetForCausalLM, RetNetModel, RWKV6ForCausalLM,
                        RWKV6Model, RWKV7ForCausalLM, RWKV7Model,
                        TransformerForCausalLM, TransformerModel)

__all__ = [
    'ABCAttention',
    'Attention',
    'BasedLinearAttention',
    'BitAttention',
    'DeltaNet',
    'GatedDeltaNet',
    'HGRNAttention',
    'HGRN2Attention',
    'GatedLinearAttention',
    'GatedSlotAttention',
    'LinearAttention',
    'MultiScaleRetention',
    'ReBasedLinearAttention',
    'RWKV6Attention',
    'RWKV7Attention',
    'ABCForCausalLM',
    'ABCModel',
    'BitNetForCausalLM',
    'BitNetModel',
    'DeltaNetForCausalLM',
    'DeltaNetModel',
    'GatedDeltaNetForCausalLM',
    'GatedDeltaNetModel',
    'HGRNForCausalLM',
    'HGRNModel',
    'HGRN2ForCausalLM',
    'HGRN2Model',
    'GLAForCausalLM',
    'GLAModel',
    'GSAForCausalLM',
    'GSAModel',
    'LinearAttentionForCausalLM',
    'LinearAttentionModel',
    'RetNetForCausalLM',
    'RetNetModel',
    'RWKV6ForCausalLM',
    'RWKV6Model',
    'RWKV7ForCausalLM',
    'RWKV7Model',
    'TransformerForCausalLM',
    'TransformerModel'
]

__version__ = '0.7'
