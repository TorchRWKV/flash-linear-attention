import os
import triton.language as tl
import triton.language.extra.libdevice as tldevice

fla_use_fast_ops = os.environ.get('FLA_USE_FAST_OPS', '0')
if fla_use_fast_ops == '1':
    exp = tldevice.fast_expf
else:
    exp = tl.exp
