"""per_token 01 (PTO). See doc/quant/per_token/01_raw_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128
SF_PAD = 64


# No out_idx: tilelang's automatic output allocation rejects float8_e4m3fn with
# "MemoryError: Unsupported code 10" on this toolchain (see
# harness/probe/fp8_out_idx.py). launch() allocates the outputs instead.
@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_pair = PAIR // group_size        # 4
    assert num_groups <= SF_PAD
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # TODO: three passes. (1) per 128 channels: x =
                #       V.vcvt(V.vload(x_ub[col], size=128), 'float32'), then
                #       V.vstore(V.vcmax(V.vabs(x, mask), mask, group=4),
                #       amax_ub[group]) -- one segmented reduce, no masks per
                #       group. (2) T.simd.mem_bar('VST_VLD'); load 64 amax at
                #       once, V.vmax against the clamp, then V.vdiv both ways and
                #       store sf/inv. (3) mem_bar again; inv =
                #       V.vload(inv_ub[group], size=128, stride=1,
                #       dist_mode='brc', group=4), multiply, and
                #       V.vstore(V.vcvt(q, 'float8_e4m3fn', rounding='R',
                #       saturate='SAT'), q_ub[col])
                raise NotImplementedError("pto/per_token/01_raw_fp32sf: implement per_token_cast")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 01", q, sf)
    return q, sf
