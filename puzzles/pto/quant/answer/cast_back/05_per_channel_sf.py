"""cast_back 05 (PTO). See doc/quant/cast_back/05_per_channel_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import status
from common.consts import BLOCK_MN

LANES = 64


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Dequantize FP8 -> bfloat16 with one scale per channel per token group."""
    assert hidden % LANES == 0 and group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, group_tokens)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, hidden), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for m_block in T.serial(num_m_blocks):
                # One scale row per 32 tokens, as in variant 04.
                T.copy(Sf[m_block, 0], sf_ub)
                for row in T.serial(group_tokens):
                    token = m_block * group_tokens + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="no broadcast this time: the scale for 64 consecutive channels is 64 consecutive float32 values, so scale = V.vload(sf_ub[col], size=64) with no dist_mode"
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        for strip in T.serial(hidden // LANES):
                            col = strip * LANES
                            values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")
                            # The scale is already laid out one value per lane.
                            scale = V.vload(sf_ub[col], size=LANES)
                            scaled = V.vmul(values, scale, mask)
                            V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 05", out)
    return out
