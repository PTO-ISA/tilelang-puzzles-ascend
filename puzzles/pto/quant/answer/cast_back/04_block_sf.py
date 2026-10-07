"""cast_back 04 (PTO). See doc/quant/cast_back/04_block_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import status
from common.consts import BLOCK_K, BLOCK_MN

LANES = 64


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Dequantize FP8 -> bfloat16 with one scale per `block`-shaped tile."""
    bm, bk = block
    assert hidden % 128 == 0 and bm == 32 and bk == 32
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)
    sf_pad = max(num_k_blocks, 16)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((sf_pad,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for m_block in T.serial(num_m_blocks):
                # Hoisted: one scale row serves all 32 tokens below.
                T.copy(Sf[m_block, 0], sf_ub[0:num_k_blocks])
                for row in T.serial(bm):
                    token = m_block * bm + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="identical VF body to variant 01 -- the only change is in the schedule above, where sf_ub is filled once per 32 tokens"
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        groups_per_strip = LANES // bk
                        for strip in T.serial(hidden // LANES):
                            col = strip * LANES
                            group = strip * groups_per_strip
                            values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")
                            scale = V.vload(sf_ub[group], size=LANES, stride=1,
                                            dist_mode="brc", group=groups_per_strip)
                            scaled = V.vmul(values, scale, mask)
                            V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 04", out)
    return out
