"""cast_back 04 (ASC). See doc/quant/cast_back/03_per_channel_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN

LANES = 64


@tilelang.jit(target="ascend", out_idx=[2])
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
                # One scale row per 32 tokens, as in variant 03.
                T.copy(Sf[m_block, 0], sf_ub)
                for row in T.serial(group_tokens):
                    token = m_block * group_tokens + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="no mask and no broadcast this time: the scale for 64 consecutive channels is 64 consecutive float32 values, so scale = S.vld(sf_ub[col]) with the default contiguous dist"
                    with T.SimdVF():
                        for strip in T.serial(hidden // LANES):
                            col = strip * LANES
                            values = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.float32)
                            # The scale is already laid out one value per lane.
                            scale = S.vld(sf_ub[col])
                            scaled = S.vmul(values, scale)
                            S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16), dist="PK_B32")
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 04", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/cast_back/04")
