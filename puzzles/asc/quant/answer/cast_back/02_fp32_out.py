"""cast_back 02 (ASC). See doc/quant/cast_back/02_fp32_out.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G

LANES = 64
SF_PAD = 64


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> float32."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.float32),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.float32)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])
                # --- BEGIN SOLUTION hint="same as variant 01, except the store is S.vsts(out_ub[col], scaled, dist='NORM_B32') with no vcvt -- the lanes are already float32"
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        group = strip * (LANES // group_size)
                        values = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.float32)
                        lo = S.vld(sf_ub[group], dist="BRC_B32")
                        hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
                        scaled = S.vmul(values, S.vsel(lo, hi, mask_low))
                        # No convert, no packing: a plain contiguous 32-bit store.
                        S.vsts(out_ub[col], scaled, dist="NORM_B32")
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 02", out)
    return out
