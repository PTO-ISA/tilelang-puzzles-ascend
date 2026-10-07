"""cast_back 03 (ASC). See doc/quant/cast_back/03_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import status
from common.consts import CANONICAL_G, PACK_FACTOR

LANES = 64
EXP_MASK = 0x7F800000       # the float32 exponent field


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with packed-UE8M0 scales."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR       # two exponents per int16
    num_tokens = T.dynamic("num_tokens")
    sf_pad = max(num_words, 16)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_words), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((sf_pad,), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_words])
                # --- BEGIN SOLUTION hint="build sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low) and exp_mask = S.vdup(0x7F800000, T.uint32); per strip, broadcast the word with S.vld(sf_ub[strip], dist='BRC_B16'), T.reinterpret it to 'uint32x64', S.vshl by sf_shift, S.vand with exp_mask, reinterpret to 'float32x64'; then multiply and store as in variant 01"
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    # Lanes 0-31 extract the low byte (shift 23); lanes 32-63
                    # extract the high byte (shift 15). One vector, built once.
                    sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low)
                    exp_mask = S.vdup(EXP_MASK, T.uint32)
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        values = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.float32)

                        # One word covers both groups this strip spans.
                        packed = S.vld(sf_ub[strip], dist="BRC_B16")
                        bits = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"),
                                             sf_shift), exp_mask)
                        # An exponent field with a zero mantissa *is* 2^(e-127).
                        scale = T.reinterpret(bits, "float32x64")

                        scaled = S.vmul(values, scale)
                        S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16), dist="PK_B32")
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    """`sf_packed` is int16; the kernel reads it as uint16 for the shifts."""
    out = compile_kernel(q.shape[1])(q, sf_packed.view(torch.uint16))
    status.assert_on_device("cast_back 03", out)
    return out
