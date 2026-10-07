"""cast_back 07 (ASC). See doc/quant/cast_back/07_col_major_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR

LANES = 64
FP4_STRIP = 128
EXP_MASK = 0x7F800000


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Packed UE8M0 + column-major scales + FP4 values -> bfloat16."""
    assert hidden % FP4_STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        SfCm: T.Tensor((num_words, num_tokens), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            # The scale tile stays transposed in UB, exactly as it is in GM.
            sf_ub = T.alloc_shared((num_words, token_block), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for blk in T.serial(num_blocks):
                T.copy(SfCm[0, blk * token_block], sf_ub)
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="combine variants 03 and 06: per 128-value FP4 strip, vcvt to bfloat16 and S.vintlv(zero, x) to get two float32x64 halves; for each half broadcast its packed word with S.vld(sf_ub[word, row], dist='BRC_B16') -- note the transposed index -- then shift/mask as in variant 03 to build the scale; multiply and store both halves"
                    with T.SimdVF():
                        mask_low = S.pset(32, "PAT_VL32")
                        sf_shift = S.vsel(S.vdup(23, T.int32),
                                          S.vdup(15, T.int32), mask_low)
                        exp_mask = S.vdup(EXP_MASK, T.uint32)
                        zero_bf16 = S.vdup(0.0, T.bfloat16)
                        for strip in T.serial(hidden // FP4_STRIP):
                            col = strip * FP4_STRIP
                            # 128 FP4 -> 128 bfloat16 -> two float32x64 halves.
                            x_bf16 = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"),
                                            T.bfloat16)
                            x_lo, x_hi = S.vintlv(zero_bf16, x_bf16)

                            # Each 64-lane half spans two groups == one packed
                            # word. Transposed index: [word, token], not [token, word].
                            # Plain Python loop, not T.unroll: `half` must be a
                            # real int so that picking x_lo vs x_hi happens at
                            # trace time. Inside T.unroll it is a symbolic var,
                            # `half == 0` is a PrimExpr rather than a bool, and
                            # the branch silently always takes the first arm.
                            for half, values_bf16 in enumerate((x_lo, x_hi)):
                                word = strip * 2 + half
                                packed = S.vld(sf_ub[word, row], dist="BRC_B16")
                                bits = S.vand(
                                    S.vshl(T.reinterpret(packed, "uint32x64"),
                                           sf_shift), exp_mask)
                                scale = T.reinterpret(bits, "float32x64")
                                values = T.reinterpret(values_bf16, "float32x64")
                                S.vsts(out_ub[col + half * LANES],
                                       S.vcvt(S.vmul(values, scale), T.bfloat16),
                                       dist="PK_B32")
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf_cm: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf_cm.view(torch.uint16))
    status.assert_on_device("cast_back 07", out)
    return out
