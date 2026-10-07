"""per_token 05 (ASC). See doc/quant/per_token/05_col_major_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128
SF_STRIDE = 64          # padded row length of the token-major scale buffer


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Quantize, writing the scales transposed as (num_groups, num_tokens)."""
    assert hidden % PAIR == 0 and group_size == 32 and token_block == 32
    num_groups = hidden // group_size
    log2_block = token_block.bit_length() - 1          # 5
    num_out_values = num_groups * token_block
    groups_per_gather = LANES // token_block           # 2
    assert num_out_values % LANES == 0, "this teaching kernel wants whole gathers"
    num_gathers = num_out_values // LANES
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        SfCm: T.Tensor((num_groups, num_tokens), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_STRIDE,), T.float32)
            inv_ub = T.alloc_shared((SF_STRIDE,), T.float32)
            # token-major scales for a block of tokens, then their transpose
            sf_dense_ub = T.alloc_shared((token_block, SF_STRIDE), T.float32)
            sf_out_ub = T.alloc_shared((num_groups, token_block), T.float32)
            idx_ub = T.alloc_shared((LANES,), T.uint32)

            # --- BEGIN SOLUTION hint="build the gather index vector once with S.vci: token = lane & (token_block-1), group = lane >> log2(token_block), idx = token*SF_STRIDE + group. Then per token compute scales into sf_dense_ub[row, :] as usual, and after the block transpose with S.vgather2(sf_dense_ub[0, base], idx) -> S.vsts(sf_out_ub[base, 0], ...)"
            # One lane-index vector serves every block; compute it once.
            with T.SimdVF():
                lane = S.vci(0, T.int32)
                token_of_lane = S.vand(T.reinterpret(lane, "uint32x64"),
                                       S.vdup(token_block - 1, T.uint32))
                group_of_lane = T.reinterpret(S.vshrs(lane, log2_block), "uint32x64")
                S.vsts(idx_ub[0],
                       S.vadd(S.vmuls(token_of_lane, SF_STRIDE), group_of_lane),
                       dist="NORM_B32")

            for blk in T.serial(num_blocks):
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(X[token, 0], x_ub)
                    with T.SimdVF():
                        mask_low = S.pset(32, "PAT_VL32")
                        mask_all = S.pset(32, "PAT_ALL")
                        mask_high = S.pnot(mask_low, mask_all)
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * (PAIR // group_size)
                            a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"),
                                               T.float32))
                            a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                               T.float32))
                            S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
                            S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
                            S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low), dist="ONEPT_B32")
                            S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                        one = S.vdup(1, T.uint32)
                        bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
                        biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                        # Scales land token-major; they are transposed after the block.
                        S.vsts(sf_dense_ub[row, 0],
                               T.reinterpret(S.vshls(biased, 23), "float32x64"))
                        S.vsts(inv_ub[0], T.reinterpret(
                            S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                            "float32x64"))
                        S.mem_bar("VST_VLD")

                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * (PAIR // group_size)
                            i0 = S.vld(inv_ub[group], dist="BRC_B32")
                            i1 = S.vld(inv_ub[group + 1], dist="BRC_B32")
                            i2 = S.vld(inv_ub[group + 2], dist="BRC_B32")
                            i3 = S.vld(inv_ub[group + 3], dist="BRC_B32")
                            x0 = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
                            x1 = S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                        T.float32)
                            S.vsts(q_ub[col],
                                   S.vcvt(S.vmul(x0, S.vsel(i0, i1, mask_low)),
                                          T.float8_e4m3fn), dist="PK4_B32")
                            S.vsts(q_ub[col + LANES],
                                   S.vcvt(S.vmul(x1, S.vsel(i2, i3, mask_low)),
                                          T.float8_e4m3fn), dist="PK4_B32")
                    T.copy(q_ub, Q[token, 0])

                # transpose the block's scales in register, then one DMA out
                with T.SimdVF():
                    S.mem_bar("VST_VLD")
                    idx = S.vld(idx_ub[0])
                    for g in T.serial(num_gathers):
                        base = g * groups_per_gather
                        S.vsts(sf_out_ub[base, 0],
                               S.vgather2(sf_dense_ub[0, base], idx), dist="NORM_B32")
                T.copy(sf_out_ub, SfCm[0, blk * token_block])
            # --- END SOLUTION

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_cm = torch.empty((num_groups, m), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf_cm)
    status.assert_on_device("per_token 05", q, sf_cm)
    return q, sf_cm
