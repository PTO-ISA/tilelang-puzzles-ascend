"""per_channel 04 (ASC). See doc/quant/per_channel/04_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR

LANES = 64              # float32 lanes, for the scale math
BF16_LANES = 128        # bfloat16 lanes, for the reduction
BYTE_LANES = 128        # channels per interleave step


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """bfloat16 reduction + power-of-two scale + UE8M0 packed along M."""
    assert hidden % BF16_LANES == 0 and group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_pairs = T.ceildiv(num_groups, PACK_FACTOR)
    num_bf16_tiles = hidden // BF16_LANES
    num_col_tiles = hidden // LANES
    num_byte_tiles = hidden // BYTE_LANES

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_pairs, hidden * PACK_FACTOR), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            amax_bf16_ub = T.alloc_shared((hidden,), T.bfloat16)
            inv_ub = T.alloc_shared((hidden,), T.float32)
            sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden + BYTE_LANES), T.uint8)
            packed_ub = T.alloc_shared((hidden * PACK_FACTOR,), T.uint8)

            for pair in T.serial(num_pairs):
                for sub in T.serial(PACK_FACTOR):
                    mg = pair * PACK_FACTOR + sub
                    T.copy(X[mg * group_tokens, 0], x_ub)
                    # --- BEGIN SOLUTION hint="stage 1: reduce in bfloat16. abs_mask = S.vdup(0x7FFF, T.uint16); per 128-channel tile keep acc = S.alloc_local((1,), T.uint16) and do acc[0] = S.vmax(acc[0], S.vand(T.reinterpret(S.vld(x_ub[row, col], dist='NORM_B16'), 'uint16x128'), abs_mask)) over the 32 rows, then store T.reinterpret(acc[0], 'bfloat16x128') to amax_bf16_ub. Barrier. stage 2: reload widened with UNPK_B16 + vcvt and run the usual exponent trick, writing the byte to sf_rows_ub[sub] and the inverse to inv_ub. Barrier, apply. Finally interleave the two byte rows as in variant 02."
                    with T.SimdVF():
                        abs_mask = S.vdup(0x7FFF, T.uint16)
                        clamp_u16 = S.vdup(0x0001, T.uint16)
                        one = S.vdup(1, T.uint32)

                        # ---- stage 1: reduce in bfloat16, on the integer unit ----
                        for bt in T.serial(num_bf16_tiles):
                            col = bt * BF16_LANES
                            acc = S.alloc_local((1,), T.uint16)
                            acc[0] = clamp_u16
                            for row in T.serial(group_tokens):
                                bits = T.reinterpret(
                                    S.vld(x_ub[row, col], dist="NORM_B16"),
                                    "uint16x128")
                                # clearing the sign bit is abs, and for
                                # non-negative floats the integer order matches
                                acc[0] = S.vmax(acc[0], S.vand(bits, abs_mask))
                            S.vsts(amax_bf16_ub[col],
                                   T.reinterpret(acc[0], "bfloat16x128"),
                                   dist="NORM_B16")
                        S.mem_bar("VST_VLD")

                        # ---- stage 2: scale math in float32, as always ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            amax = S.vcvt(S.vld(amax_bf16_ub[col], dist="UNPK_B16"),
                                          T.float32)
                            clamped = S.vmaxs(amax, E4M3_CLAMP_MIN)
                            bits = T.reinterpret(S.vmuls(clamped, 1.0 / E4M3_MAX),
                                                 "uint32x64")
                            biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                            S.vsts(sf_rows_ub[sub, col],
                                   T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                            S.vsts(inv_ub[col], T.reinterpret(
                                S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                                "float32x64"))
                        S.mem_bar("VST_VLD")

                        # ---- stage 3: apply; the scale is one value per lane ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            inv = S.vld(inv_ub[col])
                            for row in T.serial(group_tokens):
                                v = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                                           T.float32)
                                S.vsts(q_ub[row, col],
                                       S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                       dist="PK4_B32")
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                # pack the pair's two byte rows along M (variant 02)
                with T.SimdVF():
                    S.mem_bar("VST_VLD")
                    for bt in T.serial(num_byte_tiles):
                        col = bt * BYTE_LANES
                        a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
                        b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
                        lo, _ = S.vintlv(a, b)
                        S.vsts(packed_ub[col * PACK_FACTOR], lo, dist="NORM_B8")
                    # --- END SOLUTION
                T.copy(packed_ub, Sf[pair, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    pairs = m // (BLOCK_MN * PACK_FACTOR)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((pairs, hidden * PACK_FACTOR), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_channel 04", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)
