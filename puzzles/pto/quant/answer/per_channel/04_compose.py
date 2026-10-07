"""per_channel 04 (PTO). See doc/quant/per_channel/04_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR

LANES = 64              # float32 lanes, for the scale math
BF16_LANES = 128        # bfloat16 lanes, for the reduction
BYTE_LANES = 128        # channels per interleave step


@tilelang.jit(target="pto")
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
                    # --- BEGIN SOLUTION hint="stage 1: reduce in bfloat16. Per 128-channel tile keep acc = V.alloc_local((1,), V.vreg(128, T.uint16)) seeded to 1, and do acc[0] = V.vmax(acc[0], V.vand(V.vinterpret_cast(V.vload(x_ub[row, col], size=128), 'uint16'), abs_mask), bmask) over the 32 rows, then V.vstore(V.vinterpret_cast(acc[0], 'bfloat16'), amax_bf16_ub[col]). Barrier. stage 2: reload with V.vcvt(V.vload(..., size=64), 'float32') and run the exponent trick, writing the byte with V.vcvt(biased, 'uint8'). Barrier, apply. Finally interleave the two byte rows as in variant 02."
                    with T.SimdVF():
                        bmask = V.create_mask(BF16_LANES, size=BF16_LANES)
                        fmask = V.create_mask(LANES, size=LANES)
                        abs_mask = V.vbrc(T.uint16(0x7FFF), size=BF16_LANES)
                        one = V.vbrc(T.uint32(1), size=LANES)
                        shift = V.vbrc(T.uint32(23), size=LANES)
                        b254 = V.vbrc(T.uint32(254), size=LANES)

                        # ---- stage 1: reduce in bfloat16, on the integer unit ----
                        for bt in T.serial(num_bf16_tiles):
                            col = bt * BF16_LANES
                            acc = V.alloc_local((1,), V.vreg(BF16_LANES, T.uint16))
                            acc[0] = V.vbrc(T.uint16(0x0001), size=BF16_LANES)
                            for row in T.serial(group_tokens):
                                bits = V.vinterpret_cast(
                                    V.vload(x_ub[row, col], size=BF16_LANES), "uint16")
                                # clearing the sign bit is abs, and for
                                # non-negative floats the integer order matches
                                acc[0] = V.vmax(acc[0], V.vand(bits, abs_mask), bmask)
                            V.vstore(V.vinterpret_cast(acc[0], "bfloat16"),
                                     amax_bf16_ub[col])
                        T.simd.mem_bar("VST_VLD")

                        # ---- stage 2: scale math in float32, as always ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            amax = V.vcvt(V.vload(amax_bf16_ub[col], size=LANES),
                                          "float32")
                            clamped = V.vmax(
                                amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES),
                                fmask)
                            bits = V.vinterpret_cast(
                                V.vmul(clamped,
                                       V.vbrc(T.float32(1.0 / E4M3_MAX), size=LANES),
                                       fmask), "uint32")
                            biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                            V.vstore(V.vcvt(biased, "uint8"), sf_rows_ub[sub, col])
                            V.vstore(V.vinterpret_cast(
                                V.vshl(V.vsub(b254, biased), shift), "float32"),
                                inv_ub[col])
                        T.simd.mem_bar("VST_VLD")

                        # ---- stage 3: apply; the scale is one value per lane ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            inv = V.vload(inv_ub[col], size=LANES)
                            for row in T.serial(group_tokens):
                                v = V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                           "float32")
                                V.vstore(V.vcvt(V.vmul(v, inv, fmask),
                                                "float8_e4m3fn", rounding="R",
                                                saturate="SAT"), q_ub[row, col])
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                # pack the pair's two byte rows along M (variant 02)
                with T.SimdVF():
                    T.simd.mem_bar("VST_VLD")
                    byte_mask = V.create_mask(BYTE_LANES * 2, size=BYTE_LANES * 2)
                    for bt in T.serial(num_byte_tiles):
                        col = bt * BYTE_LANES
                        a = V.vload(sf_rows_ub[0, col], size=BYTE_LANES * 2)
                        b = V.vload(sf_rows_ub[1, col], size=BYTE_LANES * 2)
                        lo, _ = V.vintlv(a, b, byte_mask)
                        V.vstore(lo, packed_ub[col * PACK_FACTOR])
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
