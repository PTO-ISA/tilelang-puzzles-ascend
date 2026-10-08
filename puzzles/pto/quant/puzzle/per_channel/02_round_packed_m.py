"""per_channel 02 (PTO). See doc/quant/per_channel/02_round_packed_m.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR

LANES = 128
BYTE_LANES = 128        # uint8 lanes per interleave step


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Quantize with power-of-two scales packed along M (two m-groups per int16)."""
    assert hidden % BYTE_LANES == 0, "the interleave step covers 128 channels"
    assert group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_pairs = T.ceildiv(num_groups, PACK_FACTOR)
    num_col_tiles = hidden // LANES
    num_byte_tiles = hidden // BYTE_LANES

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        # one packed row per *pair* of m-groups: 2 bytes per channel
        Sf: T.Tensor((num_pairs, hidden * PACK_FACTOR), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            inv_ub = T.alloc_shared((hidden,), T.float32)
            # one scale-byte row per member of the pair, then their interleave
            # Padded by one byte-tile: a uint8 vector load is 256 lanes, so the
            # interleave reads 256 bytes even when only 128 are meaningful. The
            # padding keeps it from reading into the next row.
            sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden + BYTE_LANES), T.uint8)
            packed_ub = T.alloc_shared((hidden * PACK_FACTOR,), T.uint8)

            for pair in T.serial(num_pairs):
                for sub in T.serial(PACK_FACTOR):
                    mg = pair * PACK_FACTOR + sub
                    T.copy(X[mg * group_tokens, 0], x_ub)
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                        one = V.vbrc(T.uint32(1), size=LANES)
                        shift = V.vbrc(T.uint32(23), size=LANES)
                        b254 = V.vbrc(T.uint32(254), size=LANES)
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
                            acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
                            for row in T.serial(group_tokens):
                                v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                                  "float32"), mask)
                                acc[0] = V.vmax(acc[0], v, mask)
                            # the exponent trick, one value per channel
                            bits = V.vinterpret_cast(
                                V.vmul(acc[0],
                                       V.vbrc(T.float32(1.0 / E4M3_MAX), size=LANES),
                                       mask), "uint32")
                            biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                            # one convert, no reinterpret + store mode
                            V.vstore(V.vcvt(biased, "uint8"), sf_rows_ub[sub, col])
                            V.vstore(V.vinterpret_cast(
                                V.vshl(V.vsub(b254, biased), shift), "float32"),
                                inv_ub[col])
                        T.simd.mem_bar("VST_VLD")

                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            inv = V.vload(inv_ub[col], size=LANES)
                            for row in T.serial(group_tokens):
                                v = V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                           "float32")
                                V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                                rounding="R", saturate="SAT"),
                                         q_ub[row, col])
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                # TODO: the two scale-byte rows of this pair interleave into one
                #       packed row. For each 128-channel tile: a =
                #       V.vload(sf_rows_ub[0, col], size=256), b likewise from row
                #       1, then lo, _ = V.vintlv(a, b, byte_mask) and V.vstore(lo,
                #       packed_ub[col*2]). A uint8 register is 256 lanes, so lo
                #       alone covers these 128 channels' 256 output bytes; the
                #       rows are padded so the oversized load does not run into
                #       the next row.
                raise NotImplementedError("pto/per_channel/02_round_packed_m: implement per_channel_cast")
                T.copy(packed_ub, Sf[pair, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) with sf_packed int16 of shape (M/64, hidden)."""
    m, hidden = x.shape
    pairs = m // (BLOCK_MN * PACK_FACTOR)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((pairs, hidden * PACK_FACTOR), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_channel 02", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_channel/02")
