"""per_channel 02 (ASC). See doc/quant/per_channel/02_round_packed_m.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR

LANES = 64
BYTE_LANES = 128        # uint8 lanes per interleave step


@tilelang.jit(target="ascend")
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
                        qmax = S.vdup(E4M3_MAX, T.float32)
                        one = S.vdup(1, T.uint32)
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            acc = S.alloc_local((1,), T.float32)
                            acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
                            for row in T.serial(group_tokens):
                                v = S.vabs(S.vcvt(S.vld(x_ub[row, col],
                                                        dist="UNPK_B16"), T.float32))
                                acc[0] = S.vmax(acc[0], v)
                            # the exponent trick, one value per channel
                            bits = T.reinterpret(S.vmuls(acc[0], 1.0 / E4M3_MAX),
                                                 "uint32x64")
                            biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                            S.vsts(sf_rows_ub[sub, col],
                                   T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                            S.vsts(inv_ub[col], T.reinterpret(
                                S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                                "float32x64"))
                        S.mem_bar("VST_VLD")

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

                # TODO: the two scale-byte rows of this pair interleave into one
                #       packed row. For each 128-channel tile: a =
                #       S.vld(sf_rows_ub[0, col], dist='NORM_B8'), b likewise from
                #       row 1, then lo, _ = S.vintlv(a, b) and store lo at
                #       packed_ub[col*2] with dist='NORM_B8'. A uint8 register is
                #       256 lanes, so lo alone covers these 128 channels' 256
                #       output bytes; the rows are padded so the oversized load
                #       does not run into the next row.
                raise NotImplementedError("asc/per_channel/02_round_packed_m: implement per_channel_cast")
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
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_channel/02 --role puzzle")
