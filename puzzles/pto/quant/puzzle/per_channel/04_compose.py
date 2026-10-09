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
                    # TODO: stage 1: reduce in bfloat16. Per 128-channel tile keep
                    #       acc = V.alloc_local((1,), V.vreg(128, T.uint16))
                    #       seeded to 1, and do acc[0] = V.vmax(acc[0],
                    #       V.vand(V.vinterpret_cast(V.vload(x_ub[row, col],
                    #       size=128), 'uint16'), abs_mask), bmask) over the 32
                    #       rows, then V.vstore(V.vinterpret_cast(acc[0],
                    #       'bfloat16'), amax_bf16_ub[col]). Barrier. stage 2:
                    #       reload with V.vcvt(V.vload(..., size=64), 'float32')
                    #       and run the exponent trick, writing the byte with
                    #       V.vcvt(biased, 'uint8'). Barrier, apply. Finally
                    #       interleave the two byte rows as in variant 02.
                    raise NotImplementedError("pto/per_channel/04_compose: implement per_channel_cast")
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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_channel/04 --role puzzle")
