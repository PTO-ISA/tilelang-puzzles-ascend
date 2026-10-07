"""cast_back 03 (PTO). See doc/quant/cast_back/03_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import status
from common.consts import CANONICAL_G, PACK_FACTOR

LANES = 64
EXP_MASK = 0x7F800000


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with packed-UE8M0 scales."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
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
                # --- BEGIN SOLUTION hint="mask_low = V.create_mask(32, size=64); sf_shift = V.vsel(mask_low, V.vbrc(T.uint32(23), size=64), V.vbrc(T.uint32(15), size=64)) -- note the mask comes FIRST in V.vsel; per strip broadcast the word with V.vload(sf_ub[strip], size=128, stride=1, dist_mode='brc', group=1), V.vinterpret_cast to 'uint32', V.vshl by sf_shift, V.vand with the exponent mask, then vinterpret_cast to 'float32'"
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    mask_low = V.create_mask(32, size=LANES)
                    # VMI puts the mask first: vsel(mask, if_true, if_false).
                    sf_shift = V.vsel(mask_low,
                                      V.vbrc(T.uint32(23), size=LANES),
                                      V.vbrc(T.uint32(15), size=LANES))
                    exp_mask = V.vbrc(T.uint32(EXP_MASK), size=LANES)
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")

                        # 128 uint16 lanes = 64 uint32 lanes after the cast; the
                        # lane count is derived, not spelled.
                        packed = V.vload(sf_ub[strip], size=LANES * 2, stride=1,
                                         dist_mode="brc", group=1)
                        bits = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"),
                                             sf_shift), exp_mask)
                        scale = V.vinterpret_cast(bits, "float32")

                        scaled = V.vmul(values, scale, mask)
                        V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf_packed.view(torch.uint16))
    status.assert_on_device("cast_back 03", out)
    return out
