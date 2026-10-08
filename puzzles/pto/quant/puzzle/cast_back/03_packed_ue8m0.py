"""cast_back 03 (PTO). See doc/quant/cast_back/03_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import CANONICAL_G, PACK_FACTOR

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
                # TODO: mask_low = V.create_mask(32, size=64); sf_shift =
                #       V.vsel(mask_low, V.vbrc(T.uint32(23), size=64),
                #       V.vbrc(T.uint32(15), size=64)) -- note the mask comes
                #       FIRST in V.vsel; per strip broadcast the word with
                #       V.vload(sf_ub[strip], size=128, stride=1, dist_mode='brc',
                #       group=1), V.vinterpret_cast to 'uint32', V.vshl by
                #       sf_shift, V.vand with the exponent mask, then
                #       vinterpret_cast to 'float32'
                raise NotImplementedError("pto/cast_back/03_packed_ue8m0: implement cast_back")
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf_packed.view(torch.uint16))
    status.assert_on_device("cast_back 03", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/cast_back/03")
