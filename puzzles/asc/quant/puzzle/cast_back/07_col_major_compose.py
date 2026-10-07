"""cast_back 07 (ASC). See doc/quant/cast_back/07_col_major_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import status
from common.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR

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
                    # TODO: combine variants 03 and 06: per 128-value FP4 strip,
                    #       vcvt to bfloat16 and S.vintlv(zero, x) to get two
                    #       float32x64 halves; for each half broadcast its packed
                    #       word with S.vld(sf_ub[word, row], dist='BRC_B16') --
                    #       note the transposed index -- then shift/mask as in
                    #       variant 03 to build the scale; multiply and store both
                    #       halves
                    raise NotImplementedError("asc/cast_back/07_col_major_compose: implement cast_back")
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf_cm: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf_cm.view(torch.uint16))
    status.assert_on_device("cast_back 07", out)
    return out
