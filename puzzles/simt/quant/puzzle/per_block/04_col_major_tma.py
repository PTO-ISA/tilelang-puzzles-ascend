"""per_block 04 (SIMT). See doc/quant/per_block/04_col_major_tma.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """As variant 01, with the tile scales written transposed.

    All three vector tiers agree that this config is free, and for the same
    reason: a tile scale is a single scalar, so storing it transposed is a
    swapped pair of indices on the final `T.copy` and nothing else. The VF body
    is byte-identical to variant 01's.

    That is worth contrasting with per_token/05, which asks for the same
    transposed layout where the scales live one-per-group. There ASC has to
    build a lane-index vector and permute a block through a gather, PTO is
    nearly as involved, and SIMT writes one index -- a real difference. Here
    there is no difference to have, because there are no lanes in play.
    """
    bm, bk = block
    assert bm == 32 and bk == 32, "Ascend per_block granularity is 32x32"
    assert hidden % bk == 0
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        # the whole config: (num_k_blocks, num_m_blocks), not the transpose
        SfCm: T.Tensor((num_k_blocks, num_m_blocks), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((1,), T.float32)
            inv_ub = T.alloc_shared((1,), T.float32)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)

                    # TODO: paste variant 01's VF body unchanged -- the flat
                    #       reduction, the clamp, the two divisions and the apply
                    #       are all the same. That is the lesson: a tile scale is
                    #       one scalar, so a transposed layout costs nothing in
                    #       any of the three vector styles. The whole change is
                    #       outside this region: SfCm is declared (num_k_blocks,
                    #       num_m_blocks) and the final copy is T.copy(sf_ub[0:1],
                    #       SfCm[kb, mb:mb + 1]) -- a swapped pair of indices.
                    #       Contrast per_token/05, where the same config lands on
                    #       per-group scales and forces ASC into S.vci plus
                    #       S.vgather2.
                    raise NotImplementedError("simt/per_block/04_col_major_tma: implement per_block_cast")

                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    # the swapped pair of indices, and the entire variant
                    T.copy(sf_ub[0:1], SfCm[kb, mb:mb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((hidden // BLOCK_K, m // BLOCK_MN), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 04", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_block/04 --role puzzle")
