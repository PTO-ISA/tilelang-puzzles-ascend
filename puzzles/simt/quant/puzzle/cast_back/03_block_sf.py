"""cast_back 03 (SIMT). See doc/quant/cast_back/03_block_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN

THREADS = 128


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Dequantize FP8 -> bfloat16 with one scale per `block`-shaped tile."""
    bm, bk = block
    assert hidden % 128 == 0 and bm == 32 and bk == 32
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_k_blocks,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for m_block in T.serial(num_m_blocks):
                # One scale row serves all bm tokens of the block: the DMA is
                # hoisted out of the token loop, which is this variant's point.
                T.copy(Sf[m_block, 0], sf_ub)

                for row in T.serial(bm):
                    T.copy(Q[m_block * bm + row, 0], q_ub)

                    # TODO: identical to variant 01 except the index that picks
                    #       the scale: sf_ub[i // bk] instead of sf_ub[i //
                    #       group_size]. The scale row is loaded once per block of
                    #       bm tokens by the T.copy above, so the token loop never
                    #       touches it. In SIMT the coarser scale axis really is
                    #       only a different divisor -- compare the ASC file,
                    #       where it is the same vector body but a different loop
                    #       nest.
                    raise NotImplementedError("simt/cast_back/03_block_sf: implement cast_back")

                    T.copy(out_ub, Out[m_block * bm + row, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 03", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/03 --role puzzle")
