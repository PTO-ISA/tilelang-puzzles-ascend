"""cast_back 04 (SIMT). See doc/quant/cast_back/04_per_channel_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN

THREADS = 128


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Dequantize FP8 -> bfloat16 with one scale per channel per token group."""
    assert hidden % 128 == 0 and group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, group_tokens)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, hidden), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)

            for m_block in T.serial(num_m_blocks):
                T.copy(Sf[m_block, 0], sf_ub)       # one scale per channel

                for row in T.serial(group_tokens):
                    T.copy(Q[m_block * group_tokens + row, 0], q_ub)

                    # --- BEGIN SOLUTION hint="the simplest scale index in the kernel: one scale per channel, so sf_ub[i] with no divisor at all. On the vector tiers this is the variant where the broadcast machinery disappears -- ASC drops its mask, one of its two BRC_B32 loads and the vsel. In SIMT there was never any broadcast to drop, so nothing changes except the index, which is itself the honest observation: SIMT cannot show you the saving because it never paid the cost."
                    with T.SimtVF(threads=THREADS):
                        q_l = T.alloc_fragment((hidden,), T.float32)
                        o_l = T.alloc_fragment((hidden,), T.float32)
                        T.copy(q_ub, q_l)

                        for i in T.Parallel(hidden):
                            o_l[i] = q_l[i] * sf_ub[i]

                        T.copy(o_l, out_ub)
                    # --- END SOLUTION

                    T.copy(out_ub, Out[m_block * group_tokens + row, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 04", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/cast_back/04")
