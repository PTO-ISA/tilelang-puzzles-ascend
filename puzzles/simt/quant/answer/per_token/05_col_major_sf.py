"""per_token 05 (SIMT). See doc/quant/per_token/05_col_major_sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Quantize, writing the scales transposed as (num_groups, num_tokens).

    This is the variant where the thread model earns its keep. ASC cannot
    restride a vector register, so it builds a lane-index vector with `S.vci`
    and permutes a block of scales through `S.vgather2`. Here the transposed
    layout is just the index the scale is stored at -- there is no transpose
    step, and no block of scales has to be held back to be permuted.
    """
    assert hidden % group_size == 0 and group_size == 32 and token_block == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        SfCm: T.Tensor((num_groups, num_tokens), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            # Already transposed: one column per token of the current block.
            sf_out_ub = T.alloc_shared((num_groups, token_block), T.float32)

            for blk in T.serial(num_blocks):
                for sub in T.serial(token_block):
                    T.copy(X[blk * token_block + sub, 0], x_ub)

                    # --- BEGIN SOLUTION hint="as variant 02 for the maths -- one thread per channel into a (group, channel) fragment, T.reduce_absmax(dim=1), then the power-of-two exponent trick per group: bits = T.reinterpret('int32', amax/E4M3_MAX); biased = ((bits - 1) >> 23) + 1; sf = T.reinterpret('float32', biased << 23) and inv = T.reinterpret('float32', (254 - biased) << 23). The config itself is then free: store the scale at sf_out_ub[g, sub] instead of a token-major slot. There is no transpose and no gather -- compare the ASC file, which builds a lane-index vector with S.vci and permutes a whole block through S.vgather2 because a register's lanes cannot move."
                    with T.SimtVF(threads=THREADS):
                        x_l = T.alloc_fragment((num_groups, group_size), T.float32)
                        amax_l = T.alloc_fragment((num_groups,), T.float32)
                        q_l = T.alloc_fragment((hidden,), T.float32)
                        inv_l = T.alloc_fragment((num_groups,), T.float32)

                        for g, j in T.Parallel(num_groups, group_size):
                            x_l[g, j] = T.Cast("float32", x_ub[g * group_size + j])
                        T.reduce_absmax(x_l, amax_l, dim=1)

                        for g in T.Parallel(num_groups):
                            amax = T.max(amax_l[g], E4M3_CLAMP_MIN)

                            # ceil(log2(amax/448)) from the exponent field, then
                            # both scales rebuilt by writing an exponent back.
                            bits = T.reinterpret("int32", amax / E4M3_MAX)
                            biased = ((bits - 1) >> 23) + 1
                            inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
                            # The whole column-major config: a different index.
                            sf_out_ub[g, sub] = T.reinterpret("float32", biased << 23)

                        for i in T.Parallel(hidden):
                            q_l[i] = T.Cast("float32", x_ub[i]) * inv_l[i // group_size]

                        T.copy(q_l, q_ub)
                    # --- END SOLUTION

                    T.copy(q_ub, Q[blk * token_block + sub, 0])

                # One contiguous store per block: (num_groups, token_block) lands
                # in SfCm[:, blk*token_block : ...] with no permutation.
                T.copy(sf_out_ub, SfCm[0, blk * token_block])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_cm = torch.empty((num_groups, m), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf_cm)
    status.assert_on_device("per_token 05", q, sf_cm)
    return q, sf_cm

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/05")
