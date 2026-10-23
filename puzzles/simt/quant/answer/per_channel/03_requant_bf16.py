"""per_channel 03 (SIMT). See doc/quant/per_channel/03_requant_bf16.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN,
                   in_group: int = CANONICAL_G):
    """Per-token-quantized input -> per-channel-quantized output.

    The two scale layouts pull in opposite directions, and that is the lesson:
    the INPUT scale varies along K (one per group of 32 channels) while the
    OUTPUT scale varies along M (one per channel). On a vector unit those are
    two different problems -- the input scale needs a broadcast load
    (`dist_mode='brc'` with `group=`, or two loads and a select on ASC), the
    output scale needs none at all. In SIMT both are a subscript: the input
    scale is `c // in_group`, the output scale is `c`.
    """
    assert hidden % 128 == 0 and group_tokens == 32 and in_group == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_in_groups = hidden // in_group

    @T.prim_func
    def per_channel_requant(
        Q_in: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf_in: T.Tensor((num_tokens, num_in_groups), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_groups, hidden), T.float32),
    ):
        with T.Kernel(1):
            q_in_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            # The 2-D region is exactly right here: 32 tokens by num_in_groups
            # scales. (Contrast per_token/01, where a 2-D UB buffer would have
            # silently read several tokens' worth of one token's channels.)
            sf_in_ub = T.alloc_shared((group_tokens, num_in_groups), T.float32)
            val_ub = T.alloc_shared((group_tokens, hidden), T.float32)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            inv_ub = T.alloc_shared((hidden,), T.float32)

            for mg in T.serial(num_groups):
                T.copy(Q_in[mg * group_tokens, 0], q_in_ub)
                T.copy(Sf_in[mg * group_tokens, 0], sf_in_ub)

                # --- BEGIN SOLUTION hint="three stages. (1) dequantize into val_ub, a float32 scratch in shared: T.Parallel(group_tokens, hidden) with val_ub[row, c] = T.Cast('float32', q_in_ub[row, c]) * sf_in_ub[row, c // in_group]. That `c // in_group` is the whole of the input-scale broadcast -- PTO needs a dist_mode='brc' load with group=4, ASC two loads and a select. (2) variant 01's reduction, but reading val_ub: one thread per channel, down the 32 rows, and the output scale is indexed by channel alone. (3) apply. Keep the dequantized values in SHARED, not a fragment: stages 1 and 2 touch them with different index patterns, and only shared buffers allow that."
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((group_tokens, hidden), T.float32)

                    # stage 1: dequantize. The input scale varies along K.
                    for row, c in T.Parallel(group_tokens, hidden):
                        val_ub[row, c] = (T.Cast("float32", q_in_ub[row, c])
                                          * sf_in_ub[row, c // in_group])

                    # stage 2: reduce along M. The output scale varies along M,
                    # so the index is the channel and there is no broadcast.
                    for c in T.Parallel(hidden):
                        amax = T.alloc_var(T.float32)
                        amax = E4M3_CLAMP_MIN
                        for row in T.serial(group_tokens):
                            amax = T.max(amax, T.abs(val_ub[row, c]))
                        sf_ub[c] = amax / E4M3_MAX
                        inv_ub[c] = E4M3_MAX / amax

                    # stage 3: apply
                    for row, c in T.Parallel(group_tokens, hidden):
                        q_l[row, c] = val_ub[row, c] * inv_ub[c]

                    T.copy(q_l, q_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[mg * group_tokens, 0])
                T.copy(sf_ub, Sf[mg, 0])

    return per_channel_requant


def launch(q_in: torch.Tensor, sf_in: torch.Tensor):
    m, hidden = q_in.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=q_in.device)
    sf = torch.empty((m // BLOCK_MN, hidden), dtype=torch.float32, device=q_in.device)
    compile_kernel(hidden)(q_in, sf_in, q, sf)
    status.assert_on_device("per_channel 03", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_channel/03")
