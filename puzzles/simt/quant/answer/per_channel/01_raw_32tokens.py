"""per_channel 01 (SIMT). See doc/quant/per_channel/01_raw_32tokens.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Quantize bfloat16 -> FP8 with one FP32 scale per channel per token group.

    Reducing along M instead of K is the one kernel where SIMD is already in its
    happy place: a channel is a lane, so the reduction is just `V.vmax` of 32
    loaded rows into an accumulator register, with no cross-lane step at all.
    SIMT's version is the mirror image -- a channel is a thread, and the
    reduction is a serial loop over 32 rows -- and the two come out the same
    length. Expect no syntactic win here.
    """
    assert hidden % 128 == 0, "channels are processed in tiles of 128"
    assert group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_groups, hidden), T.float32),
    ):
        with T.Kernel(1):
            # a whole token group lives in UB, because the reduction spans it
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            inv_ub = T.alloc_shared((hidden,), T.float32)

            for mg in T.serial(num_groups):
                T.copy(X[mg * group_tokens, 0], x_ub)

                # --- BEGIN SOLUTION hint="one thread per CHANNEL: T.Parallel(hidden), each thread walking the 32 rows of its own column with an inner T.serial and seeding the accumulator at E4M3_CLAMP_MIN. Write sf and its inverse to shared buffers indexed by channel. Then apply with T.Parallel(group_tokens, hidden), reading inv_ub[c] -- the scale index is the channel, so there is no group arithmetic anywhere. Note this variant is a draw, not a win: because a channel is a lane, the SIMD tiers need no cross-lane reduction either -- they keep a V.vreg accumulator and V.vmax 32 rows into it. Same shape, same length."
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((group_tokens, hidden), T.float32)

                    # one thread per channel, reducing down the token axis
                    for c in T.Parallel(hidden):
                        amax = T.alloc_var(T.float32)
                        amax = E4M3_CLAMP_MIN
                        for row in T.serial(group_tokens):
                            amax = T.max(amax,
                                         T.abs(T.Cast("float32", x_ub[row, c])))
                        sf_ub[c] = amax / E4M3_MAX
                        inv_ub[c] = E4M3_MAX / amax

                    # the scale index is just the channel
                    for row, c in T.Parallel(group_tokens, hidden):
                        q_l[row, c] = T.Cast("float32", x_ub[row, c]) * inv_ub[c]

                    T.copy(q_l, q_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[mg * group_tokens, 0])
                T.copy(sf_ub, Sf[mg, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m // BLOCK_MN, hidden), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_channel 01", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_channel/01")
