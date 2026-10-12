"""per_token 01 (SIMT). See doc/quant/per_token/01_raw_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


# No out_idx, for the same reason as the ASC and PTO tiers: tilelang's automatic
# output allocation rejects float8_e4m3fn on this toolchain, so launch() allocates
# the outputs. See harness/probe/fp8_out_idx.py.
@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per 32 channels."""
    assert hidden % group_size == 0
    assert group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            # Flat, like the ASC and PTO tiers. A 2-D UB buffer here would be
            # wrong: `T.copy(X[token, 0], x_ub)` into a (num_groups, group_size)
            # buffer copies the 2-D REGION X[token:token+num_groups, 0:group_size],
            # not the token's 128 contiguous channels. The group structure belongs
            # in the index arithmetic instead, which shared buffers allow freely.
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # --- BEGIN SOLUTION hint="open `with T.SimtVF(threads=128):` and allocate float32 fragments shaped (num_groups, group_size) -- making the group the fast axis is what turns the quant group into a reduction axis. T.copy widens the bfloat16 input. Reduce with one thread per group -- T.Parallel(num_groups) with an inner T.serial(group_size) accumulating into amax_l[g], seeded at E4M3_CLAMP_MIN so the clamp is free -- and form sf and its reciprocal in the same body. T.reduce_max(dim=1) on a (num_groups, group_size) fragment also works here -- harness/probe/simt_limits.py confirms it -- but the explicit scan is what the rest of the tier is written against: it lets the accumulator be SEEDED with the clamp instead of clamping afterwards, and it is the same spelling the later variants need once the scale math grows. Then one T.Parallel(num_groups, group_size) applying inv_l[g] to x_l[g, j]; indexing a (num_groups,) fragment by [g] inside a 2-D Parallel is fine, since the one-pattern rule is per buffer. T.copy the fragments out to shared."
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((hidden,), T.float32)
                    amax_l = T.alloc_fragment((num_groups,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    # Pass 1: one thread per group, scanning its own 32 channels
                    # straight out of UB. Seeding the accumulator with the clamp
                    # makes the clamp free.
                    for g in T.Parallel(num_groups):
                        amax_l[g] = E4M3_CLAMP_MIN
                        for j in T.serial(group_size):
                            v = T.Cast("float32", x_ub[g * group_size + j])
                            amax_l[g] = T.max(amax_l[g], T.abs(v))
                        inv_l[g] = E4M3_MAX / amax_l[g]
                        sf_l[g] = amax_l[g] / E4M3_MAX

                    # Pass 2: apply. `i // group_size` picks the group's scale,
                    # which is the whole of ASC's broadcast-and-select.
                    for i in T.Parallel(hidden):
                        q_l[i] = T.Cast("float32", x_ub[i]) * inv_l[i // group_size]

                    T.copy(q_l, q_ub)               # float32 -> FP8, saturating
                    T.copy(sf_l, sf_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 01", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/01")
