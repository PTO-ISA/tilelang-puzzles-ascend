"""per_token 03 (SIMT). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize, returning each scale as a single UE8M0 exponent byte.

    The packing is free in SIMT for a reason worth noticing: the biased exponent
    is already an integer in a register, and `sf_ub` is a uint8 buffer, so
    storing it *is* the pack. ASC needs a reinterpret to `uint8x256` plus a
    matching `PK4_B32` store mode; PTO needs one `V.vcvt(..., 'uint8')`.
    """
    assert hidden % group_size == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            # uint8, so the exponent byte is the stored value.
            sf_ub = T.alloc_shared((num_groups,), T.uint8)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # --- BEGIN SOLUTION hint="variant 02 again, but keep the biased exponent instead of rebuilding a float from it. `biased` is already the UE8M0 byte value, so write it straight into a uint8 shared buffer with sf_ub[g] = T.Cast('uint8', biased) -- a store to shared, which is exempt from the one-index-pattern rule that fragments obey. There is no reinterpret and no packed store mode: the buffer's dtype is the packing. The inverse scale is still rebuilt as T.reinterpret('float32', (254 - biased) << 23)."
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

                        bits = T.reinterpret("int32", amax / E4M3_MAX)
                        biased = ((bits - 1) >> 23) + 1
                        # The byte IS the scale. No reinterpret, no pack mode.
                        sf_ub[g] = T.Cast("uint8", biased)
                        inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)

                    for i in T.Parallel(hidden):
                        q_l[i] = T.Cast("float32", x_ub[i]) * inv_l[i // group_size]

                    T.copy(q_l, q_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) where sf_packed is int16, two exponents per word."""
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m, num_groups), dtype=torch.uint8, device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_token 03", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/03")
