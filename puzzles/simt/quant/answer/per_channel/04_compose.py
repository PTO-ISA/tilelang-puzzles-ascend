"""per_channel 02 (SIMT). See doc/quant/per_channel/04_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """bfloat16 reduction + power-of-two scale + UE8M0 packed along M.

    The composition itself is free, as it is in every tier: this variant is
    variant 02's body with variant 01's reduction, and nothing new has to be
    invented. Two honest notes on where the tiers diverge.

    The bfloat16 reduction does not survive the port. The SIMD tiers earn this
    variant by reducing on the integer unit in bfloat16 -- mask off the sign bit
    and `V.vmax` the raw `uint16` patterns, which works because for
    non-negative floats the integer order matches the float order -- so one
    instruction covers 128 channels instead of 64. SIMT has no lane width to
    trade: threads are the parallelism, `T.copy` widens bfloat16 to float32 on
    the way into a fragment, and a bfloat16 cross-thread reduction ICEs. So the
    reduction below is float32. It is numerically identical anyway, and for the
    reason the harness asserts: only the *exponent* of the maximum is used, and
    bfloat16 keeps exponents exactly.

    The M-packing, on the other hand, is where SIMT wins outright -- see
    variant 02. A thread that owns channel c writes two bytes; the SIMD tiers
    need `V.vintlv` plus a padded scratch row.
    """

    assert hidden % 128 == 0 and group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_pairs = T.ceildiv(num_groups, PACK_FACTOR)

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        # one packed row per *pair* of m-groups: 2 bytes per channel
        Sf: T.Tensor((num_pairs, hidden * PACK_FACTOR), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            inv_ub = T.alloc_shared((hidden,), T.float32)
            # No padding: the pack is a scalar store, so nothing can overrun.
            sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden), T.uint8)
            packed_ub = T.alloc_shared((hidden * PACK_FACTOR,), T.uint8)

            for pair in T.serial(num_pairs):
                for sub in T.serial(PACK_FACTOR):
                    mg = pair * PACK_FACTOR + sub
                    T.copy(X[mg * group_tokens, 0], x_ub)
                    # --- BEGIN SOLUTION hint="compose variant 01's per-channel reduction with variant 02's power-of-two scale and M-packing: one thread per channel down the 32 rows, the exponent trick, the byte into sf_rows_ub[sub, c], the inverse into inv_ub[c], then the apply over T.Parallel(group_tokens, hidden). Finish with the two-bytes-per-channel pack from variant 02 -- T.Parallel(hidden) writing packed_ub[PACK_FACTOR*c] and packed_ub[PACK_FACTOR*c + 1] -- which is where SIMT wins outright, since the SIMD tiers need V.vintlv plus a padded scratch row. Read the docstring for the one thing that does NOT port: the SIMD tiers reduce in bfloat16 on the integer unit to double their lane count, and SIMT has no lane count to double, so the reduction stays float32 -- identical results, since only the exponent of the maximum is used and bfloat16 keeps exponents exactly."
                    with T.SimtVF(threads=THREADS):
                        q_l = T.alloc_fragment((group_tokens, hidden), T.float32)

                        for c in T.Parallel(hidden):
                            amax = T.alloc_var(T.float32)
                            amax = E4M3_CLAMP_MIN
                            for row in T.serial(group_tokens):
                                amax = T.max(amax,
                                             T.abs(T.Cast("float32", x_ub[row, c])))
                            bits = T.reinterpret("int32", amax / E4M3_MAX)
                            biased = ((bits - 1) >> 23) + 1
                            # this m-group's byte row; `sub` is a trace-time index
                            sf_rows_ub[sub, c] = T.Cast("uint8", biased)
                            inv_ub[c] = T.reinterpret("float32", (254 - biased) << 23)

                        for row, c in T.Parallel(group_tokens, hidden):
                            q_l[row, c] = T.Cast("float32", x_ub[row, c]) * inv_ub[c]

                        T.copy(q_l, q_ub)
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                with T.SimtVF(threads=THREADS):
                    for c in T.Parallel(hidden):
                        packed_ub[PACK_FACTOR * c] = sf_rows_ub[0, c]
                        packed_ub[PACK_FACTOR * c + 1] = sf_rows_ub[1, c]
                    # --- END SOLUTION
                T.copy(packed_ub, Sf[pair, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) with sf_packed int16 of shape (M/64, hidden)."""
    m, hidden = x.shape
    pairs = m // (BLOCK_MN * PACK_FACTOR)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((pairs, hidden * PACK_FACTOR), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_channel 04", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_channel/04")
