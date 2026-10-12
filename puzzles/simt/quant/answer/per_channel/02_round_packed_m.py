"""per_channel 02 (SIMT). See doc/quant/per_channel/02_round_packed_m.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR

THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Power-of-two scales packed along M: two m-groups share an int16.

    The packing is where this variant pays off for SIMT. The output wants byte
    2c from m-group 2i and byte 2c+1 from m-group 2i+1, which on a vector unit
    is an *interleave*: ASC and PTO load both byte rows and call `V.vintlv`.
    Worse, a uint8 vector register is 256 lanes, so the load overruns the 128
    channels being packed and both SIMD tiers have to pad `sf_rows_ub` by a
    whole tile to keep it from reading into the next row.

    A thread that owns channel c just writes two bytes at 2c and 2c+1. No
    interleave instruction, and no padding -- a scalar store cannot overrun.
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

                # --- BEGIN SOLUTION hint="the two byte rows of this pair interleave into one packed row: byte 2c is m-group 2i's exponent for channel c, byte 2c+1 is m-group 2i+1's. Give each thread one channel -- T.Parallel(hidden) writing packed_ub[PACK_FACTOR*c] and packed_ub[PACK_FACTOR*c + 1] -- and the interleave disappears into the index. This is the clearest SIMT win in the ladder: the SIMD tiers need V.vintlv on a pair of 256-lane uint8 loads, AND they must pad sf_rows_ub by a whole 128-byte tile, because a 256-lane load of 128 meaningful bytes would otherwise read into the next row. Two shared stores cannot overrun, so there is no padding here."
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
    status.assert_on_device("per_channel 02", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_channel/02")
