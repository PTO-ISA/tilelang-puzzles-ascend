"""per_token 04 (SIMT). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX

THREADS = 128


def e2m1_nibble(v):
    """float32 expression -> e2m1 nibble expression, rounded half-to-even.

    This is the torch tier's bit-exact packer transcribed onto scalars -- the
    *same* algorithm, not the SIMD tiers' round-to-odd-then-convert. Both reach
    the same codes; the difference is that the SIMD tiers have a hardware
    float32 -> e2m1 convert to finish the job, so they only need to pre-round to
    bfloat16 with a sticky bit to stop the second rounding double-rounding.
    No FP4 convert is reachable from a SimtVF region, so this does the whole
    encode by hand, which is the honest cost of the port: eighteen lines where
    PTO has four.

    One shift subtlety. The subnormal path shifts right by `126 - exps`, which
    for a zero input is 126 -- undefined past the word width. torch's packer
    relies on its own ">= 32 gives 0" behaviour; here the shift is clamped to
    31, which is provably identical because `mant_pre < 2**23`, so every shift
    of 23 or more already yields 0.
    """
    bits = T.reinterpret("int32", v)
    # bit 3 of (bits >> 28) is bit 31 of bits: the sign nibble, without
    # depending on whether >> sign-extends.
    sign_nib = (bits >> 28) & 0x8
    exps = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF

    is_sub = exps < 127                       # below 1.0 -> e2m1 subnormal row
    sh = T.min(126 - exps, 31)
    mant_pre = 0x400000 | (mant >> 1)         # the implicit 1, then make room
    # Every bit about to be shifted out has to survive as "something was here".
    sticky_sub = T.if_then_else(is_sub, (mant & 1) | (mant_pre & ((1 << sh) - 1)), 0)
    mant_eff = T.if_then_else(is_sub, mant_pre >> sh, mant)
    exp_eff = T.max(exps, 126) - 126          # rebias 127 -> 1

    m2 = (mant_eff >> 21) & 3                 # two bits, of which we keep one
    guard = m2 & 1
    lsb = (m2 >> 1) & 1
    sticky = sticky_sub | (mant_eff & 0x1FFFFF)
    inc = guard & (T.if_then_else(sticky != 0, 1, 0) | lsb)
    tmp = (((exp_eff << 2) | m2) + inc) >> 1
    return sign_nib | T.min(tmp, 7)           # saturate at the 6.0 code


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize float32 -> packed FP4 e2m1 with one FP32 scale per 32 channels."""
    assert hidden % group_size == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.float32),
        # uint8, two nibbles per byte. ASC and PTO declare float4_e2m1fn and let
        # the hardware convert write the nibbles; a SimtVF region has no such
        # convert, so the packing is explicit and the dtype says so.
        Q: T.Tensor((num_tokens, hidden // 2), T.uint8),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.float32)
            q_ub = T.alloc_shared((hidden // 2,), T.uint8)
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # --- BEGIN SOLUTION hint="float32 input needs no widening, and E2M1_MAX is 6.0 -- the reduction and the scale are variant 01's, with E2M1_CLAMP_MIN as the seed. The output is the work. The e2m1_nibble(v) helper is ALREADY WRITTEN above this region: read it, because it is the torch tier's bit-exact packer on scalars, and it is what the SIMD tiers DO NOT have to write -- they own a float32 -> e2m1 convert and only pre-round to bfloat16 with a sticky bit. Give each thread ONE OUTPUT BYTE: T.Parallel(hidden // 2), encode x_ub[2*p] and x_ub[2*p+1], and pack them as lo | (hi << 4), low nibble first. Both halves live in the same quant group, so the scale index is p // (group_size // 2). Writing a byte per thread rather than a nibble per thread is what keeps the pack free of any cross-thread exchange; indexing the shared x_ub twice is allowed, where a fragment would reject the second pattern."
                with T.SimtVF(threads=THREADS):
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    for g in T.Parallel(num_groups):
                        amax = T.alloc_var(T.float32)
                        amax = E2M1_CLAMP_MIN
                        for j in T.serial(group_size):
                            amax = T.max(amax, T.abs(x_ub[g * group_size + j]))
                        inv_l[g] = E2M1_MAX / amax
                        sf_l[g] = amax / E2M1_MAX

                    # One thread per output BYTE: it encodes both nibbles itself,
                    # so the pack needs no exchange between threads.
                    for p in T.Parallel(hidden // 2):
                        inv = inv_l[p // (group_size // 2)]
                        lo = e2m1_nibble(x_ub[2 * p] * inv)
                        hi = e2m1_nibble(x_ub[2 * p + 1] * inv)
                        q_ub[p] = T.Cast("uint8", lo | (hi << 4))

                    T.copy(sf_l, sf_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 04", q, sf)
    return q.view(torch.int8), sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/04")
