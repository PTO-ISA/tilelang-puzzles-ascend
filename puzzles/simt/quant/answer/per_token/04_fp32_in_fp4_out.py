"""per_token 04 (SIMT). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX

THREADS = 128
# bfloat16 lanes consumed by one packed-FP4 convert-and-store, as the SIMD
# tiers use it: two float32 registers' worth. Only the hw_convert spelling
# needs it.
PAIR = 128


def e2m1_nibble(v, branch_free: bool = False):
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

    ``branch_free`` picks between two spellings of the three conditionals, and
    nothing else. Both are bit-exact; they differ by **2.09x in cycles**.

    * ``False`` (the default, and what the ladder measures) uses
      ``T.if_then_else``, which is what a GPU programmer writes first. On this
      toolchain it lowers to a *real branch*, so threads whose values fall on
      different sides of the condition serialize: this variant is the only one
      in the ladder whose trace carries ``SIMT_BRANCH`` and
      ``SIMT_START_DVG`` markers in quantity -- 44,835 branches, where every
      other SIMT kernel here has none.
    * ``True`` builds a 0/all-ones mask with ``-T.Cast("int32", cond)`` and
      selects arithmetically. It issues **37% fewer instructions and runs 2.09x
      fewer cycles** (562,280 to 269,370), because the instructions it drops
      are the branching itself.

    The branch-free arm is safe only because both sides can always be evaluated:
    the shift is already clamped to 31, so ``mant_pre >> sh`` cannot be
    undefined even when the condition is false. That is a property of this
    encoder, not a general licence.

    On a GPU the distinction would not exist -- ``cond ? a : b`` over cheap
    arithmetic is predicated into a select by the compiler. So the gap is a
    codegen gap rather than a cost of the thread model, which is why the default
    stays the natural spelling and the docs quote both numbers.
    """
    bits = T.reinterpret("int32", v)
    # bit 3 of (bits >> 28) is bit 31 of bits: the sign nibble, without
    # depending on whether >> sign-extends.
    sign_nib = (bits >> 28) & 0x8
    exps = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF

    sh = T.min(126 - exps, 31)
    mant_pre = 0x400000 | (mant >> 1)         # the implicit 1, then make room
    low = (1 << sh) - 1                       # the bits about to be shifted out

    if branch_free:
        # -(cond) is 0 or all-ones. Both arms are always evaluated, which is
        # safe here precisely because `sh` is clamped above.
        m = -T.Cast("int32", exps < 127)
        sticky_sub = ((mant & 1) | (mant_pre & low)) & m
        mant_eff = ((mant_pre >> sh) & m) | (mant & ~m)
    else:
        is_sub = exps < 127                   # below 1.0 -> e2m1 subnormal row
        # Every bit about to be shifted out has to survive as "something was
        # here".
        sticky_sub = T.if_then_else(is_sub, (mant & 1) | (mant_pre & low), 0)
        mant_eff = T.if_then_else(is_sub, mant_pre >> sh, mant)

    exp_eff = T.max(exps, 126) - 126          # rebias 127 -> 1
    m2 = (mant_eff >> 21) & 3                 # two bits, of which we keep one
    guard = m2 & 1
    lsb = (m2 >> 1) & 1
    sticky = sticky_sub | (mant_eff & 0x1FFFFF)
    # `sticky` is a non-negative bit field, so min(sticky, 1) is "is it set".
    nz = T.min(sticky, 1) if branch_free else T.if_then_else(sticky != 0, 1, 0)
    tmp = (((exp_eff << 2) | m2) + (guard & (nz | lsb))) >> 1
    return sign_nib | T.min(tmp, 7)           # saturate at the 6.0 code


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   branch_free: bool = False):
    """Quantize float32 -> packed FP4 e2m1 with one FP32 scale per 32 channels.

    This is the tier's kernel: everything inside one ``T.SimtVF`` region,
    including the e2m1 encode, which is what makes it the most expensive row in
    the ladder. ``compile_hybrid_kernel`` below is the same algorithm with that
    one encode handed to the vector unit, and it is 72.29x cheaper; this
    function stays pure SIMT because the tier exists to measure what the thread
    model costs on its own.

    ``branch_free`` is a trace-time flag threaded into ``e2m1_nibble``; it
    changes no result, only how the encoder's conditionals are spelled. See that
    function's docstring, and `doc/quant/per_token/04_fp32_in_fp4_out.md` for
    the measurement.
    """
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

                # --- BEGIN SOLUTION hint="float32 input needs no widening, and E2M1_MAX is 6.0 -- the reduction and the scale are variant 01's, with E2M1_CLAMP_MIN clamping the fold's result. The output is the work. The e2m1_nibble(v, branch_free) helper is ALREADY WRITTEN above this region: read it, because it is the torch tier's bit-exact packer on scalars, and it is what the SIMD tiers DO NOT have to write -- they own a float32 -> e2m1 convert and only pre-round to bfloat16 with a sticky bit. Read its docstring too: it holds BOTH spellings of the three conditionals, and the branch-free one is 2.09x faster for reasons worth knowing. Give each thread ONE OUTPUT BYTE: T.Parallel(hidden // 2), encode x_ub[2*p] and x_ub[2*p+1], and pack them as lo | (hi << 4), low nibble first. Both halves live in the same quant group, so the scale index is p // (group_size // 2). Writing a byte per thread rather than a nibble per thread is what keeps the pack free of any cross-thread exchange; indexing the shared x_ub twice is allowed, where a fragment would reject the second pattern. Thread `branch_free` straight through to the helper -- it is a trace-time Python flag, so only one spelling is ever emitted."
                with T.SimtVF(threads=THREADS):
                    x_l = T.alloc_fragment((num_groups, group_size), T.float32)
                    amax_l = T.alloc_fragment((num_groups,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    for g, j in T.Parallel(num_groups, group_size):
                        x_l[g, j] = x_ub[g * group_size + j]
                    T.reduce_absmax(x_l, amax_l, dim=1)

                    for g in T.Parallel(num_groups):
                        amax = T.max(amax_l[g], E2M1_CLAMP_MIN)
                        inv_l[g] = E2M1_MAX / amax
                        sf_l[g] = amax / E2M1_MAX

                    # One thread per output BYTE: it encodes both nibbles
                    # itself, so the pack needs no exchange between threads.
                    for p in T.Parallel(hidden // 2):
                        inv = inv_l[p // (group_size // 2)]
                        lo = e2m1_nibble(x_ub[2 * p] * inv, branch_free)
                        hi = e2m1_nibble(x_ub[2 * p + 1] * inv, branch_free)
                        q_ub[p] = T.Cast("uint8", lo | (hi << 4))

                    T.copy(sf_l, sf_ub)
                # --- END SOLUTION

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


# --- BEGIN ALT hw_convert ---------------------------------------------------
# A SECOND SPELLING of this variant, kept in its own function rather than behind
# an `if` inside the one above: the two differ in their output stage and in
# their output dtype, and reading either should not mean stepping over the
# other. tools/vf_lines.py skips everything up to END ALT, because only one of
# the two is ever emitted and the published line counts are of the tier's own
# kernel. The harness asserts the two produce byte-identical results.

@tilelang.jit(target="ascend")
def compile_hybrid_kernel(hidden: int, group_size: int = CANONICAL_G):
    """The same quantization, with the e2m1 encode done by the vector unit.

    No `T.Cast` to FP4 lowers inside a `T.SimtVF` region, and none can: an fp4
    element is four bits, so fp4 storage is retyped to `float4_e2m1fnx2` and
    every store into it needs two lanes, where per-thread scalar code has one.
    Underneath, this toolchain's only FP4 types are the **vector** ones
    (`vector_f4e2m1x2`), and the conversion is a vector instruction over them --
    there is no scalar FP4 type to convert into at all.
    `harness/probe/simt_fp4_convert.py` records the nine spellings that fail.

    What a kernel *can* do is leave the region for that one instruction. A
    `T.SimtVF` region and a `T.SimdVF` region live in the same kernel here: the
    thread model keeps the reduction, the scales and the round-to-odd, and three
    lines of the vector model do the convert. Byte-identical output, **72.29x**
    fewer simulated VF cycles (562,280 to 7,778, or 1.26x ASC rather than
    91.3x), and eighteen lines of hand-written encoder deleted.

    It is reached through ``launch(x, hw_convert=True)`` and is **not** the
    default, because the SIMT tier measures what the thread model costs on its
    own. See `doc/quant/per_token/04_fp32_in_fp4_out.md`.
    """
    assert hidden % group_size == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast_hybrid(
        X: T.Tensor((num_tokens, hidden), T.float32),
        # float4_e2m1fn, as the ASC and PTO tiers declare it: the hardware
        # convert writes the nibbles, so the kernel does not pack bytes.
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.float32)
            bf_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                # --- BEGIN SOLUTION hint="the same algorithm as compile_kernel above, except that the e2m1 encode is handed to the vector unit instead of written out. Two regions, in order. (1) T.SimtVF: variant 01's reduction and scales with E2M1 constants, then one T.Parallel(hidden) that scales each value and rounds it to ODD on the way to bfloat16 -- bits = T.reinterpret('uint32', v); sticky = T.min(bits & 0xFFFF, 1); store T.reinterpret('bfloat16', T.Cast('uint16', (bits >> 16) | sticky)). The sticky bit is what stops the convert that follows from rounding a second time; it is the SIMD tiers' S.vdintlv + S.vor + S.vmin, on scalars. (2) T.SimdVF, OUTSIDE the first region: one S.vsts of S.vcvt(S.vld(bf_ub[col]), T.float4_e2m1fn) per PAIR lanes, dist='PK4_B32'. Declare the output T.float4_e2m1fn, as ASC and PTO do, since the hardware writes the nibbles now."
                with T.SimtVF(threads=THREADS):
                    x_l = T.alloc_fragment((num_groups, group_size), T.float32)
                    amax_l = T.alloc_fragment((num_groups,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    for g, j in T.Parallel(num_groups, group_size):
                        x_l[g, j] = x_ub[g * group_size + j]
                    T.reduce_absmax(x_l, amax_l, dim=1)

                    for g in T.Parallel(num_groups):
                        amax = T.max(amax_l[g], E2M1_CLAMP_MIN)
                        inv_l[g] = E2M1_MAX / amax
                        sf_l[g] = amax / E2M1_MAX

                    # Scale, then round to ODD on the way to bfloat16: the
                    # convert below rounds a second time, and the sticky bit is
                    # what stops that from double-rounding. This is the SIMD
                    # tiers' S.vdintlv + S.vor + S.vmin, written on scalars.
                    for i in T.Parallel(hidden):
                        bits = T.reinterpret("uint32",
                                             x_ub[i] * inv_l[i // group_size])
                        sticky = T.min(bits & 0xFFFF, T.uint32(1))
                        bf_ub[i] = T.reinterpret(
                            "bfloat16", T.Cast("uint16", (bits >> 16) | sticky))

                    T.copy(sf_l, sf_ub)

                # Out of the thread model for one instruction. PAIR is two
                # float32 registers' worth of bfloat16, which is what the
                # packed-FP4 store mode writes in one go.
                with T.SimdVF():
                    for pair in T.serial(hidden // PAIR):
                        S.vsts(q_ub[pair * PAIR],
                               S.vcvt(S.vld(bf_ub[pair * PAIR]),
                                      T.float4_e2m1fn),
                               dist="PK4_B32")
                # --- END SOLUTION

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast_hybrid
# --- END ALT ----------------------------------------------------------------


def launch(x: torch.Tensor, branch_free: bool = False, hw_convert: bool = False):
    """Two additive flags, each selecting a bit-identical alternative spelling.

    ``branch_free=True`` respells the encoder's conditionals as mask arithmetic,
    inside ``compile_kernel``. ``hw_convert=True`` calls
    ``compile_hybrid_kernel`` instead, which has no encoder at all. Both default
    to the shipped behaviour, so the harness's ``launch(x)`` is the same call it
    makes on every other tier -- and the harness exercises all three arms and
    asserts they agree byte for byte.
    """
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    if hw_convert:
        # the same bytes under the name the hardware convert writes them by
        q_fp4 = q.view(torch.float4_e2m1fn_x2)
        compile_hybrid_kernel(hidden)(x, q_fp4, sf)
        status.assert_on_device("per_token 04", q_fp4, sf)
    else:
        compile_kernel(hidden, branch_free=branch_free)(x, q, sf)
        status.assert_on_device("per_token 04", q, sf)
    return q.view(torch.int8), sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_token/04")
