"""per_token 07 (ASC) -- the bfloat16 fast path, and everything composed.

Final per_token variant, and the one that reaches production's actual hot loop.

Composed config:
    bfloat16 input -> bfloat16 compute -> power-of-two scale -> packed UE8M0
    -> FP8 e4m3 output

### Why compute in bfloat16

A register is 256 bytes: 64 float32 lanes or **128 bfloat16** lanes. Reducing in
bfloat16 halves the instruction count for the amax pass. It is only legal because
the scale is a power of two (variant 02), so applying it is exact in any float
format -- production gates the fast path on exactly that, plus `hidden % 256 == 0`:

    value_dtype = bfloat16  only if  hidden % 256 == 0
                               and  use_packed_ue8m0 and round_sf
                               and  the input is bfloat16

This variant therefore runs at **K = 256**, not the ladder's usual 128.

### Absolute value as a bitwise AND

For any IEEE-like format, clearing the sign bit *is* `abs`. In bfloat16 that is
`& 0x7FFF`, which runs on the integer unit:

    abs_x = S.vand(T.reinterpret(x, "uint16x128"), S.vdup(0x7FFF, T.uint16))

Cheaper than `vabs`, and it is why the fast path reinterprets to uint16 rather
than working in float.

### Getting groups of 32 out of a bfloat16 vector

This is the intricate part, and it is intricate because 32 bfloat16 values are
64 bytes -- neither one 32-byte hardware lane group nor one register. Production's
answer, reproduced here:

1. `S.vld2(..., dist="DINTLV_B16")` loads 256 contiguous bfloat16 and
   *deinterleaves* them into two 128-lane vectors: evens and odds.
2. `S.vmax(abs_even, abs_odd)` gives 128 values where lane `i` is
   `max(x[2i], x[2i+1])` -- each lane now covers 2 original values.
3. `S.vcgmax` is the *grouped* max: it reduces within each 32-byte hardware lane
   group, which for 16-bit elements is 16 lanes, producing 8 results. Each result
   therefore covers 16 x 2 = **32 original values** -- exactly one quant group.
4. `S.vintlv(zero, maxima)` widens those 8 bfloat16 results to float32 (the
   zero-interleave trick from cast_back/06), and a masked 8-element store puts
   them in `amax_ub`.

So one 256-value strip produces 8 group maxima in about five vector operations.
The float32 path needed 8 masked `vcmax` plus 8 single-element stores for the same
work.

The price is precision: the reduction happens in bfloat16, so `amax` carries 8
mantissa bits rather than 24. That is harmless here because the next step only
uses its *exponent* -- `ceil(log2(amax/448))` -- and bfloat16 keeps the exponent
exactly. The test asserts that the chosen exponents match the float32 path.

### Where this lands relative to production

`per_token_cast_asc.py` adds multi-core `T.Persistent` scheduling,
double-buffered UB, and the remaining config branches (FP4, column-major,
requant, the split). The vector arithmetic in its hot loop is what this file does.

Run:  python puzzles/asc/quant/answer/per_token/07_bf16_fast_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0

VARIANT = "asc/per_token/07_bf16_fast_compose"
LANES = 64
STRIP = 256            # the bf16 fast path's step: 8 groups of 32
SF_PAD = 64
BF16_K = 256           # this variant needs hidden % 256 == 0


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """bfloat16 compute + power-of-two scale + packed UE8M0 + FP8 output."""
    assert hidden % STRIP == 0, "the bfloat16 fast path steps 256 values at a time"
    assert group_size == 32
    num_groups = hidden // group_size
    groups_per_strip = STRIP // group_size          # 8
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.uint8)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # --- BEGIN SOLUTION hint="bf16 reduce: per 256-value strip, x0, x1 = S.vld2(x_ub[col], dist='DINTLV_B16'); abs via S.vand(T.reinterpret(x, 'uint16x128'), S.vdup(0x7FFF, T.uint16)); pair them with S.vmax; S.vcgmax gives 8 grouped maxima; widen with dense, _ = S.vintlv(S.vdup(0.0, T.bfloat16), T.reinterpret(maxima, 'bfloat16x128')) and store 8 elements with mask S.pset(32, 'PAT_VL8'), dist='NORM_B32', extent=8. Then the exponent trick from variant 02/03 and the apply pass from variant 01."
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)
                    mask_vl8 = S.pset(32, "PAT_VL8")
                    abs_mask = S.vdup(0x7FFF, T.uint16)
                    zero_bf16 = S.vdup(0.0, T.bfloat16)

                    # ---- pass 1: reduce in bfloat16, 256 values per step ----
                    for strip in T.serial(hidden // STRIP):
                        col = strip * STRIP
                        group = strip * groups_per_strip
                        # deinterleave 256 bf16 into evens and odds
                        x0, x1 = S.vld2(x_ub[col], dist="DINTLV_B16")
                        # clearing the sign bit is abs, on the integer unit
                        a0 = S.vand(T.reinterpret(x0, "uint16x128"), abs_mask)
                        a1 = S.vand(T.reinterpret(x1, "uint16x128"), abs_mask)
                        # each lane now covers 2 originals; vcgmax reduces 16
                        # lanes per hardware group -> 8 results of 32 originals
                        maxima = S.vcgmax(S.vmax(a0, a1))
                        # widen the 8 bf16 maxima to float32 by interleaving zeros
                        dense, _ = S.vintlv(zero_bf16,
                                            T.reinterpret(maxima, "bfloat16x128"))
                        S.vsts(amax_ub[group], T.reinterpret(dense, "float32x64"),
                               mask_vl8, dist="NORM_B32", extent=groups_per_strip)
                    S.mem_bar("VST_VLD")

                    # ---- pass 2: exponent arithmetic (variants 02/03) ----
                    clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                    one = S.vdup(1, T.uint32)
                    bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
                    biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                    S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                    S.vsts(inv_ub[0], T.reinterpret(
                        S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                        "float32x64"))
                    S.mem_bar("VST_VLD")

                    # ---- pass 3: apply, in float32 (the scale is exact) ----
                    for pair in T.serial(hidden // 128):
                        col = pair * 128
                        group = pair * 4
                        for half in range(2):
                            c = col + half * LANES
                            g = group + half * 2
                            i0 = S.vld(inv_ub[g], dist="BRC_B32")
                            i1 = S.vld(inv_ub[g + 1], dist="BRC_B32")
                            xv = S.vcvt(S.vld(x_ub[c], dist="UNPK_B16"), T.float32)
                            S.vsts(q_ub[c],
                                   S.vcvt(S.vmul(xv, S.vsel(i0, i1, mask_low)),
                                          T.float8_e4m3fn), dist="PK4_B32")
                # --- END SOLUTION
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    ng = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m, ng), dtype=torch.uint8, device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_token 07", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)


def demo_numbers() -> None:
    print("[demo] one 256-byte register holds:")
    print("[demo]   64 float32 lanes   or   128 bfloat16 lanes")
    print("[demo] reducing in bf16 halves the amax pass's instruction count.")
    print("[demo] getting groups of 32 out of a bf16 vector, step by step:")
    print("[demo]   vld2 DINTLV_B16 : 256 bf16 -> evens(128) + odds(128)")
    print("[demo]   vand 0x7FFF     : abs, on the integer unit")
    print("[demo]   vmax            : lane i now covers x[2i], x[2i+1]")
    print("[demo]   vcgmax          : reduces 16 lanes per hw group -> 8 results")
    print("[demo]                     each covering 16 x 2 = 32 originals")
    print("[demo]   vintlv(0, m)    : widen those 8 bf16 to float32")
    print("[demo] reducing in bf16 loses mantissa bits but keeps the exponent,")
    print("[demo] and only the exponent is used -- the test asserts that.")
    v = torch.tensor([3.14159265], dtype=torch.float32)
    bf = v.to(torch.bfloat16).float()
    import math
    assert math.floor(math.log2(v.item())) == math.floor(math.log2(bf.item()))
    print(f"[demo]   {v.item():.8f} -> bf16 {bf.item():.8f}: same exponent")


def test_correctness() -> None:
    m = sim.sim_shapes()[0]
    k = BF16_K
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_token(x, CANONICAL_G, round_sf=True, packed=True)
    q, packed = launch(x.npu())
    assert_same_bytes(packed.cpu(), ref_packed, f"sf_packed({m},{k})")
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    # the bf16 reduction must pick the same power-of-two exponent as float32 would
    _, ref_f32 = oracle.per_token(x, CANONICAL_G, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(packed.cpu()), ref_f32), (
        "the bfloat16 reduction changed the chosen exponent"
    )
    print(f"[check] shape=({m},{k}) packed scales byte-exact, FP8 matches, and the")
    print(f"[check] bfloat16 reduction picked the same exponents as float32 would")


def main() -> int:
    if status.unimplemented(VARIANT, lambda: compile_kernel(BF16_K)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "07_bf16_fast_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
