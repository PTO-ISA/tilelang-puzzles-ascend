"""per_token 07 (PTO) -- the bfloat16 fast path, and the ladder's biggest win.

Final per_token variant. Composed config:

    bfloat16 input -> bfloat16 compute -> power-of-two scale -> packed UE8M0
    -> FP8 e4m3 output, at K = 256

Read the ASC variant for why bfloat16 compute exists and why it is gated on a
power-of-two scale. Then compare the two amax passes, because this is the
clearest single demonstration of what VMI is for.

### The whole intricate dance disappears

ASC has to produce group maxima of **32** bfloat16 values. 32 bfloat16 is 64
bytes -- neither one 32-byte hardware lane group nor one register -- so there is no
instruction that does it. Production's workaround is four steps:

    ASC -- 7 operations, and you have to know why each is there:
        x0, x1 = S.vld2(x_ub[col], dist="DINTLV_B16")   # deinterleave 256 -> 2x128
        a0 = S.vand(T.reinterpret(x0, "uint16x128"), abs_mask)
        a1 = S.vand(T.reinterpret(x1, "uint16x128"), abs_mask)
        maxima = S.vcgmax(S.vmax(a0, a1))   # pair lanes, then reduce 16 per hw
                                            # group -> 8 results of 32 originals
        dense, _ = S.vintlv(zero_bf16, T.reinterpret(maxima, "bfloat16x128"))
        S.vsts(amax_ub[group], T.reinterpret(dense, "float32x64"), mask_vl8,
               dist="NORM_B32", extent=8)

Every step there is a workaround for a width mismatch: the deinterleave-and-pair
exists only to turn "32 values" into "16 lanes", which is a grouping the hardware
*does* have, and the zero-interleave exists only to widen the results back.

VMI asks for the grouping it actually wants:

    PTO -- 4 operations, and they say what they mean:
        raw = V.vload(x_ub[col], size=256)
        abs_u = V.vand(V.vinterpret_cast(raw, "uint16"), abs_mask)
        amax = V.vcmax(abs_u, mask, group=8)          # 8 groups of 32. Done.
        V.vstore(V.vcvt(V.vinterpret_cast(amax, "bfloat16"), "float32"),
                 amax_ub[group], group=8, stride=1)

`group=8` on a 256-lane vector means "eight independent segments of 32" -- exactly
the quant group -- so no deinterleave, no pairing, no regrouping, no re-widening.
The one remaining reinterpret is free.

This is the same `group=` that variant 01 used for groups of 32 in a 64-lane
float32 vector. One concept, and it covers every width and every element type;
ASC needs a different trick for each combination.

### The apply pass also stays in bfloat16

Production multiplies in bfloat16 too, at 256 lanes, with the inverse broadcast
eight-ways:

    inv = V.vload(inv_ub[group], size=256, dist_mode="brc", group=8)
    V.vstore(V.vcvt(V.vcvt(V.vmul(x, inv, mask), "float32"),
                    "float8_e4m3fn", rounding="R", saturate="SAT"), q_ub[col])

Safe because the scale is a power of two, so the bfloat16 multiply is exact. The
double `vcvt` is because FP8 conversion comes from float32.

### Measure it

`python tools/vf_lines.py` prints the per-variant operation counts. This variant
and `cast_back/06` are where the gap is widest, and both for the same reason: ASC
was emulating a width the hardware does not have, and VMI makes the width an
argument.

Run:  python puzzles/pto/quant/answer/per_token/07_bf16_fast_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0

VARIANT = "pto/per_token/07_bf16_fast_compose"
LANES = 64
STRIP = 256
SF_PAD = 64
BF16_K = 256


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """bfloat16 compute + power-of-two scale + packed UE8M0 + FP8 output."""
    assert hidden % STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_strip = STRIP // group_size          # 8
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.macro
    def compute_scale(amax, lanes):
        """amax -> (UE8M0 exponent byte value, bfloat16 reciprocal)."""
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes), mask)
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(
            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        inv = V.vinterpret_cast(
            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
        return biased, inv

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
            inv_ub = T.alloc_shared((SF_PAD,), T.bfloat16)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # --- BEGIN SOLUTION hint="bf16 reduce in four operations: raw = V.vload(x_ub[col], size=256); abs_u = V.vand(V.vinterpret_cast(raw, 'uint16'), V.vbrc(T.uint16(0x7FFF), size=256)); amax = V.vcmax(abs_u, mask256, group=8); then V.vstore(V.vcvt(V.vinterpret_cast(amax, 'bfloat16'), 'float32'), amax_ub[group], group=8, stride=1). No deinterleave and no pairing -- group=8 asks for exactly 8 segments of 32. Then compute_scale, and an apply pass that multiplies in bfloat16 at 256 lanes with a brc group=8 inverse."
                with T.SimdVF():
                    mask = V.create_mask(STRIP, size=STRIP)
                    abs_mask = V.vbrc(T.uint16(0x7FFF), size=STRIP)

                    # ---- pass 1: reduce in bfloat16, 8 groups of 32 per step ----
                    for strip in T.serial(hidden // STRIP):
                        col = strip * STRIP
                        group = strip * groups_per_strip
                        raw = V.vload(x_ub[col], size=STRIP)
                        # clearing the sign bit is abs, on the integer unit
                        abs_u = V.vand(V.vinterpret_cast(raw, "uint16"), abs_mask)
                        # group=8: eight independent segments of 32. That is the
                        # whole trick, and it is the same `group=` as variant 01.
                        amax = V.vcmax(abs_u, mask, group=groups_per_strip)
                        V.vstore(V.vcvt(V.vinterpret_cast(amax, "bfloat16"), "float32"),
                                 amax_ub[group], group=groups_per_strip, stride=1)
                    T.simd.mem_bar("VST_VLD")

                    # ---- pass 2: exponent arithmetic ----
                    biased, inv = compute_scale(V.vload(amax_ub[0], size=LANES), LANES)
                    V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])
                    V.vstore(V.vcvt(inv, "bfloat16"), inv_ub[0])
                    T.simd.mem_bar("VST_VLD")

                    # ---- pass 3: apply in bfloat16, 256 lanes at a time ----
                    for strip in T.serial(hidden // STRIP):
                        col = strip * STRIP
                        group = strip * groups_per_strip
                        inv_v = V.vload(inv_ub[group], size=STRIP, stride=1,
                                        dist_mode="brc", group=groups_per_strip)
                        x = V.vload(x_ub[col], size=STRIP)
                        scaled = V.vmul(x, inv_v, mask)
                        # FP8 conversion comes from float32, hence the double cvt.
                        V.vstore(V.vcvt(V.vcvt(scaled, "float32"), "float8_e4m3fn",
                                        rounding="R", saturate="SAT"), q_ub[col])
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
    print("[demo] producing group maxima of 32 bfloat16 values:")
    print("[demo]   ASC: vld2 DINTLV -> 2 vand -> vmax -> vcgmax -> vintlv -> vsts")
    print("[demo]        7 ops, because 32 bf16 is 64 bytes: neither one 32-byte")
    print("[demo]        hardware lane group nor one register. The deinterleave")
    print("[demo]        and pairing exist only to turn 32 values into 16 lanes.")
    print("[demo]   PTO: vload -> vand -> vcmax(group=8) -> vstore")
    print("[demo]        4 ops. group=8 asks for 8 segments of 32 directly.")
    print("[demo] same `group=` as variant 01's groups of 32 in a 64-lane float32")
    print("[demo] vector: one concept covering every width and element type.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()[0], BF16_K
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_token(x, CANONICAL_G, round_sf=True, packed=True)
    q, packed = launch(x.npu())
    assert_same_bytes(packed.cpu(), ref_packed, f"sf_packed({m},{k})")
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    _, ref_f32 = oracle.per_token(x, CANONICAL_G, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(packed.cpu()), ref_f32)
    print(f"[check] shape=({m},{k}) packed scales byte-exact, FP8 matches, and the")
    print(f"[check] bfloat16 reduction picked the same exponents as float32 would")


def main() -> int:
    if status.unimplemented(VARIANT, lambda: compile_kernel(BF16_K)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_token", "07_bf16_fast_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
