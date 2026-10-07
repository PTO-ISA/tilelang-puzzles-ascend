"""cast_back 06 (PTO) -- packed FP4 input, and the clearest PTO win in the ladder.

Same kernel as the ASC variant. Read that one first for why FP4 forces a width
change at all.

### PTO vs ASC: width stops dictating structure

An unpacking load of FP4 yields **128** values. The scale is float32, which is
64 lanes per 256-byte register -- so in ASC those 128 values have to be split
across two registers, and the code has to say so at every step:

    ASC (two 64-lane halves, 10 vector operations):
        x_bf16     = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.bfloat16)
        x_lo, x_hi = S.vintlv(zero_bf16, x_bf16)          # widen AND split
        s0, s1, s2, s3 = four BRC_B32 loads
        scale_lo   = S.vsel(s0, s1, mask_low)
        scale_hi   = S.vsel(s2, s3, mask_low)
        out_lo     = S.vmul(T.reinterpret(x_lo, "float32x64"), scale_lo)
        out_hi     = S.vmul(T.reinterpret(x_hi, "float32x64"), scale_hi)
        two PK_B32 stores

VMI has a real 128-lane logical float32 type -- 512 bytes, two physical registers,
one name -- so the split never happens:

    PTO (one 128-lane vector, 5 vector operations):
        x_f32 = V.vzip(zero_bf16, V.vcvt(V.vload(q_ub[col], size=128), "bfloat16"),
                       "float32")
        scale = V.vload(sf_ub[group], size=128, stride=1, dist_mode="brc", group=4)
        V.vstore(V.vcvt(V.vmul(x_f32, scale, mask), "bfloat16"), out_ub[col])

Three things collapsed at once:

1. **`vzip` takes the target dtype.** ASC's `vintlv` interleaves and leaves you to
   reinterpret each half with an explicit lane count; `V.vzip(lo, hi, "float32")`
   says "interleave these and read the result as float32" in one operation.
2. **`group=4` scales to the width.** The ASC broadcast-and-select pattern needs
   one `vsel` per 64-lane register, so widening to 128 lanes *doubles* that code.
   In VMI the group count is an argument: 2 at 64 lanes, 4 at 128.
3. **No reinterpret with a baked lane count.** `"float32x64"` cannot describe the
   128-lane case; `"float32"` describes both.

This is the property the docs keep claiming and this file demonstrates concretely:
in ASC the register width is part of the *type*, so changing element size forces
the code structure to change. In VMI width is an *argument*, so the same code
shape serves any width. It is also why production's PTO `per_token` can share one
`compute_scale` helper across 4-, 64-, 128- and 256-lane call sites while the ASC
version cannot.

### Where it is still not torch

`size=128` is still written by hand, the strip loop is still explicit, and the
zero-interleave widening idiom is still an idiom you have to know. VMI removed the
*width bookkeeping*, not the vector programming.

Run:  python puzzles/pto/quant/answer/cast_back/06_fp4_e2m1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import CANONICAL_G

VARIANT = "pto/cast_back/06_fp4_e2m1"
FP4_STRIP = 128


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize packed FP4 -> bfloat16 with one FP32 scale per 32 channels."""
    assert hidden % FP4_STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_strip = FP4_STRIP // group_size      # 4
    num_tokens = T.dynamic("num_tokens")
    sf_pad = max(num_groups, 64)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            sf_ub = T.alloc_shared((sf_pad,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])
                # --- BEGIN SOLUTION hint="stay at 128 lanes throughout: x_f32 = V.vzip(V.vbrc(T.bfloat16(0.0), size=128), V.vcvt(V.vload(q_ub[col], size=128), 'bfloat16'), 'float32'); scale = V.vload(sf_ub[group], size=128, stride=1, dist_mode='brc', group=4); then one vmul and one vstore -- no splitting into 64-lane halves"
                with T.SimdVF():
                    mask = V.create_mask(FP4_STRIP, size=FP4_STRIP)
                    zero_bf16 = V.vbrc(T.bfloat16(0.0), size=FP4_STRIP)
                    for strip in T.serial(hidden // FP4_STRIP):
                        col = strip * FP4_STRIP
                        group = strip * groups_per_strip

                        # 128 FP4 -> 128 bfloat16 -> 128 float32, one name each.
                        x_bf16 = V.vcvt(V.vload(q_ub[col], size=FP4_STRIP), "bfloat16")
                        x_f32 = V.vzip(zero_bf16, x_bf16, "float32")

                        # group=4 instead of group=2: the width is an argument.
                        scale = V.vload(sf_ub[group], size=FP4_STRIP, stride=1,
                                        dist_mode="brc", group=groups_per_strip)

                        scaled = V.vmul(x_f32, scale, mask)
                        V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf)
    status.assert_on_device("cast_back 06", out)
    return out


def demo_numbers() -> None:
    print("[demo] vector operations for one 128-value FP4 strip:")
    print("[demo]   ASC: vcvt, vld, vintlv, 4x BRC vld, 2x vsel, 2x vmul, 2x vcvt,")
    print("[demo]        2x vsts   -- plus two explicit float32x64 reinterprets")
    print("[demo]   PTO: vload, vcvt, vzip, brc vload, vmul, vcvt, vstore")
    print("[demo] the difference is entirely width bookkeeping: ASC must split 128")
    print("[demo] lanes into two 64-lane registers because the lane count is part")
    print("[demo] of its type; VMI has a 128-lane logical float32 and does not.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = torch.randn(m, k) * 3
    q_packed, sf = oracle.per_token(x, CANONICAL_G, fmt="e2m1")
    ref = oracle.cast_back(q_packed, sf, (1, CANONICAL_G), fp4=True,
                           out_dtype=torch.bfloat16)
    got = launch(q_packed.npu(), sf.npu()).cpu()
    assert_bf16_near(got, ref, f"cast_back_fp4({m},{k})", atol=0.0)
    print(f"[check] shape=({m},{k}) packed={tuple(q_packed.shape)} "
          f"matches the torch oracle exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "cast_back", "06_fp4_e2m1")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
