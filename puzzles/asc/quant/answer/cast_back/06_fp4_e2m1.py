"""cast_back 06 (ASC) -- packed FP4 (e2m1) input, two values per byte.

The quantized values are now 4 bits each (see puzzles/torch/.../06_fp4_e2m1.py
for the 16-code table). Two consequences inside the vector unit:

### A strip is now 128 values, not 64

One `S.vld` with `UNPK4_B8` on an FP4 buffer yields **128** values, because each
byte carries two. Convert them to bfloat16 and you have a 128-lane bfloat16
vector -- one full 256-byte register.

But the scale is float32, so the multiply has to happen in float32, which is
64 lanes per register. 128 bfloat16 lanes therefore have to become **two**
float32 registers.

### Widening bfloat16 to float32 without a convert

There is no `vcvt` from bfloat16 to float32 in this instruction set. The trick is
that a bfloat16 *is* the top half of the float32 with the same value: float32 is
`sign|exp(8)|mantissa(23)` and bfloat16 is `sign|exp(8)|mantissa(7)`, the
truncated form. So interleaving zeros into the low half builds the float32:

    x_lo, x_hi = S.vintlv(zero_bf16, x_bf16)        # pairs: (0, value)
    f32 = T.reinterpret(x_lo, "float32x64")         # 0 | value << 16  ==  the float

`S.vintlv` interleaves two vectors lane by lane and returns two results, so one
call both widens *and* splits the 128 lanes into the two 64-lane halves. Neat,
but you have to recognise the idiom -- nothing in the name says "widen".

Note how much of this file is bookkeeping about register widths. That is the
recurring tax of ASC: the lane count is part of the type
(`"float32x64"`, `"uint16x128"`), so a change of element size forces the code
structure to change with it. The PTO version of this file keeps one 128-lane
logical vector throughout and does the same work in one operation.

Run:  python puzzles/asc/quant/answer/cast_back/06_fp4_e2m1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import CANONICAL_G
from common.math_ops import unpack_e2m1_bytes

VARIANT = "asc/cast_back/06_fp4_e2m1"
LANES = 64          # float32 lanes per register
FP4_STRIP = 128     # FP4 values produced by one unpacking load


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize packed FP4 -> bfloat16 with one FP32 scale per 32 channels."""
    assert hidden % FP4_STRIP == 0, "the FP4 path steps 128 values at a time"
    assert group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    sf_pad = max(num_groups, LANES)

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
                # --- BEGIN SOLUTION hint="one S.vld(q_ub[col], dist='UNPK4_B8') yields 128 FP4 values; S.vcvt them to bfloat16; widen+split with x_lo, x_hi = S.vintlv(S.vdup(0.0, T.bfloat16), x_bf16) and T.reinterpret each half to 'float32x64'; build two scale vectors (groups g,g+1 for the low half and g+2,g+3 for the high) the same way as variant 01, multiply, and store both halves"
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    zero_bf16 = S.vdup(0.0, T.bfloat16)
                    for strip in T.serial(hidden // FP4_STRIP):
                        col = strip * FP4_STRIP
                        group = strip * (FP4_STRIP // group_size)   # 4 groups per strip

                        # 128 FP4 values -> 128 bfloat16 lanes.
                        x_bf16 = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.bfloat16)
                        # Interleave zeros to widen to float32, and split in two.
                        x_lo, x_hi = S.vintlv(zero_bf16, x_bf16)

                        s0 = S.vld(sf_ub[group], dist="BRC_B32")
                        s1 = S.vld(sf_ub[group + 1], dist="BRC_B32")
                        s2 = S.vld(sf_ub[group + 2], dist="BRC_B32")
                        s3 = S.vld(sf_ub[group + 3], dist="BRC_B32")
                        scale_lo = S.vsel(s0, s1, mask_low)
                        scale_hi = S.vsel(s2, s3, mask_low)

                        out_lo = S.vmul(T.reinterpret(x_lo, "float32x64"), scale_lo)
                        out_hi = S.vmul(T.reinterpret(x_hi, "float32x64"), scale_hi)
                        S.vsts(out_ub[col], S.vcvt(out_lo, T.bfloat16), dist="PK_B32")
                        S.vsts(out_ub[col + LANES], S.vcvt(out_hi, T.bfloat16),
                               dist="PK_B32")
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """`q_packed` is (M, K/2) int8; tilelang wants it as float4_e2m1fn_x2."""
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf)
    status.assert_on_device("cast_back 06", out)
    return out


def demo_numbers() -> None:
    print("[demo] widening bfloat16 to float32 by interleaving zeros:")
    v = torch.tensor([1.5], dtype=torch.bfloat16)
    bits16 = v.view(torch.int16).item() & 0xFFFF
    bits32 = bits16 << 16
    back = torch.tensor([bits32], dtype=torch.int32).view(torch.float32).item()
    print(f"[demo]   bfloat16 1.5 = 0x{bits16:04X}")
    print(f"[demo]   0 | 0x{bits16:04X} << 16 = 0x{bits32:08X} -> float32 {back:g}")
    assert back == 1.5
    print("[demo]   so vintlv(zero, x_bf16) widens without any convert instruction")
    print("[demo] register bookkeeping for one 128-value FP4 strip:")
    print("[demo]   128 FP4 values -> 128 bfloat16 lanes (1 register)")
    print("[demo]   -> 2 x 64 float32 lanes (2 registers), because the scale is float32")


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
    sim.print_banner("asc", "cast_back", "06_fp4_e2m1")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
