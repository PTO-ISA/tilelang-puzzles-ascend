"""cast_back 03 (PTO) -- packed UE8M0 decode in VMI.

Read the ASC variant first: the bit trick is the same and it is explained there
in full. An exponent byte placed at bit 23 with a zero mantissa *is* the float32
value 2^(e-127), so decoding is a shift and a mask.

### PTO vs ASC: an honest draw

This is the variant where VMI buys the least, and it is worth saying so plainly
rather than manufacturing an advantage. Both backends do the identical sequence:

    broadcast the packed word -> reinterpret as uint32 -> per-lane shift
    -> mask the exponent field -> reinterpret as float32

Line for line:

    ASC: packed = S.vld(sf_ub[strip], dist="BRC_B16")
         bits   = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"), sf_shift), exp_mask)
         scale  = T.reinterpret(bits, "float32x64")

    PTO: packed = V.vload(sf_ub[strip], size=128, stride=1, dist_mode="brc", group=1)
         bits   = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"), sf_shift), exp_mask)
         scale  = V.vinterpret_cast(bits, "float32")

Three differences, all small:

1. **Reinterpret drops the lane count.** `T.reinterpret(v, "uint32x64")` versus
   `V.vinterpret_cast(v, "uint32")` -- VMI derives the lane count from the total
   bit width, so the same expression works at any width. This is the one real
   win here, and it is the same "width is not part of the type" property that
   matters much more in the reduction variants.
2. **Mask construction.** `S.pset(32, "PAT_VL32")` names a hardware predicate
   pattern; `V.create_mask(32, size=64)` says "32 of 64 lanes". The VMI form
   composes with arbitrary widths.
3. **`vsel` argument order.** ASC is `S.vsel(if_true, if_false, mask)`; VMI puts
   the mask first, `V.vsel(mask, if_true, if_false)`. A gotcha when porting, and
   the kind of thing that silently produces wrong answers rather than an error.

What does *not* change is the important part: the per-lane variable shift is
still a hand-written trick in both. VMI does not have a "decode UE8M0"
operation, and should not -- it is a vector IR, not a quantization library.

(Note for later: production's PTO `cast_back` actually loses a capability here.
ASC can build a shift *vector* and apply `S.vshr(v, shifts)` with a different
amount per lane. VMI's `vshrs` takes a scalar shift amount only, so where
production's ASC path uses a per-lane variable shift, the PTO path computes both
byte positions and selects between them with `vcmp` + `vsel`. This variant does
not hit that -- a left-shift by a vector works fine -- but the column-major
variant does, and it is one of the few places VMI is genuinely the weaker
surface.)

Run:  python puzzles/pto/quant/answer/cast_back/03_packed_ue8m0.py
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
from common.consts import CANONICAL_G, PACK_FACTOR
from common.math_ops import decode_ue8m0, pack_ue8m0_row_major

VARIANT = "pto/cast_back/03_packed_ue8m0"
LANES = 64
EXP_MASK = 0x7F800000


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with packed-UE8M0 scales."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
    num_tokens = T.dynamic("num_tokens")
    sf_pad = max(num_words, 16)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_words), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((sf_pad,), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_words])
                # --- BEGIN SOLUTION hint="mask_low = V.create_mask(32, size=64); sf_shift = V.vsel(mask_low, V.vbrc(T.uint32(23), size=64), V.vbrc(T.uint32(15), size=64)) -- note the mask comes FIRST in V.vsel; per strip broadcast the word with V.vload(sf_ub[strip], size=128, stride=1, dist_mode='brc', group=1), V.vinterpret_cast to 'uint32', V.vshl by sf_shift, V.vand with the exponent mask, then vinterpret_cast to 'float32'"
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    mask_low = V.create_mask(32, size=LANES)
                    # VMI puts the mask first: vsel(mask, if_true, if_false).
                    sf_shift = V.vsel(mask_low,
                                      V.vbrc(T.uint32(23), size=LANES),
                                      V.vbrc(T.uint32(15), size=LANES))
                    exp_mask = V.vbrc(T.uint32(EXP_MASK), size=LANES)
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")

                        # 128 uint16 lanes = 64 uint32 lanes after the cast; the
                        # lane count is derived, not spelled.
                        packed = V.vload(sf_ub[strip], size=LANES * 2, stride=1,
                                         dist_mode="brc", group=1)
                        bits = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"),
                                             sf_shift), exp_mask)
                        scale = V.vinterpret_cast(bits, "float32")

                        scaled = V.vmul(values, scale, mask)
                        V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf_packed.view(torch.uint16))
    status.assert_on_device("cast_back 03", out)
    return out


def demo_numbers() -> None:
    e8m0 = torch.tensor([[127, 120]], dtype=torch.uint8)
    word = pack_ue8m0_row_major(e8m0)[0, 0].item() & 0xFFFF
    print(f"[demo] exponents [127, 120] pack to 0x{word:04X}")
    got = []
    for shift, which in ((23, "low byte  -> group 0"), (15, "high byte -> group 1")):
        bits = ((word | (word << 16)) << shift) & EXP_MASK
        val = torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
        got.append(val)
        print(f"[demo]   shift {shift:2d} -> 0x{bits:08X} -> {val:g}   ({which})")
    assert got == decode_ue8m0(e8m0)[0].tolist()
    print("[demo] identical bit trick to the ASC variant; VMI just spells the")
    print("[demo] reinterpret without a lane count and puts the mask first in vsel")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    e8m0 = torch.randint(100, 140, (m, k // CANONICAL_G), dtype=torch.uint8)
    packed = pack_ue8m0_row_major(e8m0)
    ref = oracle.cast_back(q, packed, (1, CANONICAL_G), packed=True,
                           out_dtype=torch.bfloat16)
    got = launch(q.npu(), packed.npu()).cpu()
    assert_bf16_near(got, ref, f"cast_back_packed({m},{k})", atol=0.0)
    print(f"[check] shape=({m},{k}) matches the torch oracle exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "cast_back", "03_packed_ue8m0")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
