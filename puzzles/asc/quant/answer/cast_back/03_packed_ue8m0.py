"""cast_back 03 (ASC) -- decode packed UE8M0 scales inside the vector unit.

The scales now arrive as one byte each, two bytes fused per int16 word (see
puzzles/torch/.../03_packed_ue8m0.py for the format). The kernel has to turn an
exponent byte into a float32 multiplier, and it does so without any arithmetic:

    a float32 is   sign | exponent(8) | mantissa(23)
    so placing byte e at bit 23, with a zero mantissa, *is* the number 2^(e-127)

That is a shift and a mask, no divide and no exp2:

    scale_bits = (word << shift) & 0x7F800000      -> reinterpret as float32

### Extracting the right byte with a per-lane shift

One int16 word holds two exponents: the low byte is the even group, the high byte
the odd one. A 64-lane strip spans exactly those two groups, so lanes 0-31 need
the low byte and lanes 32-63 need the high byte -- from the *same* word.

`BRC_B16` broadcasts the 16-bit word across the register; reinterpreting the
result as uint32 gives each lane `word | word << 16`. Then:

    shift by 23  ->  bits 0-7  (the low byte) land on bits 23-30
    shift by 15  ->  bits 8-15 (the high byte) land on bits 23-30

and the mask keeps only bits 23-30. So the byte selection is just a *different
shift amount per lane*, built once with a select:

    sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low)

A vector shift where each lane shifts by its own amount is a real instruction
here (`S.vshl` with a vector operand), which is what makes this work in one pass.

### Lazy vs eager decode

This is production's *lazy* path: the packed words stay in UB and are decoded
inside the multiply loop. The alternative is to decode once per token into a
float32 scratch buffer and then reuse variant 01 unchanged -- simpler to read,
but it costs an extra UB write, an extra read, and a memory barrier between them.
Production keeps lazy for exactly that reason (`use_lazy_packed_scale`).

Run:  python puzzles/asc/quant/answer/cast_back/03_packed_ue8m0.py
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
from common.consts import CANONICAL_G, PACK_FACTOR
from common.math_ops import decode_ue8m0, pack_ue8m0_row_major

VARIANT = "asc/cast_back/03_packed_ue8m0"
LANES = 64
EXP_MASK = 0x7F800000       # the float32 exponent field


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> bfloat16 with packed-UE8M0 scales."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR       # two exponents per int16
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
                # --- BEGIN SOLUTION hint="build sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low) and exp_mask = S.vdup(0x7F800000, T.uint32); per strip, broadcast the word with S.vld(sf_ub[strip], dist='BRC_B16'), T.reinterpret it to 'uint32x64', S.vshl by sf_shift, S.vand with exp_mask, reinterpret to 'float32x64'; then multiply and store as in variant 01"
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    # Lanes 0-31 extract the low byte (shift 23); lanes 32-63
                    # extract the high byte (shift 15). One vector, built once.
                    sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low)
                    exp_mask = S.vdup(EXP_MASK, T.uint32)
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        values = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"), T.float32)

                        # One word covers both groups this strip spans.
                        packed = S.vld(sf_ub[strip], dist="BRC_B16")
                        bits = S.vand(S.vshl(T.reinterpret(packed, "uint32x64"),
                                             sf_shift), exp_mask)
                        # An exponent field with a zero mantissa *is* 2^(e-127).
                        scale = T.reinterpret(bits, "float32x64")

                        scaled = S.vmul(values, scale)
                        S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16), dist="PK_B32")
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf_packed: torch.Tensor) -> torch.Tensor:
    """`sf_packed` is int16; the kernel reads it as uint16 for the shifts."""
    out = compile_kernel(q.shape[1])(q, sf_packed.view(torch.uint16))
    status.assert_on_device("cast_back 03", out)
    return out


def demo_numbers() -> None:
    e8m0 = torch.tensor([[127, 120]], dtype=torch.uint8)
    word = pack_ue8m0_row_major(e8m0)[0, 0].item() & 0xFFFF
    print(f"[demo] exponents [127, 120] pack to the int16 word 0x{word:04X} = {word}")
    for shift, which in ((23, "low byte  -> group 0"), (15, "high byte -> group 1")):
        bits = ((word | (word << 16)) << shift) & EXP_MASK
        val = torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
        print(f"[demo]   (word << {shift:2d}) & 0x7F800000 = 0x{bits:08X} -> {val:g}   "
              f"({which})")
    expect = decode_ue8m0(e8m0)[0].tolist()
    print(f"[demo] torch decode agrees: {expect}")
    got = [
        torch.tensor([((word | (word << 16)) << s) & EXP_MASK],
                     dtype=torch.int32).view(torch.float32).item()
        for s in (23, 15)
    ]
    assert got == expect, (got, expect)
    print("[demo] no divide, no exp2 -- a shift and a mask")


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
    print(f"[check] shape=({m},{k}) matches the torch oracle exactly "
          f"({packed.numel()} int16 words for {e8m0.numel()} scales)")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "cast_back", "03_packed_ue8m0")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
