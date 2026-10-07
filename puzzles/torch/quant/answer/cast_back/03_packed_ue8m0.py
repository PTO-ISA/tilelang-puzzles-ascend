"""cast_back 03 -- scales arrive as packed UE8M0 bytes, not float32.

New config: ``use_packed_ue8m0``. This is a memory-footprint change, and it is
one of the two features in this ladder that exist purely to save bandwidth
(the other is FP4).

A power-of-two scale needs no mantissa at all: storing the exponent is enough.
UE8M0 is exactly that -- an unsigned 8-bit exponent, no sign, no mantissa:

    stored byte  e = exponent + 127          (same bias as float32)
    decoded      scale = 2^(e - 127)

So one scale costs **1 byte instead of 4**. Ascend then packs two of those bytes
into one int16 word, because that is the narrowest type the DMA engine moves
efficiently (CUDA packs four into an int32 -- the pack factor is a per-target
constant, 2 here).

Decoding is a shift, not a divide: an fp32 bit pattern is
``sign | exponent(8) | mantissa(23)``, so placing the byte at bit 23 with a zero
mantissa *is* the float:

    scale_bits = e << 23        ->  reinterpret as float32  ->  2^(e-127)

Worked example:

    e = 120  ->  2^(120-127) = 2^-7 = 0.0078125
    e = 127  ->  2^0         = 1.0
    e = 105  ->  2^-22       (what a clamped all-zero group ends up as)

Note what UE8M0 cannot represent: zero. A byte of 0x00 decodes to 2^-127, not 0,
which is why the quantizer clamps amax from below instead of letting a zero group
produce a zero scale.

Run:  python puzzles/torch/quant/answer/cast_back/03_packed_ue8m0.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from common.demo import print_example
from common.check import assert_bf16_near
from common.math_ops import decode_packed_ue8m0, decode_ue8m0, pack_ue8m0_row_major

VARIANT = "torch/cast_back/03_packed_ue8m0"


def torch_cast_back_packed(q: torch.Tensor, sf_packed: torch.Tensor,
                           group_size: int = CANONICAL_G):
    """Dequantize with packed-UE8M0 scales.

    ``sf_packed`` is (M, K/group_size/2) int16; each word holds two exponent
    bytes, low byte first.
    """
    # --- BEGIN SOLUTION hint="decode_packed_ue8m0(sf_packed) gives (M, K/G) float32 scales; then dequantize exactly as variant 01"
    m, k = q.shape
    assert k % group_size == 0
    scale = decode_packed_ue8m0(sf_packed)
    grouped = q.float().view(m, k // group_size, group_size)
    return (grouped * scale.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    # --- END SOLUTION


def demo_numbers() -> None:
    print("[demo] UE8M0 decode table")
    for e in (105, 120, 127, 134):
        val = decode_ue8m0(torch.tensor([e], dtype=torch.uint8)).item()
        print(f"[demo]   byte {e:3d} -> 2^{e - 127:<4d} = {val:g}")

    # two groups with exponents 127 (scale 1.0) and 120 (scale 2^-7)
    e8m0 = torch.tensor([[127, 120]], dtype=torch.uint8)
    packed = pack_ue8m0_row_major(e8m0)
    q = torch.zeros(1, 64)
    q[0, 0] = 2.0
    q[0, 32] = 2.0
    out = torch_cast_back_packed(q.to(torch.float8_e4m3fn), packed)
    print(f"[demo] packed int16 word = {packed.item()} "
          f"(= {e8m0[0, 0].item()} | {e8m0[0, 1].item()} << 8)")
    print_example("cast_back 03", packed=packed, out=out[:, [0, 32]])
    assert abs(out[0, 0].item() - 2.0) < 1e-3, out[0, 0]
    assert abs(out[0, 32].item() - 2.0 * 2 ** -7) < 1e-6, out[0, 32]
    print("[demo] same quantized value, two scales, two results")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 128)):
        q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
        e8m0 = torch.randint(100, 140, (m, k // CANONICAL_G), dtype=torch.uint8)
        packed = pack_ue8m0_row_major(e8m0)
        got = torch_cast_back_packed(q, packed)
        ref = oracle.cast_back(q, packed, (1, CANONICAL_G), packed=True,
                               out_dtype=torch.bfloat16)
        assert_bf16_near(got, ref, f"cast_back_packed({m},{k})", atol=0.0)
        print(f"[check] shape=({m},{k}) ok ({packed.numel()} int16 words "
              f"vs {m * k // CANONICAL_G} float32 scales)")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
