"""cast_back 06 -- the quantized input is packed FP4 (e2m1), two values per byte.

New config: ``in_dtype = float4_e2m1fn``. Alongside UE8M0 scale packing, this is
the other pure memory-saving feature in the ladder, and the bigger one: FP4
halves the activation footprint relative to FP8.

e2m1 is 4 bits: 1 sign, 2 exponent, 1 mantissa. There is no FP4 scalar type in
torch, so a (M, K) FP4 tensor is stored as (M, K/2) int8 with two nibbles per
byte, low nibble first.

Decoding one nibble ``s eee m`` -> wait, it is ``s ee m``:

    bit 3    sign
    bits 2-1 exponent (2 bits, bias 1)
    bit 0    mantissa (1 bit)

    e == 0   subnormal:  value = +- 0.5 * m
    e  > 0   normal:     value = +- (1 + 0.5 * m) * 2^(e - 1)

The complete set of 16 codes is therefore
``{0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6}`` -- only 15 distinct values, and
the largest is 6.0. That is why the FP4 quantizer uses ``quant_max = 6.0``
where FP8 uses 448.0.

Accuracy cost: one mantissa bit means ~15% relative error after a round trip,
versus ~3% for FP8. FP4 is used where the model tolerates it.

Run:  python puzzles/torch/quant/answer/cast_back/06_fp4_e2m1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from common.demo import print_example
from common.check import assert_bf16_near
from common.math_ops import unpack_e2m1_bytes

VARIANT = "torch/cast_back/06_fp4_e2m1"


def torch_cast_back_fp4(q_packed: torch.Tensor, sf: torch.Tensor,
                        group_size: int = CANONICAL_G):
    """Dequantize packed-FP4 input.

    ``q_packed`` is (M, K/2) int8; the logical width is K = 2 * q_packed.shape[1].
    """
    # TODO: unpack_e2m1_bytes(q_packed) gives (M, K) float32; then scale per group
    #       exactly as variant 01
    raise NotImplementedError("torch/cast_back/06_fp4_e2m1: implement torch_cast_back_fp4")


def demo_numbers() -> None:
    print("[demo] the complete e2m1 code table")
    codes = torch.arange(16, dtype=torch.int16)
    packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.int8).view(1, -1)
    values = unpack_e2m1_bytes(packed)[0]
    for c, v in zip(range(16), values.tolist()):
        print(f"[demo]   0x{c:X}  s={c >> 3} e={(c >> 1) & 3} m={c & 1}  ->  {v:g}")
    distinct = sorted({abs(v) for v in values.tolist()})
    print(f"[demo] distinct magnitudes: {distinct}  (max {max(distinct)})")
    assert max(distinct) == 6.0, distinct
    print("[demo] max magnitude 6.0 -- hence quant_max=6.0 for FP4, vs 448.0 for FP8")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 64)):
        x = torch.randn(m, k) * 3
        q_packed, sf = oracle.per_token(x, CANONICAL_G, fmt="e2m1")
        assert q_packed.shape == (m, k // 2), q_packed.shape
        got = torch_cast_back_fp4(q_packed, sf)
        ref = oracle.cast_back(q_packed, sf, (1, CANONICAL_G), fp4=True,
                               out_dtype=torch.bfloat16)
        assert_bf16_near(got, ref, f"cast_back_fp4({m},{k})", atol=0.0)
        rel = (got.float() - x).abs().max().item() / x.abs().max().item()
        print(f"[check] shape=({m},{k}) packed={tuple(q_packed.shape)} ok, "
              f"round-trip rel-err {rel:.1%}")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
