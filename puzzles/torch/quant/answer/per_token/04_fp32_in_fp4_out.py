"""per_token 04 -- float32 input, FP4 (e2m1) output.

New config: the dtype axis at both ends. The algorithm does not change at all;
only the types do. Two independent knobs:

**float32 input.** On the NPU this is the cheaper case, oddly: a float32 value is
already the vector unit's native compute type, so the load needs no conversion.
A bfloat16 input has to be widened first. (The flip side is bandwidth: float32
input moves twice the bytes.)

**FP4 e2m1 output.** The quantization target changes from 448.0 to **6.0**,
because that is the largest magnitude e2m1 can represent (see cast_back 06 for
the full 16-code table). Everything else is identical:

    sf = amax / 6.0           instead of   amax / 448.0
    q  = cast_e2m1(x * (6.0 / amax))

The clamp floor changes too. For FP8 it is 1e-4; for FP4 it is
``6.0 * 2^-126``, chosen so ``sf`` stays normal rather than denormal.

Accuracy: e2m1 has a single mantissa bit, so a round trip lands around 11-15%
relative error versus ~3% for e4m3. FP4 buys a 2x memory saving over FP8 and a
4x saving over bfloat16, and you pay for it in precision.

Storage: there is no FP4 scalar type in torch, so (M, K) logical values come back
as (M, K/2) int8 with two nibbles per byte.

Run:  python puzzles/torch/quant/answer/per_token/04_fp32_in_fp4_out.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_same_bytes
from common.math_ops import pack_e2m1_from_fp32, unpack_e2m1_bytes

VARIANT = "torch/per_token/04_fp32_in_fp4_out"


def torch_per_token_cast_fp4(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize a float32 input to packed FP4. Returns ``(q_packed, sf)``."""
    # --- BEGIN SOLUTION hint="same shape as variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN instead of the e4m3 constants, and pack_e2m1_from_fp32(quant) instead of .to(float8_e4m3fn)"
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E2M1_CLAMP_MIN)
    sf = amax / E2M1_MAX
    quant = (grouped * (E2M1_MAX / amax).unsqueeze(-1)).view(m, k)
    return pack_e2m1_from_fp32(quant), sf
    # --- END SOLUTION


def demo_numbers() -> None:
    x = torch.zeros(1, 32, dtype=torch.float32)
    x[0, 0:4] = torch.tensor([6.0, 3.0, -1.5, 0.5])
    q_packed, sf = torch_per_token_cast_fp4(x)
    values = unpack_e2m1_bytes(q_packed)
    print("[demo] one group, amax=6.0 so the scale is exactly 1.0")
    print(f"[demo]   sf={sf[0, 0].item():g}")
    print_example("per_token 04", x=x[:, :4], fp4_values=values[:, :4])
    assert sf[0, 0].item() == 1.0, sf
    assert torch.allclose(values[0, :4], torch.tensor([6.0, 3.0, -1.5, 0.5]))
    print("[demo] these four values are all exactly representable in e2m1")
    # 0.75 would not be: it is the midpoint of 0.5 and 1.0 and rounds away.
    probe = torch.zeros(1, 32, dtype=torch.float32)
    probe[0, 0:2] = torch.tensor([6.0, 0.75])
    pq, _ = torch_per_token_cast_fp4(probe)
    got = unpack_e2m1_bytes(pq)[0, 1].item()
    print(f"[demo] 0.75 is not in the format: it quantizes to {got:g}")
    print(f"[demo] storage: {x.numel() * 4} bytes float32 -> "
          f"{q_packed.numel()} bytes FP4 ({x.numel() * 4 // q_packed.numel()}x smaller)")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 64)):
        x = randn_with_zero_row(m, k, torch.device("cpu"), dtype=torch.float32) * 3
        q_packed, sf = torch_per_token_cast_fp4(x)
        ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, fmt="e2m1")
        assert_same_bytes(q_packed, ref_q, f"q_packed({m},{k})")
        assert torch.equal(sf, ref_sf), "scales must match the oracle exactly"
        back = oracle.cast_back(q_packed, sf, (1, CANONICAL_G), fp4=True,
                                out_dtype=torch.float32)
        rel = (back - x).abs().max().item() / x.abs().max().item()
        print(f"[check] shape=({m},{k}) packed={tuple(q_packed.shape)} ok, "
              f"round-trip rel-err {rel:.1%}")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
