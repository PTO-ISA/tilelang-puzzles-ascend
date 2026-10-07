"""per_token 03 -- store the scale as one packed UE8M0 byte.

New config: ``use_packed_ue8m0``. This is the payoff of variant 02: now that
the scale is guaranteed to be a power of two, its mantissa is always zero, so
storing the exponent alone loses nothing.

    variant 02:  sf is float32           4 bytes per scale
    variant 03:  sf is uint8 exponent    1 byte per scale    (4x smaller)
                 then packed 2-per-int16 for the DMA engine  -> (M, K/32/2) int16

    stored byte = exp + 127        (float32's exponent bias)

Why pack at all: a single byte is an awkward DMA width. Ascend fuses two bytes
into one int16; CUDA fuses four into an int32. The factor is a per-target
constant -- ``PACK_FACTOR`` is 2 here -- and it is the reason the public scale
shape has that extra division in it.

Worked example, two groups whose scales are 2^0 and 2^-7:

    exponents     [  0,  -7 ]
    stored bytes  [127, 120 ]
    packed int16  127 | (120 << 8) = 30847

The decode is a shift, as in cast_back 03: ``byte << 23`` reinterpreted as
float32 *is* 2^(byte-127).

Note this variant only makes sense with ``round_sf`` on. A raw ``amax/448``
scale has a nonzero mantissa, and UE8M0 has nowhere to put it.

Run:  python puzzles/torch/quant/answer/per_token/03_packed_ue8m0.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_same_bytes
from common.math_ops import ceil_log2_exp, decode_packed_ue8m0, inv_pow2_from_exp, pack_ue8m0_row_major

VARIANT = "torch/per_token/03_packed_ue8m0"


def torch_per_token_cast_packed(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with packed-UE8M0 scales. Returns ``(q, sf_packed)``.

    ``sf_packed`` is (M, K/group_size/PACK_FACTOR) int16.
    """
    # --- BEGIN SOLUTION hint="as variant 02, but instead of pow2_from_exp store (exp + 127) as uint8 and call pack_ue8m0_row_major on it"
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped * inv_pow2_from_exp(exp_sf).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    e8m0 = (exp_sf + 127).to(torch.uint8)
    return q, pack_ue8m0_row_major(e8m0)
    # --- END SOLUTION


def demo_numbers() -> None:
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0] = 448.0        # group 0: amax=448 -> sf = 2^0
    x[0, 32] = 3.5         # group 1: amax=3.5 -> sf = 2^-7
    q, packed = torch_per_token_cast_packed(x)
    decoded = decode_packed_ue8m0(packed)
    print("[demo] two groups, scales 2^0 and 2^-7")
    print(f"[demo]   packed int16 = {packed[0, 0].item()}")
    print(f"[demo]   decoded scales = {decoded[0].tolist()}")
    print_example("per_token 03", q=q.float()[:, [0, 32]], packed=packed)
    assert decoded[0, 0].item() == 1.0, decoded
    assert abs(decoded[0, 1].item() - 2.0 ** -7) < 1e-9, decoded
    fp32_bytes = 2 * 4
    print(f"[demo] scale storage: {fp32_bytes} bytes as float32 -> "
          f"{packed.numel() * 2} bytes packed ({fp32_bytes // (packed.numel() * 2)}x)")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 128)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, packed = torch_per_token_cast_packed(x)
        ref_q, ref_packed = oracle.per_token(x, CANONICAL_G, round_sf=True, packed=True)
        assert_same_bytes(packed, ref_packed, f"sf_packed({m},{k})")
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        # the packed scales must decode to the same powers of two as variant 02
        _, sf_f32 = oracle.per_token(x, CANONICAL_G, round_sf=True)
        assert torch.equal(decode_packed_ue8m0(packed), sf_f32), "decode must round-trip"
        print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} int16 ok, "
              f"decodes back to the float32 scales exactly")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
