"""per_token 02 -- round the scale up to a power of two.

New config: ``round_sf``. The scale stops being ``amax/448`` and becomes the
smallest power of two that is at least that large:

    sf = 2^ceil(log2(amax / 448))

Two reasons this matters, and the second is the real one:

1. Multiplying by a power of two is exact -- it only changes the exponent field,
   so the quantization step introduces no extra rounding of its own.
2. A power-of-two scale needs no mantissa, so it can be stored as a single
   exponent byte. That is variant 03, and it is where the memory saving comes
   from. This variant is the arithmetic that makes 03 possible.

The cost: rounding the scale *up* means the group's largest value no longer lands
on 448 but somewhere in [224, 448], so up to one bit of range goes unused.

### The ceil-log2 bit trick

Kernels do not call ``log2``. For a positive float32 with bit pattern
``sign | exp(8) | mantissa(23)``:

    exp_field = bits >> 23          gives floor(log2(v)) + 127

``floor`` is the wrong direction -- it would make the scale too small and let
values saturate past 448. Subtracting 1 before the shift fixes it:

    exp = ((bits - 1) >> 23) + 1 - 127        == ceil(log2(v))

Check it by hand:

    v = 1.0   bits = 0x3F800000, minus 1 -> 0x3F7FFFFF, >>23 = 126, +1-127 =  0   2^0 = 1    exact
    v = 1.5   bits = 0x3FC00000, minus 1 -> 0x3FBFFFFF, >>23 = 127, +1-127 =  1   2^1 = 2    rounded up
    v = 2.0   bits = 0x40000000, minus 1 -> 0x3FFFFFFF, >>23 = 127, +1-127 =  1   2^1 = 2    exact
    v = 0.3   bits = 0x3E99999A, minus 1 -> 0x3E999999, >>23 = 125, +1-127 = -1   2^-1 = 0.5 rounded up

An exact power of two stays put; anything else goes up. And the reciprocal is
built by *subtracting* the exponent rather than dividing:

    sf_inv_bits = (254 - (exp + 127)) << 23       i.e. 2^-exp

Run:  python puzzles/torch/quant/answer/per_token/02_round_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_fp32_ulps
from common.math_ops import ceil_log2_exp, inv_pow2_from_exp, pow2_from_exp

VARIANT = "torch/per_token/02_round_sf"


def torch_per_token_cast_round(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Returns ``(q, sf)``, sf still float32."""
    # TODO: amax as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX); sf =
    #       pow2_from_exp(exp); multiply by inv_pow2_from_exp(exp) instead of
    #       dividing
    raise NotImplementedError("torch/per_token/02_round_sf: implement torch_per_token_cast_round")


def demo_numbers() -> None:
    print("[demo] the ceil-log2 bit trick, checked against math")
    import math

    for v in (1.0, 1.5, 2.0, 0.3, 448.0):
        t = torch.tensor([v], dtype=torch.float32)
        got = int(ceil_log2_exp(t).item())
        want = math.ceil(math.log2(v))
        bits = t.view(torch.int32).item() & 0xFFFFFFFF
        print(f"[demo]   v={v:<7g} bits=0x{bits:08X} -> exp={got:<3d} "
              f"(math.ceil(log2) = {want})  2^exp={2.0 ** got:g}")
        assert got == want, (v, got, want)

    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    x[0, 0:3] = torch.tensor([3.0, 1.0, -0.5])
    q_raw, sf_raw = oracle.per_token(x, CANONICAL_G)
    q_p2, sf_p2 = torch_per_token_cast_round(x)
    print(f"[demo] amax=3.0: raw sf={sf_raw[0, 0].item():.8f}  "
          f"pow2 sf={sf_p2[0, 0].item():.8f}")
    print(f"[demo]   raw  q[0,0]={q_raw[0, 0].float().item():g} (lands on 448)")
    print(f"[demo]   pow2 q[0,0]={q_p2[0, 0].float().item():g} (lands lower: range traded for an exact scale)")
    assert sf_p2[0, 0].item() >= sf_raw[0, 0].item()


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 64)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf = torch_per_token_cast_round(x)
        ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, round_sf=True)
        assert_fp32_ulps(sf, ref_sf, f"sf({m},{k})", max_ulps=0)
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        # every scale must be an exact power of two
        mant = sf.view(torch.int32) & 0x7FFFFF
        assert int(mant.abs().max()) == 0, "a power-of-two scale has a zero mantissa"
        print(f"[check] shape=({m},{k}) ok, all scales are exact powers of two")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
