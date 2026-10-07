"""per_token 01 -- quantize to FP8, one FP32 scale per 32 channels.

This is the first kernel in the ladder with a **reduction** in it, and that is
the one new idea. ``cast_back`` was given its scales; here we have to compute
them, which means looking at all 32 values in a group before we can write any of
them.

    x  : (M, K)      bfloat16
    q  : (M, K)      float8_e4m3fn
    sf : (M, K/32)   float32

    amax = max(|x|) over each group of 32 channels
    amax = max(amax, 1e-4)                    <- see below
    sf   = amax / 448.0                       (448 = largest finite e4m3)
    q    = cast_e4m3(x * (448.0 / amax))

The scale maps each group's largest magnitude onto the top of the FP8 range, so
every group gets the format's full precision regardless of its magnitude. That
is the entire point of block-wise quantization.

Worked example, one token, K = 64, so two groups of 32:

    group 0:  x[0, 0:4] = [1.0, 2.0, -4.0, 0.5],  rest 0
              amax = 4.0,  sf = 4/448 = 0.00892857,  448/4 = 112
              q[0, 0:4] = [112, 224, -448, 56]

    The largest magnitude lands exactly on -448, the e4m3 limit. Nothing
    saturates, nothing is wasted.

Why clamp amax at 1e-4: an all-zero group would give sf_inv = 448/0 = Inf and
then 0 * Inf = NaN, poisoning the output. The clamp costs nothing and makes a
zero row harmless. The test deliberately zeroes row 0 to exercise it.

Two passes are unavoidable. The reduction has to finish before the scale exists,
so the data is read twice (or held onto). On the NPU that shows up as a memory
barrier between the two passes -- see the ASC/PTO 01 files.

Run:  python puzzles/torch/quant/answer/per_token/01_raw_fp32sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_fp32_ulps

VARIANT = "torch/per_token/01_raw_fp32sf"


def torch_per_token_cast(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Return ``(q, sf)``: FP8 values and one FP32 scale per group of channels."""
    # TODO: view x as (M, K//G, G); amax = abs().amax(-1) clamped to
    #       E4M3_CLAMP_MIN; sf = amax/E4M3_MAX; q = (grouped *
    #       (E4M3_MAX/amax).unsqueeze(-1)) cast to float8_e4m3fn
    raise NotImplementedError("torch/per_token/01_raw_fp32sf: implement torch_per_token_cast")


def demo_numbers() -> None:
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0:4] = torch.tensor([1.0, 2.0, -4.0, 0.5])
    q, sf = torch_per_token_cast(x)
    print("[demo] one token, K=64, two groups of 32")
    print_example("per_token 01", x=x[:, :4], sf=sf, q=q[:, :4])
    print(f"[demo] group 0: amax=4.0 -> sf={sf[0, 0].item():.8f} (=4/448), "
          f"inverse={E4M3_MAX / 4.0:g}")
    assert torch.allclose(q[0, :4].float(), torch.tensor([112.0, 224.0, -448.0, 56.0]))
    print("[demo] largest magnitude maps exactly onto -448, the e4m3 limit")

    zero = torch.zeros(1, 32, dtype=torch.bfloat16)
    qz, sfz = torch_per_token_cast(zero)
    print(f"[demo] all-zero group: sf={sfz[0, 0].item():.3e} (clamped), "
          f"q has NaN: {bool(qz.float().isnan().any())}")
    assert not qz.float().isnan().any(), "the clamp is what prevents 0 * Inf = NaN"


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 64), (64, 256)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf = torch_per_token_cast(x)
        ref_q, ref_sf = oracle.per_token(x, CANONICAL_G)
        assert_fp32_ulps(sf, ref_sf, f"sf({m},{k})", max_ulps=0)
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} ok")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
