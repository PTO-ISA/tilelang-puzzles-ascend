"""per_token 06 -- the sf_only / cast_only split, and requantization.

Two related configs, both about *reusing* scales rather than always computing
them alongside the values.

### sf_only and cast_only

Production can run this kernel in three modes:

    full       compute amax -> scale -> quantized values      (variants 01-05)
    sf_only    compute amax -> scale, skip the values
    cast_only  scales are given; apply them, no amax pass at all

``sf_only`` exists because a caller may need to know the scales before deciding
how to process the values. ``cast_only`` exists for the other half of that
split, and also for requantizing with scales someone else chose.

A subtle numerical point: ``cast_only`` is **not always bit-identical** to the
full path on the same input. The full path forms the multiplier straight from
amax as ``448/amax``. ``cast_only`` only has the stored, already-rounded scale,
so it must compute ``1/sf``. Those differ in the last bit or two and can flip an
occasional FP8 code. With ``round_sf`` the difference vanishes entirely, because
a power of two and its reciprocal are both exact -- one more reason production
prefers power-of-two scales. The test below demonstrates both cases.

### requant

``in_config.with_sf``: the input is *already* quantized and carries its own
scales. To requantize (to a different format, or a different group size) you
must dequantize first:

    x        = decode(q_in) * sf_in          (dequantize)
    q, sf    = quantize(x)                   (then the ordinary path)

This cannot be fused into one pass, because the new amax is not known until the
whole group has been dequantized. On the NPU that is visible as a scratch UB
buffer holding the dequantized values between the two stages.

Run:  python puzzles/torch/quant/answer/per_token/06_split_requant.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near

VARIANT = "torch/per_token/06_split_requant"


def torch_sf_only(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Compute only the scale factors."""
    # TODO: amax over each group, clamp, divide by E4M3_MAX; return just sf
    raise NotImplementedError("torch/per_token/06_split_requant: implement torch_sf_only")


def torch_cast_only(x: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize using scales that are given, with no amax pass.

    Mirror the kernel: take the reciprocal of the stored scale and multiply.
    """
    # TODO: sf_inv = 1.0 / sf (not E4M3_MAX/amax -- we only have sf); multiply
    #       each group by sf_inv.unsqueeze(-1) and cast to float8_e4m3fn
    raise NotImplementedError("torch/per_token/06_split_requant: implement torch_cast_only")


def torch_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                  group_size: int = CANONICAL_G):
    """Dequantize an already-quantized input, then quantize it again."""
    # TODO: dequantize with oracle.cast_back(q_in, sf_in, (1, group_size),
    #       out_dtype=float32), then run the ordinary variant-01 quantize on the
    #       result
    raise NotImplementedError("torch/per_token/06_split_requant: implement torch_requant")


def demo_numbers() -> None:
    torch.manual_seed(0)
    x = randn_with_zero_row(8, 64, torch.device("cpu"))

    sf = torch_sf_only(x)
    print(f"[demo] sf_only -> scales {tuple(sf.shape)} only, no values computed")

    # The two multipliers differ by a ULP or two *always*; whether that flips an
    # FP8 code depends on the input. Show both halves of that.
    probe = torch.randn(1024, 128)
    grouped = probe.view(1024, -1, CANONICAL_G)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    mul_full = E4M3_MAX / amax            # what the full path forms
    mul_cast = 1.0 / (amax / E4M3_MAX)    # what cast_only is forced to form
    ulp = (mul_full.view(torch.int32) - mul_cast.view(torch.int32)).abs()
    print(f"[demo] the two multipliers differ in {int((ulp > 0).sum())}/{ulp.numel()} "
          f"groups, by up to {int(ulp.max())} ULP")
    print("[demo]   full:      448 / amax")
    print("[demo]   cast_only: 1 / sf, and sf is the already-rounded amax / 448")

    for dtype in (torch.bfloat16, torch.float32):
        x_s = torch.randn(1024, 128, dtype=dtype)
        q_full, sf_full = oracle.per_token(x_s, CANONICAL_G)
        d = (torch_cast_only(x_s, sf_full).view(torch.uint8).int()
             - q_full.view(torch.uint8).int()).abs()
        print(f"[demo] {str(dtype):16} input -> {int((d > 0).sum()):4}/{d.numel()} "
              f"FP8 codes flip (max {int(d.max())})")
    print("[demo]   so the divergence is real but tiny, and input-dependent")

    q_p2, sf_p2 = oracle.per_token(probe, CANONICAL_G, round_sf=True)
    d_p2 = (torch_cast_only(probe, sf_p2).view(torch.uint8).int()
            - q_p2.view(torch.uint8).int()).abs()
    print(f"[demo] with round_sf             -> {int((d_p2 > 0).sum())}/{d_p2.numel()} flip")
    print("[demo]   a power of two and its reciprocal are both exact, so they agree")
    assert int((ulp > 0).sum()) > 0, "the multipliers really should differ"
    assert int((d_p2 > 0).sum()) == 0, "round_sf should make cast_only bit-identical"


def test_correctness() -> None:
    torch.manual_seed(0)
    x = randn_with_zero_row(32, 128, torch.device("cpu"))

    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G)
    assert torch.equal(torch_sf_only(x), ref_sf), "sf_only must match the full path's scales"
    print("[check] sf_only matches the full path exactly")

    assert torch.equal(torch_cast_only(x, ref_sf).view(torch.uint8),
                       oracle.per_token_cast_only(x, ref_sf, CANONICAL_G).view(torch.uint8))
    print("[check] cast_only matches the oracle")

    q2, sf2 = torch_requant(ref_q, ref_sf)
    rq, rsf = oracle.requant_per_token(ref_q, ref_sf, CANONICAL_G)
    assert_fp8_near(q2, rq, "requant q")
    assert torch.equal(sf2, rsf), "requant scales must match the oracle"
    # requantizing an already-quantized tensor at the same granularity is lossless
    b1 = oracle.cast_back(ref_q, ref_sf, (1, CANONICAL_G), out_dtype=torch.float32)
    b2 = oracle.cast_back(q2, sf2, (1, CANONICAL_G), out_dtype=torch.float32)
    err = (b2 - b1).abs().max().item()
    print(f"[check] requant at the same granularity is lossless: max-abs-err {err:.3e}")
    assert err == 0.0, err


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
