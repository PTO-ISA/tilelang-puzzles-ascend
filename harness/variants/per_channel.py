"""per_channel variant checks.

The reduction runs along M, so the scale array is (M/32, K) -- one scale per
channel. Two variants pack their scales along M, which needs an even number of
token groups, so their shape rule rounds M up to 64.
"""

from __future__ import annotations

import torch

from harness import oracle
from harness.asserts import assert_fp8_near, assert_fp32_ulps, assert_same_bytes
from harness.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR
from harness.demo import randn_with_zero_row
from harness.math_ops import decode_packed_ue8m0_along_m

from harness.spec import Ctx, Shape, Variant, register

CPU = torch.device("cpu")
G = CANONICAL_G
LANES = 64
PACK_M = BLOCK_MN * PACK_FACTOR        # 64: packing along M needs two groups


def _v01(ctx: Ctx) -> None:
    assert ctx.m % BLOCK_MN == 0, f"needs M a multiple of {BLOCK_MN}, got {ctx.m}"
    assert ctx.k % LANES == 0, f"channels are processed {LANES} at a time"
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_sf = oracle.per_channel(x, BLOCK_MN)
    q, sf = ctx.call(ctx.dev(x))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})",
                     max_ulps=0 if ctx.is_torch else 1)
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf={tuple(sf.shape)} matches the oracle")


def _v02(ctx: Ctx) -> None:
    assert ctx.m % PACK_M == 0, (
        f"packing along M needs an even number of token groups, so M must be a "
        f"multiple of {PACK_M}; got {ctx.m}"
    )
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_packed = oracle.per_channel(x, BLOCK_MN, round_sf=True, packed=True)
    q, packed = ctx.call(ctx.dev(x))
    assert_same_bytes(ctx.host(packed), ref_packed, f"sf_packed({ctx.m},{ctx.k})")
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    _, ref_f32 = oracle.per_channel(x, BLOCK_MN, round_sf=True)
    assert torch.equal(decode_packed_ue8m0_along_m(ctx.host(packed)), ref_f32), (
        "the M-packed bytes must decode back to the float32 scales"
    )
    ctx.note(f"shape=({ctx.m},{ctx.k}) packed={tuple(packed.shape)} byte-exact, "
             f"decodes back along M to the float32 scales")


def _v03(ctx: Ctx) -> None:
    """Requantize per-token-quantized input to per-channel scales.

    The one variant whose FP8 output is deliberately not bit-exact. Requantizing
    input that already sits on the FP8 grid produces *exact ties*, so the two
    implementations break them differently. Rather than loosening a tolerance,
    this asserts both that no difference exceeds one code and that every
    differing position is a genuine midpoint -- so a real defect still fails.
    """
    x = randn_with_zero_row(ctx.m, ctx.k, CPU) * 3
    q_in, sf_in = oracle.per_token(x, G, round_sf=True)
    dq = oracle.cast_back(q_in, sf_in, (1, G), out_dtype=torch.float32)
    ref_q, ref_sf = oracle.per_channel(dq, BLOCK_MN)

    q, sf = ctx.call(ctx.dev(q_in), ctx.dev(sf_in))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})", max_ulps=0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) scales are bit-exact vs torch")

    d = (ctx.host(q).view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    n_diff, worst = int((d > 0).sum()), int(d.max())
    ctx.note(f"FP8 codes differing from torch: {n_diff}/{d.numel()} "
             f"({n_diff / d.numel():.1%}), worst difference {worst} code")
    assert worst <= 1, (
        f"differences of more than one FP8 code ({worst}) are not tie-breaking "
        f"and indicate a real bug"
    )
    # Every difference must be a genuine tie: the exact product lands exactly
    # halfway between two representable codes.
    lo = torch.minimum(ctx.host(q).float(), ref_q.float())
    hi = torch.maximum(ctx.host(q).float(), ref_q.float())
    exact = dq / ctx.host(sf).repeat_interleave(BLOCK_MN, dim=0)
    ties = int(((d > 0) & torch.isclose(exact, (lo + hi) / 2, rtol=1e-6)).sum())
    ctx.note(f"of those, {ties} are exact ties (the product is the midpoint of "
             f"two FP8 codes)")
    assert ties == n_diff, (
        "some differences are not ties, so this is not just tie-breaking"
    )

    b1 = oracle.cast_back(q_in, sf_in, (1, G), out_dtype=torch.float32)
    b2 = oracle.cast_back(ctx.host(q), ctx.host(sf), (BLOCK_MN, 1),
                          out_dtype=torch.float32)
    e1 = (b1 - x).abs().max().item() / x.abs().max().item()
    e2 = (b2 - x).abs().max().item() / x.abs().max().item()
    ctx.note(f"error vs the original: after 1 quantization {e1:.1%}, "
             f"after requantization {e2:.1%}")
    assert e2 >= e1, "requantizing cannot recover precision"


def _v04(ctx: Ctx) -> None:
    assert ctx.m % PACK_M == 0, f"needs M a multiple of {PACK_M}, got {ctx.m}"
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_packed = oracle.per_channel(x, BLOCK_MN, round_sf=True, packed=True)
    q, packed = ctx.call(ctx.dev(x))
    assert_same_bytes(ctx.host(packed), ref_packed, f"sf_packed({ctx.m},{ctx.k})")
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    _, ref_f32 = oracle.per_channel(x, BLOCK_MN, round_sf=True)
    assert torch.equal(decode_packed_ue8m0_along_m(ctx.host(packed)), ref_f32), (
        "the bfloat16 reduction changed the chosen power-of-two exponent"
    )
    ctx.note(f"shape=({ctx.m},{ctx.k}) packed={tuple(packed.shape)} byte-exact, "
             f"and the bfloat16 reduction picked the same exponents as float32")


register(
    Variant("per_channel", "01_raw_32tokens", _v01,
            torch_fn="torch_per_channel_cast",
            torch_shapes=((32, 128), (64, 64), (32, 256))),
    Variant("per_channel", "02_round_packed_m", _v02,
            torch_fn="torch_per_channel_cast_packed",
            shape=Shape(m_multiple=PACK_M),
            torch_shapes=((64, 128), (128, 64))),
    Variant("per_channel", "03_requant_bf16", _v03,
            torch_fn="torch_per_channel_requant",
            torch_shapes=((32, 128), (64, 128))),
    Variant("per_channel", "04_compose", _v04,
            torch_fn="torch_per_channel_compose",
            shape=Shape(m_multiple=PACK_M),
            torch_shapes=((64, 128), (128, 256))),
)
