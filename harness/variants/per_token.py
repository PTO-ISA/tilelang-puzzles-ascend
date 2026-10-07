"""per_token variant checks.

per_token is the first kernel with a reduction, so every body here builds one
input tensor and compares two outputs: the quantized values and the scales.

``randn_with_zero_row`` is used throughout rather than plain ``randn`` -- it
zeroes row 0, which exercises the ``clamp_min`` path. Without the clamp an
all-zero group gives 448/0 = Inf and then 0*Inf = NaN, so a test on purely random
data would never notice a missing clamp.
"""

from __future__ import annotations

import torch

from harness import oracle
from harness.asserts import assert_fp8_near, assert_fp32_ulps, assert_same_bytes
from harness.consts import BLOCK_MN, CANONICAL_G
from harness.demo import randn_with_zero_row
from harness.math_ops import decode_packed_ue8m0, unpack_e2m1_bytes

from harness.spec import Ctx, Shape, Variant, register

G = CANONICAL_G
CPU = torch.device("cpu")
BF16_K = 256          # the bfloat16 fast path steps 256 values at a time


def _x(ctx: Ctx, dtype=torch.bfloat16, scale: float = 1.0) -> torch.Tensor:
    x = randn_with_zero_row(ctx.m, ctx.k, CPU, dtype=dtype)
    return x * scale if scale != 1.0 else x


def _fp4_budget(got_packed, ref_packed, ctx: Ctx, sf) -> None:
    """FP4 has 8 magnitudes, so one differing code is a factor of up to 1.5.

    Counting them rather than hiding them behind a tolerance: the kernel's
    two-step float32 -> bfloat16 -> e2m1 rounding should track the bit-exact torch
    packer closely, and in practice matches it exactly.
    """
    got_v, ref_v = unpack_e2m1_bytes(got_packed), unpack_e2m1_bytes(ref_packed)
    differing, total = int((got_v != ref_v).sum()), got_v.numel()
    ctx.note(f"shape=({ctx.m},{ctx.k}) FP4 values differing from the torch packer: "
             f"{differing}/{total} ({differing / total:.2%})")
    assert differing / total < 0.02, (
        f"{differing}/{total} FP4 codes differ; the two-step rounding should track "
        f"the torch packer closely"
    )
    return got_v


# ---------------------------------------------------------------------------
# 01 -- raw float32 scale
# ---------------------------------------------------------------------------

def _v01(ctx: Ctx) -> None:
    x = _x(ctx)
    ref_q, ref_sf = oracle.per_token(x, G)
    q, sf = ctx.call(ctx.dev(x))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})",
                     max_ulps=0 if ctx.is_torch else 1)
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    assert not ctx.host(q).float().isnan().any(), "the zero row must not produce NaN"
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf={tuple(sf.shape)} matches the oracle; "
             f"the all-zero row clamped (no NaN)")


# ---------------------------------------------------------------------------
# 02 -- power-of-two scale
# ---------------------------------------------------------------------------

def _v02(ctx: Ctx) -> None:
    x = _x(ctx)
    ref_q, ref_sf = oracle.per_token(x, G, round_sf=True)
    q, sf = ctx.call(ctx.dev(x))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})", max_ulps=0)
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    # A power of two has a zero mantissa. Checking the bits catches a scale that
    # happens to be close to a power of two without being one.
    mant = ctx.host(sf).view(torch.int32) & 0x7FFFFF
    assert int(mant.abs().max()) == 0, "every scale must be an exact power of two"
    ctx.note(f"shape=({ctx.m},{ctx.k}) bit-exact scales, all powers of two")


# ---------------------------------------------------------------------------
# 03 -- packed UE8M0 scale
# ---------------------------------------------------------------------------

def _v03(ctx: Ctx) -> None:
    x = _x(ctx)
    ref_q, ref_packed = oracle.per_token(x, G, round_sf=True, packed=True)
    q, packed = ctx.call(ctx.dev(x))
    assert_same_bytes(ctx.host(packed), ref_packed, f"sf_packed({ctx.m},{ctx.k})")
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    _, ref_f32 = oracle.per_token(x, G, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(ctx.host(packed)), ref_f32), (
        "the packed bytes must decode back to the float32 scales"
    )
    ctx.note(f"shape=({ctx.m},{ctx.k}) packed={tuple(packed.shape)} byte-exact, "
             f"decodes back to the float32 scales")


# ---------------------------------------------------------------------------
# 04 -- float32 input, FP4 output
# ---------------------------------------------------------------------------

def _v04(ctx: Ctx) -> None:
    x = _x(ctx, dtype=torch.float32, scale=3.0)
    ref_q, ref_sf = oracle.per_token(x, G, fmt="e2m1")
    q, sf = ctx.call(ctx.dev(x))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})",
                     max_ulps=0 if ctx.is_torch else 1)
    got_v = _fp4_budget(ctx.host(q), ref_q, ctx, sf)
    back = got_v * ctx.host(sf).repeat_interleave(G, dim=1)
    rel = (back - x).abs().max().item() / x.abs().max().item()
    ctx.note(f"round-trip rel-err {rel:.1%} (FP4 has one mantissa bit)")


# ---------------------------------------------------------------------------
# 05 -- column-major scales
# ---------------------------------------------------------------------------

def _v05(ctx: Ctx) -> None:
    # An in-register transpose works a block of tokens at a time, so the NPU
    # tiers need M % 32 == 0. The torch tier is a plain `.T` and has no such
    # constraint, which is why it can sweep M=8.
    if not ctx.is_torch:
        assert ctx.m % BLOCK_MN == 0, (
            f"the in-register transpose works on blocks of {BLOCK_MN} tokens, got M={ctx.m}")
    x = _x(ctx)
    ref_q, ref_sf = oracle.per_token(x, G, round_sf=True)
    q, sf_cm = ctx.call(ctx.dev(x))
    assert sf_cm.shape == (ctx.k // G, ctx.m), (
        f"column-major scales should be (K/32, M) = ({ctx.k // G}, {ctx.m}), "
        f"got {tuple(sf_cm.shape)}"
    )
    assert_fp32_ulps(ctx.host(sf_cm).T.contiguous(), ref_sf,
                     f"sf_cm.T({ctx.m},{ctx.k})", max_ulps=0)
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf_cm={tuple(sf_cm.shape)} transposes back "
             f"to {tuple(ref_sf.shape)} bit-exactly")


# ---------------------------------------------------------------------------
# 06 -- sf_only / cast_only / requant
# ---------------------------------------------------------------------------

def _v06(ctx: Ctx) -> None:
    """Three modes, and the one place the tiers are driven differently.

    The NPU tiers take a compile-time ``mode`` on a single ``launch``; the torch
    tier exposes one function per mode. Everything compared is the same.
    """
    x = _x(ctx)
    ref_q, ref_sf = oracle.per_token(x, G)

    # sf_only: scales, no quantized output
    if ctx.is_torch:
        sf = ctx.call_named("torch_sf_only", x)
    else:
        _, sf = ctx.call(ctx.dev(x), "sf_only")
    assert_fp32_ulps(ctx.host(sf), ref_sf, "sf_only",
                     max_ulps=0 if ctx.is_torch else 1)
    ctx.note("sf_only matches the oracle's scales")

    # cast_only: scales given, no amax pass
    if ctx.is_torch:
        q = ctx.call_named("torch_cast_only", x, ref_sf)
    else:
        q, _ = ctx.call(ctx.dev(x), "cast_only", x_sf=ctx.dev(ref_sf))
    ref_co = oracle.per_token_cast_only(x, ref_sf, G)
    assert_fp8_near(ctx.host(q), ref_co, "cast_only")
    # cast_only only has the rounded scale, so it computes 1/sf where the fused
    # path forms 448/amax. Those differ in the last bit or two; report how far.
    d = (ctx.host(q).view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    ctx.note(f"cast_only matches the oracle; vs the fused path "
             f"{int((d > 0).sum())}/{d.numel()} codes differ (the 1/sf reciprocal)")

    # requant: an already-quantized input, dequantized then quantized again
    if ctx.is_torch:
        q2, sf2 = ctx.call_named("torch_requant", ref_q, ref_sf)
    else:
        q2, sf2 = ctx.call(ctx.dev(ref_q), "requant", x_sf=ctx.dev(ref_sf))
    rq, rsf = oracle.requant_per_token(ref_q, ref_sf, G)
    assert_fp32_ulps(ctx.host(sf2), rsf, "requant sf",
                     max_ulps=0 if ctx.is_torch else 1)
    assert_fp8_near(ctx.host(q2), rq, "requant q")
    ctx.note("requant matches dequantize-then-quantize in torch")


# ---------------------------------------------------------------------------
# 07 -- bfloat16 compute path, composed
# ---------------------------------------------------------------------------

def _v07(ctx: Ctx) -> None:
    x = _x(ctx)
    ref_q, ref_packed = oracle.per_token(x, G, round_sf=True, packed=True)
    q, packed = ctx.call(ctx.dev(x))
    assert_same_bytes(ctx.host(packed), ref_packed, f"sf_packed({ctx.m},{ctx.k})")
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    # The reduction runs in bfloat16, which keeps only 8 mantissa bits -- but the
    # scale uses only the *exponent*, and bfloat16 keeps that exactly. Assert it,
    # because a reduction that silently changed the exponent would still look
    # plausible.
    _, ref_f32 = oracle.per_token(x, G, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(ctx.host(packed)), ref_f32), (
        "the bfloat16 reduction changed the chosen power-of-two exponent"
    )
    ctx.note(f"shape=({ctx.m},{ctx.k}) packed scales byte-exact, FP8 matches, and "
             f"the bfloat16 reduction picked the same exponents as float32 would")


register(
    Variant("per_token", "01_raw_fp32sf", _v01, torch_fn="torch_per_token_cast",
            torch_shapes=((32, 128), (8, 64), (64, 256))),
    Variant("per_token", "02_round_sf", _v02, torch_fn="torch_per_token_cast_round",
            torch_shapes=((32, 128), (8, 64))),
    Variant("per_token", "03_packed_ue8m0", _v03,
            torch_fn="torch_per_token_cast_packed",
            torch_shapes=((32, 128), (8, 128))),
    Variant("per_token", "04_fp32_in_fp4_out", _v04,
            torch_fn="torch_per_token_cast_fp4",
            torch_shapes=((32, 128), (8, 64))),
    Variant("per_token", "05_col_major_sf", _v05,
            torch_fn="torch_per_token_cast_col_major",
            torch_shapes=((32, 128), (8, 64))),
    # the split modes are compile-time, so the TODO probe needs one of them
    Variant("per_token", "06_split_requant", _v06, torch_fn="torch_sf_only",
            torch_shapes=((32, 128),),
            probe_args=lambda m, k: (k, "full")),
    Variant("per_token", "07_bf16_fast_compose", _v07,
            torch_fn="torch_per_token_bf16_compose",
            shape=Shape(fixed_k=BF16_K),
            torch_shapes=((32, 256), (8, 512)),
            probe_args=lambda m, k: (BF16_K,)),
)
