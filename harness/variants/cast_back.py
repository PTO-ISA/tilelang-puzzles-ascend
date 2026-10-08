"""cast_back variant checks.

One body per variant, shared by all three tiers. The body builds the inputs,
calls the tier's entry point through ``ctx``, and asserts against
``harness.oracle`` -- which the torch tier's answers define.

cast_back is the only kernel with no reduction: its scale factors are an input.
So every body here is "build q and sf, dequantize, compare".
"""

from __future__ import annotations

import torch

from harness import oracle
from harness.asserts import assert_bf16_near, assert_fp32_ulps
from harness.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from harness.math_ops import pack_ue8m0_row_major

from harness.spec import Ctx, Shape, Variant, register

G = CANONICAL_G


def _fp8_and_scales(m: int, k: int, sf_shape: tuple[int, int]):
    """A quantized tensor and a plausible positive scale array."""
    q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    sf = torch.rand(*sf_shape) * 0.01 + 1e-4
    return q, sf


# ---------------------------------------------------------------------------
# 01 -- FP8 values, one FP32 scale per 32 channels, either output dtype
# ---------------------------------------------------------------------------

def _v01(ctx: Ctx) -> None:
    """Both output dtypes, because the dtype is a config of this kernel.

    It used to be variant 02. As a separate variant it was one changed line of
    arithmetic; as a switch inside this kernel the two store paths sit side by
    side, which is the actual lesson -- float32 is a plain contiguous store and
    bfloat16 is a narrowing one.
    """
    q, sf = _fp8_and_scales(ctx.m, ctx.k, (ctx.m, ctx.k // G))

    ref_bf16 = oracle.cast_back(q, sf, (1, G), out_dtype=torch.bfloat16)
    got = ctx.host(ctx.call(ctx.dev(q), ctx.dev(sf)))
    assert_bf16_near(got, ref_bf16, f"cast_back({ctx.m},{ctx.k})", atol=0.0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) bfloat16 out matches the oracle exactly")

    ref_f32 = oracle.cast_back(q, sf, (1, G), out_dtype=torch.float32)
    got_f32 = ctx.host(ctx.call(ctx.dev(q), ctx.dev(sf), out_dtype="float32"))
    # Checked explicitly: a bfloat16 result silently upcast on the way home
    # would pass the value comparison below.
    assert got_f32.dtype == torch.float32, (
        f"expected float32 output, got {got_f32.dtype}"
    )
    assert_fp32_ulps(got_f32, ref_f32, f"cast_back_f32({ctx.m},{ctx.k})",
                     max_ulps=0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) float32 out is bit-exact "
             f"(one multiply, no intermediate rounding)")


# ---------------------------------------------------------------------------
# 02 -- packed UE8M0 scales
# ---------------------------------------------------------------------------

def _v02(ctx: Ctx) -> None:
    q = (torch.randn(ctx.m, ctx.k) * 100).to(torch.float8_e4m3fn)
    e8m0 = torch.randint(100, 140, (ctx.m, ctx.k // G), dtype=torch.uint8)
    packed = pack_ue8m0_row_major(e8m0)
    ref = oracle.cast_back(q, packed, (1, G), packed=True, out_dtype=torch.bfloat16)
    got = ctx.host(ctx.call(ctx.dev(q), ctx.dev(packed)))
    assert_bf16_near(got, ref, f"cast_back_packed({ctx.m},{ctx.k})", atol=0.0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) exact; {packed.numel()} int16 words "
             f"carry {e8m0.numel()} scales")


# ---------------------------------------------------------------------------
# 03 -- one scale per 32x32 tile
# ---------------------------------------------------------------------------

def _v03(ctx: Ctx) -> None:
    assert ctx.m % BLOCK_MN == 0, f"needs M a multiple of {BLOCK_MN}, got {ctx.m}"
    assert ctx.k % BLOCK_K == 0, f"needs K a multiple of {BLOCK_K}, got {ctx.k}"
    q, sf = _fp8_and_scales(ctx.m, ctx.k, (ctx.m // BLOCK_MN, ctx.k // BLOCK_K))
    ref = oracle.cast_back(q, sf, (BLOCK_MN, BLOCK_K), out_dtype=torch.bfloat16)
    got = ctx.host(ctx.call(ctx.dev(q), ctx.dev(sf)))
    assert_bf16_near(got, ref, f"cast_back_block({ctx.m},{ctx.k})", atol=0.0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf={tuple(sf.shape)} exact")


# ---------------------------------------------------------------------------
# 04 -- one scale per channel, shared by 32 tokens
# ---------------------------------------------------------------------------

def _v04(ctx: Ctx) -> None:
    assert ctx.m % BLOCK_MN == 0, f"needs M a multiple of {BLOCK_MN}, got {ctx.m}"
    q, sf = _fp8_and_scales(ctx.m, ctx.k, (ctx.m // BLOCK_MN, ctx.k))
    ref = oracle.cast_back(q, sf, (BLOCK_MN, 1), out_dtype=torch.bfloat16)
    got = ctx.host(ctx.call(ctx.dev(q), ctx.dev(sf)))
    assert_bf16_near(got, ref, f"cast_back_channel({ctx.m},{ctx.k})", atol=0.0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf={tuple(sf.shape)} exact")


# ---------------------------------------------------------------------------
# 05 -- packed FP4 (e2m1) input
# ---------------------------------------------------------------------------

def _v05(ctx: Ctx) -> None:
    x = torch.randn(ctx.m, ctx.k) * 3
    q_packed, sf = oracle.per_token(x, G, fmt="e2m1")
    assert q_packed.shape == (ctx.m, ctx.k // 2), (
        f"packed FP4 should be (M, K/2), got {tuple(q_packed.shape)}"
    )
    ref = oracle.cast_back(q_packed, sf, (1, G), fp4=True, out_dtype=torch.bfloat16)
    got = ctx.host(ctx.call(ctx.dev(q_packed), ctx.dev(sf)))
    assert_bf16_near(got, ref, f"cast_back_fp4({ctx.m},{ctx.k})", atol=0.0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) packed={tuple(q_packed.shape)} exact")


# ---------------------------------------------------------------------------
# 06 -- column-major packed scales, FP4 values, everything composed
# ---------------------------------------------------------------------------

def _v06(ctx: Ctx) -> None:
    assert ctx.m % BLOCK_MN == 0, f"needs M a multiple of {BLOCK_MN}, got {ctx.m}"
    x = torch.randn(ctx.m, ctx.k) * 3
    q_packed, packed = oracle.per_token(x, G, fmt="e2m1", round_sf=True, packed=True)
    sf_cm = oracle.to_col_major(packed)
    ref = oracle.cast_back(q_packed, packed, (1, G), packed=True, fp4=True,
                           out_dtype=torch.bfloat16)
    got = ctx.host(ctx.call(ctx.dev(q_packed), ctx.dev(sf_cm)))
    assert_bf16_near(got, ref, f"cast_back_compose({ctx.m},{ctx.k})", atol=0.0)
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf_cm={tuple(sf_cm.shape)} exact")


register(
    Variant("cast_back", "01_e4m3_fp32sf", _v01, torch_fn="torch_cast_back",
            torch_shapes=((32, 128), (8, 64), (64, 256))),
    Variant("cast_back", "02_packed_ue8m0", _v02, torch_fn="torch_cast_back_packed",
            torch_shapes=((32, 128), (8, 128))),
    Variant("cast_back", "03_block_sf", _v03, torch_fn="torch_cast_back_block",
            torch_shapes=((32, 128), (64, 64))),
    Variant("cast_back", "04_per_channel_sf", _v04,
            torch_fn="torch_cast_back_per_channel",
            torch_shapes=((32, 128), (64, 64))),
    Variant("cast_back", "05_fp4_e2m1", _v05, torch_fn="torch_cast_back_fp4",
            torch_shapes=((32, 128), (8, 64))),
    Variant("cast_back", "06_col_major_compose", _v06,
            torch_fn="torch_cast_back_compose",
            torch_shapes=((32, 128), (64, 256))),
)
