"""per_block variant checks.

One scale per 32x32 tile, so the reduction is two-dimensional. Every body builds
one input and compares the quantized values plus a much smaller scale array --
(M/32, K/32) rather than per_token's (M, K/32).
"""

from __future__ import annotations

import torch

from harness import oracle
from harness.asserts import assert_fp8_near, assert_fp32_ulps, assert_same_bytes
from harness.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from harness.demo import randn_with_zero_row
from harness.math_ops import decode_packed_ue8m0, unpack_e2m1_bytes

from harness.spec import Ctx, Variant, register

CPU = torch.device("cpu")
BLOCK = (BLOCK_MN, BLOCK_K)


def _guards(ctx: Ctx) -> None:
    assert ctx.m % BLOCK_MN == 0, f"needs M a multiple of {BLOCK_MN}, got {ctx.m}"
    assert ctx.k % BLOCK_K == 0, f"needs K a multiple of {BLOCK_K}, got {ctx.k}"


def _v01(ctx: Ctx) -> None:
    _guards(ctx)
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_sf = oracle.per_block(x, BLOCK)
    q, sf = ctx.call(ctx.dev(x))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})",
                     max_ulps=0 if ctx.is_torch else 1)
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf={tuple(sf.shape)} matches the oracle")


def _v02(ctx: Ctx) -> None:
    _guards(ctx)
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_packed = oracle.per_block(x, BLOCK, round_sf=True, packed=True)
    q, packed = ctx.call(ctx.dev(x))
    assert_same_bytes(ctx.host(packed), ref_packed, f"sf_packed({ctx.m},{ctx.k})")
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    _, ref_f32 = oracle.per_block(x, BLOCK, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(ctx.host(packed)), ref_f32), (
        "the packed bytes must decode back to the float32 scales"
    )
    ctx.note(f"shape=({ctx.m},{ctx.k}) packed={tuple(packed.shape)} byte-exact, "
             f"decodes to the float32 scales")


def _v03(ctx: Ctx) -> None:
    _guards(ctx)
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_sf = oracle.per_block(x, BLOCK, fmt="e2m1")
    q, sf = ctx.call(ctx.dev(x))
    assert_fp32_ulps(ctx.host(sf), ref_sf, f"sf({ctx.m},{ctx.k})",
                     max_ulps=0 if ctx.is_torch else 1)
    got_v, ref_v = unpack_e2m1_bytes(ctx.host(q)), unpack_e2m1_bytes(ref_q)
    differing, total = int((got_v != ref_v).sum()), got_v.numel()
    ctx.note(f"shape=({ctx.m},{ctx.k}) FP4 values differing from the torch packer: "
             f"{differing}/{total} ({differing / total:.2%})")
    assert differing / total < 0.02, (
        f"{differing}/{total} FP4 codes differ; the two-step rounding should "
        f"track the torch packer closely"
    )


def _v04(ctx: Ctx) -> None:
    _guards(ctx)
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_sf = oracle.per_block(x, BLOCK)
    q, sf_cm = ctx.call(ctx.dev(x))
    assert sf_cm.shape == (ctx.k // BLOCK_K, ctx.m // BLOCK_MN), (
        f"column-major tile scales should be (K/32, M/32) = "
        f"({ctx.k // BLOCK_K}, {ctx.m // BLOCK_MN}), got {tuple(sf_cm.shape)}"
    )
    assert_fp32_ulps(ctx.host(sf_cm).T.contiguous(), ref_sf,
                     f"sf_cm.T({ctx.m},{ctx.k})", max_ulps=0 if ctx.is_torch else 1)
    assert_fp8_near(ctx.host(q), ref_q, f"q({ctx.m},{ctx.k})")
    ctx.note(f"shape=({ctx.m},{ctx.k}) sf_cm={tuple(sf_cm.shape)} transposes back "
             f"to {tuple(ref_sf.shape)}")


def _v05(ctx: Ctx) -> None:
    """sf_only / full / cast_only.

    The interesting assertion is the last one: with a power-of-two scale the
    reciprocal is formed by negating the exponent field, which is exact, so
    cast_only must be *bit-identical* to the fused kernel. per_token/06 could only
    promise "within one code" because its scale was an arbitrary float32.
    """
    x = randn_with_zero_row(ctx.m, ctx.k, CPU)
    ref_q, ref_packed = oracle.per_block(x, BLOCK, round_sf=True, packed=True)

    if ctx.is_torch:
        sf = ctx.call_named("torch_per_block_sf_only", x)
        packed_host = sf.T.contiguous() if sf.shape[0] != ref_packed.shape[0] else sf
        assert_same_bytes(packed_host, ref_packed, "sf_only")
        q_co = ctx.call_named("torch_per_block_cast_only", x, sf)
        assert_fp8_near(q_co, ref_q, "cast_only")
        ctx.note("sf_only and cast_only match the oracle")
        return

    _, sf = ctx.call(ctx.dev(x), "sf_only")
    assert_same_bytes(ctx.host(sf).view(torch.int16), ref_packed, "sf_only")
    ctx.note("sf_only produces the oracle's packed scales byte-exactly")

    q_full, _ = ctx.call(ctx.dev(x), "full")
    assert_fp8_near(ctx.host(q_full), ref_q, "full q")
    ctx.note("full matches the oracle")

    q_co, _ = ctx.call(ctx.dev(x), "cast_only", sf_in=sf)
    assert torch.equal(ctx.host(q_co).view(torch.uint8),
                       ctx.host(q_full).view(torch.uint8)), (
        "with a power-of-two scale, cast_only must be bit-identical to fused"
    )
    ctx.note("cast_only is bit-identical to the fused kernel "
             "(exact reciprocal of a power of two)")

    _, ref_f32 = oracle.per_block(x, BLOCK, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(ctx.host(sf).view(torch.int16)), ref_f32)
    ctx.note(f"shape=({ctx.m},{ctx.k}) scales decode to the float32 scales")


register(
    Variant("per_block", "01_raw_32x32", _v01, torch_fn="torch_per_block_cast",
            torch_shapes=((32, 128), (64, 64))),
    Variant("per_block", "02_round_packed", _v02,
            torch_fn="torch_per_block_cast_packed",
            torch_shapes=((32, 128), (64, 128))),
    Variant("per_block", "03_fp4_e2m1", _v03, torch_fn="torch_per_block_cast_fp4",
            torch_shapes=((32, 128), (64, 64))),
    Variant("per_block", "04_col_major_tma", _v04,
            torch_fn="torch_per_block_cast_col_major",
            torch_shapes=((64, 128), (128, 256))),
    Variant("per_block", "05_split_compose", _v05,
            torch_fn="torch_per_block_sf_only",
            torch_shapes=((32, 128), (64, 256)),
            probe_args=lambda m, k: (k, "full")),
)
