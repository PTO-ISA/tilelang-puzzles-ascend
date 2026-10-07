"""per_channel 04 -- everything composed.

Final per_channel variant, and the last of the torch tier.

Composed config:

    bfloat16 input -> reduce along M over 32 tokens -> bfloat16 compute
    -> power-of-two scale -> UE8M0 packed along M -> FP8 e4m3 output

That is production's per_channel configuration. What remains between this file
and ``per_channel_cast_asc.py`` is entirely scheduling: multiple vector cores, a
persistent loop with a manual wave index, double-buffered Unified Buffer
allocations, and L2 cache hints on the DMA. None of it changes a single number
this file computes -- which is the useful thing to have established before
reading the kernel.

Run:  python puzzles/torch/quant/answer/per_channel/04_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_same_bytes
from common.math_ops import (
    ceil_log2_exp,
    decode_packed_ue8m0_along_m,
    inv_pow2_from_exp,
    pack_ue8m0_along_m,
)

VARIANT = "torch/per_channel/04_compose"


def torch_per_channel_compose(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """The fully composed per_channel kernel. Returns ``(q, sf_packed)``."""
    # --- BEGIN SOLUTION hint="combine variants 01-03: reduce amax along dim=1 in bfloat16, widen, exp = ceil_log2_exp(amax/E4M3_MAX), apply inv_pow2_from_exp(exp).unsqueeze(1), and pack (exp+127) with pack_ue8m0_along_m"
    m, k = x.shape
    assert m % group_tokens == 0
    assert (m // group_tokens) % PACK_FACTOR == 0, (
        f"packing along M needs an even number of token groups; "
        f"M={m} gives {m // group_tokens}"
    )
    grouped = x.view(m // group_tokens, group_tokens, k)
    amax = grouped.to(torch.bfloat16).abs().amax(dim=1).float().clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped.float() * inv_pow2_from_exp(exp_sf).unsqueeze(1)).view(m, k)
    return q.to(torch.float8_e4m3fn), pack_ue8m0_along_m((exp_sf + 127).to(torch.uint8))
    # --- END SOLUTION


def demo_numbers() -> None:
    torch.manual_seed(0)
    x = randn_with_zero_row(64, 128, torch.device("cpu"))
    q, packed = torch_per_channel_compose(x)
    groups = 64 // BLOCK_MN
    print(f"[demo] x{tuple(x.shape)} bf16 -> q{tuple(q.shape)} e4m3, "
          f"sf{tuple(packed.shape)} int16")
    value_bytes_in, value_bytes_out = x.numel() * 2, q.numel()
    scale_bytes_f32, scale_bytes_packed = groups * 128 * 4, packed.numel() * 2
    print(f"[demo] values: {value_bytes_in} B bf16 -> {value_bytes_out} B e4m3 (2x)")
    print(f"[demo] scales: {scale_bytes_f32} B float32 -> {scale_bytes_packed} B "
          f"packed UE8M0 ({scale_bytes_f32 // scale_bytes_packed}x)")
    total_in = value_bytes_in
    total_out = value_bytes_out + scale_bytes_packed
    print(f"[demo] overall: {total_in} B -> {total_out} B "
          f"({total_in / total_out:.2f}x smaller including scales)")
    assert decode_packed_ue8m0_along_m(packed).shape == (groups, 128)


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((64, 128), (128, 256)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, packed = torch_per_channel_compose(x)
        ref_q, ref_packed = oracle.per_channel(x, BLOCK_MN, round_sf=True, packed=True)
        assert_same_bytes(packed, ref_packed, f"sf_packed({m},{k})")
        assert_fp8_near(q, ref_q, f"q({m},{k})")

        decoded = decode_packed_ue8m0_along_m(packed)
        back = oracle.cast_back(q, decoded, (BLOCK_MN, 1), out_dtype=torch.float32)
        rel = (back - x.float()).abs().max().item() / x.float().abs().max().item()
        print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} ok, "
              f"round-trip rel-err {rel:.1%}")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
