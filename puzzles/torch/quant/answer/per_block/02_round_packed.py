"""per_block 02 -- power-of-two scale, stored as packed UE8M0.

Same two configs per_token picked up in its variants 02 and 03, now on the 2-D
block layout. They are folded into one variant here because the arithmetic is
identical to per_token's -- only the reduction shape differs -- and there is no
new idea in doing them separately.

    sf      = 2^ceil(log2(amax / 448))        exact, no mantissa needed
    stored  = uint8(exp + 127), two per int16

See per_token/02 for the ceil-log2 bit trick and per_token/03 for the packing
layout; both are explained there in full.

What *is* new is the packed scale array's shape. per_token packed along K, where
there were K/32 scales per row. Here there are only K/32 scales per *tile row*,
so the packing divides a much smaller number -- which is why production's
per_block path has an alignment condition that per_token does not: the packed
scale row has to be padded out to the DMA's minimum width.

Run:  python puzzles/torch/quant/answer/per_block/02_round_packed.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_same_bytes
from common.math_ops import ceil_log2_exp, decode_packed_ue8m0, inv_pow2_from_exp, pack_ue8m0_row_major

VARIANT = "torch/per_block/02_round_packed"


def torch_per_block_cast_packed(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_packed)`` -- power-of-two scales as packed UE8M0 int16."""
    # --- BEGIN SOLUTION hint="tile-reduce as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX); multiply by inv_pow2_from_exp(exp); return pack_ue8m0_row_major((exp+127).to(uint8))"
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    quant = tiles * inv_pow2_from_exp(exp_sf).unsqueeze(-1).unsqueeze(-1)
    q = quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    return q, pack_ue8m0_row_major((exp_sf + 127).to(torch.uint8))
    # --- END SOLUTION


def demo_numbers() -> None:
    x = torch.zeros(32, 128, dtype=torch.bfloat16)
    x[0, 0] = 448.0      # tile 0 -> sf = 2^0
    x[0, 32] = 3.5       # tile 1 -> sf = 2^-7
    x[0, 64] = 1.75      # tile 2 -> sf = 2^-8
    x[0, 96] = 0.875     # tile 3 -> sf = 2^-9
    q, packed = torch_per_block_cast_packed(x)
    decoded = decode_packed_ue8m0(packed)
    print(f"[demo] 4 tiles -> {packed.numel()} int16 words (2 scales each)")
    print(f"[demo]   decoded scales: {[f'2^{int(torch.log2(v))}' for v in decoded[0]]}")
    print_example("per_block 02", packed=packed, decoded=decoded)
    assert decoded[0, 0].item() == 1.0
    mant = decoded.view(torch.int32) & 0x7FFFFF
    assert int(mant.abs().max()) == 0, "every decoded scale is an exact power of two"
    print("[demo] 4 scales: 16 bytes as float32 -> "
          f"{packed.numel() * 2} bytes packed")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 128)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, packed = torch_per_block_cast_packed(x)
        ref_q, ref_packed = oracle.per_block(x, (BLOCK_MN, BLOCK_K),
                                             round_sf=True, packed=True)
        assert_same_bytes(packed, ref_packed, f"sf_packed({m},{k})")
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        _, ref_f32 = oracle.per_block(x, (BLOCK_MN, BLOCK_K), round_sf=True)
        assert torch.equal(decode_packed_ue8m0(packed), ref_f32)
        print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} ok, decodes exactly")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
