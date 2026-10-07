"""per_block 03 -- FP4 (e2m1) output on a 2-D tile.

New config: ``out_dtype = float4_e2m1fn`` with block granularity. The format
change is exactly per_token/04's -- ``quant_max`` becomes 6.0, values pack two
per byte -- so what is worth attention here is the *interaction* with coarse
blocking, which turns out to be the sharpest accuracy trade-off in the ladder.

FP4 has one mantissa bit and only 8 distinct magnitudes
(``0, 0.5, 1, 1.5, 2, 3, 4, 6``). Spreading one scale over 1024 values means a
wider dynamic range has to fit into those 8 steps than per_token's 32 values
would need. The test measures both so the cost is a number rather than a claim.

This is why production pairs FP4 with per-token or small-block granularity for
activations and reserves coarse blocking for weights, whose distribution is
better behaved.

Run:  python puzzles/torch/quant/answer/per_block/03_fp4_e2m1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_same_bytes
from common.math_ops import pack_e2m1_from_fp32, unpack_e2m1_bytes

VARIANT = "torch/per_block/03_fp4_e2m1"


def torch_per_block_cast_fp4(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q_packed, sf)`` -- packed FP4 values, one scale per tile."""
    # TODO: tile-reduce as in variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN, then
    #       pack_e2m1_from_fp32 the permuted-back quantized values
    raise NotImplementedError("torch/per_block/03_fp4_e2m1: implement torch_per_block_cast_fp4")


def demo_numbers() -> None:
    x = torch.zeros(32, 32, dtype=torch.float32)
    x[0, 0:4] = torch.tensor([6.0, 3.0, -1.5, 0.5])
    q_packed, sf = torch_per_block_cast_fp4(x)
    values = unpack_e2m1_bytes(q_packed)
    print(f"[demo] one 32x32 tile, amax=6.0 -> sf={sf[0, 0].item():g}")
    print_example("per_block 03", x=x[:1, :4], fp4=values[:1, :4])
    assert torch.allclose(values[0, :4], torch.tensor([6.0, 3.0, -1.5, 0.5]))
    print(f"[demo] storage: {x.numel() * 4} bytes fp32 -> {q_packed.numel()} bytes FP4 "
          f"+ {sf.numel() * 4} bytes of scale")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 64)):
        x = randn_with_zero_row(m, k, torch.device("cpu"), dtype=torch.float32) * 3
        q_packed, sf = torch_per_block_cast_fp4(x)
        ref_q, ref_sf = oracle.per_block(x, (BLOCK_MN, BLOCK_K), fmt="e2m1")
        assert_same_bytes(q_packed, ref_q, f"q_packed({m},{k})")
        assert torch.equal(sf, ref_sf)
        print(f"[check] shape=({m},{k}) packed={tuple(q_packed.shape)} ok")

    # the granularity x format interaction, measured
    y = torch.randn(64, 256) * 3
    rows = []
    for label, pair, blk in (
        ("per_block e4m3", oracle.per_block(y, (BLOCK_MN, BLOCK_K)), (BLOCK_MN, BLOCK_K)),
        ("per_token e4m3", oracle.per_token(y, BLOCK_K), (1, BLOCK_K)),
        ("per_block e2m1", oracle.per_block(y, (BLOCK_MN, BLOCK_K), fmt="e2m1"), (BLOCK_MN, BLOCK_K)),
        ("per_token e2m1", oracle.per_token(y, BLOCK_K, fmt="e2m1"), (1, BLOCK_K)),
    ):
        fp4 = "e2m1" in label
        back = oracle.cast_back(pair[0], pair[1], blk, fp4=fp4, out_dtype=torch.float32)
        rel = (back - y).abs().max().item() / y.abs().max().item()
        rows.append((label, rel))
        print(f"[check] {label}: round-trip rel-err {rel:.1%}")
    by = dict(rows)
    assert by["per_block e2m1"] > by["per_token e2m1"] > by["per_token e4m3"], rows
    print("[check] FP4 widens the granularity gap, as expected")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
