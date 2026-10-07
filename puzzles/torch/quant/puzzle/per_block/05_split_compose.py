"""per_block 05 -- the sf_only / cast_only split, fully composed.

Last per_block variant. Same split as per_token/06 -- compute only the scales, or
apply scales that are given -- now on the block layout, then everything at once.

Composed config:

    bfloat16 input -> 32x32 blocks -> power-of-two scale -> packed UE8M0
    -> column-major layout -> FP8 e4m3 output

That is the configuration production runs for weight quantization, and reaching
it is the point of this sub-ladder. The remaining distance to
``per_block_cast_asc.py`` is all scheduling: multiple cores, a persistent loop
over tiles, double-buffered UB, and the ``token_group`` widening from variant 04.
None of it changes the arithmetic.

Run:  python puzzles/torch/quant/answer/per_block/05_split_compose.py
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

VARIANT = "torch/per_block/05_split_compose"


def torch_per_block_sf_only(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Compute only the packed column-major scales, no quantized values."""
    # TODO: tile-reduce for amax, exp = ceil_log2_exp(amax/E4M3_MAX), pack
    #       (exp+127) and transpose -- just skip building q
    raise NotImplementedError("torch/per_block/05_split_compose: implement torch_per_block_sf_only")


def torch_per_block_cast_only(x: torch.Tensor, sf_cm: torch.Tensor,
                              block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize with given packed column-major scales, with no amax pass."""
    # TODO: transpose sf_cm back, decode_packed_ue8m0 it, then multiply each tile
    #       by 1/scale and cast -- no reduction anywhere
    raise NotImplementedError("torch/per_block/05_split_compose: implement torch_per_block_cast_only")


def demo_numbers() -> None:
    torch.manual_seed(0)
    x = randn_with_zero_row(64, 128, torch.device("cpu"))
    sf_cm = torch_per_block_sf_only(x)
    q = torch_per_block_cast_only(x, sf_cm)
    print(f"[demo] sf_only  -> {tuple(sf_cm.shape)} int16, no values")
    print(f"[demo] cast_only -> {tuple(q.shape)} e4m3, no amax pass")
    print("[demo] the two halves together reproduce the fused kernel:")
    ref_q, _ = oracle.per_block(x, (BLOCK_MN, BLOCK_K), round_sf=True, packed=True)
    d = (q.view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    print(f"[demo]   codes differing from the fused path: {int((d > 0).sum())}/{d.numel()}")
    # Exact here, because the scale is a power of two (see per_token/06).
    assert int((d > 0).sum()) == 0


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 256)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        sf_cm = torch_per_block_sf_only(x)
        q = torch_per_block_cast_only(x, sf_cm)
        ref_q, ref_packed = oracle.per_block(x, (BLOCK_MN, BLOCK_K),
                                             round_sf=True, packed=True)
        assert_same_bytes(sf_cm.T.contiguous(), ref_packed, f"sf_only({m},{k})")
        assert_fp8_near(q, ref_q, f"cast_only({m},{k})")
        back = oracle.cast_back(q, decode_packed_ue8m0(sf_cm.T.contiguous()),
                                (BLOCK_MN, BLOCK_K), out_dtype=torch.float32)
        rel = (back - x.float()).abs().max().item() / x.float().abs().max().item()
        print(f"[check] shape=({m},{k}) split==fused, round-trip rel-err {rel:.1%}")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
