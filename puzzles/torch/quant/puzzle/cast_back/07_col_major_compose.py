"""cast_back 07 -- column-major scale layout, and everything at once.

New config: ``use_tma_aligned_col_major_sf``. Then all the configs together,
which is what production actually runs.

The scale array's *public* shape stays (M, K/32), but the kernel reads and writes
it transposed, as (K/32, M). The reason is the consumer: the GEMM wants scales
laid out so a TMA (bulk async copy) can fetch them contiguously for the tile it
is about to process. Getting that layout from the quantizer for free is cheaper
than transposing afterwards.

In torch this is one ``.T``. On the NPU it is the most involved variant in the
ladder, because a transpose inside a vector register is not a memory operation --
the kernel builds a vector of gather indices from the lane id and does an
in-register gather. Compare this file against the ASC/PTO 07 files: this is the
single widest gap between the torch tier and the NPU tiers in the whole repo,
and it is worth seeing exactly where the complexity comes from.

Composed config for this variant:
    packed UE8M0 scales + column-major layout + FP4 values + bfloat16 output

Run:  python puzzles/torch/quant/answer/cast_back/07_col_major_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_K, BLOCK_MN, CANONICAL_G
from common.demo import print_example
from common.check import assert_bf16_near
from common.math_ops import decode_packed_ue8m0, unpack_e2m1_bytes

VARIANT = "torch/cast_back/07_col_major_compose"


def torch_cast_back_compose(q_packed: torch.Tensor, sf_cm: torch.Tensor,
                            group_size: int = CANONICAL_G):
    """Dequantize FP4 values with column-major packed-UE8M0 scales.

    ``q_packed``: (M, K/2) int8, two e2m1 nibbles per byte.
    ``sf_cm``    : (K/group_size/2, M) int16 -- transposed *and* byte-packed.
    """
    # TODO: transpose sf_cm back to row-major with .T, decode_packed_ue8m0 it,
    #       unpack the FP4 values, then scale per group
    raise NotImplementedError("torch/cast_back/07_col_major_compose: implement torch_cast_back_compose")


def demo_numbers() -> None:
    x = torch.randn(32, 128) * 3
    _, sf_packed = oracle.per_token(x, CANONICAL_G, fmt="e2m1",
                                    round_sf=True, packed=True)
    sf_cm = oracle.to_col_major(sf_packed)
    print(f"[demo] public row-major scales  {tuple(sf_packed.shape)} int16")
    print(f"[demo] kernel column-major      {tuple(sf_cm.shape)} int16  (a transpose)")
    print(f"[demo] round trips back:        {torch.equal(sf_cm.T.contiguous(), sf_packed)}")
    footprint_fp32_rowmajor = 32 * (128 // CANONICAL_G) * 4
    footprint_packed = sf_cm.numel() * 2
    print(f"[demo] scale bytes: {footprint_fp32_rowmajor} as fp32 -> "
          f"{footprint_packed} as packed UE8M0 ({footprint_fp32_rowmajor // footprint_packed}x smaller)")
    print(f"[demo] value bytes: {32 * 128} as bf16 -> {32 * 128 // 4} as packed FP4 (4x smaller)")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 256)):
        x = torch.randn(m, k) * 3
        q_packed, sf_packed = oracle.per_token(x, CANONICAL_G, fmt="e2m1",
                                               round_sf=True, packed=True)
        sf_cm = oracle.to_col_major(sf_packed)
        got = torch_cast_back_compose(q_packed, sf_cm)
        ref = oracle.cast_back(q_packed, sf_packed, (1, CANONICAL_G),
                               packed=True, fp4=True, out_dtype=torch.bfloat16)
        assert_bf16_near(got, ref, f"cast_back_compose({m},{k})", atol=0.0)
        rel = (got.float() - x).abs().max().item() / x.abs().max().item()
        print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} ok, "
              f"round-trip rel-err {rel:.1%}")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
