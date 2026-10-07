"""per_block 04 -- column-major scale layout.

New config: ``use_tma_aligned_col_major_sf`` on the block layout. Same motivation
as per_token/05 (the consuming GEMM wants a tile's scales contiguous so a bulk
async copy can fetch them), but there is one extra wrinkle here that per_token
does not have.

    row-major     sf[m_tile, k_tile]   shape (M/32, K/32)
    column-major  sf[k_tile, m_tile]   shape (K/32, M/32)

### token_group

Production's per_block path groups **four** tile-rows together when the scales
are both packed and column-major (``token_group = 4`` in
``per_block_cast_asc.py``). The reason is a minimum-width constraint: after
blocking by 32 and packing 2-per-word, a tile row's worth of scales can be just
a couple of int16 words -- narrower than the DMA engine's useful transfer. Four
tile-rows at a time brings it back up to a sensible width.

This is a scheduling detail with no effect on the numbers, which is why it does
not appear in this file at all. It is called out because it *does* appear in the
ASC and PTO versions, and it is easy to mistake for part of the algorithm.

Run:  python puzzles/torch/quant/answer/per_block/04_col_major_tma.py
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

VARIANT = "torch/per_block/04_col_major_tma"


def torch_per_block_cast_col_major(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Return ``(q, sf_cm)`` with packed power-of-two scales, transposed."""
    # TODO: as variant 02 (pow2 + packed), then return oracle.to_col_major(packed)
    raise NotImplementedError("torch/per_block/04_col_major_tma: implement torch_per_block_cast_col_major")


def demo_numbers() -> None:
    x = randn_with_zero_row(128, 256, torch.device("cpu"))
    q, sf_cm = torch_per_block_cast_col_major(x)
    m_tiles, k_tiles = 128 // BLOCK_MN, 256 // BLOCK_K
    print(f"[demo] x{tuple(x.shape)} -> {m_tiles}x{k_tiles} tiles of 32x32")
    print(f"[demo]   row-major packed scales would be "
          f"({m_tiles}, {k_tiles // PACK_FACTOR}) int16")
    print(f"[demo]   column-major is {tuple(sf_cm.shape)} int16  (transposed)")
    print(f"[demo]   transposes back cleanly: "
          f"{tuple(sf_cm.T.contiguous().shape)}")
    assert sf_cm.shape == (k_tiles // PACK_FACTOR, m_tiles), sf_cm.shape


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((64, 128), (128, 256)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf_cm = torch_per_block_cast_col_major(x)
        ref_q, ref_packed = oracle.per_block(x, (BLOCK_MN, BLOCK_K),
                                             round_sf=True, packed=True)
        assert_same_bytes(sf_cm.T.contiguous(), ref_packed, f"sf_cm.T({m},{k})")
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        _, ref_f32 = oracle.per_block(x, (BLOCK_MN, BLOCK_K), round_sf=True)
        assert torch.equal(decode_packed_ue8m0(sf_cm.T.contiguous()), ref_f32)
        print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} ok")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
