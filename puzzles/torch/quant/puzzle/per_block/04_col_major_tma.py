"""per_block 04 (torch). See doc/quant/per_block/04_col_major_tma.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_block_cast_col_major(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize per tile, writing the scales transposed.

    Args:
        x: ``(M, K)`` **bfloat16** -- the values to quantize. Row 0 is all zeros
            in the tests, which is what exercises the clamp.
        block: ``(bm, bk)``, the tile the scale covers.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf_cm``: ``(K/bk, M/bm)`` **float32** -- the transpose of variant
        01's ``(M/bm, K/bk)``. A tile scale is a single scalar, so unlike
        per_token/05 this layout change costs nothing.
    """
    # TODO: tile-reduce exactly as in variant 01, then transpose the scale array
    #       instead of returning it as is: sf.T.contiguous(), shape (K/32, M/32).
    #       The kernel writes sf[k_block, m_block] so the consuming GEMM can fetch
    #       one tile column contiguously. Nothing else changes -- one scalar per
    #       tile has no interior layout to disturb.
    raise NotImplementedError("torch/per_block/04_col_major_tma: implement torch_per_block_cast_col_major")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/04 --role puzzle")
