"""cast_back 03 (torch). See doc/quant/cast_back/02_block_sf.md"""

import torch

from harness.consts import BLOCK_K, BLOCK_MN


def torch_cast_back_block(q: torch.Tensor, sf: torch.Tensor,
                          sf_block: tuple = (BLOCK_MN, BLOCK_K)):
    """Dequantize with a 2-D scale block: one scale per ``sf_block``-shaped tile.

    Args:
        q: ``(M, K)`` **float8_e4m3fn** -- the quantized values.
        sf: ``(M/bm, K/bk)`` **float32** for ``sf_block = (bm, bk)`` -- one
            positive scale per tile. At the default (32, 32) and K=128 that is
            ``(M/32, 4)``.
        sf_block: the tile shape the scales cover.

    Returns:
        ``(M, K)`` **bfloat16**.
    """
    # TODO: view q as (M/bm, bm, K/bk, bk); sf needs two unsqueezes, at dim 1 and
    #       dim -1; then flatten back
    raise NotImplementedError("torch/cast_back/03_block_sf: implement torch_cast_back_block")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/03")
