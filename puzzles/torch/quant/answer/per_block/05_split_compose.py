"""per_block 05 (torch). See doc/quant/per_block/05_split_compose.md"""

import torch

from harness import oracle
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from harness.math_ops import ceil_log2_exp, decode_packed_ue8m0, pack_ue8m0_row_major


def torch_per_block_sf_only(x: torch.Tensor, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Compute only the packed column-major scales, no quantized values."""
    # --- BEGIN SOLUTION hint="tile-reduce for amax, exp = ceil_log2_exp(amax/E4M3_MAX), pack (exp+127) and transpose -- just skip building q"
    m, k = x.shape
    bm, bk = block
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    return oracle.to_col_major(pack_ue8m0_row_major((exp_sf + 127).to(torch.uint8)))
    # --- END SOLUTION


def torch_per_block_cast_only(x: torch.Tensor, sf_cm: torch.Tensor,
                              block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize with given packed column-major scales, with no amax pass."""
    # --- BEGIN SOLUTION hint="transpose sf_cm back, decode_packed_ue8m0 it, then multiply each tile by 1/scale and cast -- no reduction anywhere"
    m, k = x.shape
    bm, bk = block
    scale = decode_packed_ue8m0(sf_cm.T.contiguous())
    tiles = x.float().view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    quant = tiles * (1.0 / scale).unsqueeze(-1).unsqueeze(-1)
    return quant.permute(0, 2, 1, 3).reshape(m, k).to(torch.float8_e4m3fn)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_block/05")
