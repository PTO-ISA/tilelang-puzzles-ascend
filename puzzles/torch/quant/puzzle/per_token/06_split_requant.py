"""per_token 06 (torch). See doc/quant/per_token/06_split_requant.md"""

import torch

from harness import oracle
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_sf_only(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Compute only the scale factors."""
    # TODO: amax over each group, clamp, divide by E4M3_MAX; return just sf
    raise NotImplementedError("torch/per_token/06_split_requant: implement torch_sf_only")


def torch_cast_only(x: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize using scales that are given, with no amax pass.

    Mirror the kernel: take the reciprocal of the stored scale and multiply.
    """
    # TODO: sf_inv = 1.0 / sf (not E4M3_MAX/amax -- we only have sf); multiply
    #       each group by sf_inv.unsqueeze(-1) and cast to float8_e4m3fn
    raise NotImplementedError("torch/per_token/06_split_requant: implement torch_cast_only")


def torch_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                  group_size: int = CANONICAL_G):
    """Dequantize an already-quantized input, then quantize it again."""
    # TODO: dequantize with oracle.cast_back(q_in, sf_in, (1, group_size),
    #       out_dtype=float32), then run the ordinary variant-01 quantize on the
    #       result
    raise NotImplementedError("torch/per_token/06_split_requant: implement torch_requant")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/06")
