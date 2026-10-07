"""cast_back 01 (torch). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back(q: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Dequantize: out[m, k] = float(q[m, k]) * sf[m, k // group_size].

    Returns bfloat16, matching what the NPU kernel writes.
    """
    # TODO: view q as (M, K//G, G), multiply by sf.unsqueeze(-1), flatten back,
    #       cast to bfloat16
    raise NotImplementedError("torch/cast_back/01_e4m3_fp32sf: implement torch_cast_back")
