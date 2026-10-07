"""cast_back 02 (torch). See doc/quant/cast_back/02_fp32_out.md"""

import torch

from common.consts import CANONICAL_G


def torch_cast_back_f32(q: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Dequantize to float32 instead of bfloat16."""
    # TODO: same as variant 01, but return float32 (no .to(bfloat16))
    raise NotImplementedError("torch/cast_back/02_fp32_out: implement torch_cast_back_f32")
