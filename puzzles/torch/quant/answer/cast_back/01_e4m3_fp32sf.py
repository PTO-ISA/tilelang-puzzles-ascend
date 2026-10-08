"""cast_back 01 (torch). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back(q: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Dequantize: out[m, k] = float(q[m, k]) * sf[m, k // group_size].

    Returns bfloat16, matching what the NPU kernel writes.
    """
    # --- BEGIN SOLUTION hint="view q as (M, K//G, G), multiply by sf.unsqueeze(-1), flatten back, cast to bfloat16"
    m, k = q.shape
    assert k % group_size == 0, f"K={k} must be a multiple of the group size {group_size}"
    grouped = q.float().view(m, k // group_size, group_size)
    out = grouped * sf.unsqueeze(-1)
    return out.view(m, k).to(torch.bfloat16)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/01")
