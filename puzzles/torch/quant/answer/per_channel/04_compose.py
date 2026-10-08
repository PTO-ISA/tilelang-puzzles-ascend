"""per_channel 04 (torch). See doc/quant/per_channel/04_compose.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_along_m


def torch_per_channel_compose(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """The fully composed per_channel kernel. Returns ``(q, sf_packed)``."""
    # --- BEGIN SOLUTION hint="combine variants 01-03: reduce amax along dim=1 in bfloat16, widen, exp = ceil_log2_exp(amax/E4M3_MAX), apply inv_pow2_from_exp(exp).unsqueeze(1), and pack (exp+127) with pack_ue8m0_along_m"
    m, k = x.shape
    assert m % group_tokens == 0
    assert (m // group_tokens) % PACK_FACTOR == 0, (
        f"packing along M needs an even number of token groups; "
        f"M={m} gives {m // group_tokens}"
    )
    grouped = x.view(m // group_tokens, group_tokens, k)
    amax = grouped.to(torch.bfloat16).abs().amax(dim=1).float().clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped.float() * inv_pow2_from_exp(exp_sf).unsqueeze(1)).view(m, k)
    return q.to(torch.float8_e4m3fn), pack_ue8m0_along_m((exp_sf + 127).to(torch.uint8))
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/04")
