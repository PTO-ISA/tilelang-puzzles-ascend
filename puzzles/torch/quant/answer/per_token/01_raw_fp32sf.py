"""per_token 01 (torch). See doc/quant/per_token/01_raw_fp32sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize each group of ``group_size`` channels against its own maximum.

    Args:
        x: ``(M, K)`` **bfloat16** -- the activations to quantize. Row 0 is all
            zeros in the tests, which is what exercises the clamp.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn** -- the quantized values.
        ``sf``: ``(M, K/group_size)`` **float32** -- ``amax/448`` per group.
    """
    # --- BEGIN SOLUTION hint="view x as (M, K//G, G); amax = abs().amax(-1) clamped to E4M3_CLAMP_MIN; sf = amax/E4M3_MAX; q = (grouped * (E4M3_MAX/amax).unsqueeze(-1)) cast to float8_e4m3fn"
    m, k = x.shape
    assert k % group_size == 0, f"K={k} must be a multiple of {group_size}"
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/01")
