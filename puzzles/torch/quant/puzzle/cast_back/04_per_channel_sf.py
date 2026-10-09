"""cast_back 04 (torch). See doc/quant/cast_back/03_per_channel_sf.md"""

import torch

from harness.consts import BLOCK_MN


def torch_cast_back_per_channel(q: torch.Tensor, sf: torch.Tensor,
                                group_tokens: int = BLOCK_MN):
    """Dequantize with per-channel scales shared over ``group_tokens`` rows.

    Args:
        q: ``(M, K)`` **float8_e4m3fn** -- the quantized values.
        sf: ``(M/group_tokens, K)`` **float32** -- one positive scale per
            channel, shared by a group of tokens. Note this is one scale per
            *column*, so the array is wide where variant 03's was narrow.
        group_tokens: rows sharing one scale. 32 on Ascend.

    Returns:
        ``(M, K)`` **bfloat16**.
    """
    # TODO: view q as (M/group_tokens, group_tokens, K); sf is (M/group_tokens, K)
    #       so unsqueeze at dim 1 to broadcast over the token axis
    raise NotImplementedError("torch/cast_back/04_per_channel_sf: implement torch_cast_back_per_channel")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/04")
