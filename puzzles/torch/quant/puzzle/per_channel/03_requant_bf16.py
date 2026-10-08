"""per_channel 03 (torch). See doc/quant/per_channel/03_requant_bf16.md"""

import torch

from harness import oracle
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_channel_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                              in_group_size: int = CANONICAL_G,
                              group_tokens: int = BLOCK_MN):
    """Requantize a per-token-quantized input to per-channel scales.

    ``q_in``/``sf_in`` are per-token: sf_in is (M, K/in_group_size).
    Returns ``(q, sf)`` per-channel: sf is (M/group_tokens, K).
    """
    # TODO: dequantize with oracle.cast_back(q_in, sf_in, (1, in_group_size),
    #       out_dtype=bfloat16); then reduce amax along dim=1 in bfloat16, widen
    #       to float32, and apply sf = amax/E4M3_MAX as in variant 01
    raise NotImplementedError("torch/per_channel/03_requant_bf16: implement torch_per_channel_requant")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/03")
