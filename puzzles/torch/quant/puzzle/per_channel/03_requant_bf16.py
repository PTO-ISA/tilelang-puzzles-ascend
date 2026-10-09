"""per_channel 03 (torch). See doc/quant/per_channel/03_requant_bf16.md"""

import torch

from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_channel_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                              in_group_size: int = CANONICAL_G,
                              group_tokens: int = BLOCK_MN):
    """Requantize a per-token-quantized input to per-channel scales.

    The layer-boundary operation: one layer emits per-token scales, the next
    wants per-channel ones.

    Args:
        q_in: ``(M, K)`` **float8_e4m3fn** -- already-quantized input, not raw
            activations.
        sf_in: ``(M, K/in_group_size)`` **float32** -- the per-token scales it
            was made with.
        in_group_size: channels per input scale. 32 on Ascend.
        group_tokens: rows per output scale. 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf``: ``(M/group_tokens, K)`` **float32** -- now per channel.

    The output is deliberately *not* bit-exact against the NPU tiers: input that
    already sits on the FP8 grid produces exact ties. See the variant page.
    """
    # TODO: first dequantize the per-token input exactly as cast_back/01 did:
    #       group q_in by in_group_size and multiply by sf_in.unsqueeze(-1),
    #       producing bfloat16. Then the ordinary per_channel pass on that result:
    #       reduce amax along dim=1, widen to float32, sf = amax/E4M3_MAX, and
    #       quantize with (E4M3_MAX/amax).unsqueeze(1).
    raise NotImplementedError("torch/per_channel/03_requant_bf16: implement torch_per_channel_requant")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/03")
