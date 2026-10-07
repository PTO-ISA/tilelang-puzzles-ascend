"""per_channel 03 -- requantize an already-quantized input, in bfloat16.

Two configs at once, because production couples them: ``in_config.with_sf``
(the input arrives quantized, with its own scales) and the bfloat16 compute path.

### Requant, and why it needs a scratch buffer

    x     = decode(q_in) * sf_in        dequantize with the *input* scales
    amax  = max(|x|) over 32 tokens     per channel
    q, sf = quantize(x)                 with fresh per-channel scales

The two stages cannot be fused. The new amax is not known until the whole group
of 32 tokens has been dequantized, so the dequantized values have to live
somewhere in between. On the NPU that is an explicit scratch buffer in Unified
Buffer (``dequant_ub`` in ``per_channel_cast_asc.py``) with a memory barrier on
each side of it.

The interesting case is a *granularity change*: the input might be quantized
per-token (one scale per 32 channels) while the output is per-channel (one scale
per 32 tokens). The two scale layouts are transposes of each other, so the kernel
is reading scales along one axis and writing them along the other.

### bfloat16 compute

As in per_token/07, bfloat16 doubles the lanes per register (128 instead of 64).
It is safe here for the same reason: with ``round_sf`` the scale is a power of
two, so applying it is exact in any float format.

On the NPU this variant is also where ``vgather`` shows up for the input scales:
the input scale for a given (token, channel) is not at a fixed stride from the
lane index when the input granularity differs from the output's, so the kernel
computes an index vector and gathers.

Run:  python puzzles/torch/quant/answer/per_channel/03_requant_bf16.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near
from common.math_ops import ceil_log2_exp, inv_pow2_from_exp, pow2_from_exp

VARIANT = "torch/per_channel/03_requant_bf16"


def torch_per_channel_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                              in_group_size: int = CANONICAL_G,
                              group_tokens: int = BLOCK_MN):
    """Requantize a per-token-quantized input to per-channel scales.

    ``q_in``/``sf_in`` are per-token: sf_in is (M, K/in_group_size).
    Returns ``(q, sf)`` per-channel: sf is (M/group_tokens, K).
    """
    # --- BEGIN SOLUTION hint="dequantize with oracle.cast_back(q_in, sf_in, (1, in_group_size), out_dtype=bfloat16); then reduce amax along dim=1 in bfloat16, widen to float32, take the power-of-two exponent, and apply it"
    x = oracle.cast_back(q_in, sf_in, (1, in_group_size), out_dtype=torch.bfloat16)
    m, k = x.shape
    assert m % group_tokens == 0
    grouped = x.view(m // group_tokens, group_tokens, k)
    # The reduction runs in bfloat16 -- 128 lanes per register instead of 64.
    amax = grouped.abs().amax(dim=1).float().clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped.float() * inv_pow2_from_exp(exp_sf).unsqueeze(1)).view(m, k)
    return q.to(torch.float8_e4m3fn), pow2_from_exp(exp_sf)
    # --- END SOLUTION


def demo_numbers() -> None:
    torch.manual_seed(0)
    x = randn_with_zero_row(32, 128, torch.device("cpu")) * 3
    q_in, sf_in = oracle.per_token(x, CANONICAL_G, round_sf=True)
    print(f"[demo] input is per-token quantized:  sf_in {tuple(sf_in.shape)} "
          "= (tokens, channel groups)")
    q, sf = torch_per_channel_requant(q_in, sf_in)
    print(f"[demo] output is per-channel:         sf    {tuple(sf.shape)} "
          "= (token groups, channels)")
    print("[demo]   the two scale arrays are transposes of each other in spirit --")
    print("[demo]   the kernel reads along one axis and writes along the other")

    back_in = oracle.cast_back(q_in, sf_in, (1, CANONICAL_G), out_dtype=torch.float32)
    back_out = oracle.cast_back(q, sf, (BLOCK_MN, 1), out_dtype=torch.float32)
    e1 = (back_in - x).abs().max().item() / x.abs().max().item()
    e2 = (back_out - x).abs().max().item() / x.abs().max().item()
    print(f"[demo] error vs the original: after 1st quantization {e1:.1%}, "
          f"after requantization {e2:.1%}")
    assert e2 >= e1, "requantizing cannot recover precision already lost"
    print("[demo] requantization is lossy on top of lossy -- error only grows")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (64, 128)):
        x = randn_with_zero_row(m, k, torch.device("cpu")) * 3
        q_in, sf_in = oracle.per_token(x, CANONICAL_G, round_sf=True)
        q, sf = torch_per_channel_requant(q_in, sf_in)

        # the reference path: dequantize, then the ordinary per_channel quantize
        dq = oracle.cast_back(q_in, sf_in, (1, CANONICAL_G), out_dtype=torch.bfloat16)
        ref_q, ref_sf = oracle.per_channel(dq, BLOCK_MN, round_sf=True)
        assert torch.equal(sf, ref_sf), "requant scales must match a plain per_channel pass"
        assert_fp8_near(q, ref_q, f"requant q({m},{k})")
        print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} ok "
              f"(matches dequantize-then-per_channel)")

    # bf16 vs float32 reduction must pick the same power-of-two exponent, because
    # a bf16 input already has only 8 significand bits
    x = randn_with_zero_row(64, 128, torch.device("cpu"))
    q_in, sf_in = oracle.per_token(x, CANONICAL_G, round_sf=True)
    dq_bf16 = oracle.cast_back(q_in, sf_in, (1, CANONICAL_G), out_dtype=torch.bfloat16)
    dq_f32 = oracle.cast_back(q_in, sf_in, (1, CANONICAL_G), out_dtype=torch.float32)
    _, sf_a = oracle.per_channel(dq_bf16, BLOCK_MN, round_sf=True)
    _, sf_b = oracle.per_channel(dq_f32, BLOCK_MN, round_sf=True)
    same = int((sf_a == sf_b).sum())
    print(f"[check] bf16 vs float32 reduction picks the same exponent for "
          f"{same}/{sf_a.numel()} channels")
    assert same == sf_a.numel(), "the bf16 reduction should not change the chosen exponent"


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
