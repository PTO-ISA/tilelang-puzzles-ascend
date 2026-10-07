"""per_token 05 -- write the scale array transposed (TMA-aligned column-major).

New config: ``use_tma_aligned_col_major_sf``. The scales are the same numbers;
only their memory layout changes.

    row-major (what variants 01-04 wrote)   sf[token, group]     shape (M, K/32)
    column-major (this variant)             sf[group, token]     shape (K/32, M)

The public API still presents (M, K/32) -- the host just takes a ``.T`` view.
So why make the kernel do the transpose?

Because of who reads the scales next. The GEMM consuming this quantized tensor
processes a tile of tokens at a time and needs that tile's scales contiguously,
so it can pull them with one bulk async copy (a TMA on GPU; the DMA engine
here). In row-major order, one group's scales across 32 tokens are strided by
K/32. In column-major they are adjacent. Transposing in the quantizer is free-ish;
transposing in the GEMM's inner loop is not.

In torch this variant is a one-line ``.T``, and that is the honest answer: the
layout question barely exists at this level of abstraction.

On the NPU it is the hardest variant in the ladder. A transpose *inside a vector
register* is not a memory move -- you cannot restride a register. The kernel
instead builds a vector of source indices arithmetically from the lane id
(``vci`` gives each lane its own index) and performs an in-register gather.
Reading the ASC and PTO 05 files next to this one shows exactly which part of
that complexity is the hardware's and which is the DSL's.

Run:  python puzzles/torch/quant/answer/per_token/05_col_major_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near

VARIANT = "torch/per_token/05_col_major_sf"


def torch_per_token_cast_col_major(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize, returning the scales in the kernel-native column-major layout.

    Returns ``(q, sf_cm)`` where ``sf_cm`` is (K/group_size, M).
    """
    # --- BEGIN SOLUTION hint="compute (q, sf) with power-of-two scales as in variant 02, then return oracle.to_col_major(sf) -- a transpose -- instead of sf"
    from common.math_ops import ceil_log2_exp, inv_pow2_from_exp, pow2_from_exp

    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    q = (grouped * inv_pow2_from_exp(exp_sf).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, oracle.to_col_major(pow2_from_exp(exp_sf))
    # --- END SOLUTION


def demo_numbers() -> None:
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    for t in range(4):
        x[t, 0] = 2.0 ** t          # group 0 amax varies per token
        x[t, 32] = 1.0              # group 1 amax is the same for all tokens
    q, sf_cm = torch_per_token_cast_col_major(x)
    print("[demo] M=4 tokens, K=64 -> two groups")
    print(f"[demo] column-major sf shape {tuple(sf_cm.shape)} = (groups, tokens)")
    print_example("per_token 05", sf_col_major=sf_cm, sf_row_major_view=sf_cm.T)
    print("[demo] row 0 of the column-major array is group 0's scale for every token:")
    print(f"[demo]   {sf_cm[0].tolist()}  <- varies, because each token's amax differs")
    print(f"[demo]   {sf_cm[1].tolist()}  <- constant, all tokens share amax=1.0 here")
    assert sf_cm.shape == (2, 4), sf_cm.shape
    assert len(set(sf_cm[1].tolist())) == 1, "group 1 should have one shared scale value"


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 128), (8, 64)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf_cm = torch_per_token_cast_col_major(x)
        ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, round_sf=True)
        assert sf_cm.shape == (k // CANONICAL_G, m), sf_cm.shape
        assert torch.equal(sf_cm.T.contiguous(), ref_sf), "transposing back must give the row-major scales"
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} ok, "
              f"transposes back to {tuple(ref_sf.shape)}")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
