"""per_token 07 -- bfloat16 compute path, and every config at once.

Final config: do the arithmetic in bfloat16 rather than float32, then compose
with everything the earlier variants added.

### Why a bfloat16 compute path exists

A vector register here is 256 bytes. That holds **64 float32 lanes** or
**128 bfloat16 lanes**. Computing in bfloat16 therefore does twice the work per
instruction. For a bandwidth-bound kernel like this one that is a real win, and
it is why production carries a separate bf16 path at all.

### Why it is only legal with a power-of-two scale

bfloat16 has 8 mantissa bits. Multiplying by an arbitrary scale would round the
product to those 8 bits and lose precision the FP8 output could have kept.
Multiplying by a *power of two* only changes the exponent field -- it is exact in
any float format. So the fast path is gated on `round_sf`:

    production gate (per_token_cast_asc.py):
        value_dtype = bfloat16  only if  hidden % 256 == 0
                                   and  use_packed_ue8m0
                                   and  round_sf
                                   and  input is bf16 (or requant from packed)

`hidden % 256 == 0` is there because the bf16 path processes 256 values per
step, not 128. In this ladder that is why the composed variant uses K = 256.

The amax reduction is also done differently on the NPU in this path: instead of
`vabs` it masks off the sign bit with `& 0x7FFF`, because for IEEE-like formats
clearing the sign bit *is* absolute value, and a bitwise AND runs on the integer
unit. In torch that optimization is invisible -- `.abs()` either way -- which is
itself worth noticing.

### Composed config

    bfloat16 input -> bfloat16 compute -> power-of-two scale -> packed UE8M0
    -> column-major scale layout -> FP8 e4m3 output

Run:  python puzzles/torch/quant/answer/per_token/07_bf16_fast_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.math_ops import ceil_log2_exp, decode_packed_ue8m0, inv_pow2_from_exp, pack_ue8m0_row_major

VARIANT = "torch/per_token/07_bf16_fast_compose"


def torch_per_token_bf16_compose(x: torch.Tensor, group_size: int = CANONICAL_G):
    """The fully composed variant: bf16 compute, pow2 packed scale, col-major.

    Returns ``(q, sf_cm_packed)`` where the scale array is
    ``(K/group_size/PACK_FACTOR, M)`` int16 -- packed *and* transposed.
    """
    # --- BEGIN SOLUTION hint="reduce amax in bfloat16 (cast grouped to bfloat16 before .abs().amax()), then widen to float32 for the exponent math; exp = ceil_log2_exp(amax/E4M3_MAX); multiply by inv_pow2_from_exp(exp); pack (exp+127) with pack_ue8m0_row_major and transpose with oracle.to_col_major"
    m, k = x.shape
    assert k % group_size == 0
    assert k % 256 == 0, "the bf16 fast path steps 256 values at a time"
    grouped = x.view(m, k // group_size, group_size)
    # The reduction itself runs in bfloat16 -- this is the part that doubles the
    # lane count on the NPU. Widen only afterwards, for the exponent math.
    amax = grouped.to(torch.bfloat16).abs().amax(dim=-1).float().clamp(min=E4M3_CLAMP_MIN)
    exp_sf = ceil_log2_exp(amax / E4M3_MAX)
    # Multiplying by a power of two is exact, so this is safe in bfloat16.
    sf_inv = inv_pow2_from_exp(exp_sf)
    q = (grouped.float() * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    packed = pack_ue8m0_row_major((exp_sf + 127).to(torch.uint8))
    return q, oracle.to_col_major(packed)
    # --- END SOLUTION


def demo_numbers() -> None:
    print("[demo] vector register = 256 bytes")
    for name, bits in (("float32", 32), ("bfloat16", 16)):
        print(f"[demo]   {name:9} -> {256 * 8 // bits:3d} lanes per register")
    print("[demo] so bf16 compute does 2x the elements per instruction")

    print("[demo] why a power-of-two scale is required for bf16 compute:")
    # Start from a value bfloat16 holds exactly, so the only rounding that can
    # occur is in the multiply itself.
    v = torch.tensor([1.0 + 2 ** -7], dtype=torch.bfloat16)
    assert v.float().item() == 1.0 + 2 ** -7, "input must be bf16-exact"
    for label, scale in (("power of two (2^-3)", 0.125), ("arbitrary (0.1)", 0.1)):
        sb = torch.tensor(scale, dtype=torch.bfloat16)
        in_bf16 = (v * sb).float().item()
        exact = v.float().item() * sb.float().item()
        print(f"[demo]   {label:22} bf16 product err = {abs(in_bf16 - exact):.3e}")
    # Prove it rather than assert it in prose.
    exact_err = abs((v * torch.tensor(0.125, dtype=torch.bfloat16)).float().item()
                    - v.float().item() * 0.125)
    assert exact_err == 0.0, "a power-of-two multiply must be exact in bfloat16"
    print("[demo]   the power of two is exact (error identically 0); the arbitrary")
    print("[demo]   scale rounds the product to bfloat16's 8 significand bits")

    x = randn_with_zero_row(32, 256, torch.device("cpu"))
    q, sf_cm = torch_per_token_bf16_compose(x)
    groups = 256 // CANONICAL_G
    print(f"[demo] composed: x{tuple(x.shape)} bf16 -> q{tuple(q.shape)} e4m3, "
          f"sf{tuple(sf_cm.shape)} int16")
    print(f"[demo]   scale array is packed ({PACK_FACTOR} bytes/word) and transposed: "
          f"(M={x.shape[0]}, groups={groups}) -> {tuple(sf_cm.shape)}")
    assert sf_cm.shape == (groups // PACK_FACTOR, 32), sf_cm.shape


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((32, 256), (8, 512)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, sf_cm = torch_per_token_bf16_compose(x)

        ref_q, ref_packed = oracle.per_token(x, CANONICAL_G, round_sf=True, packed=True)
        assert_same_bytes(sf_cm.T.contiguous(), ref_packed, f"sf_cm.T({m},{k})")
        assert_fp8_near(q, ref_q, f"q({m},{k})")

        # the bf16 reduction must not change the chosen exponent: with bf16 input
        # the values already have 8 mantissa bits, so the amax is the same number
        _, ref_sf_f32 = oracle.per_token(x, CANONICAL_G, round_sf=True)
        assert torch.equal(decode_packed_ue8m0(sf_cm.T.contiguous()), ref_sf_f32), (
            "the bf16 compute path must pick the same power-of-two scales"
        )
        print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} ok, "
              f"same scales as the float32 path")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
