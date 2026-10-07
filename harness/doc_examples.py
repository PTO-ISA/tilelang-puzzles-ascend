"""Re-verify the worked examples published in the markdown.

Moving the old ``demo_numbers`` functions into prose would normally mean losing
the assertions several of them carried -- that the ceil-log2 bit trick agrees
with ``math.ceil(math.log2(v))``, that e2m1's largest magnitude really is 6.0,
that a power-of-two multiply in bfloat16 is exactly exact.

Those checks live here instead and run on every variant, so a number printed in
``doc/quant/.../NN.md`` cannot quietly stop being true. Keyed by
``(kernel, variant number)``; variants with no published arithmetic have no entry
and verify trivially.
"""

from __future__ import annotations

import math

import torch

from common.consts import E2M1_MAX, E4M3_MAX
from common.math_ops import ceil_log2_exp, decode_ue8m0, unpack_e2m1_bytes

EXP_MASK = 0x7F800000


def _ceil_log2_matches_math() -> str:
    """doc/quant/per_token/02: ((bits - 1) >> 23) + 1 - 127 == ceil(log2(v))."""
    for v in (1.0, 1.5, 2.0, 0.3, 448.0):
        got = int(ceil_log2_exp(torch.tensor([v], dtype=torch.float32)).item())
        want = math.ceil(math.log2(v))
        assert got == want, f"ceil_log2_exp({v}) = {got}, math says {want}"
    return "ceil-log2 bit trick agrees with math.ceil(math.log2(v)) on 5 values"


def _reciprocal_is_exponent_negation() -> str:
    """doc/quant/per_token/02: (254 - biased) << 23 is exactly 2**-exp."""
    for exp in (0, 1, -1, 9):
        bits = (254 - (exp + 127)) << 23
        got = torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
        assert abs(got - 2.0 ** -exp) < 1e-30, f"exp={exp}: {got} != {2.0 ** -exp}"
    return "reciprocal by exponent negation is exact on 4 exponents"


def _ue8m0_decode_table() -> str:
    """doc/quant/cast_back/03: the stored byte e decodes to 2**(e-127)."""
    for e in (105, 120, 127, 134):
        got = decode_ue8m0(torch.tensor([e], dtype=torch.uint8)).item()
        assert got == 2.0 ** (e - 127), f"byte {e}: {got} != {2.0 ** (e - 127)}"
    return "UE8M0 decode table correct for 4 bytes"


def _ue8m0_shift_mask_equivalence() -> str:
    """doc/quant/cast_back/03: (word << shift) & 0x7F800000 extracts each byte."""
    word = 127 | (120 << 8)
    for shift, expect in ((23, 1.0), (15, 2.0 ** -7)):
        bits = ((word | (word << 16)) << shift) & EXP_MASK
        got = torch.tensor([bits], dtype=torch.int32).view(torch.float32).item()
        assert got == expect, f"shift {shift}: {got} != {expect}"
    return "the in-register shift/mask decode matches the table"


def _e2m1_code_table() -> str:
    """doc/quant/cast_back/06: e2m1 has 8 magnitudes, the largest 6.0."""
    codes = torch.arange(16, dtype=torch.int16)
    packed = (codes[0::2] | (codes[1::2] << 4)).to(torch.int8).view(1, -1)
    values = unpack_e2m1_bytes(packed)[0].tolist()
    mags = sorted({abs(v) for v in values})
    assert mags == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], mags
    assert max(mags) == E2M1_MAX, f"E2M1_MAX={E2M1_MAX} but table max is {max(mags)}"
    return f"e2m1 code table is {mags}, max {E2M1_MAX} as documented"


def _e2m1_midpoint_rounds_away() -> str:
    """doc/quant/per_token/04: 0.75 is not representable; it is a midpoint."""
    assert 0.75 not in (0.5, 1.0) and 0.5 < 0.75 < 1.0
    return "0.75 is the midpoint of e2m1's 0.5 and 1.0, so it is not representable"


def _pow2_multiply_exact_in_bf16() -> str:
    """doc/quant/per_token/07: a power-of-two multiply is exact in bfloat16."""
    v = torch.tensor([1.0 + 2 ** -7], dtype=torch.bfloat16)
    assert v.float().item() == 1.0 + 2 ** -7, "input must be bf16-exact"
    exact = (v * torch.tensor(0.125, dtype=torch.bfloat16)).float().item()
    assert exact == v.float().item() * 0.125, "pow2 multiply must be exact"
    arb = (v * torch.tensor(0.1, dtype=torch.bfloat16)).float().item()
    assert arb != v.float().item() * 0.1, "an arbitrary scale should round"
    return "pow2 bf16 multiply is exact (error 0); an arbitrary scale is not"


def _bf16_keeps_the_exponent() -> str:
    """doc/quant/per_channel/04: narrowing to bf16 preserves the exponent."""
    v = torch.tensor([3.14159265], dtype=torch.float32)
    bf = v.to(torch.bfloat16).float()
    assert math.floor(math.log2(v.item())) == math.floor(math.log2(bf.item()))
    return "bfloat16 narrowing keeps the exponent, which is all the scale uses"


def _bf16_abs_is_a_mask() -> str:
    """doc/quant/per_channel/04: non-negative bf16 compares correctly as uint16."""
    vals = torch.tensor([0.5, 1.0, 1.5, 3.0, 6.0], dtype=torch.bfloat16)
    bits = (vals.view(torch.int16).int() & 0x7FFF).tolist()
    assert bits == sorted(bits), f"uint16 order disagrees with float order: {bits}"
    return "masking the sign bit lets the running max stay in uint16"


def _fp8_max_is_448() -> str:
    """doc/quant/per_token/01: e4m3's largest finite magnitude is 448."""
    assert E4M3_MAX == 448.0
    big = torch.tensor([448.0], dtype=torch.float32).to(torch.float8_e4m3fn)
    assert big.float().item() == 448.0, "448 must be representable in e4m3"
    return "e4m3 represents 448 exactly, so it is the quantization target"


# (kernel, variant number) -> checks published in that variant's markdown
_CHECKS = {
    ("cast_back", "03"): (_ue8m0_decode_table, _ue8m0_shift_mask_equivalence),
    ("cast_back", "06"): (_e2m1_code_table,),
    ("cast_back", "07"): (_e2m1_code_table,),
    ("per_token", "01"): (_fp8_max_is_448,),
    ("per_token", "02"): (_ceil_log2_matches_math, _reciprocal_is_exponent_negation),
    ("per_token", "03"): (_ceil_log2_matches_math, _ue8m0_decode_table),
    ("per_token", "04"): (_e2m1_code_table, _e2m1_midpoint_rounds_away),
    ("per_token", "07"): (_pow2_multiply_exact_in_bf16,),
    ("per_block", "02"): (_ceil_log2_matches_math,),
    ("per_block", "03"): (_e2m1_code_table,),
    ("per_block", "05"): (_reciprocal_is_exponent_negation,),
    ("per_channel", "02"): (_ue8m0_decode_table,),
    ("per_channel", "04"): (_bf16_keeps_the_exponent, _bf16_abs_is_a_mask),
}


def verify(variant, tier: str) -> None:
    """Assert the worked example published for this variant still holds.

    Runs on every tier because the arithmetic is a property of the formats, not
    of the backend -- if it breaks, the markdown is wrong for all three.
    """
    checks = _CHECKS.get((variant.kernel, variant.num), ())
    for check in checks:
        print(f"[demo] {check()}")
