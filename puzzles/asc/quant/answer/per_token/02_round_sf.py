"""per_token 02 (ASC) -- the ceil-log2 bit trick, in vector registers.

New config: `round_sf`. The scale becomes the smallest power of two that is at
least `amax / 448`. See puzzles/torch/quant/answer/per_token/02_round_sf.py for
why (exactness, and it makes the one-byte UE8M0 scale of variant 03 possible) and
for the hand-checked arithmetic.

### The whole thing is integer arithmetic on the exponent field

A float32 is `sign | exponent(8) | mantissa(23)`. So:

    biased = ((bits - 1) >> 23) + 1          == ceil(log2(v)) + 127

The `- 1` before the shift is what turns floor into ceil. Floor would make the
scale too small and let values saturate past 448.

Then both the scale and its reciprocal are built by *placing an exponent*, with no
division anywhere:

    sf      = biased        << 23            ->  2^ceil_exp
    sf_inv  = (254 - biased) << 23           ->  2^-ceil_exp

`254 - biased` works out as `127 - ceil_exp`, i.e. the negated exponent, still
biased. Negating an exponent field is exact; dividing would not be.

Every one of these steps is a vector operation on 64 lanes at once, so 64 groups'
scales are computed in about six instructions:

    bits     = T.reinterpret(S.vmuls(clamped, 1/448), "uint32x64")
    biased   = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
    sf       = T.reinterpret(S.vshls(biased, 23), "float32x64")
    sf_inv   = T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                             "float32x64")

Note `T.reinterpret` is free -- it renames the bits, it does not move them. The
vector unit does integer and float operations on the same registers.

### A simulator note

These shifts are the reason this repo defaults to `msprof op simulator` rather
than `cannsim`. `cannsim` **hangs** on `vshr`/`vshl`, so every variant from here
on is unrunnable under that backend. See doc/known-issues.md.

Run:  python puzzles/asc/quant/answer/per_token/02_round_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_token/02_round_sf"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Scales are returned as float32."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)

                    # pass 1: unchanged from variant 01
                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32))
                        a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                           T.float32))
                        S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high),
                               dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low),
                               dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high),
                               dist="ONEPT_B32")
                    S.mem_bar("VST_VLD")

                    # --- BEGIN SOLUTION hint="replace variant 01's divides with the exponent trick: clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN); bits = T.reinterpret(S.vmuls(clamped, 1/448), 'uint32x64'); biased = S.vadds(S.vshrs(S.vsub(bits, S.vdup(1, T.uint32)), 23), 1); then sf = reinterpret(S.vshls(biased, 23), 'float32x64') and inv = reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23), 'float32x64')"
                    # pass 2: amax -> power-of-two scale, by exponent arithmetic
                    clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                    one = S.vdup(1, T.uint32)
                    bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
                    # ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
                    biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                    sf = T.reinterpret(S.vshls(biased, 23), "float32x64")
                    # 254 - biased == 127 - ceil_exp: the negated exponent
                    inv = T.reinterpret(
                        S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                        "float32x64")
                    S.vsts(sf_ub[0], sf)
                    S.vsts(inv_ub[0], inv)
                    # --- END SOLUTION
                    S.mem_bar("VST_VLD")

                    # pass 3: unchanged from variant 01
                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        i0 = S.vld(inv_ub[group], dist="BRC_B32")
                        i1 = S.vld(inv_ub[group + 1], dist="BRC_B32")
                        i2 = S.vld(inv_ub[group + 2], dist="BRC_B32")
                        i3 = S.vld(inv_ub[group + 3], dist="BRC_B32")
                        x0 = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
                        x1 = S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"), T.float32)
                        S.vsts(q_ub[col],
                               S.vcvt(S.vmul(x0, S.vsel(i0, i1, mask_low)),
                                      T.float8_e4m3fn), dist="PK4_B32")
                        S.vsts(q_ub[col + LANES],
                               S.vcvt(S.vmul(x1, S.vsel(i2, i3, mask_low)),
                                      T.float8_e4m3fn), dist="PK4_B32")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 02", q, sf)
    return q, sf


def demo_numbers() -> None:
    import math

    print("[demo] ceil(log2(v)) from the exponent field, no log2 instruction:")
    for v in (1.0, 1.5, 2.0, 0.3):
        bits = torch.tensor([v], dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF
        biased = ((bits - 1) >> 23) + 1
        exp = biased - 127
        assert exp == math.ceil(math.log2(v)), (v, exp)
        print(f"[demo]   v={v:<5g} bits=0x{bits:08X} -> biased={biased} "
              f"-> exp={exp:+d} -> 2^exp={2.0 ** exp:g}")
    print("[demo] and the reciprocal is the negated exponent, not a divide:")
    for exp in (0, 1, -1, 9):
        biased = exp + 127
        inv_bits = (254 - biased) << 23
        inv = torch.tensor([inv_bits], dtype=torch.int32).view(torch.float32).item()
        assert abs(inv - 2.0 ** -exp) < 1e-30, (exp, inv)
        print(f"[demo]   (254 - {biased}) << 23 -> {inv:g} == 2^{-exp}")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, round_sf=True)
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=0)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    mant = sf.cpu().view(torch.int32) & 0x7FFFFF
    assert int(mant.abs().max()) == 0, "every scale must be an exact power of two"
    print(f"[check] shape=({m},{k}) bit-exact scales, all powers of two")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "02_round_sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
