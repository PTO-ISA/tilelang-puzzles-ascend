"""per_token 04 (PTO) -- float32 input, FP4 output, one vector wide.

Read the ASC variant for why FP4 output needs bfloat16 in the middle and why that
intermediate must be rounded to odd. Neither backend has a float32 -> e2m1
convert; both error out if you ask.

### PTO vs ASC: the round-to-odd step keeps one vector

The round-to-odd trick needs the low and high 16-bit halves of each float32
separated. ASC's `vdintlv` deinterleaves *two registers at a time*, so the
sequence is built around combining the kernel's two 64-lane halves:

    ASC:
        low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                              T.reinterpret(q1, "uint16x128"))
        odd = S.vor(high, S.vmin(low, one_u16))
        S.vsts(q_ub[col], S.vcvt(T.reinterpret(odd, "bfloat16x128"),
                                 T.float4_e2m1fn), dist="PK4_B32")

VMI's `vunzip` splits one vector into its halves, and the vector is already 128
lanes, so there is nothing to combine:

    PTO:
        low, high = V.vunzip(q, "uint16")
        odd = V.vinterpret_cast(V.vor(high, V.vmin(low, one_u16)), "bfloat16")
        V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"), q_ub[col])

Same three steps, but ASC's version only makes sense as a *pairwise* operation on
two registers, while PTO's is a property of one value. That difference shows up as
soon as the kernel wants a different width: the ASC form is pinned to exactly two
64-lane inputs.

`V.vcvt(..., rounding="R")` also states the final rounding mode explicitly, where
ASC leaves it to the instruction's default.

### A toolchain gap worth knowing (both backends)

The sticky bit wants `min(low, 1)`. The scalar-operand spelling is what production
uses, but on this build `S.vmins` / its VMI equivalent has no uint16 overload:

    error: no matching function for call to 'asc_min_scalar'

Both files therefore use the vector form against a splatted 1, which costs one
extra broadcast. Measured result is unaffected: 0 of 4096 FP4 codes differ from
the bit-exact torch packer.

Run:  python puzzles/pto/quant/answer/per_token/04_fp32_in_fp4_out.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp32_ulps
from common.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX
from common.demo import randn_with_zero_row
from common.math_ops import unpack_e2m1_bytes

VARIANT = "pto/per_token/04_fp32_in_fp4_out"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize float32 -> packed FP4 e2m1 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_pair = PAIR // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.macro
    def to_bf16_round_odd(x, lanes):
        """float32 -> bfloat16, rounded to odd, so a later rounding cannot tie.

        `vunzip` splits one vector into its low and high 16-bit halves. The high
        halves are the truncated bfloat16 values; `min(low, 1)` is the sticky bit.
        """
        low, high = V.vunzip(x, "uint16")
        one = V.vbrc(T.uint16(1), size=lanes)
        return V.vinterpret_cast(V.vor(high, V.vmin(low, one)), "bfloat16")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.float32)
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # --- BEGIN SOLUTION hint="float32 input needs no convert on load. Use E2M1_MAX/E2M1_CLAMP_MIN. For the store, write a to_bf16_round_odd(x, lanes) macro using low, high = V.vunzip(x, 'uint16') then V.vor(high, V.vmin(low, V.vbrc(T.uint16(1), size=lanes))) reinterpreted to 'bfloat16'; then V.vcvt(..., 'float4_e2m1fn', rounding='R') and store"
                with T.SimdVF():
                    mask = V.create_mask(PAIR, size=PAIR)
                    mask64 = V.create_mask(LANES, size=LANES)
                    qmax = V.vbrc(T.float32(E2M1_MAX), size=LANES)
                    clamp_min = V.vbrc(T.float32(E2M1_CLAMP_MIN), size=LANES)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        # float32 input: a plain load, no convert.
                        x = V.vload(x_ub[col], size=PAIR)
                        V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
                                 amax_ub[group])
                    T.simd.mem_bar("VST_VLD")

                    clamped = V.vmax(V.vload(amax_ub[0], size=LANES), clamp_min, mask64)
                    V.vstore(V.vdiv(clamped, qmax, mask64), sf_ub[0])
                    V.vstore(V.vdiv(qmax, clamped, mask64), inv_ub[0])
                    T.simd.mem_bar("VST_VLD")

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        inv = V.vload(inv_ub[group], size=PAIR, stride=1,
                                      dist_mode="brc", group=groups_per_pair)
                        q = V.vmul(V.vload(x_ub[col], size=PAIR), inv, mask)
                        # One 128-lane vector all the way through: no pairing up
                        # of halves, because vunzip splits a single value.
                        odd = to_bf16_round_odd(q, PAIR)
                        V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"), q_ub[col])
                # --- END SOLUTION
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8,
                    device=x.device).view(torch.float4_e2m1fn_x2)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 04", q, sf)
    return q.view(torch.uint8).view(torch.int8), sf


def demo_numbers() -> None:
    print("[demo] separating the halves of a float32 for the round-to-odd step:")
    print("[demo]   ASC: S.vdintlv(a, b) -- deinterleaves TWO registers, so the")
    print("[demo]        kernel's two 64-lane halves must be paired up for it")
    print("[demo]   PTO: V.vunzip(x, 'uint16') -- splits ONE value, and the value")
    print("[demo]        is already 128 lanes, so there is nothing to pair")
    v = 1.25 + 2.0 ** -20
    bits = torch.tensor([v], dtype=torch.float32).view(torch.int32).item()
    high, low = (bits >> 16) & 0xFFFF, bits & 0xFFFF
    trunc = torch.tensor([high << 16], dtype=torch.int32).view(torch.float32).item()
    odd = torch.tensor([(high | min(low, 1)) << 16],
                       dtype=torch.int32).view(torch.float32).item()
    print(f"[demo] v={v!r}: truncated {trunc!r}, round-to-odd {odd!r}")
    assert odd != trunc


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"), dtype=torch.float32) * 3
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, fmt="e2m1")
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)
    got_v, ref_v = unpack_e2m1_bytes(q.cpu()), unpack_e2m1_bytes(ref_q)
    differing, total = int((got_v != ref_v).sum()), got_v.numel()
    print(f"[check] shape=({m},{k}) FP4 values differing from the torch packer: "
          f"{differing}/{total} ({differing / total:.2%})")
    assert differing / total < 0.02
    back = got_v * sf.cpu().repeat_interleave(CANONICAL_G, dim=1)
    rel = (back - x).abs().max().item() / x.abs().max().item()
    print(f"[check] round-trip rel-err {rel:.1%}")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_token", "04_fp32_in_fp4_out")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
