"""per_token 02 (PTO) -- the exponent trick, and lane count as a parameter.

Same arithmetic as the ASC variant: the scale becomes a power of two, built by
integer operations on the float32 exponent field rather than by division. Read
that file for the derivation.

### PTO vs ASC: this is where "width is an argument" becomes structural

ASC bakes the lane count into the dtype string it reinterprets through:

    bits   = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
    sf     = T.reinterpret(S.vshls(biased, 23), "float32x64")

Those `x64` suffixes mean the sequence only works at 64 lanes. A 128-lane or
4-lane version of the same six operations is *different code*, so ASC kernels end
up with the scale computation written out once per width that the kernel needs.

VMI derives the lane count from the total bit width, so the identical expression
works at any size -- which means the scale computation can be a **reusable helper
parameterised by width**:

    @T.macro
    def compute_scale(amax, lanes):
        ...
        return sf, sf_inv

This file defines exactly that and calls it at 64 lanes. Production's PTO
`per_token_cast_asc.py` defines the same helper once and calls it at 4, 64, 128
and 256 lanes across its various configurations; the ASC version of that file
cannot, and spells the arithmetic out per path. That is a large part of where the
port's 83 removed lines came from, and it is a structural advantage rather than a
cosmetic one -- it changes what can be factored out.

### A smaller difference worth noting

VMI has no scalar-operand `vsub`, so where ASC writes `S.vsub(bits, one)` with a
splatted `one`, VMI needs the same `V.vbrc(T.uint32(1), size=lanes)` -- a constant
vector built explicitly. Several VMI calls also require a mask that ASC leaves
implicit. This is the ergonomic cost measured by `tools/vf_lines.py`: PTO's
*operation* count is lower, but its *line* count often is not.

Run:  python puzzles/pto/quant/answer/per_token/02_round_sf.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_token/02_round_sf"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Scales are returned as float32."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    groups_per_pair = PAIR // group_size
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.macro
    def compute_scale(amax, lanes):
        """amax -> (power-of-two scale, its reciprocal), at any lane count.

        The same body serves 4, 64, 128 or 256 lanes because `size=` is an
        argument and the reinterprets carry no lane count. The ASC spelling of
        this cannot be written once: "uint32x64" fixes the width.
        """
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes), mask)
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(
            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
        # ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        sf = V.vinterpret_cast(V.vshl(biased, shift), "float32")
        # 254 - biased == 127 - ceil_exp: the negated exponent, still biased.
        inv = V.vinterpret_cast(
            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
        return sf, inv

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
                    mask = V.create_mask(PAIR, size=PAIR)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                        V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
                                 amax_ub[group])
                    T.simd.mem_bar("VST_VLD")

                    # TODO: write a compute_scale(amax, lanes) macro: clamp with
                    #       V.vmax against E4M3_CLAMP_MIN, bits =
                    #       V.vinterpret_cast(V.vmul(clamped, 1/448), 'uint32'),
                    #       biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                    #       with shift=23, then sf =
                    #       vinterpret_cast(V.vshl(biased, shift), 'float32') and
                    #       inv = vinterpret_cast(V.vshl(V.vsub(254, biased),
                    #       shift), 'float32'); call it on V.vload(amax_ub[0],
                    #       size=64) and store both
                    raise NotImplementedError("pto/per_token/02_round_sf: implement per_token_cast")
                    T.simd.mem_bar("VST_VLD")

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        inv_v = V.vload(inv_ub[group], size=PAIR, stride=1,
                                        dist_mode="brc", group=groups_per_pair)
                        x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                        V.vstore(V.vcvt(V.vmul(x, inv_v, mask), "float8_e4m3fn",
                                        rounding="R", saturate="SAT"), q_ub[col])
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
    print("[demo] the same compute_scale body at four widths:")
    for lanes in (4, 64, 128, 256):
        print(f"[demo]   size={lanes:3d} -> V.vbrc(T.uint32(254), size={lanes}) "
              f"and vinterpret_cast(..., 'float32')")
    print("[demo] the ASC spelling fixes the width in the type it reinterprets")
    print("[demo] through -- 'uint32x64' / 'float32x64' -- so each width needs its")
    print("[demo] own copy of the six operations. Production's PTO per_token calls")
    print("[demo] one helper at 4, 64, 128 and 256 lanes; the ASC file cannot.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, round_sf=True)
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=0)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    mant = sf.cpu().view(torch.int32) & 0x7FFFFF
    assert int(mant.abs().max()) == 0
    print(f"[check] shape=({m},{k}) bit-exact scales, all powers of two")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_token", "02_round_sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
