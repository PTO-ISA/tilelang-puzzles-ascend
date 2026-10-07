"""per_token 03 (ASC) -- write the scale as one UE8M0 byte.

New config: `use_packed_ue8m0`. Variant 02 made the scale a power of two; its
mantissa is therefore always zero, so the exponent byte alone carries all the
information. One byte instead of four.

The convenient part: variant 02 already computed exactly the byte we need.
`biased = ceil(log2(amax/448)) + 127` *is* the UE8M0 encoding. So this variant
changes only the store:

    variant 02   sf = reinterpret(biased << 23, "float32x64")   -> 4 bytes/scale
    variant 03   store the low byte of `biased` directly        -> 1 byte/scale

### Narrowing 32-bit lanes to bytes

`PK4_B32` is the "pack 4x from 32-bit" store: it takes the low byte of each
32-bit lane and writes them contiguously, so 64 lanes become 64 bytes. The
reinterpret to `"uint8x256"` is what tells the store the destination element
width; it moves no data.

    S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"), dist="PK4_B32")

### Where the int16 packing happens

The kernel writes a flat array of bytes. The "two bytes per int16" packing that
the public API presents is then just a reinterpretation of that same memory --
`uint8[num_groups]` viewed as `int16[num_groups/2]`, little-endian, so byte 2i
lands in the low half of word i. No kernel work at all; `launch()` does it with a
`.view()`.

That is worth noticing because it is easy to assume the packing needs vector
work. It only would if the pack axis were not the fastest-varying one -- which is
exactly the case in per_channel, where the scales pack along M and a real
`vintlv` is needed.

Run:  python puzzles/asc/quant/answer/per_token/03_packed_ue8m0.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0

VARIANT = "asc/per_token/03_packed_ue8m0"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize with power-of-two scales stored as UE8M0 bytes."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.uint8)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        a0 = S.vabs(S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32))
                        a1 = S.vabs(S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"),
                                           T.float32))
                        S.vsts(amax_ub[group], S.vcmax(a0, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 1], S.vcmax(a0, mask_high), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 2], S.vcmax(a1, mask_low), dist="ONEPT_B32")
                        S.vsts(amax_ub[group + 3], S.vcmax(a1, mask_high), dist="ONEPT_B32")
                    S.mem_bar("VST_VLD")

                    clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                    one = S.vdup(1, T.uint32)
                    bits = T.reinterpret(S.vmuls(clamped, inv_qmax), "uint32x64")
                    biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                    inv = T.reinterpret(
                        S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23), "float32x64")
                    # --- BEGIN SOLUTION hint="`biased` already IS the UE8M0 byte (ceil_exp + 127), so store its low byte: S.vsts(sf_ub[0], T.reinterpret(biased, 'uint8x256'), dist='PK4_B32'). PK4_B32 narrows each 32-bit lane to one byte."
                    # `biased` is already the UE8M0 encoding: ceil_exp + 127.
                    S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                    # --- END SOLUTION
                    S.vsts(inv_ub[0], inv)
                    S.mem_bar("VST_VLD")

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * (PAIR // group_size)
                        i0 = S.vld(inv_ub[group], dist="BRC_B32")
                        i1 = S.vld(inv_ub[group + 1], dist="BRC_B32")
                        i2 = S.vld(inv_ub[group + 2], dist="BRC_B32")
                        i3 = S.vld(inv_ub[group + 3], dist="BRC_B32")
                        x0 = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)
                        x1 = S.vcvt(S.vld(x_ub[col + LANES], dist="UNPK_B16"), T.float32)
                        S.vsts(q_ub[col], S.vcvt(S.vmul(x0, S.vsel(i0, i1, mask_low)),
                                                 T.float8_e4m3fn), dist="PK4_B32")
                        S.vsts(q_ub[col + LANES],
                               S.vcvt(S.vmul(x1, S.vsel(i2, i3, mask_low)),
                                      T.float8_e4m3fn), dist="PK4_B32")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) where sf_packed is int16, two exponents per word."""
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m, num_groups), dtype=torch.uint8, device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_token 03", q, sf_bytes)
    # The int16 packing is a reinterpretation of the same bytes, not kernel work.
    return q, sf_bytes.view(torch.int16)


def demo_numbers() -> None:
    print("[demo] the exponent byte is already computed by variant 02's trick:")
    for amax in (448.0, 3.5, 1.75):
        v = amax / E4M3_MAX
        bits = torch.tensor([v], dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF
        biased = ((bits - 1) >> 23) + 1
        print(f"[demo]   amax={amax:<7g} -> biased exponent byte {biased} "
              f"-> scale 2^{biased - 127}")
        assert 0 <= biased <= 255
    print("[demo] storage: 4 bytes/scale as float32 -> 1 byte/scale as UE8M0")
    print("[demo] the 'two bytes per int16' public layout is a .view() on the")
    print("[demo] same memory -- no kernel work, because the pack axis is the")
    print("[demo] fastest-varying one. (per_channel is the case where it is not.)")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_token(x, CANONICAL_G, round_sf=True, packed=True)
    q, packed = launch(x.npu())
    assert_same_bytes(packed.cpu(), ref_packed, f"sf_packed({m},{k})")
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    _, ref_f32 = oracle.per_token(x, CANONICAL_G, round_sf=True)
    assert torch.equal(decode_packed_ue8m0(packed.cpu()), ref_f32)
    print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} int16 byte-exact, "
          f"decodes back to the float32 scales")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "03_packed_ue8m0")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
