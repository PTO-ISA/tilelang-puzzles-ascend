"""per_token 03 (PTO) -- the UE8M0 byte store.

New config: `use_packed_ue8m0`. Read the ASC variant for why the exponent byte is
already in hand after variant 02's trick, and for why the "two bytes per int16"
public layout needs no kernel work.

### PTO vs ASC: a conversion instead of a reinterpret plus a store mode

ASC narrows 32-bit lanes to bytes by reinterpreting the vector to the destination
element width and asking the store to pack:

    S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"), dist="PK4_B32")

Two things have to be right together there: the reinterpret's lane count
(`uint8x256` -- 64 lanes of 32 bits seen as 256 bytes) and the matching store mode
(`PK4_B32`, "pack 4x from 32-bit"). Get either wrong and it still compiles.

VMI converts, and the destination buffer's dtype decides the packing:

    V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])

One operation, no lane bookkeeping, and the intent is legible. This is the same
pattern as cast_back/02's store: ASC selects an instruction by width, VMI infers
it from the buffer.

### compute_scale returns the exponent directly

Because `biased` is the UE8M0 encoding, the packed path wants the *integer*
exponent rather than the float32 scale. The lane-parameterised helper from variant
02 takes a flag and returns one or the other -- which is exactly the shape of
production's `compute_scale`, and only possible because the helper is reusable in
the first place.

Run:  python puzzles/pto/quant/answer/per_token/03_packed_ue8m0.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_same_bytes
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0

VARIANT = "pto/per_token/03_packed_ue8m0"
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
    def compute_scale(amax, lanes, packed: bool):
        """amax -> (scale, 1/scale), at any lane count.

        With `packed`, the first result is the *biased exponent* -- the UE8M0
        byte -- instead of the float32 scale. Same helper, one flag, which is the
        shape production's compute_scale has and which only works because the
        body is width-independent.
        """
        mask = V.create_mask(lanes, size=lanes)
        clamped = V.vmax(amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=lanes), mask)
        one = V.vbrc(T.uint32(1), size=lanes)
        shift = V.vbrc(T.uint32(23), size=lanes)
        bits = V.vinterpret_cast(
            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=lanes), mask), "uint32")
        # ((bits - 1) >> 23) + 1  ==  ceil(log2(v)) + 127
        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
        sf = biased if packed else V.vinterpret_cast(V.vshl(biased, shift), "float32")
        # 254 - biased == 127 - ceil_exp: the negated exponent, still biased.
        inv = V.vinterpret_cast(
            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=lanes), biased), shift), "float32")
        return sf, inv

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
                    mask = V.create_mask(PAIR, size=PAIR)

                    for pair in T.serial(hidden // PAIR):
                        col = pair * PAIR
                        group = pair * groups_per_pair
                        x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                        V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
                                 amax_ub[group])
                    T.simd.mem_bar("VST_VLD")

                    # TODO: call compute_scale(..., packed=True) so it returns the
                    #       biased exponent, then store it as bytes with one
                    #       conversion: V.vstore(V.vcvt(exponent, 'uint8'),
                    #       sf_ub[0]). No reinterpret and no PK4_B32 store mode --
                    #       the destination buffer dtype decides the packing.
                    raise NotImplementedError("pto/per_token/03_packed_ue8m0: implement per_token_cast")
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
    """Returns (q, sf_packed) where sf_packed is int16, two exponents per word."""
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m, num_groups), dtype=torch.uint8, device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_token 03", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)


def demo_numbers() -> None:
    print("[demo] narrowing 64 x 32-bit lanes to 64 bytes:")
    print("[demo]   ASC: T.reinterpret(v, 'uint8x256') + dist='PK4_B32'")
    print("[demo]        -- lane count and store mode must agree, and both compile")
    print("[demo]           even when they do not")
    print("[demo]   PTO: V.vcvt(v, 'uint8')  -- the buffer dtype decides the rest")
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
    sim.print_banner("pto", "per_token", "03_packed_ue8m0")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
