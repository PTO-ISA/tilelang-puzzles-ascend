"""per_channel 04 (ASC) -- everything composed: bfloat16 reduction, pow2, packed-M.

Final variant of the final kernel. Composed config:

    bfloat16 input -> bfloat16 reduction -> power-of-two scale
    -> UE8M0 packed along M -> FP8 e4m3 output

That is production's per_channel configuration. What remains between this file and
`per_channel_cast_asc.py` is scheduling -- multiple vector cores with a manual wave
index, double-buffered UB, and L2 cache hints on the DMA -- plus the requant path,
which is variant 03.

### Reducing in bfloat16 without leaving the integer unit

per_token/07 reduced in bfloat16 and needed an elaborate dance to get groups of 32
out of it. per_channel needs no grouping at all -- the reduction is along M, so
lanes stay lanes -- which makes the bfloat16 reduction almost free here:

    abs_bits   = S.vand(T.reinterpret(x, "uint16x128"), 0x7FFF)
    acc[0]     = S.vmax(acc[0], abs_bits)        # 128 channels per step

Two things are going on in that `vmax`:

1. Clearing the sign bit is `abs` (per_token/07).
2. Comparing the *bit patterns* of non-negative floats as unsigned integers gives
   the same ordering as comparing the floats. So the running maximum can be kept
   in `uint16` and never touch the float unit.

That second point is the trick worth remembering. It only works because the values
are known non-negative after the mask -- for signed input the integer ordering and
the float ordering disagree.

### The scale math still runs in float32

The maxima are written out as bfloat16 and reloaded widened, so the exponent
arithmetic is the same 64-lane float32 sequence as every other variant. Reducing
in bfloat16 costs nothing in the scale, because only its exponent is used and
bfloat16 keeps the exponent exactly -- the test asserts that the chosen exponents
match what a float32 reduction would pick.

Run:  python puzzles/asc/quant/answer/per_channel/04_compose.py
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
from common.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0_along_m

VARIANT = "asc/per_channel/04_compose"
LANES = 64              # float32 lanes, for the scale math
BF16_LANES = 128        # bfloat16 lanes, for the reduction
BYTE_LANES = 128        # channels per interleave step


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """bfloat16 reduction + power-of-two scale + UE8M0 packed along M."""
    assert hidden % BF16_LANES == 0 and group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_pairs = T.ceildiv(num_groups, PACK_FACTOR)
    num_bf16_tiles = hidden // BF16_LANES
    num_col_tiles = hidden // LANES
    num_byte_tiles = hidden // BYTE_LANES

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_pairs, hidden * PACK_FACTOR), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            amax_bf16_ub = T.alloc_shared((hidden,), T.bfloat16)
            inv_ub = T.alloc_shared((hidden,), T.float32)
            sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden + BYTE_LANES), T.uint8)
            packed_ub = T.alloc_shared((hidden * PACK_FACTOR,), T.uint8)

            for pair in T.serial(num_pairs):
                for sub in T.serial(PACK_FACTOR):
                    mg = pair * PACK_FACTOR + sub
                    T.copy(X[mg * group_tokens, 0], x_ub)
                    # --- BEGIN SOLUTION hint="stage 1: reduce in bfloat16. abs_mask = S.vdup(0x7FFF, T.uint16); per 128-channel tile keep acc = S.alloc_local((1,), T.uint16) and do acc[0] = S.vmax(acc[0], S.vand(T.reinterpret(S.vld(x_ub[row, col], dist='NORM_B16'), 'uint16x128'), abs_mask)) over the 32 rows, then store T.reinterpret(acc[0], 'bfloat16x128') to amax_bf16_ub. Barrier. stage 2: reload widened with UNPK_B16 + vcvt and run the usual exponent trick, writing the byte to sf_rows_ub[sub] and the inverse to inv_ub. Barrier, apply. Finally interleave the two byte rows as in variant 02."
                    with T.SimdVF():
                        abs_mask = S.vdup(0x7FFF, T.uint16)
                        clamp_u16 = S.vdup(0x0001, T.uint16)
                        one = S.vdup(1, T.uint32)

                        # ---- stage 1: reduce in bfloat16, on the integer unit ----
                        for bt in T.serial(num_bf16_tiles):
                            col = bt * BF16_LANES
                            acc = S.alloc_local((1,), T.uint16)
                            acc[0] = clamp_u16
                            for row in T.serial(group_tokens):
                                bits = T.reinterpret(
                                    S.vld(x_ub[row, col], dist="NORM_B16"),
                                    "uint16x128")
                                # clearing the sign bit is abs, and for
                                # non-negative floats the integer order matches
                                acc[0] = S.vmax(acc[0], S.vand(bits, abs_mask))
                            S.vsts(amax_bf16_ub[col],
                                   T.reinterpret(acc[0], "bfloat16x128"),
                                   dist="NORM_B16")
                        S.mem_bar("VST_VLD")

                        # ---- stage 2: scale math in float32, as always ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            amax = S.vcvt(S.vld(amax_bf16_ub[col], dist="UNPK_B16"),
                                          T.float32)
                            clamped = S.vmaxs(amax, E4M3_CLAMP_MIN)
                            bits = T.reinterpret(S.vmuls(clamped, 1.0 / E4M3_MAX),
                                                 "uint32x64")
                            biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                            S.vsts(sf_rows_ub[sub, col],
                                   T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                            S.vsts(inv_ub[col], T.reinterpret(
                                S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                                "float32x64"))
                        S.mem_bar("VST_VLD")

                        # ---- stage 3: apply; the scale is one value per lane ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            inv = S.vld(inv_ub[col])
                            for row in T.serial(group_tokens):
                                v = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                                           T.float32)
                                S.vsts(q_ub[row, col],
                                       S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                       dist="PK4_B32")
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                # pack the pair's two byte rows along M (variant 02)
                with T.SimdVF():
                    S.mem_bar("VST_VLD")
                    for bt in T.serial(num_byte_tiles):
                        col = bt * BYTE_LANES
                        a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
                        b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
                        lo, _ = S.vintlv(a, b)
                        S.vsts(packed_ub[col * PACK_FACTOR], lo, dist="NORM_B8")
                    # --- END SOLUTION
                T.copy(packed_ub, Sf[pair, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    pairs = m // (BLOCK_MN * PACK_FACTOR)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((pairs, hidden * PACK_FACTOR), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_channel 04", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)


def demo_numbers() -> None:
    print("[demo] reducing in bfloat16 on the integer unit:")
    print("[demo]   1. clearing the sign bit (& 0x7FFF) is abs")
    print("[demo]   2. for non-negative floats, comparing bit patterns as")
    print("[demo]      unsigned integers gives the same order as comparing floats")
    vals = torch.tensor([0.5, 1.0, 1.5, 3.0, 6.0], dtype=torch.bfloat16)
    bits = (vals.view(torch.int16).int() & 0x7FFF).tolist()
    print(f"[demo]   bf16 {vals.tolist()}")
    print(f"[demo]   bits {bits}   -- increasing, same order")
    assert bits == sorted(bits)
    print("[demo]   so the running max never leaves uint16.")
    print("[demo] this only holds after the mask: for signed values the integer")
    print("[demo] order and the float order disagree.")
    print("[demo] per_channel needs no grouping, so unlike per_token/07 the bf16")
    print("[demo] reduction needs no deinterleave/regroup dance at all.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    step = BLOCK_MN * PACK_FACTOR
    if m % step:
        m = (m // step + 1) * step
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_channel(x, BLOCK_MN, round_sf=True, packed=True)
    q, packed = launch(x.npu())
    assert_same_bytes(packed.cpu(), ref_packed, f"sf_packed({m},{k})")
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    _, ref_f32 = oracle.per_channel(x, BLOCK_MN, round_sf=True)
    assert torch.equal(decode_packed_ue8m0_along_m(packed.cpu()), ref_f32), (
        "the bfloat16 reduction changed the chosen exponent"
    )
    print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} byte-exact, and")
    print(f"[check] the bfloat16 reduction picked the same exponents as float32")


def main() -> int:
    k = sim.sim_shapes()[1]
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_channel", "04_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
