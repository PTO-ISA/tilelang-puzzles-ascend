"""per_channel 04 (PTO) -- everything composed. The last file in the ladder.

Composed config:

    bfloat16 input -> bfloat16 reduction -> power-of-two scale
    -> UE8M0 packed along M -> FP8 e4m3 output

Read the ASC variant for the integer-unit reduction trick (clearing the sign bit
is abs, and non-negative floats compare correctly as unsigned integers).

### PTO vs ASC: a fair summary of the whole ladder

This variant is close to a draw, and by now the reason should be predictable. Its
three stages are:

    reduce along M      lanes stay lanes, no grouping    -> no VMI advantage
    scale math          one value per channel            -> no VMI advantage
    pack along M        a plain interleave                -> no VMI advantage

VMI pays where ASC has to *emulate* something: a segment width the hardware does
not have (`group=` on reduces and broadcasts), a conversion ASC expresses as a
distribution mode, or a vector wider than one register. per_channel has none of
those, because its reduction axis already lines up with the register's lanes --
which is the same property that makes it the easy kernel on this hardware and the
hard one on a GPU.

Measured over all 23 paired variants, `python tools/vf_lines.py` puts PTO at
roughly two thirds of ASC's static vector-operation count, concentrated in
per_token (where `group=` replaces mask-and-select) and in the FP4 and bfloat16
paths (where width is the problem). cast_back/03, cast_back/05, cast_back/07,
per_token/05 and all of per_block and per_channel are draws or near-draws. Both
numbers are worth stating: the advantage is real, large where it applies, and
absent where it does not.

Run:  python puzzles/pto/quant/answer/per_channel/04_compose.py
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
from common.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import randn_with_zero_row
from common.math_ops import decode_packed_ue8m0_along_m

VARIANT = "pto/per_channel/04_compose"
LANES = 64              # float32 lanes, for the scale math
BF16_LANES = 128        # bfloat16 lanes, for the reduction
BYTE_LANES = 128        # channels per interleave step


@tilelang.jit(target="pto")
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
                    # --- BEGIN SOLUTION hint="stage 1: reduce in bfloat16. Per 128-channel tile keep acc = V.alloc_local((1,), V.vreg(128, T.uint16)) seeded to 1, and do acc[0] = V.vmax(acc[0], V.vand(V.vinterpret_cast(V.vload(x_ub[row, col], size=128), 'uint16'), abs_mask), bmask) over the 32 rows, then V.vstore(V.vinterpret_cast(acc[0], 'bfloat16'), amax_bf16_ub[col]). Barrier. stage 2: reload with V.vcvt(V.vload(..., size=64), 'float32') and run the exponent trick, writing the byte with V.vcvt(biased, 'uint8'). Barrier, apply. Finally interleave the two byte rows as in variant 02."
                    with T.SimdVF():
                        bmask = V.create_mask(BF16_LANES, size=BF16_LANES)
                        fmask = V.create_mask(LANES, size=LANES)
                        abs_mask = V.vbrc(T.uint16(0x7FFF), size=BF16_LANES)
                        one = V.vbrc(T.uint32(1), size=LANES)
                        shift = V.vbrc(T.uint32(23), size=LANES)
                        b254 = V.vbrc(T.uint32(254), size=LANES)

                        # ---- stage 1: reduce in bfloat16, on the integer unit ----
                        for bt in T.serial(num_bf16_tiles):
                            col = bt * BF16_LANES
                            acc = V.alloc_local((1,), V.vreg(BF16_LANES, T.uint16))
                            acc[0] = V.vbrc(T.uint16(0x0001), size=BF16_LANES)
                            for row in T.serial(group_tokens):
                                bits = V.vinterpret_cast(
                                    V.vload(x_ub[row, col], size=BF16_LANES), "uint16")
                                # clearing the sign bit is abs, and for
                                # non-negative floats the integer order matches
                                acc[0] = V.vmax(acc[0], V.vand(bits, abs_mask), bmask)
                            V.vstore(V.vinterpret_cast(acc[0], "bfloat16"),
                                     amax_bf16_ub[col])
                        T.simd.mem_bar("VST_VLD")

                        # ---- stage 2: scale math in float32, as always ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            amax = V.vcvt(V.vload(amax_bf16_ub[col], size=LANES),
                                          "float32")
                            clamped = V.vmax(
                                amax, V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES),
                                fmask)
                            bits = V.vinterpret_cast(
                                V.vmul(clamped,
                                       V.vbrc(T.float32(1.0 / E4M3_MAX), size=LANES),
                                       fmask), "uint32")
                            biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                            V.vstore(V.vcvt(biased, "uint8"), sf_rows_ub[sub, col])
                            V.vstore(V.vinterpret_cast(
                                V.vshl(V.vsub(b254, biased), shift), "float32"),
                                inv_ub[col])
                        T.simd.mem_bar("VST_VLD")

                        # ---- stage 3: apply; the scale is one value per lane ----
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            inv = V.vload(inv_ub[col], size=LANES)
                            for row in T.serial(group_tokens):
                                v = V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                           "float32")
                                V.vstore(V.vcvt(V.vmul(v, inv, fmask),
                                                "float8_e4m3fn", rounding="R",
                                                saturate="SAT"), q_ub[row, col])
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                # pack the pair's two byte rows along M (variant 02)
                with T.SimdVF():
                    T.simd.mem_bar("VST_VLD")
                    byte_mask = V.create_mask(BYTE_LANES * 2, size=BYTE_LANES * 2)
                    for bt in T.serial(num_byte_tiles):
                        col = bt * BYTE_LANES
                        a = V.vload(sf_rows_ub[0, col], size=BYTE_LANES * 2)
                        b = V.vload(sf_rows_ub[1, col], size=BYTE_LANES * 2)
                        lo, _ = V.vintlv(a, b, byte_mask)
                        V.vstore(lo, packed_ub[col * PACK_FACTOR])
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
    print("[demo] reduction needs no deinterleave/regroup dance at all -- which")
    print("[demo] is also why VMI has no advantage to offer in this kernel.")


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
    sim.print_banner("pto", "per_channel", "04_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
