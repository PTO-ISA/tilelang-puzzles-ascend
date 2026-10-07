"""per_channel 02 (PTO) -- UE8M0 packed along M, in VMI.

Read the ASC variant: it explains why per_channel is the one kernel whose scale
bytes pack along M rather than along the fastest-varying axis, and therefore the
one where packing costs an instruction instead of a host-side `.view()`.

### PTO vs ASC

The interleave itself is a transliteration -- `V.vintlv` takes the same two
vectors and returns the same two results, with a mask added:

    ASC:  lo, _ = S.vintlv(a, b)
    PTO:  lo, _ = V.vintlv(a, b, mask)

No saving, and the same width subtlety applies: a uint8 vector is 256 lanes, so
`lo` alone carries the 256 output bytes for 128 channels, the rows are padded so
the oversized load cannot reach the next row, and the second result is discarded.

Where VMI does help in this kernel is the surrounding code rather than the pack:
the reduction runs at 128 channels per tile instead of 64 (variant 01), and the
scale byte is written with one `V.vcvt(biased, "uint8")` instead of a reinterpret
plus a `PK4_B32` store mode (per_token/03).

### An aside worth carrying forward

`vintlv` appears three times in this ladder doing three different jobs:

    cast_back/06       vintlv(zero, x_bf16)   widen bfloat16 -> float32
    per_token/04       vdintlv(q0, q1)        split float32 into its 16-bit halves
    per_channel/02     vintlv(row_a, row_b)   pack two byte rows into one

Interleaving with zeros builds wider elements; interleaving two real vectors packs
narrower ones; de-interleaving splits. The instruction name describes the data
movement and says nothing about which of those you are doing, so these are worth
recognising as idioms rather than deriving each time.

Run:  python puzzles/pto/quant/answer/per_channel/02_round_packed_m.py
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

VARIANT = "pto/per_channel/02_round_packed_m"
LANES = 128
BYTE_LANES = 128        # uint8 lanes per interleave step


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Quantize with power-of-two scales packed along M (two m-groups per int16)."""
    assert hidden % BYTE_LANES == 0, "the interleave step covers 128 channels"
    assert group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_pairs = T.ceildiv(num_groups, PACK_FACTOR)
    num_col_tiles = hidden // LANES
    num_byte_tiles = hidden // BYTE_LANES

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        # one packed row per *pair* of m-groups: 2 bytes per channel
        Sf: T.Tensor((num_pairs, hidden * PACK_FACTOR), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            inv_ub = T.alloc_shared((hidden,), T.float32)
            # one scale-byte row per member of the pair, then their interleave
            # Padded by one byte-tile: a uint8 vector load is 256 lanes, so the
            # interleave reads 256 bytes even when only 128 are meaningful. The
            # padding keeps it from reading into the next row.
            sf_rows_ub = T.alloc_shared((PACK_FACTOR, hidden + BYTE_LANES), T.uint8)
            packed_ub = T.alloc_shared((hidden * PACK_FACTOR,), T.uint8)

            for pair in T.serial(num_pairs):
                for sub in T.serial(PACK_FACTOR):
                    mg = pair * PACK_FACTOR + sub
                    T.copy(X[mg * group_tokens, 0], x_ub)
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                        one = V.vbrc(T.uint32(1), size=LANES)
                        shift = V.vbrc(T.uint32(23), size=LANES)
                        b254 = V.vbrc(T.uint32(254), size=LANES)
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
                            acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
                            for row in T.serial(group_tokens):
                                v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                                  "float32"), mask)
                                acc[0] = V.vmax(acc[0], v, mask)
                            # the exponent trick, one value per channel
                            bits = V.vinterpret_cast(
                                V.vmul(acc[0],
                                       V.vbrc(T.float32(1.0 / E4M3_MAX), size=LANES),
                                       mask), "uint32")
                            biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                            # one convert, no reinterpret + store mode
                            V.vstore(V.vcvt(biased, "uint8"), sf_rows_ub[sub, col])
                            V.vstore(V.vinterpret_cast(
                                V.vshl(V.vsub(b254, biased), shift), "float32"),
                                inv_ub[col])
                        T.simd.mem_bar("VST_VLD")

                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            inv = V.vload(inv_ub[col], size=LANES)
                            for row in T.serial(group_tokens):
                                v = V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                           "float32")
                                V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                                rounding="R", saturate="SAT"),
                                         q_ub[row, col])
                    T.copy(q_ub, Q[mg * group_tokens, 0])

                # TODO: the two scale-byte rows of this pair interleave into one
                #       packed row. For each 128-channel tile: a =
                #       V.vload(sf_rows_ub[0, col], size=256), b likewise from row
                #       1, then lo, _ = V.vintlv(a, b, byte_mask) and V.vstore(lo,
                #       packed_ub[col*2]). A uint8 register is 256 lanes, so lo
                #       alone covers these 128 channels' 256 output bytes; the
                #       rows are padded so the oversized load does not run into
                #       the next row.
                raise NotImplementedError("pto/per_channel/02_round_packed_m: implement per_channel_cast")
                T.copy(packed_ub, Sf[pair, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) with sf_packed int16 of shape (M/64, hidden)."""
    m, hidden = x.shape
    pairs = m // (BLOCK_MN * PACK_FACTOR)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((pairs, hidden * PACK_FACTOR), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_channel 02", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)


def demo_numbers() -> None:
    print("[demo] which axis the UE8M0 bytes pack along:")
    print("[demo]   per_token   sf (M, K/32)     -> along K, adjacent, free")
    print("[demo]   per_block   sf (M/32, K/32)  -> along K, adjacent, free")
    print("[demo]   per_channel sf (M/32, K)     -> along M, `hidden` bytes apart")
    print("[demo] so this is the one kernel where packing needs an instruction:")
    print("[demo]   lo, _ = V.vintlv(row_2i, row_2i+1, mask)  # [a0,b0,a1,b1,...]")
    print("[demo] and m-groups must be processed in PAIRS, so M must be a")
    print(f"[demo] multiple of {BLOCK_MN * PACK_FACTOR}, not {BLOCK_MN}.")
    print("[demo] note cast_back/06 used the same vintlv to *widen* bf16->f32 by")
    print("[demo] interleaving zeros. One instruction, two unrelated uses.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    m = max(m, BLOCK_MN * PACK_FACTOR)
    if m % (BLOCK_MN * PACK_FACTOR):
        m = (m // (BLOCK_MN * PACK_FACTOR) + 1) * BLOCK_MN * PACK_FACTOR
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_channel(x, BLOCK_MN, round_sf=True, packed=True)
    q, packed = launch(x.npu())
    assert_same_bytes(packed.cpu(), ref_packed, f"sf_packed({m},{k})")
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    _, ref_f32 = oracle.per_channel(x, BLOCK_MN, round_sf=True)
    assert torch.equal(decode_packed_ue8m0_along_m(packed.cpu()), ref_f32)
    print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} byte-exact, "
          f"decodes back along M to the float32 scales")


def main() -> int:
    k = sim.sim_shapes()[1]
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_channel", "02_round_packed_m")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
