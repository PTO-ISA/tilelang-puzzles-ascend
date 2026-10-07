"""per_channel 02 (ASC) -- UE8M0 packed along M, where packing finally costs work.

New configs: `round_sf` and `use_packed_ue8m0` -- but packed along **M**, which
production calls `sf_col_pack`. This is the variant that makes the packing
interesting.

### Why the pack axis is M here

    per_token    sf is (M, K/32)     K/32 scales per row  -> pack along K
    per_block    sf is (M/32, K/32)  one per tile         -> pack along K
    per_channel  sf is (M/32, K)     only M/32 rows       -> pack along M

In the first two, the pack axis was the fastest-varying one, so the "two bytes per
int16" layout was a host-side `.view()` on adjacent bytes and the kernel did
nothing. Here the two bytes that must share a word come from **different scale
rows** -- m-group 2i and 2i+1 -- which are `hidden` bytes apart. Adjacent in the
output, far apart in the input.

### Interleaving two rows

That is exactly what an interleave instruction does:

    lo, hi = S.vintlv(row_2i, row_2i_plus_1)

`S.vintlv` takes two vectors and returns two, with the elements alternating:
`lo = [a0, b0, a1, b1, ...]`. On uint8 vectors that *is* the packed layout, so one
instruction per 128 channels replaces what would otherwise be a byte-by-byte
scatter.

A width detail that costs a debugging session if missed: a uint8 vector load is
**256 lanes**, so `lo` alone already carries the interleave of 128 channels from
each row -- all 256 output bytes -- and `hi` interleaves whatever followed. The
scale-byte rows are padded by 128 bytes so that oversized load cannot run into the
next row. Reading the second result instead of discarding it, or forgetting the
padding, produces output that is mostly right and wrong in about a third of its
bytes, which is exactly the kind of bug a byte-exact test catches and an
approximate one does not.

The consequence for the loop structure: m-groups have to be processed **in pairs**,
because a pair is what produces one packed row. That is why this variant requires
M to be a multiple of 64 rather than 32, and why the test checks that an odd
number of groups is rejected rather than silently mispacked.

### The same interleave, used for something else earlier

cast_back/06 used `S.vintlv(zero, x_bf16)` to *widen* bfloat16 to float32. Same
instruction, completely different purpose -- interleaving with zeros builds wider
elements; interleaving two real vectors packs narrower ones. Worth recognising,
because the name says neither.

Run:  python puzzles/asc/quant/answer/per_channel/02_round_packed_m.py
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

VARIANT = "asc/per_channel/02_round_packed_m"
LANES = 64
BYTE_LANES = 128        # uint8 lanes per interleave step


@tilelang.jit(target="ascend")
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
                        qmax = S.vdup(E4M3_MAX, T.float32)
                        one = S.vdup(1, T.uint32)
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            acc = S.alloc_local((1,), T.float32)
                            acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
                            for row in T.serial(group_tokens):
                                v = S.vabs(S.vcvt(S.vld(x_ub[row, col],
                                                        dist="UNPK_B16"), T.float32))
                                acc[0] = S.vmax(acc[0], v)
                            # the exponent trick, one value per channel
                            bits = T.reinterpret(S.vmuls(acc[0], 1.0 / E4M3_MAX),
                                                 "uint32x64")
                            biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                            S.vsts(sf_rows_ub[sub, col],
                                   T.reinterpret(biased, "uint8x256"), dist="PK4_B32")
                            S.vsts(inv_ub[col], T.reinterpret(
                                S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                                "float32x64"))
                        S.mem_bar("VST_VLD")

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

                # --- BEGIN SOLUTION hint="the two scale-byte rows of this pair interleave into one packed row. For each 128-channel tile: a = S.vld(sf_rows_ub[0, col], dist='NORM_B8'), b likewise from row 1, then lo, _ = S.vintlv(a, b) and store lo at packed_ub[col*2] with dist='NORM_B8'. A uint8 register is 256 lanes, so lo alone covers these 128 channels' 256 output bytes; the rows are padded so the oversized load does not run into the next row."
                # Pack: byte 2c of the output is m-group 2i's exponent for channel
                # c, byte 2c+1 is m-group 2i+1's. That is an interleave.
                with T.SimdVF():
                    S.mem_bar("VST_VLD")
                    for bt in T.serial(num_byte_tiles):
                        col = bt * BYTE_LANES
                        a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
                        b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
                        # A uint8 register is 256 lanes, so `lo` already holds the
                        # interleave of the first 128 elements of each input --
                        # exactly the 256 output bytes for these 128 channels.
                        # `hi` interleaves the padding and is discarded.
                        lo, _ = S.vintlv(a, b)
                        S.vsts(packed_ub[col * PACK_FACTOR], lo, dist="NORM_B8")
                # --- END SOLUTION
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
    print("[demo]   lo, hi = S.vintlv(row_2i, row_2i+1)   # [a0,b0,a1,b1,...]")
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
    sim.print_banner("asc", "per_channel", "02_round_packed_m")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
