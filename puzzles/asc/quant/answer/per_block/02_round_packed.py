"""per_block 02 (ASC) -- power-of-two scale, stored as a UE8M0 byte.

Both configs from per_token 02 and 03, on the block layout. The arithmetic is
identical -- see those files for the ceil-log2 exponent trick and the byte
encoding -- so what is worth attention here is a detail of the *layout*.

### The packing is still free, but for a different reason

per_token had K/32 scales per row, packed two-per-int16 along K, and the packing
was a host-side `.view()` because the pack axis was the fastest-varying one.

per_block has only K/32 scales per *tile row*, and the kernel produces them one at
a time -- one scale per tile. So the kernel writes single bytes at `Sf[mb, kb]`,
and the int16 pairing is again just a reinterpretation of adjacent bytes. Still no
vector work.

The case where this stops being free is per_channel, whose scales pack along M --
a direction that is not adjacent in memory, and which therefore needs a real
interleave instruction. That is per_channel/02.

### Only one lane matters

The scale computation here runs on a vector whose lane 0 holds the tile maximum
and whose other 63 lanes hold whatever was left in the register. That is fine --
the arithmetic is lane-wise and only lane 0 is ever stored -- but it is worth
noticing, because it means the same six-operation sequence serves a 64-group
per_token row and a single-scale per_block tile without modification.

Run:  python puzzles/asc/quant/answer/per_block/02_round_packed.py
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
from common.math_ops import decode_packed_ue8m0
from common.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_block/02_round_packed"
LANES = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per `block` tile."""
    bm, bk = block
    assert bm == 32 and bk == 32, "Ascend per_block granularity is 32x32"
    assert hidden % bk == 0
    tile_values = bm * bk                       # 1024
    num_chunks = tile_values // LANES           # 16 reduction steps
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)
    run_pad = max(num_chunks, LANES)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            run_ub = T.alloc_shared((run_pad,), T.float32)   # per-chunk maxima
            sf_ub = T.alloc_shared((LANES,), T.uint8)
            inv_ub = T.alloc_shared((LANES,), T.float32)
            # Flat views of the UB tiles: contiguous, so 64-lane ops can walk them.
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float8_e4m3fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    # --- BEGIN SOLUTION hint="two-level reduction. (1) for each of the 16 chunks of 64, load from flat_x with dist='UNPK_B16', vcvt to float32, vabs, and store S.vcmax(..., mask_all) to run_ub[chunk] with dist='ONEPT_B32'. (2) barrier, then reduce the 16 partials with S.vcmax(S.vld(run_ub[0]), mask_vl16), clamp, divide both ways, store the scale and keep the inverse. (3) barrier, then reload each chunk, multiply by the broadcast inverse and store FP8 with dist='PK4_B32'."
                    with T.SimdVF():
                        mask_all = S.pset(32, "PAT_ALL")
                        mask_chunks = S.pset(32, f"PAT_VL{num_chunks}")
                        qmax = S.vdup(E4M3_MAX, T.float32)

                        # ---- pass 1: 16 partial maxima ----
                        for chunk in T.serial(num_chunks):
                            v = S.vabs(S.vcvt(S.vld(flat_x[chunk * LANES],
                                                    dist="UNPK_B16"), T.float32))
                            S.vsts(run_ub[chunk], S.vcmax(v, mask_all),
                                   dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        # ---- pass 2: reduce the partials, then one scale ----
                        tile_amax = S.vcmax(S.vld(run_ub[0]), mask_chunks)
                        clamped = S.vmaxs(tile_amax, E4M3_CLAMP_MIN)
                        # the exponent trick; only lane 0 is ever stored
                        one = S.vdup(1, T.uint32)
                        bits = T.reinterpret(S.vmuls(clamped, 1.0 / E4M3_MAX),
                                             "uint32x64")
                        biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                        S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"),
                               dist="PK4_B32")
                        S.vsts(inv_ub[0], T.reinterpret(
                            S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23),
                            "float32x64"))
                        S.mem_bar("VST_VLD")

                        # ---- pass 3: one scale for the whole tile ----
                        inv = S.vld(inv_ub[0], dist="BRC_B32")
                        for chunk in T.serial(num_chunks):
                            v = S.vcvt(S.vld(flat_x[chunk * LANES], dist="UNPK_B16"),
                                       T.float32)
                            S.vsts(flat_q[chunk * LANES],
                                   S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                   dist="PK4_B32")
                    # --- END SOLUTION
                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    """Returns (q, sf_packed) with sf_packed int16, two tile exponents per word."""
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_bytes = torch.empty((m // BLOCK_MN, hidden // BLOCK_K), dtype=torch.uint8,
                           device=x.device)
    compile_kernel(hidden)(x, q, sf_bytes)
    status.assert_on_device("per_block 02", q, sf_bytes)
    return q, sf_bytes.view(torch.int16)


def demo_numbers() -> None:
    print("[demo] one scale per 32x32 tile, stored as one UE8M0 byte:")
    for amax in (448.0, 3.5, 1.75):
        v = amax / E4M3_MAX
        bits = torch.tensor([v], dtype=torch.float32).view(torch.int32).item() & 0xFFFFFFFF
        biased = ((bits - 1) >> 23) + 1
        print(f"[demo]   tile amax={amax:<7g} -> byte {biased} -> 2^{biased - 127}")
    print("[demo] the int16 pairing is a .view() on adjacent bytes -- the pack")
    print("[demo] axis is the fastest-varying one. per_channel/02 is the case")
    print("[demo] where it is not, and a real interleave is needed.")
    print("[demo] a 32x32 bfloat16 tile is 1024 values.")
    print("[demo] the tile is 32 wide, but there is no 32-lane vector type:")
    print("[demo]   legal lane counts are {1, 2, 4, 8, 64, 128, 256}")
    print("[demo]   32 float32 = 128 bytes = half a register: not a legal width")
    print("[demo] and 8 lanes (one 32-byte slice) does not compile for bf16->f32:")
    print("[demo]   VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout")
    print("[demo]   support   (see common/probe/vf_lane_limits.py)")
    print("[demo] so: flatten the tile and reduce 64 at a time, 16 steps,")
    print("[demo] then reduce the 16 partials. Pick the width from the hardware")
    print("[demo] and reshape the problem, not the other way round.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0 and k % BLOCK_K == 0
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_packed = oracle.per_block(x, (BLOCK_MN, BLOCK_K),
                                        round_sf=True, packed=True)
    q, packed = launch(x.npu())
    assert_same_bytes(packed.cpu(), ref_packed, f"sf_packed({m},{k})")
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    _, ref_f32 = oracle.per_block(x, (BLOCK_MN, BLOCK_K), round_sf=True)
    assert torch.equal(decode_packed_ue8m0(packed.cpu()), ref_f32)
    print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} byte-exact, "
          f"decodes to the float32 scales")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_block", "02_round_packed")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
