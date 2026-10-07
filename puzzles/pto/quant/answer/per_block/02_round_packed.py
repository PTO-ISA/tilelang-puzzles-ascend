"""per_block 02 (PTO) -- power-of-two scale as a UE8M0 byte.

Read the ASC variant for the layout observation (the int16 pairing is free here
because the pack axis is the fastest-varying one) and per_token/02-03 for the
exponent trick itself.

### PTO vs ASC

The same two small differences as per_token/03, for the same reasons:

    scale byte:  ASC  S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"),
                             dist="PK4_B32")       # reinterpret + store mode
                 PTO  V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])   # one convert

    exponent:    ASC reinterprets through "uint32x64" / "float32x64", so the
                 sequence is pinned to 64 lanes
                 PTO reinterprets without a lane count, so the same expressions
                 would serve any width

Neither matters much in a kernel that computes one scale per tile. They matter in
production, where the same scale arithmetic is reached from several differently
shaped paths and ASC has to repeat it per width.

Run:  python puzzles/pto/quant/answer/per_block/02_round_packed.py
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
from common.math_ops import decode_packed_ue8m0
from common.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_block/02_round_packed"
LANES = 128   # VMI reaches 128 float32 lanes in one logical vector


@tilelang.jit(target="pto")
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
                    # --- BEGIN SOLUTION hint="two-level reduction at 128 lanes, so 8 chunks not 16. (1) per chunk: v = V.vabs(V.vcvt(V.vload(flat_x[chunk*128], size=128), 'float32'), mask) then V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk]). (2) barrier; reduce the 8 partials with V.vcmax(partials, V.create_mask(8, size=128), group=1), clamp at 1 lane, divide both ways, store scale and inverse. (3) barrier; inv = V.vbrc(V.vload(sf_ub[1], size=1), size=128), then reload each chunk, multiply and V.vstore the FP8 convert."
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        mask_chunks = V.create_mask(num_chunks, size=LANES)
                        qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                        one1 = V.create_mask(1, size=1)

                        # ---- pass 1: 8 partial maxima, 128 lanes each ----
                        for chunk in T.serial(num_chunks):
                            v = V.vabs(V.vcvt(V.vload(flat_x[chunk * LANES],
                                                      size=LANES), "float32"), mask)
                            V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk])
                        T.simd.mem_bar("VST_VLD")

                        # ---- pass 2: reduce the partials, then one scale ----
                        partials = V.vload(run_ub[0], size=LANES)
                        tile_amax = V.vcmax(partials, mask_chunks, group=1)
                        clamped = V.vmax(tile_amax,
                                         V.vbrc(T.float32(E4M3_CLAMP_MIN), size=1),
                                         one1)
                        # the exponent trick at a single lane
                        one_u = V.vbrc(T.uint32(1), size=1)
                        shift = V.vbrc(T.uint32(23), size=1)
                        bits = V.vinterpret_cast(
                            V.vmul(clamped,
                                   V.vbrc(T.float32(1.0 / E4M3_MAX), size=1), one1),
                            "uint32")
                        biased = V.vadd(V.vshr(V.vsub(bits, one_u), shift), one_u)
                        # one convert, no reinterpret and no store mode
                        V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])
                        V.vstore(V.vinterpret_cast(
                            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=1), biased),
                                   shift), "float32"), inv_ub[0])
                        T.simd.mem_bar("VST_VLD")

                        # ---- pass 3: one scale for the whole tile ----
                        inv = V.vbrc(V.vload(inv_ub[0], size=1), size=LANES)
                        for chunk in T.serial(num_chunks):
                            v = V.vcvt(V.vload(flat_x[chunk * LANES], size=LANES),
                                       "float32")
                            V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                            rounding="R", saturate="SAT"),
                                     flat_q[chunk * LANES])
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
    print("[demo] a 32x32 bfloat16 tile is 1024 values.")
    print("[demo] the tile is 32 wide, but there is no 32-lane vector type:")
    print("[demo]   legal lane counts are {1, 2, 4, 8, 64, 128, 256}")
    print("[demo]   32 float32 = 128 bytes = half a register: not a legal width")
    print("[demo] and 8 lanes (one 32-byte slice) does not compile for bf16->f32:")
    print("[demo]   VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout")
    print("[demo]   support   (see common/probe/vf_lane_limits.py)")
    print("[demo] the scale byte is the biased exponent, stored with one")
    print("[demo] V.vcvt(biased, 'uint8') where ASC needs a reinterpret to")
    print("[demo] uint8x256 plus a matching PK4_B32 store mode.")
    print("[demo] so: flatten the tile and reduce at the widest legal width.")
    print("[demo]   ASC: its widening load gives 64 float32 lanes -> 16 steps")
    print("[demo]   PTO: one convert reaches 128 float32 lanes   ->  8 steps")
    print("[demo] pick the width from the hardware and reshape the problem,")
    print("[demo] not the other way round.")


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
    sim.print_banner("pto", "per_block", "02_round_packed")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
