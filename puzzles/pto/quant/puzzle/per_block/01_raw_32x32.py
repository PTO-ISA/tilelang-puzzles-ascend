"""per_block 01 (PTO) -- one scale per 32x32 tile, eight steps instead of sixteen.

Read the ASC variant for the two-level reduction and, importantly, for why
neither 32 nor 8 lanes is usable here (32 is not a legal vector width; an 8-lane
bfloat16-to-float32 convert fails with `VMI-LAYOUT-CONTRACT`).

### PTO vs ASC: the reduction loop is half as long

ASC's widening load produces 64 float32 lanes, so a 1024-value tile is 16
reduction steps. VMI has a 128-lane logical float32, and one convert reaches it:

    ASC: S.vcvt(S.vld(flat_x[c], dist="UNPK_B16"), T.float32)   ->  64 lanes
    PTO: V.vcvt(V.vload(flat_x[c], size=128), "float32")        -> 128 lanes

So the same work is 8 steps rather than 16, and the partial-maxima buffer is half
the size. Nothing clever is going on -- it is the same "width is an argument"
property as everywhere else, applied to a reduction. ASC *could* process 128
channels per iteration, but only by issuing two loads and two reduces and
tracking two registers by hand, which is exactly what per_token's ASC files do.

### A note on the accumulator

Both files reduce into a UB scratch buffer and then reduce the partials. The
alternative is to keep a running maximum in a register across the loop, which VMI
supports through a register array:

    acc = V.alloc_local((1,), V.vreg(128, T.float32))
    acc[0] = V.vmax(acc[0], V.vabs(v, mask), mask)

That needs `alloc_local` because **SIMD values are immutable** -- a plain value
cannot be reassigned across loop iterations. Trying it gives

    Immutable variable `acc` is used outside its defining region

which is a confusing error for what is really "use a register array". The UB
scratch version used here avoids the issue and makes the two-level structure
visible, which is why it is the teaching form; production uses `alloc_local`.

Run:  python puzzles/pto/quant/answer/per_block/01_raw_32x32.py
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
from common.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_block/01_raw_32x32"
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
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            run_ub = T.alloc_shared((run_pad,), T.float32)   # per-chunk maxima
            sf_ub = T.alloc_shared((LANES,), T.float32)
            # Flat views of the UB tiles: contiguous, so 64-lane ops can walk them.
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float8_e4m3fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    # TODO: two-level reduction at 128 lanes, so 8 chunks not 16.
                    #       (1) per chunk: v =
                    #       V.vabs(V.vcvt(V.vload(flat_x[chunk*128], size=128),
                    #       'float32'), mask) then V.vstore(V.vcmax(v, mask,
                    #       group=1), run_ub[chunk]). (2) barrier; reduce the 8
                    #       partials with V.vcmax(partials, V.create_mask(8,
                    #       size=128), group=1), clamp at 1 lane, divide both
                    #       ways, store scale and inverse. (3) barrier; inv =
                    #       V.vbrc(V.vload(sf_ub[1], size=1), size=128), then
                    #       reload each chunk, multiply and V.vstore the FP8
                    #       convert.
                    raise NotImplementedError("pto/per_block/01_raw_32x32: implement per_block_cast")
                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m // BLOCK_MN, hidden // BLOCK_K), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 01", q, sf)
    return q, sf


def demo_numbers() -> None:
    print("[demo] a 32x32 bfloat16 tile is 1024 values.")
    print("[demo] the tile is 32 wide, but there is no 32-lane vector type:")
    print("[demo]   legal lane counts are {1, 2, 4, 8, 64, 128, 256}")
    print("[demo]   32 float32 = 128 bytes = half a register: not a legal width")
    print("[demo] and 8 lanes (one 32-byte slice) does not compile for bf16->f32:")
    print("[demo]   VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout")
    print("[demo]   support   (see common/probe/vf_lane_limits.py)")
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
    ref_q, ref_sf = oracle.per_block(x, (BLOCK_MN, BLOCK_K))
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} matches the torch oracle")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_block", "01_raw_32x32")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
