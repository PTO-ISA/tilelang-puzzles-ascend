"""per_block 01 (ASC) -- one scale per 32x32 tile.

Read puzzles/torch/quant/answer/per_block/01_raw_32x32.py for the maths and the
accuracy trade-off, and per_token/01 for the three-pass reduce/scale/apply
structure. What is new here is that the reduction is 2-D, and that the obvious
lane width for it does not exist.

### The tile is flattened, and reduced 64 lanes at a time

A 32x32 bf16 tile is 1024 values. Its rows are not contiguous in global memory
(stride K), so the DMA brings the tile into a UB buffer of shape (32, 32) -- which
*is* contiguous -- and the vector code then treats it as one flat run of 1024:

    flat = T.Tensor((BM * BK,), T.bfloat16, x_ub.data)   # a view, no copy

Reducing 1024 values 64 at a time takes 16 steps. Each step's maximum goes to a
scratch buffer, and a final reduction over those 16 values gives the tile maximum.
That two-level shape (reduce within a register, then reduce the partials through
UB) is the standard way any cross-register reduction is written here.

### Why not 32 lanes, or 8

The tile is 32 wide, so a 32-lane vector looks natural. There is no such type:
the legal lane counts are {1, 2, 4, 8, 64, 128, 256} and 32 is absent -- it is half
a 256-byte register, neither one 32-byte slice nor one whole register. See
doc/vf-lane-widths-and-limits.md.

The next guess, 8 lanes (one 32-byte slice), **does not compile**: an 8-lane
bfloat16-to-float32 convert is rejected with

    VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout support

`common/probe/vf_lane_limits.py` reproduces that, and it is why this kernel -- and
production's -- reduces at 64 lanes over a flattened tile rather than following
the tile's own 32-wide geometry. Worth internalising as a general rule: **pick the
lane width from the hardware, then reshape the problem to fit it**, not the other
way round.

Run:  python puzzles/asc/quant/answer/per_block/01_raw_32x32.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_block/01_raw_32x32"
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
                        S.vsts(sf_ub[0], S.vdiv(clamped, qmax), dist="ONEPT_B32")
                        S.vsts(sf_ub[1], S.vdiv(qmax, clamped), dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        # ---- pass 3: one scale for the whole tile ----
                        inv = S.vld(sf_ub[1], dist="BRC_B32")
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
    print("[demo] so: flatten the tile and reduce 64 at a time, 16 steps,")
    print("[demo] then reduce the 16 partials. Pick the width from the hardware")
    print("[demo] and reshape the problem, not the other way round.")


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
    sim.print_banner("asc", "per_block", "01_raw_32x32")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
