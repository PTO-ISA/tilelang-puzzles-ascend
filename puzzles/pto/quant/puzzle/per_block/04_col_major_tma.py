"""per_block 04 (PTO) -- column-major scales, which cost nothing here.

New config: `use_tma_aligned_col_major_sf`. The scales are written as
`sf_cm[k_block, m_block]` instead of `sf[m_block, k_block]`, for the usual reason
(the consuming GEMM wants a tile's scales contiguous).

### Why this is free here, when per_token/05 needed a gather

per_token's kernel holds a whole token's worth of scales in a register -- 64 of
them -- and the transposed layout wants them strided apart in memory. A register
lane cannot move, so the kernel has to compute per-lane source indices and gather.

per_block's kernel produces **one scalar per tile**. A single value has no layout,
so writing it to `sf_cm[kb, mb]` instead of `sf[mb, kb]` is purely a change of
address. No gather, no index arithmetic, no extra instruction.

The general rule: the cost of a transposed output depends on how many values the
producing loop holds at once. One at a time is free; a vector-full needs a
transpose. Worth knowing before assuming a layout change is cheap or expensive.

### What is deliberately not here

Production combines column-major *with* packed UE8M0 on this kernel, and that
needs a further trick: after blocking by 32 and packing two-per-word, a tile row's
scales can be just a couple of int16 words -- narrower than the DMA engine moves
efficiently -- so production groups four tile-rows together
(`token_group = 4` in `per_block_cast_asc.py`). That is a multi-core scheduling
concern with no effect on the arithmetic, so this single-core ladder covers packing
in variant 02 and the layout here, and does not combine them. Variant 05 composes
everything else.

### PTO vs ASC

Nothing. The change is one index expression in the schedule, which the two backends share. Compare per_token/05, where the equivalent config needed an in-register gather and the two backends were still a draw.

### What `tools/vf_lines.py` says, and why it understates this kernel

per_block is the one kernel where PTO's *static* operation count comes out
slightly **higher** than ASC's. That is a real property of the source -- VMI needs
explicit `size=` and mask operands, and per_block's reduction is a whole-vector
`group=1` reduce, so the segmented-operation advantage that drives the savings in
per_token does not apply.

It is also misleading as a measure of work. The count is of operations *written*,
not operations *executed*: PTO reduces 128 lanes per iteration against ASC's 64,
so it runs half as many iterations of the reduction loop and issues fewer
instructions at runtime. A static count cannot see that.

The honest summary for this kernel: VMI is not shorter here, it is wider.

Run:  python puzzles/pto/quant/answer/per_block/04_col_major_tma.py
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

VARIANT = "pto/per_block/04_col_major_tma"
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
        SfCm: T.Tensor((num_k_blocks, num_m_blocks), T.float32),
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
                    raise NotImplementedError("pto/per_block/04_col_major_tma: implement per_block_cast")
                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    # the transposed index is the entire change
                    T.copy(sf_ub[0:1], SfCm[kb, mb:mb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((hidden // BLOCK_K, m // BLOCK_MN), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 04", q, sf)
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
    q, sf_cm = launch(x.npu())
    assert sf_cm.shape == (k // BLOCK_K, m // BLOCK_MN), sf_cm.shape
    assert_fp32_ulps(sf_cm.cpu().T.contiguous(), ref_sf, f"sf_cm.T({m},{k})",
                     max_ulps=1)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} transposes back "
          f"to {tuple(ref_sf.shape)}")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_block", "04_col_major_tma")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
