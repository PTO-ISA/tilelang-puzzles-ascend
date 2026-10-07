"""per_block 03 (ASC) -- FP4 (e2m1) output on a 32x32 tile.

`quant_max` becomes 6.0 and the values pack two per byte. The conversion path is
the one established in per_token/04: there is no float32 -> e2m1 instruction, so
it goes float32 -> bfloat16 (rounded to odd) -> e2m1. Read that file for why the
intermediate must be rounded to odd.

### The tile makes the pairing natural

per_token/04 had to pair up two 64-lane registers to feed `S.vdintlv`. Here the
tile is already a flat 1024-value run, so the two halves of each 128-value chunk
are simply adjacent -- the pairing is an artefact of the data layout rather than
something the kernel has to arrange.

That is a small thing, but it is the sort of reason production has a separate code
path per granularity: the same arithmetic wants a different loop shape depending
on how the data happens to be laid out in UB.

### Accuracy

One scale over 1024 values *and* one mantissa bit is the ladder's coarsest
combination. The torch variant measures it: about 16% round-trip error against
12% for per_token FP4 and 3% for per_block FP8. The test here checks the kernel
against the bit-exact torch packer rather than re-measuring that.

Run:  python puzzles/asc/quant/answer/per_block/03_fp4_e2m1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp32_ulps, assert_same_bytes
from common.math_ops import unpack_e2m1_bytes
from common.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_block/03_fp4_e2m1"
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
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float4_e2m1fn)
            run_ub = T.alloc_shared((run_pad,), T.float32)   # per-chunk maxima
            sf_ub = T.alloc_shared((LANES,), T.float32)
            # Flat views of the UB tiles: contiguous, so 64-lane ops can walk them.
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float4_e2m1fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    # TODO: two-level reduction. (1) for each of the 16 chunks of
                    #       64, load from flat_x with dist='UNPK_B16', vcvt to
                    #       float32, vabs, and store S.vcmax(..., mask_all) to
                    #       run_ub[chunk] with dist='ONEPT_B32'. (2) barrier, then
                    #       reduce the 16 partials with S.vcmax(S.vld(run_ub[0]),
                    #       mask_vl16), clamp, divide both ways, store the scale
                    #       and keep the inverse. (3) barrier, then reload each
                    #       chunk, multiply by the broadcast inverse and store FP8
                    #       with dist='PK4_B32'.
                    raise NotImplementedError("asc/per_block/03_fp4_e2m1: implement per_block_cast")
                    T.copy(q_ub, Q[mb * bm, kb * bk])
                    T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8,
                    device=x.device).view(torch.float4_e2m1fn_x2)
    sf = torch.empty((m // BLOCK_MN, hidden // BLOCK_K), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 03", q, sf)
    return q.view(torch.uint8).view(torch.int8), sf


def demo_numbers() -> None:
    print("[demo] FP4 output: quant_max 6.0, two values per byte, and no")
    print("[demo] float32 -> e2m1 instruction, so the path is")
    print("[demo]   float32 -> bfloat16 (round to odd) -> e2m1")
    print("[demo] the tile is already flat, so the two halves each vdintlv needs")
    print("[demo] are simply adjacent -- per_token/04 had to pair them up.")
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
    ref_q, ref_sf = oracle.per_block(x, (BLOCK_MN, BLOCK_K), fmt="e2m1")
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)
    got_v, ref_v = unpack_e2m1_bytes(q.cpu()), unpack_e2m1_bytes(ref_q)
    differing, total = int((got_v != ref_v).sum()), got_v.numel()
    print(f"[check] shape=({m},{k}) FP4 values differing from the torch packer: "
          f"{differing}/{total} ({differing / total:.2%})")
    assert differing / total < 0.02


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_block", "03_fp4_e2m1")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
