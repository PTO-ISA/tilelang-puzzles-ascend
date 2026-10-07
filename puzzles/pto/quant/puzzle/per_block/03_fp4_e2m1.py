"""per_block 03 (PTO) -- FP4 output on a tile, one vector per store.

Read the ASC variant, and per_token/04 for the round-to-odd requirement.

### PTO vs ASC

The same difference as per_token/04, and it is structural rather than cosmetic.
ASC's `vdintlv` is a *pairwise* operation on two registers, so its loop is written
around producing two 64-lane halves and combining them. VMI's `vunzip` splits one
value, and the value is already 128 lanes, so the loop body has no pairing in it
at all:

    ASC per 128 values:  2 loads, 2 converts, 2 multiplies, vdintlv, vor, vmin,
                         reinterpret, convert, store
    PTO per 128 values:  1 load, 1 convert, 1 multiply, vunzip, vor, vmin,
                         convert, store

The saving is the duplicated half of the value path -- exactly the pattern from
cast_back/06, where ASC had to split 128 FP4 values into two float32 registers and
duplicate everything downstream.

Run:  python puzzles/pto/quant/answer/per_block/03_fp4_e2m1.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp32_ulps, assert_same_bytes
from common.math_ops import unpack_e2m1_bytes
from common.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_block/03_fp4_e2m1"
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
                    raise NotImplementedError("pto/per_block/03_fp4_e2m1: implement per_block_cast")
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
    print("[demo] a 32x32 bfloat16 tile is 1024 values.")
    print("[demo] the tile is 32 wide, but there is no 32-lane vector type:")
    print("[demo]   legal lane counts are {1, 2, 4, 8, 64, 128, 256}")
    print("[demo]   32 float32 = 128 bytes = half a register: not a legal width")
    print("[demo] and 8 lanes (one 32-byte slice) does not compile for bf16->f32:")
    print("[demo]   VMI-LAYOUT-CONTRACT: pto.vmi.extf has no registered layout")
    print("[demo]   support   (see common/probe/vf_lane_limits.py)")
    print("[demo] FP4 store: ASC pairs two 64-lane registers for vdintlv;")
    print("[demo] VMI vunzips one 128-lane value, so no pairing appears.")
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
    sim.print_banner("pto", "per_block", "03_fp4_e2m1")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
