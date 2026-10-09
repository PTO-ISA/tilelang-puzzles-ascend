"""per_block 03 (ASC). See doc/quant/per_block/03_fp4_e2m1.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX

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
                    # TODO: stages (1) and (2) are variant 01's with E2M1_MAX /
                    #       E2M1_CLAMP_MIN in place of the e4m3 constants -- still
                    #       S.vdiv both ways into sf_ub[0] and sf_ub[1]. Stage (3)
                    #       is the new part: there is no float32 to e2m1 convert,
                    #       so go via bfloat16 with round-to-odd, two chunks at a
                    #       time (loop num_chunks//2). Multiply both chunks by the
                    #       broadcast inverse, then low, high =
                    #       S.vdintlv(T.reinterpret(q0, 'uint16x128'),
                    #       T.reinterpret(q1, 'uint16x128')); odd = S.vor(high,
                    #       S.vmin(low, one_u16)) forces the last mantissa bit
                    #       when anything was dropped, so the second rounding
                    #       cannot double-round; finally S.vsts(...,
                    #       S.vcvt(T.reinterpret(odd, 'bfloat16x128'),
                    #       T.float4_e2m1fn), dist='PK4_B32').
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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_block/03 --role puzzle")
