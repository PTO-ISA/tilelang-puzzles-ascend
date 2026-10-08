"""per_block 03 (PTO). See doc/quant/per_block/03_fp4_e2m1.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX

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
                    # --- BEGIN SOLUTION hint="two-level reduction at 128 lanes, so 8 chunks not 16. (1) per chunk: v = V.vabs(V.vcvt(V.vload(flat_x[chunk*128], size=128), 'float32'), mask) then V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk]). (2) barrier; reduce the 8 partials with V.vcmax(partials, V.create_mask(8, size=128), group=1), clamp at 1 lane, divide both ways, store scale and inverse. (3) barrier; inv = V.vbrc(V.vload(sf_ub[1], size=1), size=128), then reload each chunk, multiply and V.vstore the FP8 convert."
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        mask_chunks = V.create_mask(num_chunks, size=LANES)
                        qmax = V.vbrc(T.float32(E2M1_MAX), size=LANES)
                        one_u16 = V.vbrc(T.uint16(1), size=LANES)
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
                                         V.vbrc(T.float32(E2M1_CLAMP_MIN), size=1),
                                         one1)
                        qmax1 = V.vbrc(T.float32(E2M1_MAX), size=1)
                        V.vstore(V.vdiv(clamped, qmax1, one1), sf_ub[0])
                        V.vstore(V.vdiv(qmax1, clamped, one1), sf_ub[1])
                        T.simd.mem_bar("VST_VLD")

                        # ---- pass 3: one scale for the whole tile ----
                        inv = V.vbrc(V.vload(sf_ub[1], size=1), size=LANES)
                        for chunk in T.serial(num_chunks):
                            v = V.vcvt(V.vload(flat_x[chunk * LANES], size=LANES),
                                       "float32")
                            q = V.vmul(v, inv, mask)
                            # one 128-lane value, so vunzip needs no pairing
                            low, high = V.vunzip(q, "uint16")
                            odd = V.vinterpret_cast(
                                V.vor(high, V.vmin(low, one_u16)), "bfloat16")
                            V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"),
                                     flat_q[chunk * LANES])
                    # --- END SOLUTION
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
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_block/03")
