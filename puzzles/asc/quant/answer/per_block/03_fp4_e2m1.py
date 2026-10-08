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
                    # --- BEGIN SOLUTION hint="two-level reduction. (1) for each of the 16 chunks of 64, load from flat_x with dist='UNPK_B16', vcvt to float32, vabs, and store S.vcmax(..., mask_all) to run_ub[chunk] with dist='ONEPT_B32'. (2) barrier, then reduce the 16 partials with S.vcmax(S.vld(run_ub[0]), mask_vl16), clamp, divide both ways, store the scale and keep the inverse. (3) barrier, then reload each chunk, multiply by the broadcast inverse and store FP8 with dist='PK4_B32'."
                    with T.SimdVF():
                        mask_all = S.pset(32, "PAT_ALL")
                        mask_chunks = S.pset(32, f"PAT_VL{num_chunks}")
                        qmax = S.vdup(E2M1_MAX, T.float32)
                        one_u16 = S.vdup(1, T.uint16)

                        # ---- pass 1: 16 partial maxima ----
                        for chunk in T.serial(num_chunks):
                            v = S.vabs(S.vcvt(S.vld(flat_x[chunk * LANES],
                                                    dist="UNPK_B16"), T.float32))
                            S.vsts(run_ub[chunk], S.vcmax(v, mask_all),
                                   dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        # ---- pass 2: reduce the partials, then one scale ----
                        tile_amax = S.vcmax(S.vld(run_ub[0]), mask_chunks)
                        clamped = S.vmaxs(tile_amax, E2M1_CLAMP_MIN)
                        S.vsts(sf_ub[0], S.vdiv(clamped, qmax), dist="ONEPT_B32")
                        S.vsts(sf_ub[1], S.vdiv(qmax, clamped), dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        # ---- pass 3: one scale for the whole tile ----
                        inv = S.vld(sf_ub[1], dist="BRC_B32")
                        # 128 FP4 values per store, so 8 chunks of two halves
                        for chunk in T.serial(num_chunks // 2):
                            base = chunk * 2 * LANES
                            v0 = S.vcvt(S.vld(flat_x[base], dist="UNPK_B16"), T.float32)
                            v1 = S.vcvt(S.vld(flat_x[base + LANES], dist="UNPK_B16"),
                                        T.float32)
                            q0 = S.vmul(v0, inv)
                            q1 = S.vmul(v1, inv)
                            # float32 -> bfloat16 (round to odd) -> e2m1
                            low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                                                  T.reinterpret(q1, "uint16x128"))
                            odd = S.vor(high, S.vmin(low, one_u16))
                            S.vsts(flat_q[base],
                                   S.vcvt(T.reinterpret(odd, "bfloat16x128"),
                                          T.float4_e2m1fn), dist="PK4_B32")
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
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_block/03")
