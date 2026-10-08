"""per_block 02 (ASC). See doc/quant/per_block/02_round_packed.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

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
                    # --- BEGIN SOLUTION hint="stages (1) and (3) are variant 01's unchanged -- reuse them. Stage (2) is the new part: after the partials reduce to tile_amax and you clamp with S.vmaxs, do NOT divide. Take the exponent instead: bits = T.reinterpret(S.vmuls(clamped, 1.0/E4M3_MAX), 'uint32x64'); biased = S.vadds(S.vshrs(S.vsub(bits, S.vdup(1, T.uint32)), 23), 1). `biased` already IS the UE8M0 byte, so store it narrowed with S.vsts(sf_ub[0], T.reinterpret(biased, 'uint8x256'), dist='PK4_B32'), and keep the inverse as T.reinterpret(S.vshls(S.vsub(S.vdup(254, T.uint32), biased), 23), 'float32x64') -- negating an exponent field is exact, so no divide is needed anywhere."
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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_block/02")
