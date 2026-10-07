"""per_block 05 (ASC). See doc/quant/per_block/05_split_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, mode: str = "full",
                   block: tuple = (BLOCK_MN, BLOCK_K)):
    """mode in {"full", "sf_only", "cast_only"}."""
    bm, bk = block
    assert bm == 32 and bk == 32 and hidden % bk == 0
    assert mode in ("full", "sf_only", "cast_only")
    need_amax = mode in ("full", "sf_only")
    need_quant = mode in ("full", "cast_only")
    tile_values = bm * bk
    num_chunks = tile_values // LANES
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)
    run_pad = max(num_chunks, LANES)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        SfIn: T.Tensor((num_m_blocks, num_k_blocks), T.uint8),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.uint8),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk), T.float8_e4m3fn)
            run_ub = T.alloc_shared((run_pad,), T.float32)
            sf_ub = T.alloc_shared((LANES,), T.uint8)
            inv_ub = T.alloc_shared((LANES,), T.float32)
            flat_x = T.Tensor((tile_values,), T.bfloat16, x_ub.data)
            flat_q = T.Tensor((tile_values,), T.float8_e4m3fn, q_ub.data)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)
                    if mode == "cast_only":
                        T.copy(SfIn[mb, kb], sf_ub[0:1])
                    # --- BEGIN SOLUTION hint="gate the passes on the mode. need_amax emits the two-level reduction and the exponent trick; cast_only instead loads the given byte with S.vld(sf_ub[0], dist='BRC_B32'), converts it to uint32, and forms the inverse by negating the exponent: (254 - byte) << 23 reinterpreted to float32 -- exact, no division. need_quant emits the apply pass."
                    with T.SimdVF():
                        mask_all = S.pset(32, "PAT_ALL")
                        mask_chunks = S.pset(32, f"PAT_VL{num_chunks}")
                        bias254 = S.vdup(254, T.uint32)

                        if need_amax:
                            for chunk in T.serial(num_chunks):
                                v = S.vabs(S.vcvt(S.vld(flat_x[chunk * LANES],
                                                        dist="UNPK_B16"), T.float32))
                                S.vsts(run_ub[chunk], S.vcmax(v, mask_all),
                                       dist="ONEPT_B32")
                            S.mem_bar("VST_VLD")

                            tile_amax = S.vcmax(S.vld(run_ub[0]), mask_chunks)
                            clamped = S.vmaxs(tile_amax, E4M3_CLAMP_MIN)
                            one = S.vdup(1, T.uint32)
                            bits = T.reinterpret(S.vmuls(clamped, 1.0 / E4M3_MAX),
                                                 "uint32x64")
                            biased = S.vadds(S.vshrs(S.vsub(bits, one), 23), 1)
                            S.vsts(sf_ub[0], T.reinterpret(biased, "uint8x256"),
                                   dist="PK4_B32")
                        else:
                            # cast_only: the byte is given. Negating its exponent
                            # inverts the scale exactly -- no division.
                            #
                            # BRC_B32 on a uint8 buffer broadcasts the 32 bits at
                            # that address, i.e. four bytes, so mask down to the
                            # one we want. (Converting is not an option: the load
                            # already yields uint32 lanes, and S.vcvt rejects
                            # uint32->uint32.)
                            raw = T.reinterpret(S.vld(sf_ub[0], dist="BRC_B32"),
                                                "uint32x64")
                            biased = S.vand(raw, S.vdup(0xFF, T.uint32))

                        if need_quant:
                            S.vsts(inv_ub[0], T.reinterpret(
                                S.vshls(S.vsub(bias254, biased), 23), "float32x64"))
                            S.mem_bar("VST_VLD")
                            inv = S.vld(inv_ub[0], dist="BRC_B32")
                            for chunk in T.serial(num_chunks):
                                v = S.vcvt(S.vld(flat_x[chunk * LANES],
                                                 dist="UNPK_B16"), T.float32)
                                S.vsts(flat_q[chunk * LANES],
                                       S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                       dist="PK4_B32")
                    # --- END SOLUTION
                    if need_quant:
                        T.copy(q_ub, Q[mb * bm, kb * bk])
                    if need_amax:
                        T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor, mode: str = "full", sf_in: torch.Tensor | None = None):
    m, hidden = x.shape
    nm, nk = m // BLOCK_MN, hidden // BLOCK_K
    dev = x.device
    given = sf_in if sf_in is not None else torch.zeros((nm, nk), dtype=torch.uint8,
                                                        device=dev)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((nm, nk), dtype=torch.uint8, device=dev)
    compile_kernel(hidden, mode)(x, given, q, sf)
    status.assert_on_device(f"per_block 05 {mode}", q, sf)
    return q, sf
