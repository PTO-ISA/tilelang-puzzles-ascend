"""per_block 05 (PTO). See doc/quant/per_block/05_split_compose.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 128


@tilelang.jit(target="pto")
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
                    # --- BEGIN SOLUTION hint="gate the passes on the mode. need_amax emits the two-level reduction at 128 lanes and the exponent trick; cast_only instead reads the given byte with V.vcvt(V.vload(sf_ub[0], size=128), 'uint32') -- widening on conversion needs no mask -- and forms the inverse by negating the exponent: (254 - byte) << 23 reinterpreted to float32. need_quant emits the apply pass."
                    with T.SimdVF():
                        mask = V.create_mask(LANES, size=LANES)
                        mask_chunks = V.create_mask(num_chunks, size=LANES)
                        one1 = V.create_mask(1, size=1)
                        # The scale math works on a single lane: there is one
                        # scale per tile. VMI needs every operand at that width,
                        # so the constants are size=1 too.
                        shift = V.vbrc(T.uint32(23), size=1)
                        one_u = V.vbrc(T.uint32(1), size=1)
                        bias254 = V.vbrc(T.uint32(254), size=1)

                        if need_amax:
                            for chunk in T.serial(num_chunks):
                                v = V.vabs(V.vcvt(V.vload(flat_x[chunk * LANES],
                                                          size=LANES), "float32"), mask)
                                V.vstore(V.vcmax(v, mask, group=1), run_ub[chunk])
                            T.simd.mem_bar("VST_VLD")

                            partials = V.vload(run_ub[0], size=LANES)
                            tile_amax = V.vcmax(partials, mask_chunks, group=1)
                            clamped = V.vmax(
                                tile_amax,
                                V.vbrc(T.float32(E4M3_CLAMP_MIN), size=1), one1)
                            bits = V.vinterpret_cast(
                                V.vmul(clamped,
                                       V.vbrc(T.float32(1.0 / E4M3_MAX), size=1),
                                       one1), "uint32")
                            biased = V.vadd(V.vshr(V.vsub(bits, one_u), shift), one_u)
                            V.vstore(V.vcvt(biased, "uint8"), sf_ub[0])
                        else:
                            # cast_only: widening on conversion respects the
                            # element boundary, so no mask is needed.
                            biased = V.vcvt(V.vload(sf_ub[0], size=1), "uint32")

                        if need_quant:
                            V.vstore(V.vinterpret_cast(
                                V.vshl(V.vsub(bias254, biased), shift), "float32"),
                                inv_ub[0])
                            T.simd.mem_bar("VST_VLD")
                            inv = V.vbrc(V.vload(inv_ub[0], size=1), size=LANES)
                            for chunk in T.serial(num_chunks):
                                v = V.vcvt(V.vload(flat_x[chunk * LANES], size=LANES),
                                           "float32")
                                V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                                rounding="R", saturate="SAT"),
                                         flat_q[chunk * LANES])
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

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_block/05")
