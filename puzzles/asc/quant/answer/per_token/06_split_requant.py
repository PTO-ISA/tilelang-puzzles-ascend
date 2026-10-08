"""per_token 06 (ASC). See doc/quant/per_token/06_split_requant.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, mode: str = "full", group_size: int = CANONICAL_G):
    """mode in {"full", "sf_only", "cast_only", "requant"}."""
    assert hidden % PAIR == 0 and group_size == 32
    assert mode in ("full", "sf_only", "cast_only", "requant")
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    groups_per_pair = PAIR // group_size

    # Trace-time choices: only the selected branches are ever emitted.
    need_amax = mode in ("full", "sf_only", "requant")
    need_quant = mode in ("full", "cast_only", "requant")
    sf_is_input = mode == "cast_only"
    x_dtype = T.float8_e4m3fn if mode == "requant" else T.bfloat16

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), x_dtype),
        XSf: T.Tensor((num_tokens, num_groups), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), x_dtype)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            val_ub = T.alloc_shared((hidden,), T.float32)   # dequantized scratch
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)
            xsf_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                if mode in ("cast_only", "requant"):
                    T.copy(XSf[token, 0], xsf_ub[0:num_groups])
                # --- BEGIN SOLUTION hint="gate the three passes on the mode. requant first dequantizes x_ub into val_ub using the input scales (broadcast + vsel as in cast_back/01) with a barrier after. sf_only emits passes 1-2 only. cast_only skips pass 1 and sets inv = S.vdiv(1.0, given sf). Everything else is variants 01/02."
                with T.SimdVF():
                    mask_low = S.pset(32, "PAT_VL32")
                    mask_all = S.pset(32, "PAT_ALL")
                    mask_high = S.pnot(mask_low, mask_all)
                    qmax = S.vdup(E4M3_MAX, T.float32)

                    if mode == "requant":
                        # stage 0: dequantize into the scratch buffer
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * groups_per_pair
                            for half in range(2):
                                c = col + half * LANES
                                g = group + half * 2
                                s0 = S.vld(xsf_ub[g], dist="BRC_B32")
                                s1 = S.vld(xsf_ub[g + 1], dist="BRC_B32")
                                raw = S.vcvt(S.vld(x_ub[c], dist="UNPK4_B8"), T.float32)
                                S.vsts(val_ub[c], S.vmul(raw, S.vsel(s0, s1, mask_low)))
                        S.mem_bar("VST_VLD")

                    if need_amax:
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * groups_per_pair
                            for half in range(2):
                                c = col + half * LANES
                                g = group + half * 2
                                if mode == "requant":
                                    a = S.vabs(S.vld(val_ub[c]))
                                else:
                                    a = S.vabs(S.vcvt(S.vld(x_ub[c], dist="UNPK_B16"),
                                                      T.float32))
                                S.vsts(amax_ub[g], S.vcmax(a, mask_low),
                                       dist="ONEPT_B32")
                                S.vsts(amax_ub[g + 1], S.vcmax(a, mask_high),
                                       dist="ONEPT_B32")
                        S.mem_bar("VST_VLD")

                        clamped = S.vmaxs(S.vld(amax_ub[0]), E4M3_CLAMP_MIN)
                        S.vsts(sf_ub[0], S.vdiv(clamped, qmax))
                        S.vsts(inv_ub[0], S.vdiv(qmax, clamped))
                    elif sf_is_input:
                        # cast_only: the scale is given, so invert it. This is the
                        # reciprocal that makes cast_only differ from the fused
                        # path by the occasional FP8 code.
                        S.vsts(inv_ub[0], S.vdiv(S.vdup(1.0, T.float32),
                                                 S.vld(xsf_ub[0])))
                    S.mem_bar("VST_VLD")

                    if need_quant:
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * groups_per_pair
                            for half in range(2):
                                c = col + half * LANES
                                g = group + half * 2
                                i0 = S.vld(inv_ub[g], dist="BRC_B32")
                                i1 = S.vld(inv_ub[g + 1], dist="BRC_B32")
                                if mode == "requant":
                                    xv = S.vld(val_ub[c])
                                else:
                                    xv = S.vcvt(S.vld(x_ub[c], dist="UNPK_B16"),
                                                T.float32)
                                S.vsts(q_ub[c],
                                       S.vcvt(S.vmul(xv, S.vsel(i0, i1, mask_low)),
                                              T.float8_e4m3fn), dist="PK4_B32")
                # --- END SOLUTION
                if need_quant:
                    T.copy(q_ub, Q[token, 0])
                if need_amax:
                    T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor, mode: str = "full", x_sf: torch.Tensor | None = None):
    m, hidden = x.shape
    ng = hidden // CANONICAL_G
    dev = x.device
    xsf = x_sf if x_sf is not None else torch.zeros((m, ng), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((m, ng), dtype=torch.float32, device=dev)
    compile_kernel(hidden, mode)(x, xsf, q, sf)
    status.assert_on_device(f"per_token 06 {mode}", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_token/06")
