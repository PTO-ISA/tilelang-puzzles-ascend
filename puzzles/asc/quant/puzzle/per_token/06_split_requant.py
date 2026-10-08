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
                # TODO: gate the three passes on the mode. requant first
                #       dequantizes x_ub into val_ub using the input scales
                #       (broadcast + vsel as in cast_back/01) with a barrier
                #       after. sf_only emits passes 1-2 only. cast_only skips pass
                #       1 and sets inv = S.vdiv(1.0, given sf). Everything else is
                #       variants 01/02.
                raise NotImplementedError("asc/per_token/06_split_requant: implement per_token_cast")
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
