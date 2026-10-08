"""cast_back 07 (torch). See doc/quant/cast_back/07_col_major_compose.md"""

import torch

from harness.consts import CANONICAL_G
from harness.math_ops import decode_packed_ue8m0, unpack_e2m1_bytes


def torch_cast_back_compose(q_packed: torch.Tensor, sf_cm: torch.Tensor,
                            group_size: int = CANONICAL_G):
    """Dequantize FP4 values with column-major packed-UE8M0 scales.

    ``q_packed``: (M, K/2) int8, two e2m1 nibbles per byte.
    ``sf_cm``    : (K/group_size/2, M) int16 -- transposed *and* byte-packed.
    """
    # TODO: transpose sf_cm back to row-major with .T, decode_packed_ue8m0 it,
    #       unpack the FP4 values, then scale per group
    raise NotImplementedError("torch/cast_back/07_col_major_compose: implement torch_cast_back_compose")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/07")
