"""Launch one variant's kernel as the FIRST device op, for cycle measurement.

    python tools/vf_drive.py asc/per_token/06 full

Why this exists. ``msprof op simulator`` profiles a single operator -- in
practice the first one the process launches -- so the ordinary measurement path
(``tools/vf_cycles.py`` running ``python -m harness.check <id>``) only produces
the kernel's own numbers when the kernel is the first thing to reach the device.

Two variants break that. ``per_token/06`` and ``per_block/05`` take an *input*
scale tensor that their unused modes do not read, and their ``launch()``
allocates a placeholder for it with ``torch.zeros(..., device=...)``. On the NPU
that is a fill **kernel**, and it runs before the variant's own kernel -- so all
three tiers measured the same aten fill and reported identical cycles, which is
how the mistake was caught: three different implementations cannot cost the same.

This driver builds every tensor on the host, moves them with a plain copy, and
then calls ``compile_kernel`` directly, once. The variant's kernel is the first
operator, and the mode is explicit rather than whichever one the harness happened
to run first.

It deliberately does **not** re-verify numerics: ``python -m harness.check`` is
the only thing allowed to say a variant is correct, and ``tools/vf_cycles.py``
will not report a driven measurement for an id that did not PASS there.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                   # noqa: E402

from harness import sim                                        # noqa: E402
from harness.consts import BLOCK_K, BLOCK_MN, CANONICAL_G      # noqa: E402

# How to build arguments for the variants this driver knows about. Each entry
# returns the positional args for compile_kernel(...)(...), host-side.
M, K = 32, 128


def _load(tier: str, kernel: str, stem: str):
    path = next((ROOT / "puzzles" / tier / "quant/answer" / kernel).glob(f"{stem}_*.py"))
    spec = importlib.util.spec_from_file_location(f"drv_{tier}_{kernel}_{stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def drive_per_token_06(mod, mode: str):
    ng = K // CANONICAL_G
    x_dtype = torch.float8_e4m3fn if mode == "requant" else torch.bfloat16
    x = torch.randn(M, K, dtype=torch.float32).to(x_dtype)
    xsf = torch.zeros((M, ng), dtype=torch.float32)       # built on the HOST
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn)
    sf = torch.empty((M, ng), dtype=torch.float32)
    args = [t.npu() for t in (x, xsf, q, sf)]             # plain copies, no kernel
    mod.compile_kernel(K, mode)(*args)
    return args


def drive_per_token_04(mod, mode: str):
    """per_token/04, whose three modes are three spellings of its output stage.

    Unlike the two variants above, this one's launch() runs no device op first,
    so the harness path measures it correctly. It is driven only because the
    *alternative* spellings have no harness entry point of their own -- the
    harness calls launch(x), which is the default arm.

    ``hw_convert`` is a different entry point -- ``compile_hybrid_kernel`` --
    and declares its output as fp4 rather than packed uint8, which is the same
    bytes under a different name, so the buffer is allocated once and viewed.
    """
    ng = K // CANONICAL_G
    x = torch.randn(M, K, dtype=torch.float32) * 3.0
    q = torch.empty((M, K // 2), dtype=torch.uint8)
    sf = torch.empty((M, ng), dtype=torch.float32)
    if mode == "hw_convert":
        q = q.view(torch.float4_e2m1fn_x2)      # the same bytes, named as fp4
        args = [t.npu() for t in (x, q, sf)]
        mod.compile_hybrid_kernel(K)(*args)
    else:
        args = [t.npu() for t in (x, q, sf)]
        mod.compile_kernel(K, branch_free=(mode == "branch_free"))(*args)
    return args


def drive_per_block_05(mod, mode: str):
    nm, nk = M // BLOCK_MN, K // BLOCK_K
    x = torch.randn(M, K, dtype=torch.float32).to(torch.bfloat16)
    given = torch.zeros((nm, nk), dtype=torch.uint8)
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn)
    sf = torch.empty((nm, nk), dtype=torch.uint8)
    args = [t.npu() for t in (x, given, q, sf)]
    mod.compile_kernel(K, mode)(*args)
    return args


DRIVERS = {
    "per_token/06": drive_per_token_06,
    "per_token/04": drive_per_token_04,
    "per_block/05": drive_per_block_05,
}
DEFAULT_MODE = "full"


def main() -> int:
    if not 2 <= len(sys.argv) <= 3:
        print(__doc__)
        return 2
    vid, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) == 3 else DEFAULT_MODE)
    tier, kernel, stem = vid.split("/")
    key = f"{kernel}/{stem}"
    if key not in DRIVERS:
        print(f"[check] no driver for {key}; this driver exists only for the "
              f"variants whose launch() runs a device op first: "
              f"{', '.join(sorted(DRIVERS))}")
        return 2

    if sim.should_reexec():
        raise SystemExit(sim.reexec_argv([str(Path(__file__).resolve()), vid, mode]))

    torch.manual_seed(0)
    mod, path = _load(tier, kernel, stem)
    DRIVERS[key](mod, mode)
    print(f"[check] drove {vid} mode={mode} from {path.relative_to(ROOT)}")
    print(f"PASS drove {vid} mode={mode} (cycles come from the trace, not from here)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
