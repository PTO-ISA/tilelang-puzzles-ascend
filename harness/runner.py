"""Run one variant: import its module, drive it, emit the status line.

This preserves the ordering the old per-file ``main()`` established, because
each step exists for a reason:

1. **Probe on the host first.** ``compile_kernel`` is traced without a device so
   that an unwritten kernel reports TODO in a few seconds instead of paying ~25 s
   for a simulator launch it will not use.
2. **Then hand off to the simulator.** There is no NPU here, so the process
   re-execs itself under ``msprof op simulator`` and exits with the inner run's
   verdict. The re-exec must happen after the probe and before any device work.
3. **Then run the checks**, wrapped by ``status.run_variant`` so that exactly one
   ``[status]`` line is printed, last. ``harness.check`` parses that line and
   nothing else.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch

from common import sim, status

from harness import doc_examples
from harness.spec import Ctx, Variant

ROOT = Path(__file__).resolve().parents[1]


def module_path(variant: Variant, tier: str, role: str) -> Path:
    return ROOT / "puzzles" / tier / "quant" / role / variant.kernel / f"{variant.stem}.py"


def load_module(path: Path) -> ModuleType:
    """Import a variant file by path, without requiring it to be importable as a package."""
    name = "variant_" + path.stem + "_" + path.parent.name + "_" + path.parts[-4]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _shapes(variant: Variant, tier: str) -> list[tuple[int, int]]:
    if tier == "torch":
        return list(variant.torch_shapes)
    m, k = sim.sim_shapes()
    return [variant.shape.resolve(m, k)]


def _probe(variant: Variant, tier: str, mod: ModuleType, m: int, k: int):
    """A zero-arg callable that traces the kernel on the host, or None.

    Returning None skips the fast TODO path, which is right for the torch tier:
    there is no compile step there and the body itself runs in milliseconds.
    """
    if tier == "torch":
        return None
    compile_kernel = getattr(mod, "compile_kernel", None)
    if compile_kernel is None:
        return None
    args = variant.probe_args(m, k)
    return lambda: compile_kernel(*args)


def run_one(variant: Variant, tier: str, role: str = "answer") -> int:
    """Drive one (variant, tier, role) and return a process exit code."""
    vid = variant.id(tier)
    path = module_path(variant, tier, role)
    if not path.exists():
        print(f"[check] missing file: {path.relative_to(ROOT)}")
        status.emit(vid, status.FAIL, 0.0, "missing file")
        print(status.FAIL)
        return 1

    mod = load_module(path)
    shapes = _shapes(variant, tier)
    m0, k0 = shapes[0]

    # 1. cheap host-side TODO check, before any simulator cost
    probe = _probe(variant, tier, mod, m0, k0)
    if probe is not None and status.unimplemented(vid, probe):
        return 0

    # 2. hand off to the CPU simulator (NPU tiers only)
    if tier != "torch" and sim.should_reexec():
        argv = ["-m", "harness.check", vid]
        if role != "answer":
            argv += ["--role", role]
        raise SystemExit(sim.reexec_argv(argv))

    if tier != "torch":
        sim.print_banner(tier, variant.kernel, variant.stem)

    # 3. run the published worked example, then the variant's own checks
    def body() -> None:
        doc_examples.verify(variant, tier)
        for m, k in shapes:
            # Seed per shape rather than once per file: the result is
            # reproducible for a single variant run in isolation, which the old
            # seed-then-loop arrangement was not.
            torch.manual_seed(0)
            variant.body(Ctx(tier=tier, m=m, k=k, module=mod, variant=variant))

    return status.run_variant(vid, body, xfail_reason=variant.xfail)
