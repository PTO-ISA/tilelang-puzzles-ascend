"""What the harness knows about each variant.

One record per variant, shared by all three tiers. That sharing is possible
because the ASC and PTO tiers are semantically identical in how they are driven
and checked -- the only differences between their old ``test_correctness``
functions were dropped comments and inlined tuples. The torch tier differs in
two specific ways (it sweeps several shapes, and its tolerances are tighter),
and those are expressed as fields rather than as a separate code path.

Each variant supplies exactly one ``body(ctx)`` function. It builds the inputs,
invokes the tier's entry point through ``ctx``, and asserts. Writing it once per
variant rather than once per (variant, tier) is what collapses 66
``test_correctness`` functions into 22.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Sequence

import torch

TIERS = ("torch", "asc", "pto")
KERNELS = ("cast_back", "per_token", "per_block", "per_channel")


# ---------------------------------------------------------------------------
# shape rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """How to pick (M, K) for the NPU tiers from the configured sim shape.

    ``fixed_k`` pins K regardless of TLP_SIM_K -- per_token/07 needs K=256
    because the bfloat16 fast path steps 256 values at a time.

    ``m_multiple`` rounds M *up* to a multiple -- the per_channel variants that
    pack along M need an even number of token groups, so M=64 at the default
    SIM_M=32.
    """

    fixed_k: int | None = None
    m_multiple: int | None = None

    def resolve(self, m: int, k: int) -> tuple[int, int]:
        if self.fixed_k is not None:
            k = self.fixed_k
        if self.m_multiple is not None and m % self.m_multiple:
            m = (m // self.m_multiple + 1) * self.m_multiple
        return m, k


DEFAULT_SHAPE = Shape()


# ---------------------------------------------------------------------------
# the per-run context handed to a variant's body
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    """Everything a variant body needs, so the body itself stays tier-agnostic."""

    tier: str
    m: int
    k: int
    module: ModuleType
    variant: "Variant"

    @property
    def is_torch(self) -> bool:
        return self.tier == "torch"

    def dev(self, t: torch.Tensor) -> torch.Tensor:
        """Move a host tensor to where this tier's entry point expects it."""
        return t if self.is_torch else t.npu()

    def host(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.is_torch else t.cpu()

    def call(self, *args, **kwargs):
        """Invoke this tier's entry point.

        torch tier: the reference function named by ``Variant.torch_fn``.
        asc / pto : ``launch``, which also allocates outputs and asserts the
                    results came back from the device.
        """
        name = self.variant.torch_fn if self.is_torch else "launch"
        fn = getattr(self.module, name, None)
        if fn is None:
            raise AttributeError(
                f"{self.variant.id(self.tier)}: module has no '{name}'. "
                f"The torch tier entry point is named by Variant.torch_fn; "
                f"the NPU tiers must expose 'launch'."
            )
        return fn(*args, **kwargs)

    def call_named(self, torch_name: str, *args, **kwargs):
        """Invoke a specific torch-tier function, or ``launch`` on the NPU tiers.

        Needed by the split/requant variants, where the torch tier exposes one
        function per mode (``torch_sf_only``, ``torch_cast_only``,
        ``torch_requant``) while the NPU tiers take a ``mode`` argument on a
        single ``launch``. The body decides which shape it wants; this just
        resolves the name.
        """
        if not self.is_torch:
            return self.call(*args, **kwargs)
        fn = getattr(self.module, torch_name, None)
        if fn is None:
            raise AttributeError(
                f"{self.variant.id(self.tier)}: module has no '{torch_name}'")
        return fn(*args, **kwargs)

    def note(self, msg: str) -> None:
        """Print a progress line.

        The '[check]' tag is load-bearing: harness/sim.py forwards only tagged
        lines out of the simulator subprocess, so an untagged print vanishes.
        """
        print(f"[check] {msg}")


# ---------------------------------------------------------------------------
# the variant record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    kernel: str
    stem: str                                   # "01_e4m3_fp32sf"
    body: Callable[[Ctx], None]                 # builds inputs, calls, asserts
    torch_fn: str = ""                          # torch tier entry point name
    shape: Shape = DEFAULT_SHAPE                # NPU tier shape rule
    torch_shapes: Sequence[tuple[int, int]] = ((32, 128),)
    probe_args: Callable[[int, int], tuple] = lambda m, k: (k,)
    tiers: Sequence[str] = TIERS
    xfail: str | None = None                    # set only for a real toolchain limit

    @property
    def num(self) -> str:
        return self.stem.split("_", 1)[0]

    def id(self, tier: str) -> str:
        """The canonical variant id, also the string the TODO marker must use."""
        return f"{tier}/{self.kernel}/{self.stem}"


REGISTRY: list[Variant] = []


def register(*variants: Variant) -> None:
    REGISTRY.extend(variants)


def all_variants() -> list[Variant]:
    """Every registered variant, in ladder order."""
    if not REGISTRY:
        from harness import variants as _  # noqa: F401  (populates REGISTRY)
    order = {k: i for i, k in enumerate(KERNELS)}
    return sorted(REGISTRY, key=lambda v: (order.get(v.kernel, 99), v.stem))
