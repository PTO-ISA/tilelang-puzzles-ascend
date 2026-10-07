"""Tiny printed examples and random inputs."""

import torch


def fmt_row(values) -> str:
    return "[" + ", ".join(f"{float(v):8.4g}" for v in values) + "]"


def print_example(title: str, **named_tensors: torch.Tensor) -> None:
    """Print small tensors readably.

    Every line carries the ``[demo]`` marker so it survives the simulator log
    relay in ``common/sim.py``, which forwards only tagged lines (the camodel
    prints thousands of its own).
    """
    print(f"[demo] --- {title} ---")
    for name, t in named_tensors.items():
        cpu = t.detach().float().cpu()
        print(f"[demo]   {name:10s} dtype={t.dtype} shape={tuple(t.shape)}")
        if cpu.ndim == 0:
            print(f"[demo]     {float(cpu):.6g}")
        elif cpu.ndim == 1:
            print(f"[demo]     {fmt_row(cpu.tolist())}")
        else:
            for row in cpu.tolist()[:4]:
                print(f"[demo]     {fmt_row(row[:8])}{' ...' if len(row) > 8 else ''}")


def randn_with_zero_row(m: int, k: int, device, dtype=torch.bfloat16) -> torch.Tensor:
    x = torch.randn((m, k), dtype=dtype, device=device)
    x[0].zero_()
    return x
