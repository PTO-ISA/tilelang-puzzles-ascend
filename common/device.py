"""Device selection: NPU under the simulator, CPU when explicitly requested."""

import os

import torch


def cpu_only() -> bool:
    """True when the caller asked to stay on CPU (TLP_CPU_ONLY=1)."""
    return os.environ.get("TLP_CPU_ONLY") == "1"


def get_device() -> torch.device:
    """``npu:0`` under the simulator or a real NPU; ``cpu`` when TLP_CPU_ONLY=1."""
    if cpu_only():
        return torch.device("cpu")
    import torch_npu  # noqa: F401

    return torch.device("npu:0")


def sync() -> None:
    if cpu_only():
        return
    torch.npu.synchronize()
