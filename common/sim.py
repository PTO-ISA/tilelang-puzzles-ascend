"""CPU-simulator launcher. Every NPU puzzle file calls ``maybe_reexec()`` first.

There is no NPU in the teaching container, so each kernel variant re-launches
itself under a cycle-accurate CPU model of the chip.

Two runners exist:

``msprof op simulator`` (default, SoC ``Ascend950PR_9599``)
    The one that works for this whole ladder. In particular it executes the
    ``vshr`` / ``vshl`` ceil-log2 bit-trick that every ``round_sf`` variant needs.

``cannsim`` / ``npusim`` (fallback, ``TLP_SIMULATOR=cannsim``)
    Kept for comparison only. It **hangs** on those shift ops, so it cannot run
    the ``round_sf``, packed-UE8M0, or compose variants.

Environment knobs:
    TLP_SIMULATOR   msopprof (default) | cannsim
    TLP_SIM_SOC     override the SoC version
    TLP_SIM_M       token count       (default consts.SIM_M)
    TLP_SIM_K       hidden size       (default consts.SIM_K)
    TLP_CPU_ONLY=1  skip the simulator entirely and stay on CPU (torch tier)
    TLP_SIM_ROOT    where msprof writes its output (default ~/.cache/tlp_sim)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .consts import SIM_CEILING_SECONDS, SIM_K, SIM_M, SIM_TARGET_SECONDS

DEFAULT_MSPROF_SOC = "Ascend950PR_9599"
DEFAULT_CANNSIM_SOC = "Ascend950"

# The repo root. The simulator subprocess runs from $HOME (msprof refuses a
# group-writable cwd), so `python -m harness.check` there cannot find the
# package unless we put the root on PYTHONPATH explicitly.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _child_env() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def sim_shapes() -> tuple[int, int]:
    return int(os.environ.get("TLP_SIM_M", SIM_M)), int(os.environ.get("TLP_SIM_K", SIM_K))


def simulator_name() -> str:
    return os.environ.get("TLP_SIMULATOR", "msopprof").strip().lower()


def npu_ready() -> bool:
    try:
        import torch
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def print_banner(backend: str, kernel: str, variant: str) -> None:
    m, k = sim_shapes()
    print(f"[sim] backend={backend} kernel={kernel} variant={variant} "
          f"shape=({m},{k}) cores=1 runner={simulator_name()}")
    print(f"[sim] time budget: ~{SIM_TARGET_SECONDS}s target, {SIM_CEILING_SECONDS}s ceiling")


# --------------------------------------------------------------------------
# log handling
# --------------------------------------------------------------------------

_KEEP = (
    "[status]", "[check]", "[demo]", "[sim]", "PASS", "FAIL", "XFAIL", "XPASS",
    "AssertionError", "Traceback", "Error", "error:", "VMI-", "--- ",
)


def _relay(out: str) -> None:
    """Forward the lines a reader cares about; the camodel prints thousands."""
    for line in out.splitlines():
        if any(k in line for k in _KEEP):
            print(line)


def _verdict(out: str, returncode: int) -> int:
    """Decide the exit code from what the user program actually printed.

    Deliberately strict: a PASS must be present. The simulator is known to
    SIGSEGV during teardown *after* the program exits, and that specific case is
    tolerated -- but a run that produced no verdict at all is a failure, not a
    pass. Trusting the subprocess return code alone would let silent breakage
    through, because the teardown crash makes that code unreliable in both
    directions.
    """
    if "\nFAIL" in out or out.strip().endswith("FAIL"):
        return 1
    passed = "\nPASS" in out or out.strip().endswith("PASS")
    xfailed = "\nXFAIL" in out or "result=XFAIL" in out
    if passed or xfailed:
        if returncode != 0:
            print("[sim] ignoring simulator teardown status "
                  f"({returncode}) -- known SIGSEGV after the program exits")
        return 0
    print("[sim] no PASS/XFAIL verdict in output -- treating as failure")
    return 1


def _out_dir() -> Path:
    root = Path(os.environ.get("TLP_SIM_ROOT", Path.home() / ".cache" / "tlp_sim"))
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    d = Path(tempfile.mkdtemp(prefix="run_", dir=root))
    try:
        d.chmod(0o700)
    except OSError:
        pass
    return d


# --------------------------------------------------------------------------
# runners
# --------------------------------------------------------------------------

def run_under_msopprof(argv: list[str] | Path | str, soc: str | None = None) -> int | None:
    """Return an exit code, or None if msprof is unavailable (caller falls back).

    ``argv`` is the python argument list to run under the simulator -- e.g.
    ``["-m", "harness.check", "asc/per_token/05"]``. A bare path is accepted and
    treated as ``[path]``, which is how the older script-per-variant flow used it.
    """
    msprof = shutil.which("msprof")
    ascend = os.environ.get("ASCEND_HOME_PATH")
    if not msprof or not ascend:
        return None
    soc = soc or os.environ.get("TLP_SIM_SOC", DEFAULT_MSPROF_SOC)
    sim_lib = Path(ascend) / "tools" / "simulator" / soc / "lib"
    if not sim_lib.is_dir():
        print(f"[sim] no simulator libs for {soc} at {sim_lib}")
        return None

    env = _child_env()
    env["LD_LIBRARY_PATH"] = f"{sim_lib}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    out_dir = _out_dir()
    cmd = [msprof, "op", "simulator", f"--soc-version={soc}",
           f"--output={out_dir}", sys.executable, *_as_argv(argv)]

    start = time.time()
    # msprof refuses to run from a group/other-writable directory, and the repo
    # usually lives on a shared mount -- so run from $HOME.
    proc = subprocess.run(cmd, env=env, cwd=str(Path.home()),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    _relay(proc.stdout)
    print(f"[sim] wall time {elapsed:.1f}s (msprof op simulator, soc={soc}, exit={proc.returncode})")
    if elapsed > SIM_CEILING_SECONDS:
        print(f"[sim] WARNING: exceeded the {SIM_CEILING_SECONDS}s ceiling -- shrink the shape")
    return _verdict(proc.stdout, proc.returncode)


def run_under_cannsim(argv: list[str] | Path | str, soc: str | None = None) -> int:
    cannsim = shutil.which("cannsim") or shutil.which("npusim")
    if not cannsim:
        print("[sim] neither cannsim nor npusim on PATH")
        return 1
    soc = soc or os.environ.get("TLP_SIM_SOC", DEFAULT_CANNSIM_SOC)
    # cannsim launches python with cwd set to the interpreter's bindir, so every
    # path in the argument list must be absolute.
    cmd = [cannsim, "record", sys.executable, "-s", soc, "-u", " ".join(_as_argv(argv))]
    start = time.time()
    proc = subprocess.run(cmd, env=_child_env(), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - start
    _relay(proc.stdout)
    print(f"[sim] wall time {elapsed:.1f}s (cannsim, soc={soc}, exit={proc.returncode})")
    return _verdict(proc.stdout, proc.returncode)


def _as_argv(argv: list[str] | Path | str) -> list[str]:
    """Accept either a python argv list or a single script path."""
    if isinstance(argv, (str, Path)):
        return [str(Path(argv).resolve())]
    return [str(a) for a in argv]


def run_under_simulator(argv: list[str] | Path | str) -> int:
    if simulator_name() in ("msopprof", "msprof", "camodel"):
        status = run_under_msopprof(argv)
        if status is not None:
            return status
        print("[sim] msprof unavailable, falling back to cannsim")
    return run_under_cannsim(argv)


def should_reexec() -> bool:
    """True when this process must hand off to the CPU simulator."""
    if os.environ.get("TLP_CPU_ONLY") == "1":
        return False
    if os.environ.get("TLP_IN_SIMULATOR") == "1":
        return False       # already the inner process
    return not npu_ready()


def reexec_argv(argv: list[str]) -> int:
    """Re-run ``python <argv>`` under the simulator and return its verdict."""
    os.environ["TLP_IN_SIMULATOR"] = "1"
    return run_under_simulator(argv)


def maybe_reexec() -> None:
    """Re-launch this script under the CPU simulator when no NPU is visible.

    Call this as the first statement of ``main()`` in every NPU puzzle file. It
    either returns (we are already on a device, or the caller asked for CPU) or
    exits the process with the simulated run's status.
    """
    if not should_reexec():
        return
    raise SystemExit(reexec_argv([str(Path(sys.argv[0]).resolve())]))
