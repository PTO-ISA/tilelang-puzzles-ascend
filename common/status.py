"""Explicit per-variant run status.

The failure mode this exists to prevent: a variant whose kernel fails to compile
quietly falls back to ``torch_*_cast(x.cpu())`` inside ``launch()`` and still
prints ``PASS``. A host-emulated variant and a real NPU variant would then be
indistinguishable by exit code, and ``PASS`` would carry no information at all.

So the rule here is absolute: **a kernel variant either runs on the device or it
reports XFAIL.** No ``launch()`` may compute the answer on the host. A variant
that cannot compile raises; the runner catches it, prints the verbatim compiler
diagnostic, and records XFAIL.

Machine-readable summary lines look like::

    [status] variant=pto/per_token/01_raw_fp32sf result=PASS seconds=18.4
    [status] variant=pto/per_block/09_x result=XFAIL seconds=3.1 note=VMI-LAYOUT-CONTRACT

``run_all.py`` parses those lines; nothing else in the output is contractual.
"""

from __future__ import annotations

import time
import traceback

PASS = "PASS"
XFAIL = "XFAIL"
FAIL = "FAIL"
TODO = "TODO"


def emit(variant: str, result: str, seconds: float, note: str = "") -> None:
    """Print the one line run_all.py parses."""
    tail = f" note={note}" if note else ""
    print(f"[status] variant={variant} result={result} seconds={seconds:.1f}{tail}")


def assert_on_device(name: str, *tensors) -> None:
    """Guard against a result that was quietly computed on the host.

    Call this on whatever ``launch()`` returns. Under the simulator the kernel
    outputs must come back from ``npu``; a ``cpu`` tensor means someone slipped a
    torch fallback in, which is exactly the failure mode this repo refuses.
    """
    from .device import cpu_only

    if cpu_only():
        return
    for t in tensors:
        dev = getattr(t, "device", None)
        if dev is not None and dev.type != "npu":
            raise AssertionError(
                f"{name}: result came back on '{dev.type}', expected 'npu'. "
                "A kernel variant must not compute its answer on the host -- if the "
                "kernel cannot compile, let it raise so the runner records XFAIL."
            )


def _first_diagnostic(exc: BaseException) -> str:
    """Pull the most useful single line out of a compiler error."""
    text = str(exc)
    for marker in ("VMI-", "error:", "LAYOUT", "RESIDUAL", "UNSUPPORTED"):
        for line in text.splitlines():
            if marker in line:
                return line.strip()[:200]
    return f"{type(exc).__name__}: {text.splitlines()[0][:160]}" if text else type(exc).__name__


def run_variant(variant: str, body, *, xfail_reason: str | None = None) -> int:
    """Run one kernel variant's checks and report an honest status.

    ``body`` does the work and raises on any failure. ``xfail_reason`` is set only
    for a variant known to hit a real toolchain limit; it is documented in that
    variant's doc with the verbatim error. If such a variant unexpectedly starts
    working, that is reported loudly -- a toolchain fix should not stay invisible.
    """
    start = time.time()
    try:
        body()
    except NotImplementedError as exc:
        # An unsolved puzzle is not a failure -- it just has not been written yet.
        # Only our own marker counts: torch raises NotImplementedError for
        # genuinely unsupported ops too (e.g. fancy-indexing a float8 CPU
        # tensor), and that is a real bug we must not hide.
        if not str(exc).startswith(f"{variant}: implement "):
            elapsed = time.time() - start
            print("\n--- FAIL ---")
            traceback.print_exc()
            emit(variant, FAIL, elapsed, _first_diagnostic(exc))
            print(FAIL)
            return 1
        elapsed = time.time() - start
        print(f"\n--- not implemented yet: {exc} ---")
        print("Fill in the TODO above. The reference answer for this variant is the")
        print("same file under answer/ instead of puzzle/, if you want to compare.")
        emit(variant, TODO, elapsed)
        print(TODO)
        return 0
    except Exception as exc:  # noqa: BLE001 - we classify everything
        elapsed = time.time() - start
        note = _first_diagnostic(exc)
        if xfail_reason is not None:
            print(f"\n--- XFAIL (known limitation: {xfail_reason}) ---")
            print("Verbatim diagnostic:")
            print(str(exc)[:3000])
            emit(variant, XFAIL, elapsed, note)
            print(XFAIL)
            return 0
        print("\n--- FAIL ---")
        traceback.print_exc()
        emit(variant, FAIL, elapsed, note)
        print(FAIL)
        return 1
    elapsed = time.time() - start
    if xfail_reason is not None:
        print(f"\n--- XPASS: this variant was marked XFAIL ({xfail_reason}) but now works ---")
        print("The toolchain limitation appears to be fixed. Drop the xfail marker")
        print("and update the variant's doc with the new measurement.")
        emit(variant, PASS, elapsed, "was-xfail-now-passes")
        print(PASS)
        return 0
    emit(variant, PASS, elapsed)
    print(PASS)
    return 0


def unimplemented(variant: str, probe) -> bool:
    """Report TODO immediately if this variant's kernel is still a stub.

    NPU variants relaunch themselves under the CPU simulator, which costs ~25s.
    A student who has not written the kernel yet should not wait for that just to
    be told the body is missing, so every NPU ``main()`` calls this first: it
    traces the kernel on the host (no device needed) and, if that hits the
    puzzle's own NotImplementedError marker, prints the TODO and returns True.

    Any other exception returns False, so a genuine compile error is reported by
    the normal path with its full diagnostic rather than being swallowed here.
    """
    try:
        probe()
    except NotImplementedError as exc:
        if str(exc).startswith(f"{variant}: implement "):
            print(f"--- not implemented yet: {exc} ---")
            print("Fill in the TODO in the kernel body. The reference answer is the")
            print("same path with answer/ instead of puzzle/.")
            emit(variant, TODO, 0.0)
            print(TODO)
            return True
    except Exception:  # noqa: BLE001 - let the real run surface and classify it
        return False
    return False
