"""Test harness for the quant puzzle ladder.

Students do not read this package. It owns everything that is not a kernel:
the torch oracle, the numeric assertions, the CPU-simulator launcher, and the
per-variant checks.

A variant's ``.py`` file under ``puzzles/`` contains only ``compile_kernel`` and
``launch`` (or, in the torch tier, the reference function). Everything about how
to build its inputs, what to compare against and what else to assert lives in
``harness/variants/<kernel>.py``.

Entry point::

    python -m harness.check                   # every variant
    python -m harness.check asc/per_token/05  # one
"""
