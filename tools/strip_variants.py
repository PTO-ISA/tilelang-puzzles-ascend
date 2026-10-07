"""One-shot migration: reduce a variant file to its kernel.

Removes everything the harness and the markdown now own:

* the module docstring            -> a one-line pointer to the variant's page
* ``demo_numbers``                -> doc/quant/<kernel>/<stem>.md "Worked example"
* ``test_correctness``            -> harness/variants/<kernel>.py
* ``main`` and the ``__main__`` guard
* the ``sys.path.insert(..., parents[5])`` shim, which only existed so the file
  could be run as a script
* import lines left unused afterwards

What stays is ``compile_kernel`` (with its BEGIN/END SOLUTION sentinels) and
``launch`` -- the kernel and its calling contract.

Operates on answer files; ``tools/make_puzzles.py`` regenerates the puzzles from
them afterwards. AST-driven so the line ranges are exact, but edits are applied
to the source lines so formatting and comments survive.

    python tools/strip_variants.py --kernel cast_back          # dry run
    python tools/strip_variants.py --kernel cast_back --write
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DROP_FUNCS = {"demo_numbers", "test_correctness", "main"}
# The variant id is derived from the path by both the harness and
# tools/make_puzzles.py, so a module-level copy of it is dead weight that can
# drift from the filename.
DROP_ASSIGNS = {"VARIANT"}
# Modules only the removed code used. `oracle` is deliberately NOT here: some
# torch reference implementations legitimately build on it (to_col_major,
# cast_back), so whether it survives is decided by usage analysis like everything
# else.
HARNESS_ONLY_MODULES = {"sim", "demo"}


def _spans_to_drop(tree: ast.Module, src_lines: list[str]) -> list[tuple[int, int]]:
    """1-based inclusive line ranges to delete."""
    spans: list[tuple[int, int]] = []

    for node in tree.body:
        # the module docstring
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str) and node is tree.body[0]):
            spans.append((node.lineno, node.end_lineno))
            continue

        # the functions the harness now owns
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in DROP_FUNCS:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            spans.append((start, node.end_lineno))
            continue

        # the __main__ guard
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            if "__name__" in test:
                spans.append((node.lineno, node.end_lineno))
                continue

        # the now-dead VARIANT constant
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id in DROP_ASSIGNS:
                spans.append((node.lineno, node.end_lineno))
                continue

        # the sys.path shim
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = ast.unparse(node.value)
            if call.startswith("sys.path.insert"):
                spans.append((node.lineno, node.end_lineno))
                continue

    return spans


def _apply(src: str, doc_pointer: str) -> str:
    tree = ast.parse(src)
    lines = src.splitlines()
    drop = set()
    for a, b in _spans_to_drop(tree, lines):
        drop.update(range(a, b + 1))

    kept = [ln for i, ln in enumerate(lines, 1) if i not in drop]
    body = "\n".join(kept)

    # Drop imports whose bound names are no longer referenced.
    for _ in range(3):                      # a couple of passes: imports can chain
        tree2 = ast.parse(body)
        lines2 = body.splitlines()
        remove: set[int] = set()
        rewrite: dict[int, str] = {}
        for node in tree2.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = []
            for alias in node.names:
                names.append(alias.asname or alias.name.split(".")[0])
            span = range(node.lineno, node.end_lineno + 1)
            rest = "\n".join(ln for i, ln in enumerate(lines2, 1) if i not in span)
            mod = getattr(node, "module", "") or ""
            forced_mod = (isinstance(node, ast.ImportFrom)
                          and mod.startswith("harness.")
                          and mod.split(".")[-1] in HARNESS_ONLY_MODULES)
            live = [
                a for a in node.names
                if not (
                    (a.asname or a.name.split(".")[0]) in HARNESS_ONLY_MODULES
                    and isinstance(node, ast.ImportFrom) and mod == "harness"
                )
                and re.search(rf"\b{re.escape(a.asname or a.name.split('.')[0])}\b", rest)
            ]
            if forced_mod or not live:
                remove.update(span)
            elif len(live) != len(node.names):
                # keep the line but only the names that are still referenced
                kept_names = ", ".join(
                    a.name if not a.asname else f"{a.name} as {a.asname}" for a in live)
                if isinstance(node, ast.ImportFrom):
                    rewrite[node.lineno] = f"from {mod} import {kept_names}"
                else:
                    rewrite[node.lineno] = f"import {kept_names}"
                remove.update(r for r in span if r != node.lineno)
        if not remove and not rewrite:
            break
        body = "\n".join(rewrite.get(i, ln) for i, ln in enumerate(lines2, 1)
                          if i not in remove)

    body = f'"""{doc_pointer}"""\n\n' + body.lstrip("\n")
    body = re.sub(r"\n{3,}", "\n\n\n", body).rstrip() + "\n"
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    total_before = total_after = 0
    for tier in ("torch", "asc", "pto"):
        d = ROOT / "puzzles" / tier / "quant" / "answer" / args.kernel
        for path in sorted(d.glob("[0-9]*.py")):
            label = {"torch": "torch", "asc": "ASC", "pto": "PTO"}[tier]
            pointer = (f"{args.kernel} {path.stem.split('_')[0]} ({label}). "
                       f"See doc/quant/{args.kernel}/{path.stem}.md")
            src = path.read_text()
            out = _apply(src, pointer)
            before, after = src.count("\n") + 1, out.count("\n") + 1
            total_before += before
            total_after += after
            print(f"  {before:4} -> {after:3} lines  {path.relative_to(ROOT)}")
            if args.write:
                path.write_text(out)
    verb = "rewrote" if args.write else "would rewrite"
    print(f"\n{verb} {args.kernel}: {total_before} -> {total_after} lines "
          f"({100 * (1 - total_after / total_before):.0f}% smaller)")
    if not args.write:
        print("dry run; pass --write to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
