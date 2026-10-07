r"""Lint the Markdown for LaTeX that GitHub will not render.

GitHub runs GFM *before* MathJax, so a lot of valid TeX never reaches the math
renderer: GFM eats a backslash before `\\`, `_`, `,`, `;`, `!` and `#` inside `$`/`$$`,
treats `$20` as currency, and reads an indented `$$` as list continuation. The
result renders as raw TeX source rather than failing loudly, which is why this
is a lint and not something you notice by reading the file.

Every rule below corresponds to a documented pitfall. Run it on every commit
that touches a `.md`:

    python tools/check_math.py            # lint every tracked .md
    python tools/check_math.py doc/x.md   # just one

This is a lint, not a renderer: it proves the known pitfalls are absent, not
that the output looks right. That is the reason to keep math minimal here.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
BAD_DELIMS = re.compile(r"\\[\[\]()]")
TEXTY = re.compile(r"\\(?:text|texttt|mathrm)\s*\{([^}]*)\}")


class Finding(list):
    def add(self, path: Path, line: int, rule: str, detail: str) -> None:
        self.append((path, line, rule, detail))


def _segments(lines: list[str]):
    """Yield (lineno, text, kind) with kind in {prose, code, display}.

    Fenced code is skipped by every rule: raw TeX in a code block is legitimate
    (it is showing source), and `$` in a shell snippet is a prompt.
    """
    in_code = False
    in_display = False
    fence = ""
    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not in_display and (stripped.startswith("```") or stripped.startswith("~~~")):
            if not in_code:
                in_code, fence = True, stripped[:3]
            elif stripped.startswith(fence):
                in_code = False
            yield i, raw, "fence"
            continue
        if in_code:
            yield i, raw, "code"
            continue
        if stripped == "$$":
            in_display = not in_display
            yield i, raw, "display-delim"
            continue
        yield i, raw, "display" if in_display else "prose"


def check_file(path: Path, out: Finding) -> None:
    lines = path.read_text().splitlines()
    display_open_line = None
    in_display = False

    for lineno, raw, kind in _segments(lines):
        if kind == "code":
            continue

        if kind == "display-delim":
            if raw != "$$":
                out.add(path, lineno, "display-indent",
                        "`$$` must sit at column 0 with nothing else on the line; "
                        "indented `$$` is read as list continuation and renders as raw TeX")
            if not in_display:
                in_display, display_open_line = True, lineno
                prev = lines[lineno - 2].strip() if lineno >= 2 else ""
                if prev:
                    out.add(path, lineno, "display-blankline",
                            "a display block needs a blank line before it")
            else:
                in_display = False
                nxt = lines[lineno].strip() if lineno < len(lines) else ""
                if nxt:
                    out.add(path, lineno, "display-blankline",
                            "a display block needs a blank line after it")
            continue

        if BAD_DELIMS.search(raw):
            out.add(path, lineno, "bad-delimiter",
                    r"never use \[ \] \( \) for math; CommonMark eats them. Use $ / $$")

        math_spans: list[str] = []
        if kind == "display":
            math_spans.append(raw)
            if raw.rstrip().endswith("\\\\"):
                out.add(path, lineno, "row-break",
                        r"GFM turns \\ into a single \; use \cr for matrix/aligned rows")
            if raw.strip() and set(raw.strip()) <= set("=") :
                out.add(path, lineno, "setext-split",
                        "a line of only `=` inside $$ is parsed as a setext heading and "
                        "splits the math block; keep `=` on the previous line")
            if raw.strip() and set(raw.strip()) <= set("-"):
                out.add(path, lineno, "setext-split",
                        "a line of only `-` inside $$ is parsed as a setext heading")
        else:
            for m in INLINE.finditer(raw):
                span = m.group(1)
                math_spans.append(span)
                if span != span.strip():
                    out.add(path, lineno, "inline-space",
                            f"`$ x $` does not render; write `${span.strip()}$`")
                if span[:1].isdigit():
                    out.add(path, lineno, "digit-initial",
                            f"inline math starting with a digit is read as currency "
                            f"(`${span[:6]}...`); move it to a $$ block or rewrite")
                if span.count("_") >= 2:
                    out.add(path, lineno, "double-subscript",
                            "two `_` in one inline span: GFM makes it <em> before MathJax "
                            "sees it; use a $$ block")
                start, end = m.start(), m.end()
                before = raw[start - 1] if start else " "
                after = raw[end] if end < len(raw) else " "
                if before.isalnum() or before == "-":
                    out.add(path, lineno, "glued-dollar",
                            f"`{before}$` -- GitHub skips inline math glued to a word "
                            f"or hyphen; add a space")
                if after.isalnum() or after == "-":
                    out.add(path, lineno, "glued-dollar",
                            f"`${after}` -- same on the closing side")
            stray = raw.count("$") - 2 * len(list(INLINE.finditer(raw)))
            if stray > 0 and "$$" not in raw and "`" not in raw:
                out.add(path, lineno, "stray-dollar",
                        f"{stray} unmatched `$` on this line; escape a literal one as \\$")

        for span in math_spans:
            for bad, rule, why in (
                ("#", "hash-in-math", r"GFM strips \# and TeX then errors on #"),
                ("@", "at-in-math", "TeX treats @ as special; put code identifiers in backticks"),
                (r"\,", "thin-space", r"GFM strips the backslash, leaving a literal comma"),
                (r"\;", "thin-space", r"GFM strips the backslash"),
                (r"\!", "thin-space", r"GFM strips the backslash"),
                (r"\texttt", "texttt-in-math", "\\texttt does not work in GitHub math; use backticks"),
            ):
                if bad in span:
                    out.add(path, lineno, rule, f"`{bad}` in math -- {why}")
            for inner in TEXTY.findall(span):
                if "_" in inner:
                    out.add(path, lineno, "underscore-in-text",
                            r"`_` inside \text/\mathrm is text mode and illegal in TeX; "
                            r"use a math subscript or backticks")
                if "$" in inner:
                    out.add(path, lineno, "nested-math",
                            r"no nested math inside \text{...}; flatten it")

    if in_display:
        out.add(path, display_open_line or len(lines), "unclosed-display",
                "a `$$` block is never closed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="default: every .md in the repo")
    args = ap.parse_args()

    if args.paths:
        files = args.paths
    else:
        files = sorted(p for p in ROOT.rglob("*.md")
                       if "third_party" not in p.parts and ".git" not in p.parts)

    out = Finding()
    for path in files:
        check_file(path, out)

    if not out:
        print(f"{len(files)} markdown files: no LaTeX rendering pitfalls found")
        return 0

    by_rule: dict[str, int] = {}
    for path, line, rule, detail in out:
        by_rule[rule] = by_rule.get(rule, 0) + 1
        rel = path.relative_to(ROOT) if ROOT in path.parents or path.parent == ROOT else path
        print(f"{rel}:{line}: [{rule}] {detail}")
    print(f"\n{len(out)} finding(s) in {len(files)} files: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items())))
    return 1


if __name__ == "__main__":
    sys.exit(main())
