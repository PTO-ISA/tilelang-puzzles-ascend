# VF cycles from `msprof op simulator`

Compare two kernels by **simulated VF cycles**: a TileLang SIMT kernel (`T.SimtVF`) against a PTO / SIMD kernel (`T.SimdVF` or hand-written PTO-DSL). The number comes from the simulator's per-instruction CSV, not from a device `msprof` run and not from decoding `instr.bin`.

**VF cycles** = sum of the `cycles` column over every VF launch row on one vector core.

| Kernel | Row to sum (`instr`) |
|---|---|
| SIMT VF | `VF_SIMT` |
| SIMD / PTO VF | `VF`, or `VF_SIMD` if that is the name in the file |

Both names are **confirmed on `Ascend950PR_9599`**: a `T.SimdVF` kernel emits
`VF`, and a `T.SimtVF` kernel emits `VF_SIMT`. (`VF_SIMT` is absent from every
trace taken before this repo ran a SIMT kernel, which is why it reads as
unverified in older notes -- nothing had emitted one yet, the name was never
wrong.)

`cycles` on a row is already the total for that row. Do not multiply it by `call_count`. Add a row only when the same launch name appears again (another PC / another launch).

PTO speed-up at the same shape:

```text
pto_speedup = simt_vf_cycles / pto_vf_cycles
```

A value above 1 means the PTO kernel spent fewer simulated VF cycles. Wall microseconds are a second number; they include scalar issue, DMA, and flags, so they are not the VF comparison.

## 0. In this repo, use the tool

`tools/vf_cycles.py` implements everything below against the puzzle ladder:

```bash
python tools/vf_cycles.py cast_back/01     # every vector tier of one variant
python tools/vf_cycles.py --all            # the whole ASC : PTO : SIMT table
python tools/vf_cycles.py --table          # re-print without re-measuring
```

It runs each variant through `python -m harness.check`, which re-execs under the
simulator, reads the `--output` path that `harness/sim.py` prints when
`TLP_KEEP_SIM_OUT=1`, and sums the launch rows. It **withholds the cycle count
for any variant that did not pass**, because a number from a kernel that computes
the wrong answer is worse than no number.

The rest of this page is the underlying method, for use outside this repo.

## 1. Run both kernels the same way

Use one soc, one core, and the same shape, dtype, and launch count. Source the CANN environment first. The simulator libraries for that soc must be on `LD_LIBRARY_PATH`.

```bash
source "$ASCEND_HOME_PATH/bin/setenv.bash"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:${LD_LIBRARY_PATH:-}"

# The profiled program is the last argument.
msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --output=./out_simt \
  ./run_simt.sh

msprof op simulator \
  --soc-version=Ascend950PR_9599 \
  --output=./out_pto \
  ./run_pto.sh
```

`scripts/sim_dsl.sh` in a PTOAS checkout is the same command. It stages `msprof op simulator`, writes `msprof.stdout.log`, and copies the tree to `--output`. It also requires a Python that can import `ptodsl`, so a pure TileLang launcher is usually started with `msprof` directly.

Keep each kernel in its own `--output` directory. A second run into the same directory mixes operator dumps.

## 2. Find the trace

Under `--output`, the parsed simulator tree looks like this (single operator):

```text
OPPROF_<timestamp>_*/
└── simulator/
    └── core0.veccore0/
        ├── core0.veccore0_instr_exe.csv   # VF cycle source
        ├── core0.veccore0_code_exe.csv    # per source line, not the VF total
        └── trace.json                     # pipeline chart, not the cycle total
```

Use `core0.veccore0` unless you deliberately measured another sub-core, and then use that same core for both kernels. Multi-operator runs add a timestamp to the CSV name (`core0.veccore0_instr_exe_<timestamp>.csv`) and nest a directory per operator. Pick the CSV for the kernel you launched.

`trace.json` is the instruction pipeline. In Chrome (`chrome://tracing`) or MindStudio Insight, VF/SIMT issue shows up on the **PUSHQ** lane. Use that view to confirm you found the right launch. The reported cycle count still comes from the CSV.

## 3. Extract

Columns in `*_instr_exe.csv`:

| Field | Use |
|---|---|
| `instr` | `VF_SIMT` or `VF` / `VF_SIMD` |
| `cycles` | Add these cells. This is the metric. |
| `call_count` | How many times that row's instruction ran. Do not scale `cycles` by it. |
| `running_time(us)` | Optional VF microseconds, summed over the same rows. |
| `pipe` | **`PUSHQ`** on VF launch rows -- not `VEC`. Do not filter on this column: `MOVEMASK` and `PUSH_PB` sit on PUSHQ too, so `instr` is the only safe key. MTE/CUBE rows are not VF cycles. |
| `addr` | PC. Distinct PCs are distinct rows and are all included. |

Inner opcodes (`vadd`, loads, stores, `RVECEX`, and so on) sit inside the VF. Their `cycles` overlap. Summing them is not the VF total. Whole-core fields such as `aiv_total_cycles`, `aiv_vec_simt_vf_ratio`, and the `duration_time` line for `core0.veccore0` in `msprof.stdout.log` are also different measurements.

Confirm the launch name once per kernel. Save this as `list_vf_instr.py` and run `python3 list_vf_instr.py out_simt`, then the same command on `out_pto`:

```python
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
hits = sorted(p for p in root.rglob("*instr_exe*.csv") if "veccore0" in str(p))
path = hits[0] if hits else next(root.rglob("*instr_exe*.csv"), None)
print("csv", path if path else "NONE")
if path:
    names = {}
    for row in csv.DictReader(path.open(newline="", encoding="utf-8", errors="replace")):
        name = (row.get("instr") or "").strip()
        if "VF" in name.upper() or name.upper() in {"EXIPC", "IFU"}:
            names[name] = names.get(name, 0) + float(row.get("cycles") or 0)
    for name, cyc in sorted(names.items(), key=lambda kv: -kv[1]):
        print(f"{name}\t{cyc:.0f}")
```

The comparison below classifies each tree by which launch name it actually contains. Save it as `extract_vf_cycles.py` and run:

```bash
python3 extract_vf_cycles.py out_simt out_pto
```

```python
"""Sum VF launch cycles from two `msprof op simulator` output directories."""
import csv
import sys
from pathlib import Path

SIMT_NAMES = {"vf_simt"}
SIMD_NAMES = {"vf", "vf_simd"}


def instr_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob("*instr_exe*.csv"))


def prefer_veccore0(paths: list[Path]) -> Path | None:
    for path in paths:
        if "core0.veccore0" in path.name or "core0.veccore0" in str(path.parent):
            return path
    return paths[0] if paths else None


def vf_total(root: Path) -> dict:
    path = prefer_veccore0(instr_csvs(root))
    out = {
        "root": str(root),
        "csv": None if path is None else str(path),
        "kind": None,
        "instr": None,
        "rows": 0,
        "call_count": 0,
        "cycles": None,
        "us": None,
    }
    if path is None:
        return out
    grouped: dict[str, list[float]] = {}
    calls: dict[str, int] = {}
    for row in csv.DictReader(path.open(newline="", encoding="utf-8", errors="replace")):
        name = (row.get("instr") or "").strip()
        key = name.lower()
        if key not in SIMT_NAMES and key not in SIMD_NAMES:
            continue
        try:
            cyc = float(row.get("cycles") or 0)
            us = float(row.get("running_time(us)") or 0)
            cc = int(float(row.get("call_count") or 0))
        except ValueError:
            continue
        bucket = grouped.setdefault(name, [0.0, 0.0])
        bucket[0] += cyc
        bucket[1] += us
        calls[name] = calls.get(name, 0) + cc
        out["rows"] += 1
    if not grouped:
        return out
    # One kernel is one kind. If both names appear, keep the kind with more cycles.
    simt = {n: v for n, v in grouped.items() if n.lower() in SIMT_NAMES}
    simd = {n: v for n, v in grouped.items() if n.lower() in SIMD_NAMES}
    chosen = simt if sum(v[0] for v in simt.values()) >= sum(v[0] for v in simd.values()) else simd
    if not chosen:
        chosen = grouped
    out["kind"] = "SIMT" if chosen is simt else "SIMD"
    out["instr"] = ",".join(sorted(chosen))
    out["cycles"] = sum(v[0] for v in chosen.values())
    out["us"] = sum(v[1] for v in chosen.values())
    out["call_count"] = sum(calls[n] for n in chosen)
    return out


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: extract_vf_cycles.py <simt-or-a-outdir> <pto-or-b-outdir>")
    a = vf_total(Path(sys.argv[1]))
    b = vf_total(Path(sys.argv[2]))
    # Order the report as SIMT then PTO when the trees identify themselves.
    pair = [a, b]
    if a["kind"] == "SIMD" and b["kind"] == "SIMT":
        pair = [b, a]
    print(f"{'role':<8} {'kind':<6} {'instr':<16} {'vf_cycles':>12} {'vf_us':>10} {'calls':>8}  csv")
    for label, item in zip(("A", "B"), pair):
        cyc = "MISSING" if item["cycles"] is None else f"{item['cycles']:.0f}"
        us = "-" if item["us"] is None else f"{item['us']:.3f}"
        print(
            f"{label:<8} {item['kind'] or '-':<6} {item['instr'] or '-':<16} "
            f"{cyc:>12} {us:>10} {item['call_count']:>8}  {item['csv']}"
        )
    simt = next((x for x in pair if x["kind"] == "SIMT" and x["cycles"]), None)
    simd = next((x for x in pair if x["kind"] == "SIMD" and x["cycles"]), None)
    if simt and simd and simd["cycles"]:
        speedup = simt["cycles"] / simd["cycles"]
        print(f"pto_speedup = {simt['cycles']:.0f} / {simd['cycles']:.0f} = {speedup:.2f}")
        print("pto_speedup > 1 means the SIMD/PTO VF used fewer simulated cycles.")
    else:
        print("pto_speedup skipped: need one SIMT tree (VF_SIMT) and one SIMD tree (VF or VF_SIMD).")


if __name__ == "__main__":
    main()
```

`MISSING` means that CSV has no `VF_SIMT`, `VF`, or `VF_SIMD` row: open the file and read the `instr` column before inventing another name. `EXIPC` / `IFU` cycle totals are a different counter; use them only when you have already checked that this soc's SIMD trace has no VF launch row.

## 4. Report

State soc, CANN build, shape, dtype, which core's CSV, and that both kernels matched on correctness. Then:

| Kernel | `instr` | VF cycles | VF µs (`running_time`) | PTO speed-up |
|---|---|---:|---:|---:|
| SIMT | `VF_SIMT` | 727 | 0.42 | |
| PTO / SIMD | `VF` | 59 | 0.03 | 727/59 = 12.32 |

Speed-up is unitless simulated-cycle ratio. Quote the absolute cycles next to it. If several VF rows were summed, say how many. A whole-core `duration_time(us)` from `msprof.stdout.log` may be listed beside this table and labeled as vec-core time, not as VF cycles.

Check the two CSVs describe the same work: same grid, same trip count, and a `call_count` that matches the launches you expect. A large speed-up with a mismatched `call_count` is a different kernel, not a faster one.
