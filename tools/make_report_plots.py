"""Render the charts used by presentation/, from the measured data.

    python tools/make_report_plots.py           # write presentation/plots/
    python tools/make_report_plots.py --check   # fail if the data has moved

Three charts, all derived from `perf/quant_vf_cycles.json` and the static counts
printed by `tools/vf_lines.py`:

* `correlation.png` -- the deck's central result. Operations (or lines) written
  against measured cycles, per variant, with the Pearson r. Across 22 variants
  r is -0.09 for PTO and -0.01 for SIMT once the FP4 outlier is removed: source
  size carries essentially no information about cycle count.
* `pto_cycles.png`  -- PTO/ASC cycle ratio per variant, sorted, against the 1.00
  line.
* `simt_cycles.png` -- SIMT/ASC cycle ratio per variant, log axis, with the
  `per_token/04` outlier marked separately.

PNG rather than PDF so the same markdown renders under both the Beamer and the
Marp builds. A sidecar `plots/data.json` records the exact series plotted;
`--check` recomputes from the measurements and fails if they differ, which is
what keeps a chart from outliving the numbers it draws.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "plots"
FP4 = "per_token/04_fp32_in_fp4_out"


def load() -> dict:
    cycles = json.loads((ROOT / "perf/quant_vf_cycles.json").read_text())
    out = subprocess.run([sys.executable, str(ROOT / "tools/vf_lines.py")],
                         capture_output=True, text=True, cwd=ROOT).stdout
    data = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 9 and "/" in p[0]:
            v = p[0]
            data[v] = {
                "ops_asc": int(p[1]), "ops_pto": int(p[2]),
                "ln_asc": int(p[3]), "ln_pto": int(p[4]), "ln_simt": int(p[5]),
                "cyc_asc": cycles[f"asc/{v}"]["cycles"],
                "cyc_pto": cycles[f"pto/{v}"]["cycles"],
                "cyc_simt": cycles[f"simt/{v}"]["cycles"],
            }
    if len(data) != 22:
        raise SystemExit(f"expected 22 variants, got {len(data)}")
    return data


def pearson(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def series(data: dict) -> dict:
    v = list(data)
    pto_ops = [data[k]["ops_pto"] / data[k]["ops_asc"] for k in v]
    pto_cyc = [data[k]["cyc_pto"] / data[k]["cyc_asc"] for k in v]
    simt_ln = [data[k]["ln_simt"] / data[k]["ln_asc"] for k in v]
    simt_cyc = [data[k]["cyc_simt"] / data[k]["cyc_asc"] for k in v]
    keep = [i for i, k in enumerate(v) if k != FP4]
    return {
        "variants": v,
        "pto_ops_ratio": pto_ops, "pto_cycle_ratio": pto_cyc,
        "simt_line_ratio": simt_ln, "simt_cycle_ratio": simt_cyc,
        "r_pto": pearson(pto_ops, pto_cyc),
        "r_simt": pearson(simt_ln, simt_cyc),
        "r_simt_no_fp4": pearson([simt_ln[i] for i in keep],
                                 [simt_cyc[i] for i in keep]),
    }


def short(v: str) -> str:
    kern, stem = v.split("/")
    return f"{kern}/{stem.split('_')[0]}"


def draw(s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight",
    })
    names = [short(v) for v in s["variants"]]

    # ---- 1. the central result: size does not predict cycles ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].scatter(s["pto_ops_ratio"], s["pto_cycle_ratio"], s=44,
                  color="#1f4e8c", zorder=3)
    ax[0].axhline(1.0, color="#999", lw=1, ls="--", zorder=1)
    lo = min(s["pto_ops_ratio"]); hi = max(s["pto_ops_ratio"])
    ax[0].plot([lo, hi], [lo, hi], color="#c0392b", lw=1.2, ls=":",
               label="if size predicted cycles", zorder=2)
    ax[0].set_xlabel("PTO / ASC  vector operations written")
    ax[0].set_ylabel("PTO / ASC  simulated cycles")
    ax[0].set_title(f"PTO vs ASC   (Pearson r = {s['r_pto']:+.2f}, n=22)")
    ax[0].legend(frameon=False, fontsize=9, loc="upper left")

    keep = [i for i, v in enumerate(s["variants"]) if v != FP4]
    ax[1].scatter([s["simt_line_ratio"][i] for i in keep],
                  [s["simt_cycle_ratio"][i] for i in keep],
                  s=44, color="#1f4e8c", zorder=3)
    ax[1].axhline(1.0, color="#999", lw=1, ls="--", zorder=1)
    ax[1].set_xlabel("SIMT / ASC  source lines written")
    ax[1].set_ylabel("SIMT / ASC  simulated cycles")
    ax[1].set_title(f"SIMT vs ASC   (r = {s['r_simt_no_fp4']:+.2f}, n=21,\n"
                    f"per_token/04 excluded)")
    fig.tight_layout()
    fig.savefig(OUT / "correlation.png")
    plt.close(fig)

    # ---- 2. PTO/ASC cycle ratio per variant ---------------------------
    order = sorted(range(22), key=lambda i: s["pto_cycle_ratio"][i])
    fig, ax = plt.subplots(figsize=(9, 5.0))
    vals = [s["pto_cycle_ratio"][i] for i in order]
    cols = ["#2e7d32" if x < 0.98 else "#c0392b" if x > 1.02 else "#8a8a8a"
            for x in vals]
    ax.barh(range(22), vals, color=cols)
    ax.axvline(1.0, color="#333", lw=1.2)
    ax.set_yticks(range(22))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("PTO cycles / ASC cycles  (simulated VF cycles)")
    ax.set_title("PTO is faster on 9 of 22 variants, slower on 6, "
                 "within 2% on 7")
    ax.set_xlim(0, 1.25)
    for y, x in enumerate(vals):
        ax.text(x + 0.012, y, f"{x:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pto_cycles.png")
    plt.close(fig)

    # ---- 3. SIMT/ASC cycle ratio per variant --------------------------
    order = sorted(range(22), key=lambda i: s["simt_cycle_ratio"][i])
    fig, ax = plt.subplots(figsize=(9, 5.0))
    vals = [s["simt_cycle_ratio"][i] for i in order]
    cols = ["#c0392b" if s["variants"][i] == FP4 else "#1f4e8c" for i in order]
    ax.barh(range(22), vals, color=cols)
    ax.axvline(1.0, color="#333", lw=1.2)
    ax.set_xscale("log")
    ax.set_yticks(range(22))
    ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("SIMT cycles / ASC cycles, log scale  (simulated VF cycles)")
    ax.set_title("SIMT costs 3.2x-21.2x; per_token/04 (red) is 99.6x, "
                 "a hand-written FP4 encoder")
    for y, x in enumerate(vals):
        ax.text(x * 1.06, y, f"{x:.1f}", va="center", fontsize=8)
    ax.set_xlim(1, 400)
    fig.tight_layout()
    fig.savefig(OUT / "simt_cycles.png")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="fail if the plotted series no longer match the data")
    args = ap.parse_args()

    s = series(load())
    sidecar = OUT / "data.json"

    if args.check:
        if not sidecar.exists():
            print(f"{sidecar.relative_to(ROOT)} missing -- run without --check")
            return 1
        old = json.loads(sidecar.read_text())
        drift = [k for k in s if json.dumps(old.get(k)) != json.dumps(s[k])]
        missing = [p for p in ("correlation.png", "pto_cycles.png",
                               "simt_cycles.png") if not (OUT / p).exists()]
        if drift or missing:
            for k in drift:
                print(f"STALE   plotted series {k!r} no longer matches the data")
            for p in missing:
                print(f"MISSING {p}")
            print("\nre-run without --check to redraw")
            return 1
        print(f"all 3 charts match the measured data "
              f"(r_pto={s['r_pto']:+.3f}, r_simt_ex_fp4={s['r_simt_no_fp4']:+.3f})")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    draw(s)
    sidecar.write_text(json.dumps(s, indent=1, sort_keys=True) + "\n")
    print(f"wrote 3 charts to {OUT.relative_to(ROOT)}")
    print(f"  Pearson r, PTO ops vs cycles      {s['r_pto']:+.3f}")
    print(f"  Pearson r, SIMT lines vs cycles   {s['r_simt']:+.3f}")
    print(f"  the same, excluding per_token/04  {s['r_simt_no_fp4']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
