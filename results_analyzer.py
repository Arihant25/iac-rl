"""
Results analyzer for IaC-Bench paper (IaC-RL project).

Reproduces the statistics reported in the Results section of the paper:
  - Per-dataset mean/median ICS and SRS (Table 1)
  - ICS distribution breakdown (ICS=0, ICS=1, intermediate)
  - SRS distribution breakdown (SRS=1, SRS<0.7)
  - Overall cross-dataset aggregates
  - Low-SRS prompt examples cited in the paper

Also generates publication-quality figures (300 DPI PDF) for inclusion in the paper:
  - figures/fig_ics_distribution.pdf  — stacked bar: ICS bucket breakdown per dataset
  - figures/fig_srs_distribution.pdf  — histogram of SRS values per dataset
  - figures/fig_ics_vs_srs.pdf        — scatter of ICS vs SRS coloured by dataset

Input:  results/final_results.json
Output: printed report + figures/ directory
"""

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = Path("results/final_results.json")
FIGURES_DIR = Path("figures")

# Colorblind-friendly palette (Wong 2011)
COLORS = {
    "iac_eval": "#0072B2",   # blue
    "llm_iac":  "#D55E00",   # vermilion
}

DATASET_LABELS = {
    "iac_eval": "IaC-Eval",
    "llm_iac":  "llm-iac",
}

# ICS bucket colours for stacked bars
BUCKET_COLORS = {
    "ICS = 1.0":        "#2ca02c",   # green
    "0.5 < ICS < 1.0":  "#98df8a",   # light green
    "ICS = 0.5":        "#ff7f0e",   # orange
    "0 < ICS < 0.5":    "#ffbb78",   # light orange
    "ICS = 0.0":        "#d62728",   # red
}

FIG_STYLE = {
    "font.family":  "serif",
    "font.size":    10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def ics_bucket(v: float) -> str:
    if v == 0.0:
        return "zero"
    if v < 0.5:
        return "low"
    if v == 0.5:
        return "mid"
    if v < 1.0:
        return "high"
    return "full"


def analyze_dataset(name: str, entries: list[dict]) -> dict:
    ics_vals = [e["mean_ics"] for e in entries]
    srs_vals = [e["srs"] for e in entries]
    n = len(entries)

    buckets = {"zero": 0, "low": 0, "mid": 0, "high": 0, "full": 0}
    for v in ics_vals:
        buckets[ics_bucket(v)] += 1

    srs_full = sum(1 for v in srs_vals if v == 1.0)
    srs_low = sum(1 for v in srs_vals if v < 0.7)

    return {
        "name": name,
        "label": DATASET_LABELS.get(name, name),
        "n": n,
        "ics_vals": ics_vals,
        "srs_vals": srs_vals,
        "mean_ics": statistics.mean(ics_vals),
        "median_ics": statistics.median(ics_vals),
        "mean_srs": statistics.mean(srs_vals),
        "median_srs": statistics.median(srs_vals),
        "ics_full": buckets["full"],
        "ics_full_pct": 100 * buckets["full"] / n,
        "ics_zero": buckets["zero"],
        "ics_zero_pct": 100 * buckets["zero"] / n,
        "ics_mid": buckets["mid"],
        "ics_mid_pct": 100 * buckets["mid"] / n,
        "ics_high": buckets["high"],
        "ics_high_pct": 100 * buckets["high"] / n,
        "ics_low": buckets["low"],
        "ics_low_pct": 100 * buckets["low"] / n,
        "srs_full": srs_full,
        "srs_full_pct": 100 * srs_full / n,
        "srs_low": srs_low,
        "srs_low_pct": 100 * srs_low / n,
        "low_srs_entries": [
            (e.get("prompt", "")[:80], round(e["srs"], 3))
            for e in entries
            if e.get("srs", 1.0) < 0.7
        ],
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_table1(stats: list[dict]) -> None:
    print("\n=== Table 1: ICS and SRS distribution by dataset ===")
    header = (
        f"{'Dataset':<12} {'n':>5} {'Mean ICS':>10} {'Median ICS':>11} "
        f"{'ICS=1 (%)':>10} {'ICS=0 (%)':>10} {'Mean SRS':>10} {'SRS=1 (%)':>10}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['label']:<12} {s['n']:>5} {s['mean_ics']:>10.3f} "
            f"{s['median_ics']:>11.3f} {s['ics_full_pct']:>9.1f}% "
            f"{s['ics_zero_pct']:>9.1f}% {s['mean_srs']:>10.3f} "
            f"{s['srs_full_pct']:>9.1f}%"
        )


def print_overall(stats: list[dict]) -> None:
    all_n = sum(s["n"] for s in stats)
    all_ics = sum(s["mean_ics"] * s["n"] for s in stats) / all_n
    all_srs = sum(s["mean_srs"] * s["n"] for s in stats) / all_n
    print(f"\n=== Overall (n={all_n}) ===")
    print(f"  Mean ICS : {all_ics:.3f}")
    print(f"  Mean SRS : {all_srs:.3f}")


def print_low_srs(stats: list[dict]) -> None:
    print("\n=== Low-SRS prompts (SRS < 0.7) ===")
    for s in stats:
        print(f"\n{s['label']} — {s['srs_low']} entries ({s['srs_low_pct']:.1f}%)")
        for prompt, srs in sorted(s["low_srs_entries"], key=lambda x: x[1]):
            print(f"  SRS={srs:.3f}: {prompt}")


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved {path}")


def fig_ics_distribution(stats: list[dict]) -> None:
    """
    Stacked horizontal bar chart: ICS bucket breakdown per dataset.
    Bucket order (left→right): ICS=0, 0<ICS<0.5, ICS=0.5, 0.5<ICS<1, ICS=1
    """
    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 2.4))

        bucket_keys  = ["ics_zero_pct", "ics_low_pct", "ics_mid_pct", "ics_high_pct", "ics_full_pct"]
        bucket_labels = ["ICS = 0.0", "0 < ICS < 0.5", "ICS = 0.5", "0.5 < ICS < 1.0", "ICS = 1.0"]
        colours = [BUCKET_COLORS[l] for l in bucket_labels]

        y_pos = np.arange(len(stats))
        bar_h = 0.45

        for i, s in enumerate(stats):
            left = 0.0
            for key, colour, lbl in zip(bucket_keys, colours, bucket_labels):
                val = s[key]
                ax.barh(y_pos[i], val, bar_h, left=left, color=colour,
                        edgecolor="white", linewidth=0.4)
                if val >= 4:
                    ax.text(left + val / 2, y_pos[i], f"{val:.0f}%",
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold")
                left += val

        ax.set_yticks(y_pos)
        ax.set_yticklabels([s["label"] for s in stats])
        ax.set_xlabel("Percentage of prompts (%)")
        ax.set_xlim(0, 100)
        ax.set_title("ICS Distribution by Dataset")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colours, bucket_labels)]
        ax.legend(handles=patches, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False,
                  fontsize=8)

        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_ics_distribution.pdf")


def fig_srs_distribution(stats: list[dict]) -> None:
    """
    Side-by-side histograms of SRS values, one subplot per dataset.
    Annotates SRS=1 percentage and mean.
    """
    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, len(stats), figsize=(6.5, 2.8), sharey=False)
        if len(stats) == 1:
            axes = [axes]

        bins = np.linspace(0, 1, 21)  # 20 bins of width 0.05

        for ax, s in zip(axes, stats):
            colour = COLORS.get(s["name"], "#333333")
            vals = s["srs_vals"]
            n = s["n"]

            counts, edges = np.histogram(vals, bins=bins)
            pcts = 100 * counts / n

            ax.bar(edges[:-1], pcts, width=np.diff(edges),
                   align="edge", color=colour, alpha=0.85,
                   edgecolor="white", linewidth=0.3)

            ax.axvline(x=s["mean_srs"], color="black", linestyle="--",
                       linewidth=1.0, label=f"Mean = {s['mean_srs']:.3f}")
            ax.set_title(s["label"])
            ax.set_xlabel("SRS")
            ax.set_ylabel("Prompts (%)" if ax is axes[0] else "")
            ax.legend(frameon=False, fontsize=8)
            ax.set_xlim(0, 1.02)

            # Annotate SRS=1 bar
            ax.annotate(
                f"SRS=1: {s['srs_full_pct']:.1f}%",
                xy=(0.99, pcts[-1]),
                xytext=(0.72, pcts[-1] + max(pcts) * 0.06),
                fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )

        fig.suptitle("SRS Distribution by Dataset", y=1.01)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_srs_distribution.pdf")


def fig_ics_vs_srs(stats: list[dict]) -> None:
    """
    Scatter plot: ICS (x) vs SRS (y) for every prompt, coloured by dataset.
    Jitter applied to reduce overplotting on the dense ICS=0/1, SRS=1 corners.
    """
    rng = np.random.default_rng(42)

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 4.0))

        for s in stats:
            ics = np.array(s["ics_vals"])
            srs = np.array(s["srs_vals"])
            colour = COLORS.get(s["name"], "#333333")

            # Small jitter so stacked points are visible
            jx = rng.normal(0, 0.008, size=len(ics))
            jy = rng.normal(0, 0.008, size=len(srs))

            ax.scatter(ics + jx, srs + jy,
                       s=12, alpha=0.35, color=colour,
                       linewidths=0, label=s["label"])

        ax.set_xlabel("Intent Coverage Score (ICS)")
        ax.set_ylabel("Semantic Robustness Score (SRS)")
        ax.set_title("ICS vs. SRS per Prompt")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(y=0.7, color="gray", linestyle=":", linewidth=0.8,
                   label="SRS = 0.7 threshold")
        ax.legend(frameon=False)
        ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_ics_vs_srs.pdf")


def generate_figures(stats: list[dict]) -> None:
    print("\n=== Generating figures ===")
    fig_ics_distribution(stats)
    fig_srs_distribution(stats)
    fig_ics_vs_srs(stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    datasets = data["datasets"]
    stats = []
    for name, content in datasets.items():
        if "entries" not in content:
            continue
        stats.append(analyze_dataset(name, content["entries"]))

    print_table1(stats)
    print_overall(stats)

    # ICS distribution detail (used in Results §5.1)
    print("\n=== ICS distribution detail (Results §5.1) ===")
    for s in stats:
        n = s["n"]
        print(f"\n{s['label']} (n={n}):")
        print(f"  ICS = 1.0 : {s['ics_full']:>4} / {n}  ({s['ics_full_pct']:.1f}%)")
        print(f"  ICS = 0.0 : {s['ics_zero']:>4} / {n}  ({s['ics_zero_pct']:.1f}%)")
        print(f"  ICS = 0.5 : {s['ics_mid']:>4} / {n}  ({s['ics_mid_pct']:.1f}%)")

    # SRS detail (used in Results §5.2)
    print("\n=== SRS distribution detail (Results §5.2) ===")
    for s in stats:
        n = s["n"]
        print(f"\n{s['label']} (n={n}):")
        print(f"  SRS = 1.0 : {s['srs_full']:>4} / {n}  ({s['srs_full_pct']:.1f}%)")
        print(f"  SRS < 0.7 : {s['srs_low']:>4} / {n}  ({s['srs_low_pct']:.1f}%)")

    print_low_srs(stats)

    generate_figures(stats)


if __name__ == "__main__":
    main()
