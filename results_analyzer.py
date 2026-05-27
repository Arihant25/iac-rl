"""
Results analyzer for IaC-Bench paper (IaC-RL project).

Reproduces the statistics reported in the Results section of the paper:
  - Per-dataset mean/median ICS and SRS (Table 1)
  - ICS distribution breakdown (ICS=0, ICS=1, intermediate)
  - SRS distribution breakdown (SRS=1, SRS<0.7)
  - Overall cross-dataset aggregates
  - Low-SRS prompt examples cited in the paper

Also generates publication-quality figures (300 DPI PDF) for inclusion in the paper:
  ICS / SRS
    figures/fig_ics_distribution.pdf  — stacked bar: ICS bucket breakdown per dataset
    figures/fig_srs_distribution.pdf  — histogram of SRS values per dataset
    figures/fig_ics_vs_srs.pdf        — scatter of ICS vs SRS coloured by dataset
  Hard-gate validation & TerraMetrics (from results/summary.csv)
    figures/fig_validation_rate.pdf   — hard-gate pass rate per model × dataset
    figures/fig_structural_quality.pdf— gen vs ref structural metrics per dataset
    figures/fig_risk_indicators.pdf   — security-smell indicators per model (heatmap)

Note: TFLint and Trivy output is not present in results/summary.csv; those figures
      would require running linting/security scans and adding the columns.

Input:  results/final_results.json  +  results/summary.csv
Output: printed report + figures/ directory
"""

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_PATH = Path("results/final_results.json")
SUMMARY_PATH = Path("results/summary.csv")
ICS_PER_PROMPT_PATH = Path("results/ics_per_prompt.csv")
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


def analyze_dataset(
    name: str,
    entries: list[dict],
    ics_vals_override: list[float] | None = None,
    ics_scatter_override: list[float] | None = None,
) -> dict:
    # flat ICS observations (for distribution figure)
    ics_vals = ics_vals_override if ics_vals_override is not None else [e["mean_ics"] for e in entries]
    # per-scenario ICS means (for ICS vs SRS scatter — must match SRS count)
    if ics_scatter_override is not None:
        n_min = min(len(ics_scatter_override), len(entries))
        ics_scatter = ics_scatter_override[:n_min]
        srs_vals = [e["srs"] for e in entries[:n_min]]
    else:
        ics_scatter = [e["mean_ics"] for e in entries]
        srs_vals = [e["srs"] for e in entries]
    n_ics = len(ics_vals)
    n_srs = len(srs_vals)

    buckets = {"zero": 0, "low": 0, "mid": 0, "high": 0, "full": 0}
    for v in ics_vals:
        buckets[ics_bucket(v)] += 1

    srs_full = sum(1 for v in srs_vals if v == 1.0)
    srs_low = sum(1 for v in srs_vals if v < 0.7)

    return {
        "name": name,
        "label": DATASET_LABELS.get(name, name),
        "n": n_srs,
        "n_ics": n_ics,
        "ics_vals": ics_vals,
        "ics_scatter": ics_scatter,
        "srs_vals": srs_vals,
        "mean_ics": statistics.mean(ics_vals),
        "median_ics": statistics.median(ics_vals),
        "mean_srs": statistics.mean(srs_vals),
        "median_srs": statistics.median(srs_vals),
        "ics_full": buckets["full"],
        "ics_full_pct": 100 * buckets["full"] / n_ics,
        "ics_zero": buckets["zero"],
        "ics_zero_pct": 100 * buckets["zero"] / n_ics,
        "ics_mid": buckets["mid"],
        "ics_mid_pct": 100 * buckets["mid"] / n_ics,
        "ics_high": buckets["high"],
        "ics_high_pct": 100 * buckets["high"] / n_ics,
        "ics_low": buckets["low"],
        "ics_low_pct": 100 * buckets["low"] / n_ics,
        "srs_full": srs_full,
        "srs_full_pct": 100 * srs_full / n_srs,
        "srs_low": srs_low,
        "srs_low_pct": 100 * srs_low / n_srs,
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
            f"{s['label']:<12} {s['n_ics']:>5} {s['mean_ics']:>10.3f} "
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
            ics = np.array(s["ics_scatter"])
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


# ---------------------------------------------------------------------------
# Summary CSV helpers
# ---------------------------------------------------------------------------

MODEL_SHORT = {
    "claude-4.5-sonnet": "Claude",
    "gemini-3-flash":    "Gemini",
    "gemma-3-27b":       "Gemma",
    "glm-4.7":           "GLM",
    "grok-4.1-fast":     "Grok",
    "kimi-k2.5":         "Kimi",
    "ministral-8b":      "Mistral",
    "phi-4":             "Phi-4",
    "qwen3-235b":        "Qwen3",
}

# Prompt-strategy colours
PT_COLORS = {
    "zero-shot": "#4477AA",
    "few-shot":  "#EE6677",
    "cot":       "#228833",
}


def load_summary() -> list[dict]:
    with open(SUMMARY_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def fig_validation_rate(summary: list[dict]) -> None:
    """
    Hard-gate TerraMetrics parse/validation success rate per model,
    grouped by base dataset (iac_eval vs llm_iac), for all three prompt
    strategies side-by-side within each model group.
    """
    base_datasets = ["iac_eval", "llm_iac"]
    prompt_types  = ["zero-shot", "few-shot", "cot"]
    models = list(MODEL_SHORT.keys())
    model_labels = [MODEL_SHORT[m] for m in models]

    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)

        for ax, ds in zip(axes, base_datasets):
            x = np.arange(len(models))
            width = 0.26
            offsets = np.array([-1, 0, 1]) * width

            for offset, pt in zip(offsets, prompt_types):
                rates = []
                for m in models:
                    row = next(
                        (r for r in summary
                         if r["dataset"] == ds and r["model"] == m and r["prompt_type"] == pt),
                        None,
                    )
                    rates.append(float(row["gen_success_rate"]) * 100 if row else 0.0)

                bars = ax.bar(x + offset, rates, width,
                              color=PT_COLORS[pt], label=pt,
                              edgecolor="white", linewidth=0.3)

            ax.set_title("IaC-Eval" if ds == "iac_eval" else "llm-iac")
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=8)
            ax.set_ylim(96, 101)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v:.0f}%")
            )
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if ax is axes[0]:
                ax.set_ylabel("Pass rate (%)")

        handles = [
            mpatches.Patch(color=PT_COLORS[pt], label=pt) for pt in prompt_types
        ]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        fig.suptitle("Hard-gate Validation Pass Rate by Model and Prompting Strategy",
                     y=1.08, fontsize=11)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_validation_rate.pdf")


def fig_structural_quality(summary: list[dict]) -> None:
    """
    Gen vs. Ref comparison of four key TerraMetrics structural metrics,
    aggregated across all models and prompt types per base dataset.
    Shows whether LLMs over/under-generate compared to reference configs.
    """
    metrics = [
        ("mean_gen_num_resources",      "mean_ref_num_resources",      "Num Resources"),
        ("mean_gen_num_lines_of_code",  "mean_ref_num_lines_of_code",  "Lines of Code"),
        ("mean_gen_avgMccabeCC",        "mean_ref_avgMccabeCC",        "Avg McCabe CC"),
        ("mean_gen_maxDepthNestedBlocks","mean_ref_maxDepthNestedBlocks","Max Nesting Depth"),
    ]
    base_datasets = ["iac_eval", "llm_iac"]
    ds_labels = {"iac_eval": "IaC-Eval", "llm_iac": "llm-iac"}

    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, len(metrics), figsize=(8.5, 3.0))

        for ax, (gen_col, ref_col, label) in zip(axes, metrics):
            gen_vals, ref_vals = [], []
            for ds in base_datasets:
                rows = [r for r in summary
                        if isinstance(r["dataset"], str) and r["dataset"] == ds]
                gen_vals.append(np.mean([float(r[gen_col]) for r in rows if r[gen_col] != ""]))
                ref_vals.append(float(rows[0][ref_col]) if rows else 0.0)

            x = np.arange(len(base_datasets))
            width = 0.35
            ax.bar(x - width / 2, gen_vals, width, label="Generated",
                   color="#0072B2", alpha=0.85, edgecolor="white")
            ax.bar(x + width / 2, ref_vals, width, label="Reference",
                   color="#009E73", alpha=0.85, edgecolor="white")

            ax.set_title(label, fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels([ds_labels[d] for d in base_datasets], fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)

        handles = [
            mpatches.Patch(color="#0072B2", label="Generated"),
            mpatches.Patch(color="#009E73", label="Reference"),
        ]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        fig.suptitle("Structural Quality: Generated vs. Reference (TerraMetrics)",
                     y=1.08, fontsize=11)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_structural_quality.pdf")


def fig_risk_indicators(summary: list[dict]) -> None:
    """
    Heatmap of per-model security-smell / risk indicators on iac_eval (zero-shot).
    Metrics: wildcard-suffix strings, star strings, deprecated functions.
    Lower values are better; reference baseline annotated per column.
    """
    risk_cols = [
        ("mean_gen_numWildCardSuffixString_sum", "mean_ref_numWildCardSuffixString_sum",
         "Wildcard\nSuffix Strings"),
        ("mean_gen_numStarString_sum",           "mean_ref_numStarString_sum",
         "Star Strings\n(* perms)"),
        ("mean_gen_numDeprecatedFunctions_sum",  "mean_ref_numDeprecatedFunctions_sum",
         "Deprecated\nFunctions"),
        ("mean_gen_numEmptyString_sum",          "mean_ref_numEmptyString_sum",
         "Empty\nStrings"),
    ]
    models = list(MODEL_SHORT.keys())
    model_labels = [MODEL_SHORT[m] for m in models]

    prompt_type = "zero-shot"
    dataset = "iac_eval"

    rows_zs = [r for r in summary
               if isinstance(r["dataset"], str)
               and r["dataset"] == dataset
               and r["prompt_type"] == prompt_type]

    # Build matrix: rows = models, cols = risk metrics
    matrix = np.zeros((len(models), len(risk_cols)))
    ref_vals = np.zeros(len(risk_cols))
    col_labels = []

    for j, (gen_col, ref_col, lbl) in enumerate(risk_cols):
        col_labels.append(lbl)
        ref_row = next((r for r in rows_zs), None)
        ref_vals[j] = float(ref_row[ref_col]) if ref_row else 0.0
        for i, m in enumerate(models):
            row = next((r for r in rows_zs if r["model"] == m), None)
            matrix[i, j] = float(row[gen_col]) if row else 0.0

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.2))

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label="Mean count per config", pad=0.02)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=9)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(risk_cols)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if matrix[i, j] > matrix[:, j].max() * 0.6 else "black")

        # Reference baseline as text below x-axis
        for j, rv in enumerate(ref_vals):
            ax.text(j, len(models) - 0.3, f"ref={rv:.2f}",
                    ha="center", va="bottom", fontsize=6.5,
                    color="gray", style="italic")

        ax.set_title(
            "Risk Indicators per Model — IaC-Eval, Zero-Shot (TerraMetrics)",
            fontsize=10,
        )
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_risk_indicators.pdf")


# ---------------------------------------------------------------------------
# Lint / Security figures  (from results/lint_security_summary.csv)
# ---------------------------------------------------------------------------

LINT_SUMMARY_PATH = Path("results/lint_security_summary.csv")


def load_lint_summary() -> list[dict]:
    with open(LINT_SUMMARY_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def fig_tflint_violations(lint_summary: list[dict]) -> None:
    """
    Grouped bar chart: mean TFLint violations per config per model,
    stacked by severity (ERROR + WARNING), grouped by dataset.
    """
    base_datasets = ["iac_eval", "llm_iac"]
    ds_labels = {"iac_eval": "IaC-Eval", "llm_iac": "llm-iac"}
    models = list(MODEL_SHORT.keys())
    model_labels = [MODEL_SHORT[m] for m in models]
    prompt_type = "zero-shot"

    SEV_COLORS = {"error": "#d62728", "warning": "#ff7f0e",
                  "notice": "#aec7e8", "info": "#c7c7c7"}
    show_sevs = ["error", "warning", "notice"]

    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)

        for ax, ds in zip(axes, base_datasets):
            x = np.arange(len(models))
            width = 0.55

            bottoms = np.zeros(len(models))
            for sev in show_sevs:
                vals = []
                for m in models:
                    row = next(
                        (r for r in lint_summary
                         if isinstance(r.get("dataset"), str)
                         and r["dataset"] == ds
                         and r.get("model") == m
                         and r.get("prompt_type") == prompt_type),
                        None,
                    )
                    vals.append(float(row[f"mean_tflint_{sev}"]) if row else 0.0)

                vals = np.array(vals)
                ax.bar(x, vals, width, bottom=bottoms,
                       color=SEV_COLORS[sev], label=sev.capitalize(),
                       edgecolor="white", linewidth=0.3)
                bottoms += vals

            ax.set_title(ds_labels[ds])
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if ax is axes[0]:
                ax.set_ylabel("Mean violations per config")

        handles = [mpatches.Patch(color=SEV_COLORS[s], label=s.capitalize())
                   for s in show_sevs]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
        fig.suptitle("TFLint Violations per Config by Model (Zero-Shot)",
                     y=1.08, fontsize=11)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_tflint_violations.pdf")


def fig_trivy_misconfigs(lint_summary: list[dict]) -> None:
    """
    Grouped bar chart: mean Trivy misconfigurations per config per model,
    stacked by severity (CRITICAL, HIGH, MEDIUM), for zero-shot only.
    """
    base_datasets = ["iac_eval", "llm_iac"]
    ds_labels = {"iac_eval": "IaC-Eval", "llm_iac": "llm-iac"}
    models = list(MODEL_SHORT.keys())
    model_labels = [MODEL_SHORT[m] for m in models]
    prompt_type = "zero-shot"

    SEV_COLORS = {
        "critical": "#7b0000", "high": "#d62728",
        "medium": "#ff7f0e",   "low": "#ffbb78",
    }
    show_sevs = ["critical", "high", "medium", "low"]

    with plt.rc_context(FIG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), sharey=True)

        for ax, ds in zip(axes, base_datasets):
            x = np.arange(len(models))
            width = 0.55
            bottoms = np.zeros(len(models))

            for sev in show_sevs:
                vals = []
                for m in models:
                    row = next(
                        (r for r in lint_summary
                         if isinstance(r.get("dataset"), str)
                         and r["dataset"] == ds
                         and r.get("model") == m
                         and r.get("prompt_type") == prompt_type),
                        None,
                    )
                    vals.append(float(row[f"mean_trivy_{sev}"]) if row else 0.0)

                vals = np.array(vals)
                ax.bar(x, vals, width, bottom=bottoms,
                       color=SEV_COLORS[sev], label=sev.capitalize(),
                       edgecolor="white", linewidth=0.3)
                bottoms += vals

            ax.set_title(ds_labels[ds])
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)
            if ax is axes[0]:
                ax.set_ylabel("Mean misconfigs per config")

        handles = [mpatches.Patch(color=SEV_COLORS[s], label=s.capitalize())
                   for s in show_sevs]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False)
        fig.suptitle("Trivy Security Misconfigurations per Config by Model (Zero-Shot)",
                     y=1.08, fontsize=11)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_trivy_misconfigs.pdf")


def fig_lint_vs_prompting(lint_summary: list[dict]) -> None:
    """
    Line chart: mean total TFLint violations across prompting strategies
    (zero-shot, few-shot, cot) for each model, aggregated over both datasets.
    Shows whether prompting strategy influences linting quality.
    """
    models = list(MODEL_SHORT.keys())
    model_labels = [MODEL_SHORT[m] for m in models]
    prompt_types = ["zero-shot", "few-shot", "cot"]
    pt_labels = {"zero-shot": "Zero-Shot", "few-shot": "Few-Shot", "cot": "CoT"}
    base_datasets = ["iac_eval", "llm_iac"]

    with plt.rc_context(FIG_STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        x = np.arange(len(models))
        width = 0.26
        offsets = np.array([-1, 0, 1]) * width

        for offset, pt in zip(offsets, prompt_types):
            vals = []
            for m in models:
                # Average total violations over both base datasets
                total, count = 0.0, 0
                for ds in base_datasets:
                    row = next(
                        (r for r in lint_summary
                         if isinstance(r.get("dataset"), str)
                         and r["dataset"] == ds
                         and r.get("model") == m
                         and r.get("prompt_type") == pt),
                        None,
                    )
                    if row:
                        total += float(row.get("mean_tflint_total", 0))
                        count += 1
                vals.append(total / count if count else 0.0)

            ax.bar(x + offset, vals, width,
                   color=PT_COLORS[pt], label=pt_labels[pt],
                   edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("Mean TFLint violations per config")
        ax.set_title("TFLint Violations by Model and Prompting Strategy")
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        _save(fig, FIGURES_DIR / "fig_lint_vs_prompting.pdf")


def generate_figures(stats: list[dict]) -> None:
    print("\n=== Generating figures ===")
    fig_ics_distribution(stats)
    fig_srs_distribution(stats)
    fig_ics_vs_srs(stats)

    summary = load_summary()
    fig_validation_rate(summary)
    fig_structural_quality(summary)
    fig_risk_indicators(summary)

    if LINT_SUMMARY_PATH.exists():
        lint_summary = load_lint_summary()
        fig_tflint_violations(lint_summary)
        fig_trivy_misconfigs(lint_summary)
        fig_lint_vs_prompting(lint_summary)
    else:
        print(f"  Skipping lint/security figures — {LINT_SUMMARY_PATH} not found yet.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_revised_ics() -> dict[str, dict]:
    """
    Load ICS from ics_per_prompt.csv (revised attribute-level extraction).
    Returns per-dataset dicts with:
      "flat"        — every (scenario, model, strategy) ICS observation
      "per_scenario"— mean ICS per unique scenario_id
    """
    if not ICS_PER_PROMPT_PATH.exists():
        return {}
    from collections import defaultdict
    scenario_vals: dict[str, dict[str, list[float]]] = {}
    flat: dict[str, list[float]] = {}
    with open(ICS_PER_PROMPT_PATH, newline="") as f:
        for row in csv.DictReader(f):
            ds = row["dataset"]
            sid = row["scenario_id"]
            v = float(row["ics"])
            flat.setdefault(ds, []).append(v)
            scenario_vals.setdefault(ds, {}).setdefault(sid, []).append(v)
    result = {}
    for ds in flat:
        per_scenario = [sum(vs) / len(vs) for vs in scenario_vals[ds].values()]
        result[ds] = {"flat": flat[ds], "per_scenario": per_scenario}
    return result


def main() -> None:
    with open(RESULTS_PATH) as f:
        data = json.load(f)

    revised_ics = load_revised_ics()

    datasets = data["datasets"]
    stats = []
    for name, content in datasets.items():
        if "entries" not in content:
            continue
        rev = revised_ics.get(name, {})
        stats.append(analyze_dataset(
            name,
            content["entries"],
            ics_vals_override=rev.get("flat"),
            ics_scatter_override=rev.get("per_scenario"),
        ))

    print_table1(stats)
    print_overall(stats)

    # ICS distribution detail (used in Results §5.1)
    print("\n=== ICS distribution detail (Results §5.1) ===")
    for s in stats:
        n = s["n_ics"]
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
