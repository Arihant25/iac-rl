#!/usr/bin/env python3
"""
Analyze terraform evaluation results for paper integration.
Produces tables and figures covering fmt, init, and validate pass rates.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

MODEL_DISPLAY = {
    "claude-4.5-sonnet": "Claude Sonnet 4.5",
    "grok-4.1-fast":     "Grok 4.1 Fast",
    "gemini-3-flash":    "Gemini 3 Flash",
    "kimi-k2.5":         "Kimi K2.5",
    "glm-4.7":           "GLM-4.7",
    "qwen3-235b":        "Qwen3-235B",
    "ministral-8b":      "Ministral-8B",
    "phi-4":             "Phi-4",
    "gemma-3-27b":       "Gemma-3-27B",
}
MODEL_ORDER = list(MODEL_DISPLAY.keys())

STRATEGY_DISPLAY = {"zero-shot": "Zero", "few-shot": "Few", "cot": "CoT"}


def load_summary() -> list[dict]:
    path = RESULTS_DIR / "terraform_summary.csv"
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_by_model_dataset(rows: list[dict]) -> dict:
    """Average over prompt strategies for each (model, dataset) pair."""
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["model"], r["dataset"])
        buckets[key].append(r)

    out = {}
    for (model, dataset), group in buckets.items():
        n = len(group)
        out[(model, dataset)] = {
            "fmt":      round(sum(float(x["fmt_pass_rate"])      for x in group) / n, 4),
            "init":     round(sum(float(x["init_pass_rate"])     for x in group) / n, 4),
            "validate": round(sum(float(x["validate_pass_rate"]) for x in group) / n, 4),
            "mean_err": round(sum(float(x["mean_validate_errors"]) for x in group) / n, 4),
        }
    return out


def aggregate_by_model_strategy_dataset(rows: list[dict]) -> dict:
    """Direct lookup keyed by (model, strategy, dataset)."""
    out = {}
    for r in rows:
        key = (r["model"], r["prompt_type"], r["dataset"])
        out[key] = {
            "fmt":      float(r["fmt_pass_rate"]),
            "init":     float(r["init_pass_rate"]),
            "validate": float(r["validate_pass_rate"]),
            "mean_err": float(r["mean_validate_errors"]),
        }
    return out


# ---------------------------------------------------------------------------
# Table 1: per-model, averaged over prompt strategies, for iac_eval & llm_iac
# ---------------------------------------------------------------------------

def print_terraform_table(agg: dict) -> None:
    print("\n=== Terraform CLI Pass Rates (avg over prompt strategies) ===")
    header = f"{'Model':<22}  {'IaC-Eval':>8}  {'':>8}  {'':>8}  {'llm-iac':>8}  {'':>8}  {'':>8}"
    sub    = f"{'':22}  {'fmt':>8}  {'init':>8}  {'val':>8}  {'fmt':>8}  {'init':>8}  {'val':>8}"
    print(header)
    print(sub)
    print("-" * 80)
    for model in MODEL_ORDER:
        ie  = agg.get((model, "iac_eval"),  {})
        li  = agg.get((model, "llm_iac"),   {})
        name = MODEL_DISPLAY.get(model, model)
        print(
            f"{name:<22}  "
            f"{ie.get('fmt', float('nan')):>8.1%}  "
            f"{ie.get('init', float('nan')):>8.1%}  "
            f"{ie.get('validate', float('nan')):>8.1%}  "
            f"{li.get('fmt', float('nan')):>8.1%}  "
            f"{li.get('init', float('nan')):>8.1%}  "
            f"{li.get('validate', float('nan')):>8.1%}"
        )


# ---------------------------------------------------------------------------
# Table 2: per-strategy effect on validate pass rate (all models, avg datasets)
# ---------------------------------------------------------------------------

def print_strategy_effect(rows: list[dict]) -> None:
    print("\n=== Validate Pass Rate by Prompting Strategy (avg over models & datasets) ===")
    strats = ["zero-shot", "few-shot", "cot"]
    datasets = ["iac_eval", "llm_iac"]
    for ds in datasets:
        print(f"\n  Dataset: {ds}")
        for strat in strats:
            vals = [float(r["validate_pass_rate"])
                    for r in rows
                    if r["dataset"] == ds and r["prompt_type"] == strat]
            print(f"    {strat:<12}: {np.mean(vals):.3f}  (n={len(vals)} models)")


# ---------------------------------------------------------------------------
# Figure: grouped bar chart of validate pass rate per model, per dataset
# ---------------------------------------------------------------------------

def plot_validate_pass_rate(agg_by_msd: dict) -> Path:
    datasets = ["iac_eval", "llm_iac"]
    strategies = ["zero-shot", "few-shot", "cot"]
    colors = {"zero-shot": "#4C72B0", "few-shot": "#DD8452", "cot": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, dataset in zip(axes, datasets):
        ds_label = "IaC-Eval" if dataset == "iac_eval" else "llm-iac"
        x = np.arange(len(MODEL_ORDER))
        width = 0.25
        offsets = [-width, 0, width]

        for strat, offset in zip(strategies, offsets):
            vals = []
            for model in MODEL_ORDER:
                key = (model, strat, dataset)
                v = agg_by_msd.get(key, {}).get("validate", np.nan)
                vals.append(v * 100 if not np.isnan(v) else np.nan)
            ax.bar(x + offset, vals, width * 0.9,
                   label=STRATEGY_DISPLAY[strat], color=colors[strat], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m).replace(" ", "\n") for m in MODEL_ORDER],
            fontsize=7.5,
        )
        ax.set_ylabel("Validate pass rate (%)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(color=colors[s], label=STRATEGY_DISPLAY[s]) for s in strategies]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_terraform_validate.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: fmt pass rate comparison
# ---------------------------------------------------------------------------

def plot_fmt_pass_rate(agg_by_msd: dict) -> Path:
    datasets = ["iac_eval", "llm_iac"]
    strategies = ["zero-shot", "few-shot", "cot"]
    colors = {"zero-shot": "#4C72B0", "few-shot": "#DD8452", "cot": "#55A868"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, dataset in zip(axes, datasets):
        ds_label = "IaC-Eval" if dataset == "iac_eval" else "llm-iac"
        x = np.arange(len(MODEL_ORDER))
        width = 0.25
        offsets = [-width, 0, width]

        for strat, offset in zip(strategies, offsets):
            vals = []
            for model in MODEL_ORDER:
                key = (model, strat, dataset)
                v = agg_by_msd.get(key, {}).get("fmt", np.nan)
                vals.append(v * 100 if not np.isnan(v) else np.nan)
            ax.bar(x + offset, vals, width * 0.9,
                   label=STRATEGY_DISPLAY[strat], color=colors[strat], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY.get(m, m).replace(" ", "\n") for m in MODEL_ORDER],
            fontsize=7.5,
        )
        ax.set_ylabel("fmt pass rate (%)")
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f%%"))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(color=colors[s], label=STRATEGY_DISPLAY[s]) for s in strategies]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), fontsize=10)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_terraform_fmt.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: init pass rate heatmap
# ---------------------------------------------------------------------------

def plot_init_heatmap(agg: dict) -> Path:
    datasets = ["iac_eval", "llm_iac"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, dataset in zip(axes, datasets):
        ds_label = "IaC-Eval" if dataset == "iac_eval" else "llm-iac"
        data = np.array([
            agg.get((model, dataset), {}).get("init", np.nan) * 100
            for model in MODEL_ORDER
        ]).reshape(-1, 1)

        im = ax.imshow(data, vmin=60, vmax=100, cmap="YlGn", aspect="auto")
        ax.set_xticks([0])
        ax.set_xticklabels(["Init pass\nrate (%)"])
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER], fontsize=9)
        for i, model in enumerate(MODEL_ORDER):
            v = agg.get((model, dataset), {}).get("init", np.nan)
            if not np.isnan(v):
                ax.text(0, i, f"{v*100:.1f}%", ha="center", va="center",
                        fontsize=8, color="black")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_terraform_init.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


# ---------------------------------------------------------------------------
# LaTeX table helpers
# ---------------------------------------------------------------------------

def latex_terraform_table(agg: dict) -> str:
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Terraform CLI pass rates per model averaged over the three prompting strategies. "
                 r"\texttt{fmt} = formatting compliance (\texttt{terraform fmt -check}); "
                 r"\texttt{init} = provider initialisation (\texttt{terraform init -backend=false}); "
                 r"\texttt{val} = static validation (\texttt{terraform validate}). "
                 r"Values are fractions in $[0,1]$.}\label{tab:terraform}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{IaC-Eval} & \multicolumn{3}{c}{llm-iac} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"Model & fmt & init & val & fmt & init & val \\")
    lines.append(r"\midrule")
    for model in MODEL_ORDER:
        ie = agg.get((model, "iac_eval"), {})
        li = agg.get((model, "llm_iac"), {})
        name = MODEL_DISPLAY.get(model, model)

        def fmt_val(d: dict, k: str) -> str:
            v = d.get(k)
            return f"{v:.2f}" if v is not None else "---"

        lines.append(
            f"{name} & "
            f"{fmt_val(ie,'fmt')} & {fmt_val(ie,'init')} & {fmt_val(ie,'validate')} & "
            f"{fmt_val(li,'fmt')} & {fmt_val(li,'init')} & {fmt_val(li,'validate')} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_terraform_strategy_table(agg_by_msd: dict) -> str:
    """Validate pass rate broken down by strategy, avg over models."""
    datasets = ["iac_eval", "llm_iac"]
    strategies = ["zero-shot", "few-shot", "cot"]
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(
        r"\caption{\texttt{terraform validate} pass rate by prompting strategy, "
        r"averaged over all nine models. CoT generally reduces validation errors "
        r"compared to few-shot, mirroring the pattern observed for TFLint "
        r"violations in Table~\ref{tab:tflint}.}\label{tab:terraform_strategy}"
    )
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{IaC-Eval} & \multicolumn{3}{c}{llm-iac} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"Strategy & fmt & init & val & fmt & init & val \\")
    lines.append(r"\midrule")
    for strat in strategies:
        row_parts = []
        row_parts.append(strat)
        for ds in datasets:
            for metric in ["fmt", "init", "validate"]:
                vals = [
                    agg_by_msd[(m, strat, ds)][metric]
                    for m in MODEL_ORDER
                    if (m, strat, ds) in agg_by_msd
                ]
                mean_v = np.mean(vals) if vals else float("nan")
                row_parts.append(f"{mean_v:.2f}")
        lines.append(" & ".join(row_parts) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    rows = load_summary()

    # Split into original and paraphrased
    orig_rows = [r for r in rows if "paraphrased" not in r["dataset"]]
    para_rows = [r for r in rows if "paraphrased" in r["dataset"]]

    agg = aggregate_by_model_dataset(orig_rows)
    agg_by_msd = aggregate_by_model_strategy_dataset(orig_rows)

    print_terraform_table(agg)
    print_strategy_effect(orig_rows)

    # --- summary stats for paper text ---
    print("\n=== Key statistics for paper text ===")

    # Overall validate pass rate per dataset
    for ds in ["iac_eval", "llm_iac"]:
        vals = [v["validate"] for (m, d), v in agg.items() if d == ds]
        print(f"Mean validate pass rate {ds}: {np.mean(vals):.3f}  "
              f"range [{min(vals):.3f}, {max(vals):.3f}]")

    # fmt pass rate per dataset
    for ds in ["iac_eval", "llm_iac"]:
        vals = [v["fmt"] for (m, d), v in agg.items() if d == ds]
        print(f"Mean fmt pass rate {ds}: {np.mean(vals):.3f}  "
              f"range [{min(vals):.3f}, {max(vals):.3f}]")

    # init pass rate per dataset
    for ds in ["iac_eval", "llm_iac"]:
        vals = [v["init"] for (m, d), v in agg.items() if d == ds]
        print(f"Mean init pass rate {ds}: {np.mean(vals):.3f}  "
              f"range [{min(vals):.3f}, {max(vals):.3f}]")

    # best and worst validate per dataset
    for ds in ["iac_eval", "llm_iac"]:
        subset = {m: v["validate"] for (m, d), v in agg.items() if d == ds}
        best = max(subset, key=subset.__getitem__)
        worst = min(subset, key=subset.__getitem__)
        print(f"  {ds} best validate: {MODEL_DISPLAY[best]} {subset[best]:.3f}, "
              f"worst: {MODEL_DISPLAY[worst]} {subset[worst]:.3f}")

    # strategy effect on validate
    for ds in ["iac_eval", "llm_iac"]:
        for strat in ["zero-shot", "few-shot", "cot"]:
            vals = [float(r["validate_pass_rate"])
                    for r in orig_rows
                    if r["dataset"] == ds and r["prompt_type"] == strat]
            print(f"  validate {ds} {strat}: {np.mean(vals):.3f}")

    # mean validate errors
    for ds in ["iac_eval", "llm_iac"]:
        vals = [v["mean_err"] for (m, d), v in agg.items() if d == ds]
        print(f"Mean validate errors {ds}: {np.mean(vals):.3f}")

    # original vs paraphrased validate pass rate comparison
    agg_orig = aggregate_by_model_dataset(orig_rows)
    # remap paraphrased dataset names
    para_rows_remapped = []
    for r in para_rows:
        rr = dict(r)
        rr["dataset"] = rr["dataset"].replace("_paraphrased", "").replace("paraphrased_", "")
        para_rows_remapped.append(rr)
    agg_para = aggregate_by_model_dataset(para_rows_remapped)
    print("\n=== Validate pass rate: original vs paraphrased ===")
    for ds in ["iac_eval", "llm_iac"]:
        orig_vals = [agg_orig.get((m, ds), {}).get("validate", np.nan) for m in MODEL_ORDER]
        para_vals = [agg_para.get((m, ds), {}).get("validate", np.nan) for m in MODEL_ORDER]
        diffs = [p - o for o, p in zip(orig_vals, para_vals) if not (np.isnan(o) or np.isnan(p))]
        print(f"  {ds}: orig={np.nanmean(orig_vals):.3f}, para={np.nanmean(para_vals):.3f}, "
              f"mean diff={np.mean(diffs):+.3f}")

    # latex tables
    print("\n\n=== LaTeX: tab:terraform ===")
    print(latex_terraform_table(agg))

    print("\n\n=== LaTeX: tab:terraform_strategy ===")
    print(latex_terraform_strategy_table(agg_by_msd))

    # figures
    plot_validate_pass_rate(agg_by_msd)
    plot_fmt_pass_rate(agg_by_msd)
    plot_init_heatmap(agg)


if __name__ == "__main__":
    main()
