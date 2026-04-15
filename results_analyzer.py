"""
Results analyzer for IaC-Bench paper (IaC-RL project).

Reproduces the statistics reported in the Results section of the paper:
  - Per-dataset mean/median ICS and SRS (Table 1)
  - ICS distribution breakdown (ICS=0, ICS=1, intermediate)
  - SRS distribution breakdown (SRS=1, SRS<0.7)
  - Overall cross-dataset aggregates
  - Low-SRS prompt examples cited in the paper

Input:  results/final_results.json
Output: printed report (copy numbers directly into the paper)
"""

import json
import statistics
from pathlib import Path

RESULTS_PATH = Path("results/final_results.json")


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
        "n": n,
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


def print_table1(stats: list[dict]) -> None:
    """Reproduce Table 1 from the paper."""
    print("\n=== Table 1: ICS and SRS distribution by dataset ===")
    header = f"{'Dataset':<12} {'n':>5} {'Mean ICS':>10} {'Median ICS':>11} {'ICS=1 (%)':>10} {'ICS=0 (%)':>10} {'Mean SRS':>10} {'SRS=1 (%)':>10}"
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['name']:<12} {s['n']:>5} {s['mean_ics']:>10.3f} "
            f"{s['median_ics']:>11.3f} {s['ics_full_pct']:>9.1f}% "
            f"{s['ics_zero_pct']:>9.1f}% {s['mean_srs']:>10.3f} "
            f"{s['srs_full_pct']:>9.1f}%"
        )


def print_overall(stats: list[dict]) -> None:
    """Overall aggregates cited in the paper."""
    all_n = sum(s["n"] for s in stats)
    all_ics = sum(s["mean_ics"] * s["n"] for s in stats) / all_n
    all_srs = sum(s["mean_srs"] * s["n"] for s in stats) / all_n
    print(f"\n=== Overall (n={all_n}) ===")
    print(f"  Mean ICS : {all_ics:.3f}")
    print(f"  Mean SRS : {all_srs:.3f}")


def print_low_srs(stats: list[dict]) -> None:
    """Low-SRS prompts mentioned in the paper's Results section."""
    print("\n=== Low-SRS prompts (SRS < 0.7) ===")
    for s in stats:
        print(f"\n{s['name']} — {s['srs_low']} entries ({s['srs_low_pct']:.1f}%)")
        for prompt, srs in sorted(s["low_srs_entries"], key=lambda x: x[1]):
            print(f"  SRS={srs:.3f}: {prompt}")


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
        print(f"\n{s['name']} (n={n}):")
        print(f"  ICS = 1.0 : {s['ics_full']:>4} / {n}  ({s['ics_full_pct']:.1f}%)")
        print(f"  ICS = 0.0 : {s['ics_zero']:>4} / {n}  ({s['ics_zero_pct']:.1f}%)")
        print(f"  ICS = 0.5 : {s['ics_mid']:>4} / {n}  ({s['ics_mid_pct']:.1f}%)")

    # SRS detail (used in Results §5.2)
    print("\n=== SRS distribution detail (Results §5.2) ===")
    for s in stats:
        n = s["n"]
        print(f"\n{s['name']} (n={n}):")
        print(f"  SRS = 1.0 : {s['srs_full']:>4} / {n}  ({s['srs_full_pct']:.1f}%)")
        print(f"  SRS < 0.7 : {s['srs_low']:>4} / {n}  ({s['srs_low_pct']:.1f}%)")

    print_low_srs(stats)


if __name__ == "__main__":
    main()
