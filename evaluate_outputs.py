#!/usr/bin/env python3
"""
Evaluate Terraform Outputs with TerraMetrics

Reads all CSVs from outputs/, runs the TerraMetrics JAR on each row's
extracted_code and reference, extracts selected quality metrics, and writes
per-file result CSVs + an aggregated summary to results/.

Supports resume: skips rows whose scenario_id already exists in the output CSV.

Usage:
  python evaluate_outputs.py                           # Run all CSVs
  python evaluate_outputs.py --csv outputs/some.csv    # Run a specific CSV
  python evaluate_outputs.py --samples 5               # Only first N rows per CSV
  python evaluate_outputs.py --workers 4               # Parallelism (default: 4)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
JAR_PATH = SCRIPT_DIR / "terraform_metrics-1.0.jar"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "results"

# Selected TerraMetrics from Metrics.txt
# Keys map: friendly_name → (source: "head"|"data", json_key, aggregation: "direct"|"sum"|"avg"|"max"|"min")
HEAD_METRICS = {
    "num_lines_of_code": "num_lines_of_code",
    "num_providers": "num_providers",
    "num_resources": "num_resources",
    "num_outputs": "num_outputs",
    "num_variables": "num_variables",
    "num_modules": "num_modules",
    "num_data": "num_data",
    "num_blocks": "num_blocks",
    "num_locals": "num_locals",
    "num_terraform": "num_terraform",
}

# data-level metrics: (json_key, aggregation_method)
# aggregation_method: "sum", "avg", "max", "min"
DATA_METRICS = {
    # Structural & Maintainability
    "numImplicitDependentEach_avg": ("numImplicitDependentEach", "avg"),
    "numExplicitResourceDependency_sum": ("numExplicitResourceDependency", "sum"),
    "numImplicitDependentResources_sum": ("numImplicitDependentResources", "sum"),
    # Complexity & Readability
    "avgMccabeCC": ("avgMccabeCC", "avg"),
    "maxMccabeCC": ("maxMccabeCC", "max"),
    "sumMccabeCC": ("sumMccabeCC", "sum"),
    "numComparisonOperators_sum": ("numComparisonOperators", "sum"),
    "numConditions_sum": ("numConditions", "sum"),
    "numLoops_sum": ("numLoops", "sum"),
    "maxDepthNestedBlocks": ("maxDepthNestedBlocks", "max"),
    "numDynamicBlocks_sum": ("numDynamicBlocks", "sum"),
    "numNestedBlocks_sum": ("numNestedBlocks", "sum"),
    "loc_sum": ("loc", "sum"),
    # Documentation & Professionalism
    "containDescriptionField_sum": ("containDescriptionField", "sum"),
    "numStringValues_sum": ("numStringValues", "sum"),
    "minAttrsTextEntropy": ("minAttrsTextEntropy", "min"),
    "maxAttrsTextEntropy": ("maxAttrsTextEntropy", "max"),
    "avgAttrsTextEntropy": ("avgAttrsTextEntropy", "avg"),
    "textEntropyMeasure_avg": ("textEntropyMeasure", "avg"),
    # Risk & Anti-Pattern
    "numDeprecatedFunctions_sum": ("numDeprecatedFunctions", "sum"),
    "numDebuggingFunctions_sum": ("numDebuggingFunctions", "sum"),
    "numWildCardSuffixString_sum": ("numWildCardSuffixString", "sum"),
    "numEmptyString_sum": ("numEmptyString", "sum"),
    "numStarString_sum": ("numStarString", "sum"),
    # Additional useful metrics
    "numTokens_sum": ("numTokens", "sum"),
    "numAttrs_sum": ("numAttrs", "sum"),
    "numVars_sum": ("numVars", "sum"),
    "numReferences_sum": ("numReferences", "sum"),
    "numFunctionCall_sum": ("numFunctionCall", "sum"),
    "numLiteralExpression_sum": ("numLiteralExpression", "sum"),
    "numMetaArg_sum": ("numMetaArg", "sum"),
    "numTemplateExpression_sum": ("numTemplateExpression", "sum"),
    "numHereDocs_sum": ("numHereDocs", "sum"),
    "numObjects_sum": ("numObjects", "sum"),
    "numTuples_sum": ("numTuples", "sum"),
    "numLogiOpers_sum": ("numLogiOpers", "sum"),
    "numMathOperations_sum": ("numMathOperations", "sum"),
    "numIndexAccess_sum": ("numIndexAccess", "sum"),
    "numSplatExpressions_sum": ("numSplatExpressions", "sum"),
    "numLookUpFunctionCall_sum": ("numLookUpFunctionCall", "sum"),
    "numParams_sum": ("numParams", "sum"),
}


# ─────────────────────────────────────────────────────────────
# CSV helpers (matching generate_baselines.py escaping)
# ─────────────────────────────────────────────────────────────

def unescape_newlines(value: str) -> str:
    """Unescape \\n back to real newlines (matching generate_baselines.py)."""
    return value.replace("\\n", "\n")


def escape_newlines(value: Any) -> Any:
    """Escape newlines for CSV storage."""
    if isinstance(value, str):
        return value.replace("\n", "\\n").replace("\r", "")
    return value


# ─────────────────────────────────────────────────────────────
# TerraMetrics runner
# ─────────────────────────────────────────────────────────────

def run_terrametrics_on_code(code: str, jar_path: Path) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Write code to a temp .tf file, run the JAR, parse the JSON result.
    Returns (success, parsed_json_or_None, error_message).
    """
    if not code or not code.strip():
        return False, None, "empty code"

    # Create a temp dir so the JAR can write its output
    with tempfile.TemporaryDirectory(prefix="terametrics_") as tmpdir:
        tf_path = Path(tmpdir) / "main.tf"
        json_path = Path(tmpdir) / "metrics.json"

        tf_path.write_text(code, encoding="utf-8")

        cmd = [
            "java", "-jar", str(jar_path),
            "--file", str(tf_path),
            "-b",
            "--target", str(json_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmpdir,
            )
        except subprocess.TimeoutExpired:
            return False, None, "JAR timed out after 60s"
        except Exception as e:
            return False, None, f"subprocess error: {e}"

        # The JAR writes results to --target file
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                return True, data, ""
            except json.JSONDecodeError as e:
                return False, None, f"JSON parse error: {e}"

        # Fallback: try parsing stdout
        if result.stdout.strip():
            try:
                data = json.loads(result.stdout.strip())
                return True, data, ""
            except json.JSONDecodeError:
                pass

        stderr_snippet = (result.stderr or "")[:200]
        return False, None, f"JAR exit={result.returncode}; stderr={stderr_snippet}"


# ─────────────────────────────────────────────────────────────
# Metric extraction
# ─────────────────────────────────────────────────────────────

def _agg(data_blocks: List[Dict], key: str, method: str) -> Optional[float]:
    """Aggregate a field across data blocks."""
    vals = []
    for block in data_blocks:
        v = block.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    if method == "sum":
        return sum(vals)
    elif method == "avg":
        return sum(vals) / len(vals)
    elif method == "max":
        return max(vals)
    elif method == "min":
        return min(vals)
    return None


def extract_metrics(tm_json: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the selected metrics from TerraMetrics JSON output."""
    head = tm_json.get("head", {}) or {}
    data = tm_json.get("data", []) or []

    metrics: Dict[str, Any] = {}

    # Head-level metrics
    for name, key in HEAD_METRICS.items():
        metrics[name] = head.get(key)

    # Data-level aggregated metrics
    for name, (key, method) in DATA_METRICS.items():
        metrics[name] = _agg(data, key, method)

    return metrics


def get_all_metric_columns() -> List[str]:
    """Return all metric column names in deterministic order."""
    return list(HEAD_METRICS.keys()) + list(DATA_METRICS.keys())


# ─────────────────────────────────────────────────────────────
# Result CSV I/O
# ─────────────────────────────────────────────────────────────

def get_result_columns() -> List[str]:
    """Return the full ordered column list for the result CSV."""
    meta_cols = ["scenario_id", "model", "prompt_type", "dataset"]
    metric_names = get_all_metric_columns()
    gen_cols = [f"gen_{m}" for m in metric_names]
    ref_cols = [f"ref_{m}" for m in metric_names]
    status_cols = ["gen_terametrics_ok", "gen_terametrics_error",
                   "ref_terametrics_ok", "ref_terametrics_error"]
    return meta_cols + status_cols + gen_cols + ref_cols


def load_existing_results(result_path: Path) -> Dict[str, Dict]:
    """Load already-computed results, keyed by scenario_id."""
    existing = {}
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("scenario_id", "")
                if sid:
                    existing[sid] = row
    return existing


def save_results(result_path: Path, columns: List[str], rows: List[Dict]):
    """Save results to CSV."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────
# Process a single row
# ─────────────────────────────────────────────────────────────

def process_row(row: Dict[str, str], jar_path: Path) -> Dict[str, Any]:
    """Run TerraMetrics on both extracted_code and reference for one CSV row."""
    result: Dict[str, Any] = {
        "scenario_id": row.get("scenario_id", ""),
        "model": row.get("model", ""),
        "prompt_type": row.get("prompt_type", ""),
        "dataset": row.get("dataset", ""),
    }

    metric_names = get_all_metric_columns()

    # --- Generated code ---
    gen_code = unescape_newlines(row.get("extracted_code", ""))
    gen_ok, gen_json, gen_err = run_terrametrics_on_code(gen_code, jar_path)
    result["gen_terametrics_ok"] = gen_ok
    result["gen_terametrics_error"] = gen_err

    if gen_ok and gen_json:
        gen_metrics = extract_metrics(gen_json)
        for m in metric_names:
            result[f"gen_{m}"] = gen_metrics.get(m)
    else:
        for m in metric_names:
            result[f"gen_{m}"] = None

    # --- Reference code ---
    ref_code = unescape_newlines(row.get("reference", ""))
    ref_ok, ref_json, ref_err = run_terrametrics_on_code(ref_code, jar_path)
    result["ref_terametrics_ok"] = ref_ok
    result["ref_terametrics_error"] = ref_err

    if ref_ok and ref_json:
        ref_metrics = extract_metrics(ref_json)
        for m in metric_names:
            result[f"ref_{m}"] = ref_metrics.get(m)
    else:
        for m in metric_names:
            result[f"ref_{m}"] = None

    return result


# ─────────────────────────────────────────────────────────────
# Process a single CSV file
# ─────────────────────────────────────────────────────────────

def process_csv(
    csv_path: Path,
    jar_path: Path,
    results_dir: Path,
    samples: Optional[int] = None,
    workers: int = 4,
) -> Path:
    """Process one CSV: run TerraMetrics on each row, write results."""
    result_filename = csv_path.stem + "_metrics.csv"
    result_path = results_dir / result_filename
    columns = get_result_columns()

    # Load existing results for resume
    existing = load_existing_results(result_path)
    print(f"\n{'='*70}")
    print(f"  CSV: {csv_path.name}")
    print(f"  Already completed: {len(existing)} rows")
    print(f"{'='*70}")

    # Read input CSV
    all_rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_rows.append(row)

    if samples:
        all_rows = all_rows[:samples]

    # Filter to only rows that need processing
    todo_rows = [r for r in all_rows if r.get("scenario_id", "") not in existing]
    print(f"  Total rows: {len(all_rows)}, to process: {len(todo_rows)}")

    if not todo_rows:
        print("  Nothing to do — all rows already processed.")
        return result_path

    # Collect all results (existing + new)
    all_results = list(existing.values())

    # Process with thread pool
    start_time = time.time()
    done_count = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_row = {
            executor.submit(process_row, row, jar_path): row
            for row in todo_rows
        }

        for future in as_completed(future_to_row):
            row = future_to_row[future]
            sid = row.get("scenario_id", "?")
            try:
                result = future.result()
                all_results.append(result)
                done_count += 1

                # Progress
                elapsed = time.time() - start_time
                rate = done_count / elapsed if elapsed > 0 else 0
                remaining = len(todo_rows) - done_count
                eta = remaining / rate if rate > 0 else 0
                eta_str = f"{int(eta//60)}m{int(eta%60)}s" if eta > 0 else "--"

                gen_ok = "✓" if result.get("gen_terametrics_ok") else "✗"
                ref_ok = "✓" if result.get("ref_terametrics_ok") else "✗"
                print(f"  [{done_count}/{len(todo_rows)}] {sid}  gen={gen_ok} ref={ref_ok}  ETA: {eta_str}")

                # Save periodically (every 10 rows) for resume safety
                if done_count % 10 == 0:
                    save_results(result_path, columns, all_results)

            except Exception as e:
                print(f"  [{done_count}/{len(todo_rows)}] {sid}  ERROR: {e}")

    # Final save
    save_results(result_path, columns, all_results)
    total_time = time.time() - start_time
    print(f"  Done! {done_count} rows processed in {total_time:.1f}s → {result_path.name}")
    return result_path


# ─────────────────────────────────────────────────────────────
# Summary aggregation
# ─────────────────────────────────────────────────────────────

def build_summary(results_dir: Path) -> Path:
    """Aggregate all per-file result CSVs into a single summary CSV with means."""
    summary_path = results_dir / "summary.csv"
    metric_names = get_all_metric_columns()

    # Group rows by (dataset, model, prompt_type)
    groups: Dict[Tuple[str, str, str], List[Dict]] = {}

    for csv_file in sorted(results_dir.glob("*_metrics.csv")):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get("dataset", ""), row.get("model", ""), row.get("prompt_type", ""))
                groups.setdefault(key, []).append(row)

    # Compute means
    summary_rows = []
    for (dataset, model, prompt_type), rows in sorted(groups.items()):
        summary: Dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "prompt_type": prompt_type,
            "num_samples": len(rows),
        }

        # Count successes
        gen_ok_count = sum(1 for r in rows if r.get("gen_terametrics_ok") == "True")
        ref_ok_count = sum(1 for r in rows if r.get("ref_terametrics_ok") == "True")
        summary["gen_success_rate"] = round(gen_ok_count / len(rows), 4) if rows else 0
        summary["ref_success_rate"] = round(ref_ok_count / len(rows), 4) if rows else 0

        # Mean of each metric
        for prefix in ("gen_", "ref_"):
            for m in metric_names:
                col = f"{prefix}{m}"
                vals = []
                for r in rows:
                    v = r.get(col, "")
                    if v and v not in ("None", ""):
                        try:
                            vals.append(float(v))
                        except (ValueError, TypeError):
                            pass
                summary[f"mean_{col}"] = round(sum(vals) / len(vals), 6) if vals else None

        summary_rows.append(summary)

    # Write summary
    if summary_rows:
        all_keys = list(summary_rows[0].keys())
        # Ensure consistent columns across all rows
        for sr in summary_rows:
            for k in sr:
                if k not in all_keys:
                    all_keys.append(k)

        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for sr in summary_rows:
                writer.writerow(sr)

    print(f"\nSummary written to: {summary_path}")
    return summary_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Terraform outputs with TerraMetrics"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to a specific CSV file to process (default: all CSVs in outputs/)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Only process first N rows per CSV (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--jar",
        type=str,
        default=None,
        help=f"Path to TerraMetrics JAR (default: {JAR_PATH})",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only regenerate the summary from existing result CSVs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    jar_path = Path(args.jar) if args.jar else JAR_PATH
    if not jar_path.exists():
        print(f"ERROR: JAR not found at {jar_path}")
        sys.exit(1)

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        build_summary(results_dir)
        return

    # Determine which CSVs to process
    if args.csv:
        csv_files = [Path(args.csv)]
    else:
        csv_files = sorted(OUTPUTS_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {OUTPUTS_DIR}")
        sys.exit(1)

    print(f"TerraMetrics Evaluation")
    print(f"  JAR: {jar_path}")
    print(f"  CSVs: {len(csv_files)} files")
    print(f"  Workers: {args.workers}")
    if args.samples:
        print(f"  Samples per CSV: {args.samples}")

    # Process each CSV
    overall_start = time.time()
    for csv_path in csv_files:
        process_csv(
            csv_path=csv_path,
            jar_path=jar_path,
            results_dir=results_dir,
            samples=args.samples,
            workers=args.workers,
        )

    # Build summary
    build_summary(results_dir)

    total = time.time() - overall_start
    mins = int(total // 60)
    secs = int(total % 60)
    print(f"\nAll done! Total time: {mins}m {secs}s")


if __name__ == "__main__":
    main()
