#!/usr/bin/env python3
"""
Terraform CLI Evaluation for IaC-Bench

For every generated Terraform configuration in outputs/*.csv, runs:
  - terraform fmt -check      : formatting compliance
  - terraform init -backend=false : populate the provider cache
                                   (offline; no backend, no modules)
  - terraform validate -json  : static validation of the configuration

`terraform plan` is intentionally NOT run: it requires real cloud credentials
and network access to the corresponding providers, which cannot be produced
in a reproducible, offline-verifiable manner for a research benchmark.

Writes per-file result CSVs to results/ (e.g. iac_eval_claude_zero-shot_tf.csv)
and an aggregated summary to results/terraform_summary.csv.

Supports resume: already-processed scenario_ids are skipped.

Usage:
  uv run evaluate_terraform.py                     # all CSVs
  uv run evaluate_terraform.py --csv outputs/x.csv # single CSV
  uv run evaluate_terraform.py --samples 5         # first N rows per CSV
  uv run evaluate_terraform.py --workers 4         # parallelism
  uv run evaluate_terraform.py --summary-only      # rebuild summary only
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

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "results"

# Shared plugin cache so `terraform init` only downloads each provider once.
PLUGIN_CACHE_DIR = SCRIPT_DIR / ".terraform-plugin-cache"
PLUGIN_CACHE_DIR.mkdir(exist_ok=True)

TERRAFORM_EXE = os.environ.get("TERRAFORM_EXE", "terraform")
FMT_TIMEOUT = 20
INIT_TIMEOUT = 180
VALIDATE_TIMEOUT = 60


def unescape_newlines(value: str) -> str:
    return value.replace("\\n", "\n")


def load_existing(path: Path) -> Dict[str, Dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    existing = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("scenario_id", "")
                if sid:
                    existing[sid] = row
    return existing


def save_csv(path: Path, columns: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=columns, quoting=csv.QUOTE_ALL, extrasaction="ignore"
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _run(cmd: List[str], cwd: Path, timeout: int, env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, env=env,
        capture_output=True, text=True, timeout=timeout,
    )


def _env_for_terraform() -> Dict[str, str]:
    env = dict(os.environ)
    env["TF_IN_AUTOMATION"] = "1"
    env["TF_INPUT"] = "0"
    env["TF_CLI_ARGS"] = "-no-color"
    env["TF_PLUGIN_CACHE_DIR"] = str(PLUGIN_CACHE_DIR)
    env["CHECKPOINT_DISABLE"] = "1"
    return env


def run_terraform(code: str) -> Dict[str, Any]:
    """Run fmt, init (backend=false, get=false), and validate on a single HCL snippet."""
    out: Dict[str, Any] = {
        "fmt_ok": False,
        "fmt_error": "",
        "init_ok": False,
        "init_error": "",
        "validate_ok": False,
        "validate_error_count": 0,
        "validate_warning_count": 0,
        "validate_error": "",
    }

    if not code or not code.strip():
        out["fmt_error"] = "empty code"
        return out

    env = _env_for_terraform()

    with tempfile.TemporaryDirectory(prefix="tfeval_") as tmp:
        tmpdir = Path(tmp)
        tf_path = tmpdir / "main.tf"
        tf_path.write_text(code, encoding="utf-8")

        # --- fmt ---
        try:
            r = _run(
                [TERRAFORM_EXE, "fmt", "-check=true", "-diff=false", "-write=false", str(tf_path)],
                tmpdir, FMT_TIMEOUT, env,
            )
            # `terraform fmt -check` returns 0 if formatted, 3 if changes needed, nonzero on error
            out["fmt_ok"] = r.returncode == 0
            if r.returncode not in (0, 3):
                out["fmt_error"] = (r.stderr or r.stdout or "").strip()[:300]
        except subprocess.TimeoutExpired:
            out["fmt_error"] = "fmt timed out"
            return out
        except FileNotFoundError:
            out["fmt_error"] = "terraform executable not found"
            return out

        # --- init (backend disabled, plugin cache shared, modules skipped) ---
        try:
            r = _run(
                [TERRAFORM_EXE, "init",
                 "-backend=false", "-input=false", "-no-color",
                 "-get=false"],
                tmpdir, INIT_TIMEOUT, env,
            )
            out["init_ok"] = r.returncode == 0
            if not out["init_ok"]:
                out["init_error"] = (r.stderr or r.stdout or "").strip()[:400]
                return out
        except subprocess.TimeoutExpired:
            out["init_error"] = "init timed out"
            return out

        # --- validate ---
        try:
            r = _run(
                [TERRAFORM_EXE, "validate", "-json", "-no-color"],
                tmpdir, VALIDATE_TIMEOUT, env,
            )
            try:
                vjson = json.loads(r.stdout or "{}")
                out["validate_ok"] = bool(vjson.get("valid", False))
                out["validate_error_count"] = int(vjson.get("error_count", 0))
                out["validate_warning_count"] = int(vjson.get("warning_count", 0))
                if not out["validate_ok"]:
                    diags = vjson.get("diagnostics", []) or []
                    msgs = [d.get("summary", "") for d in diags if d.get("severity") == "error"]
                    out["validate_error"] = " | ".join(msgs)[:400]
            except json.JSONDecodeError:
                out["validate_error"] = (r.stderr or r.stdout or "").strip()[:400]
        except subprocess.TimeoutExpired:
            out["validate_error"] = "validate timed out"

    return out


RESULT_COLS = [
    "scenario_id", "model", "prompt_type", "dataset",
    "fmt_ok", "fmt_error",
    "init_ok", "init_error",
    "validate_ok", "validate_error_count", "validate_warning_count", "validate_error",
]


def process_row(row: Dict[str, str]) -> Dict[str, Any]:
    code = unescape_newlines(row.get("extracted_code", ""))
    res = run_terraform(code)
    return {
        "scenario_id": row.get("scenario_id", ""),
        "model": row.get("model", ""),
        "prompt_type": row.get("prompt_type", ""),
        "dataset": row.get("dataset", ""),
        **res,
    }


def process_csv(csv_path: Path, samples: Optional[int], workers: int) -> Path:
    stem = csv_path.stem
    out_path = RESULTS_DIR / f"{stem}_tf.csv"
    existing = load_existing(out_path)

    csv.field_size_limit(10 * 1024 * 1024)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if samples is not None:
        rows = rows[:samples]

    todo = [r for r in rows if r.get("scenario_id", "") not in existing]
    print(f"[{csv_path.name}] {len(rows)} rows total, {len(todo)} to run")

    results: List[Dict[str, Any]] = list(existing.values())

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_row, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(futures) - i) / rate if rate > 0 else 0
                print(f"  [{csv_path.name}] {i}/{len(futures)}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min")
            if i % 50 == 0:
                save_csv(out_path, RESULT_COLS, results)

    save_csv(out_path, RESULT_COLS, results)
    return out_path


def build_summary() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "terraform_summary.csv"
    csv.field_size_limit(10 * 1024 * 1024)

    agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for p in sorted(RESULTS_DIR.glob("*_tf.csv")):
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            key = (r.get("dataset", ""), r.get("model", ""), r.get("prompt_type", ""))
            d = agg.setdefault(key, {
                "dataset": key[0], "model": key[1], "prompt_type": key[2],
                "n": 0,
                "fmt_ok": 0, "init_ok": 0, "validate_ok": 0,
                "validate_error_count_total": 0,
            })
            d["n"] += 1
            d["fmt_ok"] += 1 if r.get("fmt_ok") == "True" else 0
            d["init_ok"] += 1 if r.get("init_ok") == "True" else 0
            d["validate_ok"] += 1 if r.get("validate_ok") == "True" else 0
            try:
                d["validate_error_count_total"] += int(r.get("validate_error_count") or 0)
            except ValueError:
                pass

    out_cols = [
        "dataset", "model", "prompt_type", "n",
        "fmt_pass_rate", "init_pass_rate", "validate_pass_rate",
        "mean_validate_errors",
    ]
    out_rows: List[Dict[str, Any]] = []
    for key in sorted(agg):
        d = agg[key]
        n = d["n"] or 1
        out_rows.append({
            "dataset": d["dataset"],
            "model": d["model"],
            "prompt_type": d["prompt_type"],
            "n": d["n"],
            "fmt_pass_rate": round(d["fmt_ok"] / n, 4),
            "init_pass_rate": round(d["init_ok"] / n, 4),
            "validate_pass_rate": round(d["validate_ok"] / n, 4),
            "mean_validate_errors": round(d["validate_error_count_total"] / n, 4),
        })
    save_csv(summary_path, out_cols, out_rows)
    print(f"Wrote {summary_path} ({len(out_rows)} rows)")


def parse_args():
    p = argparse.ArgumentParser(description="Run terraform fmt+validate on generated outputs")
    p.add_argument("--csv", type=Path, help="Single CSV to process")
    p.add_argument("--samples", type=int, help="First N rows per CSV")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--summary-only", action="store_true",
                   help="Rebuild results/terraform_summary.csv from existing *_tf.csv files")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.summary_only:
        build_summary()
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csvs = [args.csv] if args.csv else sorted(OUTPUTS_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {OUTPUTS_DIR}")
        return 1

    for csv_path in csvs:
        process_csv(csv_path, args.samples, args.workers)

    build_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
