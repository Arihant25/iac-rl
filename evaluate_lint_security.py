#!/usr/bin/env python3
"""
Lint and Security Evaluation for IaC-Bench

For every generated Terraform configuration in outputs/*.csv, runs:
  - TFLint  : linting rule violations (total count + per-severity counts)
  - Trivy   : security misconfigurations (total count + per-severity counts)

Writes per-file result CSVs to results/ (e.g. iac_eval_claude_zero-shot_lint.csv)
and an aggregated summary to results/lint_security_summary.csv.

Supports resume: already-processed scenario_ids are skipped.

Usage:
  python evaluate_lint_security.py                     # all CSVs
  python evaluate_lint_security.py --csv outputs/x.csv # single CSV
  python evaluate_lint_security.py --samples 5         # first N rows
  python evaluate_lint_security.py --workers 4         # parallelism
  python evaluate_lint_security.py --summary-only      # rebuild summary only
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

# Locate tools — prefer PATH, then ~/bin fallback
def _find_exe(name: str) -> str:
    user_bin = Path.home() / "bin" / f"{name}.exe"
    if user_bin.exists():
        return str(user_bin)
    return name  # rely on PATH


TFLINT_EXE   = _find_exe("tflint")
TRIVY_EXE    = _find_exe("trivy")
TRIVY_CACHE  = Path.home() / ".trivy-cache"
TRIVY_SERVER = os.environ.get("TRIVY_SERVER", "http://localhost:4954")

TFLINT_CONFIG = SCRIPT_DIR / ".tflint.hcl"

SEVERITIES = ["ERROR", "WARNING", "NOTICE", "INFO"]
TRIVY_SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]


# ─────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────

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
        w = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL,
                           extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ─────────────────────────────────────────────────────────────
# TFLint runner
# ─────────────────────────────────────────────────────────────

def run_tflint(code: str) -> Dict[str, Any]:
    """
    Write code to a temp dir, run TFLint, parse JSON output.
    Returns dict with keys: ok, total, ERROR, WARNING, NOTICE, INFO, raw_error.
    """
    result = {k: 0 for k in ["total"] + SEVERITIES}
    result["ok"] = False
    result["raw_error"] = ""

    if not code or not code.strip():
        result["raw_error"] = "empty code"
        return result

    with tempfile.TemporaryDirectory(prefix="tflint_") as tmpdir:
        tf_path = Path(tmpdir) / "main.tf"
        tf_path.write_text(code, encoding="utf-8")

        cmd = [TFLINT_EXE, "--format=json", "--no-color"]
        if TFLINT_CONFIG.exists():
            cmd += [f"--config={TFLINT_CONFIG}"]
        # Disable slow network calls: no module inspection
        cmd += ["--disable-rule=terraform_module_pinned_source"]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=tmpdir
            )
        except subprocess.TimeoutExpired:
            result["raw_error"] = "tflint timed out"
            return result
        except Exception as e:
            result["raw_error"] = str(e)
            return result

        # tflint exits 0 (no issues), 2 (issues found), or 1 (error)
        if proc.returncode not in (0, 2):
            result["raw_error"] = (proc.stderr or proc.stdout or "")[:300]
            return result

        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            result["raw_error"] = f"json parse error: {proc.stdout[:200]}"
            return result

        issues = data.get("issues", []) or []
        result["ok"] = True
        result["total"] = len(issues)
        for issue in issues:
            sev = (issue.get("rule", {}).get("severity") or "").upper()
            if sev in result:
                result[sev] += 1

    return result


# ─────────────────────────────────────────────────────────────
# Trivy runner
# ─────────────────────────────────────────────────────────────

def run_trivy(code: str) -> Dict[str, Any]:
    """
    Write code to a temp dir, run Trivy misconfiguration scan, parse JSON.
    Returns dict with keys: ok, total, CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN, raw_error.
    """
    result = {k: 0 for k in ["total"] + TRIVY_SEVERITIES}
    result["ok"] = False
    result["raw_error"] = ""

    if not code or not code.strip():
        result["raw_error"] = "empty code"
        return result

    with tempfile.TemporaryDirectory(prefix="trivy_") as tmpdir:
        tf_path = Path(tmpdir) / "main.tf"
        tf_path.write_text(code, encoding="utf-8")

        cmd = [
            TRIVY_EXE, "config",
            "--format", "json",
            "--quiet",
            "--exit-code", "0",   # never fail on findings
            "--skip-check-update",
            "--skip-version-check",
            tmpdir,
        ]

        # Strip TRIVY_SERVER from env — if inherited it causes "unknown flag: --server"
        clean_env = {k: v for k, v in os.environ.items() if k != "TRIVY_SERVER"}
        clean_env.update({"TRIVY_NO_PROGRESS": "true", "NO_COLOR": "1"})

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd=tmpdir,
                env=clean_env
            )
        except subprocess.TimeoutExpired:
            result["raw_error"] = "trivy timed out"
            return result
        except Exception as e:
            result["raw_error"] = str(e)
            return result

        if proc.returncode != 0:
            result["raw_error"] = (proc.stderr or proc.stdout or "")[:300]
            return result

        try:
            data = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            result["raw_error"] = f"json parse error: {proc.stdout[:200]}"
            return result

        result["ok"] = True
        for res in data.get("Results", []) or []:
            for misconfig in res.get("Misconfigurations", []) or []:
                result["total"] += 1
                sev = (misconfig.get("Severity") or "UNKNOWN").upper()
                if sev in result:
                    result[sev] += 1

    return result


# ─────────────────────────────────────────────────────────────
# Result schema
# ─────────────────────────────────────────────────────────────

def get_columns() -> List[str]:
    meta = ["scenario_id", "model", "prompt_type", "dataset"]
    tflint_cols = (
        ["gen_tflint_ok", "gen_tflint_total"]
        + [f"gen_tflint_{s.lower()}" for s in SEVERITIES]
        + ["gen_tflint_error"]
    )
    trivy_cols = (
        ["gen_trivy_ok", "gen_trivy_total"]
        + [f"gen_trivy_{s.lower()}" for s in TRIVY_SEVERITIES]
        + ["gen_trivy_error"]
    )
    return meta + tflint_cols + trivy_cols


# ─────────────────────────────────────────────────────────────
# Per-row processor
# ─────────────────────────────────────────────────────────────

def process_row(row: Dict[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "scenario_id": row.get("scenario_id", ""),
        "model":       row.get("model", ""),
        "prompt_type": row.get("prompt_type", ""),
        "dataset":     row.get("dataset", ""),
    }

    code = unescape_newlines(row.get("extracted_code", ""))

    lint  = run_tflint(code)
    sec   = run_trivy(code)

    result["gen_tflint_ok"]    = lint["ok"]
    result["gen_tflint_total"] = lint["total"]
    for s in SEVERITIES:
        result[f"gen_tflint_{s.lower()}"] = lint[s]
    result["gen_tflint_error"] = lint["raw_error"]

    result["gen_trivy_ok"]    = sec["ok"]
    result["gen_trivy_total"] = sec["total"]
    for s in TRIVY_SEVERITIES:
        result[f"gen_trivy_{s.lower()}"] = sec[s]
    result["gen_trivy_error"] = sec["raw_error"]

    return result


# ─────────────────────────────────────────────────────────────
# Process a single CSV
# ─────────────────────────────────────────────────────────────

def process_csv(
    csv_path: Path,
    results_dir: Path,
    samples: Optional[int] = None,
    workers: int = 4,
) -> Path:
    result_filename = csv_path.stem + "_lint.csv"
    result_path = results_dir / result_filename
    columns = get_columns()

    existing = load_existing(result_path)
    print(f"\n{'=' * 70}")
    print(f"  CSV: {csv_path.name}")
    print(f"  Already done: {len(existing)} rows")
    print(f"{'=' * 70}")

    all_rows: List[Dict] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)
    if samples:
        all_rows = all_rows[:samples]

    todo = [r for r in all_rows if r.get("scenario_id", "") not in existing]
    print(f"  Total: {len(all_rows)}, to process: {len(todo)}")
    if not todo:
        print("  Nothing to do.")
        return result_path

    all_results = list(existing.values())
    start = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_row, r): r for r in todo}
        for future in as_completed(futures):
            row = futures[future]
            sid = row.get("scenario_id", "?")
            try:
                res = future.result()
                all_results.append(res)
                done += 1
                elapsed = time.time() - start
                rate = done / elapsed if elapsed else 0
                eta = (len(todo) - done) / rate if rate else 0
                eta_str = f"{int(eta//60)}m{int(eta%60)}s"
                lint_ok = "ok" if res.get("gen_tflint_ok") else "err"
                sec_ok  = "ok" if res.get("gen_trivy_ok") else "err"
                print(
                    f"  [{done}/{len(todo)}] {sid}  "
                    f"tflint={lint_ok}({res.get('gen_tflint_total',0)}) "
                    f"trivy={sec_ok}({res.get('gen_trivy_total',0)})  ETA:{eta_str}"
                )
                if done % 20 == 0:
                    save_csv(result_path, columns, all_results)
            except Exception as e:
                print(f"  [{done}/{len(todo)}] {sid}  ERROR: {e}")

    save_csv(result_path, columns, all_results)
    print(f"  Done -> {result_path.name}  ({time.time()-start:.1f}s)")
    return result_path


# ─────────────────────────────────────────────────────────────
# Summary aggregation
# ─────────────────────────────────────────────────────────────

def build_summary(results_dir: Path) -> Path:
    summary_path = results_dir / "lint_security_summary.csv"

    count_cols = (
        ["tflint_total"] + [f"tflint_{s.lower()}" for s in SEVERITIES]
        + ["trivy_total"] + [f"trivy_{s.lower()}" for s in TRIVY_SEVERITIES]
    )

    groups: Dict[Tuple[str, str, str], List[Dict]] = {}
    for lint_csv in sorted(results_dir.glob("*_lint.csv")):
        with open(lint_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row.get("dataset",""), row.get("model",""), row.get("prompt_type",""))
                groups.setdefault(key, []).append(row)

    summary_rows = []
    for (dataset, model, prompt_type), rows in sorted(groups.items()):
        s: Dict[str, Any] = {
            "dataset": dataset, "model": model, "prompt_type": prompt_type,
            "num_samples": len(rows),
        }
        tflint_ok = sum(1 for r in rows if r.get("gen_tflint_ok") == "True")
        trivy_ok  = sum(1 for r in rows if r.get("gen_trivy_ok")  == "True")
        s["tflint_run_rate"] = round(tflint_ok / len(rows), 4) if rows else 0
        s["trivy_run_rate"]  = round(trivy_ok  / len(rows), 4) if rows else 0

        for col in count_cols:
            vals = []
            for r in rows:
                v = r.get(f"gen_{col}", "")
                if v not in ("", "None", "False", "True"):
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
            s[f"mean_{col}"] = round(sum(vals)/len(vals), 4) if vals else 0.0

        summary_rows.append(s)

    if summary_rows:
        cols = list(summary_rows[0].keys())
        save_csv(summary_path, cols, summary_rows)
    print(f"\nLint/security summary -> {summary_path}")
    return summary_path


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TFLint + Trivy evaluation")
    p.add_argument("--csv", default=None)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        build_summary(RESULTS_DIR)
        return

    csv_files = [Path(args.csv)] if args.csv else sorted(OUTPUTS_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSVs in {OUTPUTS_DIR}"); sys.exit(1)

    print(f"TFLint: {TFLINT_EXE}")
    print(f"Trivy:  {TRIVY_EXE}")
    print(f"CSVs:   {len(csv_files)}")

    t0 = time.time()
    for csv_path in csv_files:
        process_csv(csv_path, RESULTS_DIR, args.samples, args.workers)

    build_summary(RESULTS_DIR)
    total = time.time() - t0
    print(f"\nAll done in {int(total//60)}m{int(total%60)}s")


if __name__ == "__main__":
    main()
