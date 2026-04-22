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

Supports resume: already-processed rows are loaded, and failed checks can be retried
without rerunning successful checks.

Usage:
  uv run evaluate_terraform.py                     # all CSVs
  uv run evaluate_terraform.py --csv outputs/x.csv # single CSV
  uv run evaluate_terraform.py --samples 5         # first N rows per CSV
  uv run evaluate_terraform.py --workers 4         # parallelism
  uv run evaluate_terraform.py --file-workers 4    # process multiple CSVs in parallel
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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Semaphore limiting concurrent `terraform init` calls to reduce plugin-cache
# file-lock contention (the main cause of 180 s timeouts under parallel load).
_init_sem: threading.Semaphore = threading.Semaphore(1)

_NETWORK_ERRORS = (
    "could not connect",
    "failed to request discover",
    "failed to query available provider",
    "failed to install provider",
    "no such host",
    "connection refused",
    "connection reset",
    "i/o timeout",
    "context deadline exceeded",
)


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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def coerce_existing_result(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "fmt_ok": _parse_bool(row.get("fmt_ok", False)),
        "fmt_error": str(row.get("fmt_error", "") or ""),
        "init_ok": _parse_bool(row.get("init_ok", False)),
        "init_error": str(row.get("init_error", "") or ""),
        "validate_ok": _parse_bool(row.get("validate_ok", False)),
        "validate_error_count": _parse_int(row.get("validate_error_count", 0)),
        "validate_warning_count": _parse_int(row.get("validate_warning_count", 0)),
        "validate_error": str(row.get("validate_error", "") or ""),
    }


def _is_transient_fmt_error(err: str) -> bool:
    low = err.lower()
    return "timed out" in low or "terraform executable not found" in low


def _is_transient_init_error(err: str) -> bool:
    low = err.lower()
    return "timed out" in low or _is_network_error(err)


def _is_transient_validate_error(err: str) -> bool:
    low = err.lower()
    return (
        not err  # no error message recorded — unknown cause, retry to capture stderr
        or "timed out" in low
        or "validate produced no output" in low
        or "does not match any of the checksums" in low
        or "failed to load plugin schemas" in low
    )


def failed_checks(existing_row: Dict[str, Any]) -> Set[str]:
    """Return the subset of checks that failed due to transient execution issues.

    Permanent Terraform errors (bad HCL, unsupported arguments, fmt style
    violations, etc.) are excluded — re-running them produces the same result.
    Only infrastructure-side failures (timeouts, registry/network errors,
    checksum races, missing provider schemas) are worth retrying.
    """
    checks: Set[str] = set()

    if not _parse_bool(existing_row.get("fmt_ok", False)):
        if _is_transient_fmt_error(str(existing_row.get("fmt_error", "") or "")):
            checks.add("fmt")

    if not _parse_bool(existing_row.get("init_ok", False)):
        if _is_transient_init_error(str(existing_row.get("init_error", "") or "")):
            checks.add("init")

    if not _parse_bool(existing_row.get("validate_ok", False)):
        ve = str(existing_row.get("validate_error", "") or "")
        if _is_transient_validate_error(ve):
            checks.add("validate")
        elif "init prerequisite failed" in ve.lower() and "init" in checks:
            # validate was skipped because init failed transiently; retry both
            checks.add("validate")
    return checks


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
    # Prevents checksum-mismatch failures caused by concurrent workers writing
    # the same provider to the shared cache simultaneously.
    env["TF_PLUGIN_CACHE_MAY_BREAK_DEPENDENCY_LOCK_FILE"] = "1"
    return env


def _is_network_error(msg: str) -> bool:
    low = msg.lower()
    return any(p in low for p in _NETWORK_ERRORS)


def _run_init(tmpdir: Path, env: Dict[str, str], timeout: int, retries: int) -> Tuple[bool, str]:
    """Run `terraform init`, retrying on transient network errors."""
    cmd = [TERRAFORM_EXE, "init", "-backend=false", "-input=false", "-no-color", "-get=false"]
    last_err = ""
    for attempt in range(retries + 1):
        if attempt > 0:
            time.sleep(10 * attempt)
        try:
            r = _run(cmd, tmpdir, timeout, env)
            if r.returncode == 0:
                return True, ""
            err = (r.stderr or r.stdout or "").strip().replace("\n", " ")[:400]
            if not _is_network_error(err):
                return False, err
            last_err = err
        except subprocess.TimeoutExpired:
            return False, "init timed out"
        except FileNotFoundError:
            return False, "terraform executable not found"
    return False, last_err


def run_terraform(
    code: str,
    checks_to_run: Optional[Set[str]] = None,
    previous: Optional[Dict[str, Any]] = None,
    init_retries: int = 2,
) -> Dict[str, Any]:
    """Run selected checks on a single HCL snippet, preserving previous successful checks."""
    defaults: Dict[str, Any] = {
        "fmt_ok": False,
        "fmt_error": "",
        "init_ok": False,
        "init_error": "",
        "validate_ok": False,
        "validate_error_count": 0,
        "validate_warning_count": 0,
        "validate_error": "",
    }

    out: Dict[str, Any] = {**defaults}
    if previous:
        out.update(coerce_existing_result(previous))

    checks = set(checks_to_run or {"fmt", "init", "validate"})

    if not code or not code.strip():
        if "fmt" in checks:
            out["fmt_ok"] = False
            out["fmt_error"] = "empty code"
        if "init" in checks:
            out["init_ok"] = False
            out["init_error"] = "empty code"
        if "validate" in checks:
            out["validate_ok"] = False
            out["validate_error_count"] = 1
            out["validate_error"] = "empty code"
        return out

    env = _env_for_terraform()

    with tempfile.TemporaryDirectory(prefix="tfeval_") as tmp:
        tmpdir = Path(tmp)
        tf_path = tmpdir / "main.tf"
        tf_path.write_text(code, encoding="utf-8")

        # --- fmt ---
        if "fmt" in checks:
            out["fmt_error"] = ""
            try:
                r = _run(
                    [TERRAFORM_EXE, "fmt", "-check=true", "-diff=false", "-write=false", str(tf_path)],
                    tmpdir, FMT_TIMEOUT, env,
                )
                # `terraform fmt -check` returns 0 if formatted, 3 if changes needed, nonzero on error
                out["fmt_ok"] = r.returncode == 0
                if r.returncode not in (0, 3):
                    out["fmt_error"] = (r.stderr or r.stdout or "").strip().replace("\n", " ")[:300]
            except subprocess.TimeoutExpired:
                out["fmt_ok"] = False
                out["fmt_error"] = "fmt timed out"
                return out
            except FileNotFoundError:
                out["fmt_ok"] = False
                out["fmt_error"] = "terraform executable not found"
                return out

        # validate requires init in a fresh temp directory.
        need_init = "init" in checks or "validate" in checks
        init_ok_for_validate = True
        init_err_for_validate = ""

        if need_init:
            if "init" in checks:
                out["init_error"] = ""

            # Serialize init calls to avoid concurrent file-lock contention on
            # the shared plugin cache, which is the primary cause of timeouts.
            with _init_sem:
                init_ok_for_validate, init_err_for_validate = _run_init(
                    tmpdir, env, INIT_TIMEOUT, init_retries
                )

            if "init" in checks:
                out["init_ok"] = init_ok_for_validate
                if not init_ok_for_validate:
                    out["init_error"] = init_err_for_validate
                    return out
            if not init_ok_for_validate:
                pass  # handled in validate block below

        # --- validate ---
        if "validate" in checks:
            out["validate_error"] = ""
            if not init_ok_for_validate:
                out["validate_ok"] = False
                out["validate_error_count"] = 1
                out["validate_warning_count"] = 0
                out["validate_error"] = f"init prerequisite failed: {init_err_for_validate}"[:400]
                return out

            try:
                r = _run(
                    [TERRAFORM_EXE, "validate", "-json", "-no-color"],
                    tmpdir, VALIDATE_TIMEOUT, env,
                )
                stdout = (r.stdout or "").strip()
                if not stdout:
                    # validate produced no JSON output; the error is in stderr
                    out["validate_ok"] = False
                    out["validate_error_count"] = max(out.get("validate_error_count", 0), 1)
                    out["validate_error"] = (
                        (r.stderr or "").strip().replace("\n", " ") or "validate produced no output"
                    )[:400]
                else:
                    try:
                        vjson = json.loads(stdout)
                        out["validate_ok"] = bool(vjson.get("valid", False))
                        out["validate_error_count"] = int(vjson.get("error_count", 0))
                        out["validate_warning_count"] = int(vjson.get("warning_count", 0))
                        if not out["validate_ok"]:
                            diags = vjson.get("diagnostics", []) or []
                            msgs = [d.get("summary", "") for d in diags if d.get("severity") == "error"]
                            out["validate_error"] = " | ".join(msgs)[:400]
                    except json.JSONDecodeError:
                        out["validate_ok"] = False
                        out["validate_error_count"] = max(out.get("validate_error_count", 0), 1)
                        out["validate_error"] = (r.stderr or stdout or "").strip().replace("\n", " ")[:400]
            except subprocess.TimeoutExpired:
                out["validate_ok"] = False
                out["validate_error_count"] = max(out.get("validate_error_count", 0), 1)
                out["validate_error"] = "validate timed out"
            except FileNotFoundError:
                out["validate_ok"] = False
                out["validate_error_count"] = max(out.get("validate_error_count", 0), 1)
                out["validate_error"] = "terraform executable not found"

    return out


RESULT_COLS = [
    "scenario_id", "model", "prompt_type", "dataset",
    "fmt_ok", "fmt_error",
    "init_ok", "init_error",
    "validate_ok", "validate_error_count", "validate_warning_count", "validate_error",
]


def process_row(
    row: Dict[str, str],
    checks_to_run: Optional[Set[str]] = None,
    previous: Optional[Dict[str, Any]] = None,
    init_retries: int = 2,
) -> Dict[str, Any]:
    code = unescape_newlines(row.get("extracted_code", ""))
    res = run_terraform(code, checks_to_run=checks_to_run, previous=previous, init_retries=init_retries)
    return {
        "scenario_id": row.get("scenario_id", ""),
        "model": row.get("model", ""),
        "prompt_type": row.get("prompt_type", ""),
        "dataset": row.get("dataset", ""),
        **res,
    }


def process_csv(
    csv_path: Path,
    samples: Optional[int],
    workers: int,
    rerun_failed: bool,
    init_retries: int = 2,
) -> Path:
    stem = csv_path.stem
    out_path = RESULTS_DIR / f"{stem}_tf.csv"
    existing = load_existing(out_path)

    csv.field_size_limit(10 * 1024 * 1024)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if samples is not None:
        rows = rows[:samples]

    todo: List[Tuple[Dict[str, str], Optional[Set[str]], Optional[Dict[str, Any]]]] = []
    rerun_count = 0
    for r in rows:
        sid = r.get("scenario_id", "")
        previous = existing.get(sid)
        if previous is None:
            todo.append((r, None, None))
            continue
        if rerun_failed:
            checks = failed_checks(previous)
            if checks:
                todo.append((r, checks, previous))
                rerun_count += 1

    print(
        f"[{csv_path.name}] {len(rows)} rows total, {len(todo)} to run "
        f"({rerun_count} failed-row retries, {len(todo) - rerun_count} new)"
    )

    results_by_sid: Dict[str, Dict[str, Any]] = dict(existing)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(process_row, row, checks, previous, init_retries): (row, checks)
            for row, checks, previous in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            sid = res.get("scenario_id", "")
            if sid:
                results_by_sid[sid] = res
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(futures) - i) / rate if rate > 0 else 0
                print(f"  [{csv_path.name}] {i}/{len(futures)}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min")
            if i % 50 == 0:
                save_csv(out_path, RESULT_COLS, list(results_by_sid.values()))

    save_csv(out_path, RESULT_COLS, list(results_by_sid.values()))
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
    p.add_argument("--file-workers", type=int, default=1,
                   help="Number of CSV files to process concurrently")
    p.add_argument("--summary-only", action="store_true",
                   help="Rebuild results/terraform_summary.csv from existing *_tf.csv files")
    p.add_argument(
        "--rerun-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, rerun only failed checks for existing rows (default: enabled)",
    )
    p.add_argument(
        "--init-concurrency", type=int, default=1,
        help="Max concurrent `terraform init` calls (default 1). "
             "Higher values risk plugin-cache file-lock contention and timeouts.",
    )
    p.add_argument(
        "--init-retries", type=int, default=2,
        help="Retry count for transient network errors during init (default 2, 10s backoff).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.summary_only:
        build_summary()
        return 0

    global _init_sem
    _init_sem = threading.Semaphore(max(1, args.init_concurrency))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csvs = [args.csv] if args.csv else sorted(OUTPUTS_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {OUTPUTS_DIR}")
        return 1

    file_workers = max(1, int(args.file_workers))
    had_errors = False
    if len(csvs) == 1 or file_workers == 1:
        for csv_path in csvs:
            try:
                process_csv(csv_path, args.samples, args.workers, args.rerun_failed, args.init_retries)
            except Exception as exc:
                had_errors = True
                print(f"[error] {csv_path.name}: {exc}")
    else:
        print(f"Processing {len(csvs)} CSV files with file-workers={file_workers}, "
              f"row-workers={args.workers}")
        with ThreadPoolExecutor(max_workers=min(file_workers, len(csvs))) as ex:
            futures = {
                ex.submit(
                    process_csv, csv_path, args.samples, args.workers,
                    args.rerun_failed, args.init_retries,
                ): csv_path
                for csv_path in csvs
            }
            for fut in as_completed(futures):
                csv_path = futures[fut]
                try:
                    out_path = fut.result()
                    print(f"[done] {csv_path.name} -> {out_path.name}")
                except Exception as exc:
                    had_errors = True
                    print(f"[error] {csv_path.name}: {exc}")

    build_summary()
    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
