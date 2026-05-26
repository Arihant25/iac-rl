#!/usr/bin/env python3
"""
Statistical significance tests for IaC-Bench.

For each comparative claim in the paper—model rankings by ICS, dataset effects,
and prompting-strategy effects—this script runs appropriate non-parametric tests,
corrects for multiple comparisons, computes effect sizes, and produces:

  results/ics_per_prompt.csv           per-prompt ICS for all (model × strategy × dataset)
  results/statistical_tests.json       full test statistics
  results/tab_statistical_significance.tex  LaTeX table for inclusion in paper

Tests applied:
  1. Kruskal-Wallis H across 9 models (per dataset) — overall model effect
  2. Dunn's post-hoc pairwise tests with Holm-Bonferroni correction (36 pairs × 2 datasets)
  3. Rank-biserial correlation as effect size for each pair
  4. Friedman χ² for prompting-strategy effect (per dataset)
  5. Bootstrap 95 % CIs for mean ICS per model per dataset (B = 2000)

ICS computation is reproduced from srs.py using the same constraint-extraction and
satisfaction-checking logic so that the test data exactly match the paper's reported values.

Usage:
    python statistical_tests.py [--samples N]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
DATASETS_DIR = ROOT / "datasets"
IAC_EVAL_JSON = DATASETS_DIR / "iac_eval_dataset.json"
LLM_IAC_CSV = DATASETS_DIR / "llm-iac.csv"

MODELS = [
    "claude-4.5-sonnet",
    "grok-4.1-fast",
    "gemini-3-flash",
    "kimi-k2.5",
    "glm-4.7",
    "qwen3-235b",
    "ministral-8b",
    "phi-4",
    "gemma-3-27b",
]
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
STRATEGIES = ["zero-shot", "few-shot", "cot"]
DATASETS = ["iac_eval", "llm_iac"]

# ---------------------------------------------------------------------------
# ICS computation (reproduced from srs.py verbatim)
# ---------------------------------------------------------------------------

RESOURCE_MAP = {
    "s3 bucket": "aws_s3_bucket", "s3": "aws_s3_bucket",
    "vpc": "aws_vpc", "subnet": "aws_subnet", "ec2": "aws_instance",
    "security group": "aws_security_group", "lambda": "aws_lambda_function",
    "load balancer": "aws_lb", "application load balancer": "aws_lb",
    "elb": "aws_lb", "alb": "aws_lb", "nlb": "aws_lb",
    "autoscaling": "aws_autoscaling_group", "auto scaling": "aws_autoscaling_group",
    "rds": "aws_db_instance", "database": "aws_db_instance",
    "db instance": "aws_db_instance", "dynamodb": "aws_dynamodb_table",
    "elasticache": "aws_elasticache_cluster", "redshift": "aws_redshift_cluster",
    "route 53": "aws_route53_zone", "route53": "aws_route53_zone",
    "dns record": "aws_route53_record", "nat gateway": "aws_nat_gateway",
    "internet gateway": "aws_internet_gateway", "igw": "aws_internet_gateway",
    "route table": "aws_route_table", "transit gateway": "aws_ec2_transit_gateway",
    "vpn": "aws_vpn_gateway", "peering": "aws_vpc_peering_connection",
    "cloudwatch": "aws_cloudwatch_log_group", "cloudtrail": "aws_cloudtrail",
    "ssm": "aws_ssm_parameter", "parameter store": "aws_ssm_parameter",
    "systems manager": "aws_ssm_parameter",
    "eks": "aws_eks_cluster", "ecs": "aws_ecs_cluster",
    "ecr": "aws_ecr_repository", "fargate": "aws_ecs_cluster",
    "sqs": "aws_sqs_queue", "sns": "aws_sns_topic",
    "kinesis": "aws_kinesis_stream", "eventbridge": "aws_cloudwatch_event_rule",
    "event bridge": "aws_cloudwatch_event_rule", "msk": "aws_msk_cluster",
    "iam role": "aws_iam_role", "iam policy": "aws_iam_policy",
    "iam user": "aws_iam_user", "iam group": "aws_iam_group",
    "kms": "aws_kms_key", "waf": "aws_wafv2_web_acl",
    "secret": "aws_secretsmanager_secret", "secrets manager": "aws_secretsmanager_secret",
    "certificate": "aws_acm_certificate",
    "elastic beanstalk": "aws_elastic_beanstalk_environment",
    "beanstalk": "aws_elastic_beanstalk_environment",
    "cloudfront": "aws_cloudfront_distribution",
    "api gateway": "aws_api_gateway_rest_api",
    "step function": "aws_sfn_state_machine", "sfn": "aws_sfn_state_machine",
    "emr": "aws_emr_cluster", "glue": "aws_glue_job", "athena": "aws_athena_database",
    "codepipeline": "aws_codepipeline", "codecommit": "aws_codecommit_repository",
    "codebuild": "aws_codebuild_project", "cloudformation": "aws_cloudformation_stack",
    "gcp": "google_compute_instance", "google cloud": "google_compute_instance",
    "gke": "google_container_cluster", "cloud run": "google_cloud_run_service",
    "bigquery": "google_bigquery_dataset", "cloud storage": "google_storage_bucket",
    "bigtable": "google_bigtable_instance", "pub/sub": "google_pubsub_topic",
    "pubsub": "google_pubsub_topic", "cloud sql": "google_sql_database_instance",
    "cloud function": "google_cloudfunctions_function",
    "cloud dns": "google_dns_managed_zone", "cloud nat": "google_compute_router_nat",
    "azure": "azurerm_virtual_machine", "azure vm": "azurerm_virtual_machine",
    "azure storage": "azurerm_storage_account", "azure sql": "azurerm_sql_server",
    "azure function": "azurerm_function_app", "aks": "azurerm_kubernetes_cluster",
    "azure kubernetes": "azurerm_kubernetes_cluster",
    "network watcher": "azurerm_network_watcher", "azure firewall": "azurerm_firewall",
}

_IMPLICIT_CONSTRAINTS = [
    ("encrypt",      {"type": "implicit_property", "keyword": "encrypt",      "hcl_patterns": [r'kms_key_id\s*=', r'sse_algorithm\s*=', r'encrypt_at_rest\s*=\s*true', r'encrypted\s*=\s*true', r'server_side_encryption']}),
    ("kms",          {"type": "implicit_property", "keyword": "kms",          "hcl_patterns": [r'kms_key_id\s*=', r'aws_kms_key']}),
    ("multi-az",     {"type": "implicit_property", "keyword": "multi-az",     "hcl_patterns": [r'multi_az\s*=\s*true', r'availability_zones\s*=']}),
    ("highly available", {"type": "implicit_property", "keyword": "highly available", "hcl_patterns": [r'multi_az\s*=\s*true', r'desired_capacity\s*=\s*[2-9]', r'availability_zones\s*=']}),
    ("high availability", {"type": "implicit_property", "keyword": "high availability", "hcl_patterns": [r'multi_az\s*=\s*true', r'desired_capacity\s*=\s*[2-9]', r'availability_zones\s*=']}),
    ("monitoring",   {"type": "implicit_property", "keyword": "monitoring",   "hcl_patterns": [r'aws_cloudwatch', r'monitoring\s*=\s*true', r'enable_monitoring\s*=\s*true']}),
    ("logging",      {"type": "implicit_property", "keyword": "logging",      "hcl_patterns": [r'aws_cloudwatch_log', r'aws_cloudtrail', r'logging\s*\{', r'access_logs\s*\{']}),
    ("backup",       {"type": "implicit_property", "keyword": "backup",       "hcl_patterns": [r'backup_retention_period\s*=\s*[1-9]', r'aws_backup']}),
    ("lifecycle",    {"type": "implicit_property", "keyword": "lifecycle",    "hcl_patterns": [r'lifecycle_rule\s*\{', r'aws_s3_bucket_lifecycle']}),
    ("versioning",   {"type": "implicit_property", "keyword": "versioning",   "hcl_patterns": [r'versioning\s*\{', r'enabled\s*=\s*true']}),
]

_INTENT_ATTR_EQUALS = re.compile(r'with\s+(?:a specified |one |multiple |an? )?"([^"]+)"\s+equal\s+to\s+"([^"]+)"')
_INTENT_ATTR_EXISTS = re.compile(r'with\s+(?:a specified |one |multiple |an? )?"([^"]+)"')
_INTENT_SKIP = re.compile(r"referencing|enabling|block\s+that\s+contains|block\s+with")


def extract_constraints(prompt: str) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    seen_resources: set = set()
    p_lower = prompt.lower()
    for keyword in sorted(RESOURCE_MAP, key=len, reverse=True):
        if keyword in p_lower:
            rtype = RESOURCE_MAP[keyword]
            if rtype not in seen_resources:
                seen_resources.add(rtype)
                constraints.append({"type": "resource_exists", "resource_type": rtype})
    match = re.search(r"\b([a-z][0-9][a-z]{0,3}(?:dn|en)?\.[0-9]{0,2}(?:nano|micro|small|medium|large|xlarge|metal))", p_lower)
    if match:
        constraints.append({"type": "property_equals", "resource_type": "aws_instance", "property": "instance_type", "value": match.group(1)})
    match = re.search(r"(us|eu|ap|sa|ca|me|af|us-central|us-east|us-west|europe-west|asia-east)-[a-z]+-[0-9]", p_lower)
    if match:
        constraints.append({"type": "provider_region", "value": match.group(0)})
    if "private s3" in p_lower or "private bucket" in p_lower:
        constraints.append({"type": "property_equals", "resource_type": "aws_s3_bucket", "property": "acl", "value": "private"})
    if "ssh" in p_lower:
        constraints.append({"type": "security_ingress", "port": 22})
    if "https" in p_lower:
        constraints.append({"type": "security_ingress", "port": 443})
    elif "http" in p_lower:
        constraints.append({"type": "security_ingress", "port": 80})
    match = re.search(r"(\d+)\s+to\s+(\d+)", p_lower)
    if match:
        constraints.append({"type": "autoscaling_min", "value": int(match.group(1))})
        constraints.append({"type": "autoscaling_max", "value": int(match.group(2))})
    seen_implicit: set = set()
    for keyword, constraint in _IMPLICIT_CONSTRAINTS:
        if keyword in p_lower and keyword not in seen_implicit:
            seen_implicit.add(keyword)
            constraints.append(dict(constraint))
    return constraints


def parse_intent_field(intent_str: str) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    current_resource: Optional[str] = None
    for raw_line in intent_str.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Has") and "resource" in line:
            m = re.search(r'"([^"]+)"', line)
            if m:
                current_resource = m.group(1)
                constraints.append({"type": "resource_exists", "resource_type": current_resource})
            continue
        if line.startswith("with") and current_resource:
            if _INTENT_SKIP.search(line):
                continue
            m = _INTENT_ATTR_EQUALS.search(line)
            if m:
                constraints.append({"type": "attribute_equals", "resource_type": current_resource, "attribute": m.group(1), "value": m.group(2)})
                continue
            m = _INTENT_ATTR_EXISTS.search(line)
            if m:
                constraints.append({"type": "attribute_exists", "resource_type": current_resource, "attribute": m.group(1)})
    return constraints if constraints else extract_constraints(intent_str)


def _resource_block_content(tf_code: str, resource_type: str) -> str:
    out: List[str] = []
    pattern = re.compile(r'resource\s+"' + re.escape(resource_type) + r'"\s+"[^"]*"\s*\{')
    for header_m in pattern.finditer(tf_code):
        start = header_m.end()
        depth = 1
        i = start
        while i < len(tf_code) and depth > 0:
            if tf_code[i] == "{":
                depth += 1
            elif tf_code[i] == "}":
                depth -= 1
            i += 1
        out.append(tf_code[start : i - 1])
    return "\n".join(out)


def terraform_to_facts(tf_code: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {"resources": set(), "properties": {}, "region": None, "ports": set(), "_raw": tf_code}
    if not isinstance(tf_code, str) or not tf_code.strip():
        return facts
    for m in re.finditer(r'resource\s+"([^"]+)"\s+"[^"]+"', tf_code):
        facts["resources"].add(m.group(1))
    m = re.search(r'instance_type\s*=\s*"([^"]+)"', tf_code)
    if m:
        facts["properties"]["instance_type"] = m.group(1)
    m = re.search(r'acl\s*=\s*"([^"]+)"', tf_code)
    if m:
        facts["properties"]["acl"] = m.group(1)
    if "versioning" in tf_code and re.search(r"enabled\s*=\s*true", tf_code, re.I):
        facts["properties"]["versioning.enabled"] = True
    m = re.search(r'region\s*=\s*"([^"]+)"', tf_code)
    if m:
        facts["region"] = m.group(1)
    for m in re.finditer(r"from_port\s*=\s*(\d+)", tf_code):
        facts["ports"].add(int(m.group(1)))
    m = re.search(r"min_size\s*=\s*(\d+)", tf_code)
    if m:
        facts["properties"]["min_size"] = int(m.group(1))
    m = re.search(r"max_size\s*=\s*(\d+)", tf_code)
    if m:
        facts["properties"]["max_size"] = int(m.group(1))
    return facts


def is_satisfied(constraint: Dict[str, Any], facts: Dict[str, Any]) -> bool:
    ctype = constraint.get("type")
    if ctype == "resource_exists":
        return constraint["resource_type"] in facts["resources"]
    if ctype == "property_equals":
        return facts["properties"].get(constraint["property"]) == constraint["value"]
    if ctype == "provider_region":
        return facts.get("region") == constraint["value"]
    if ctype == "security_ingress":
        return constraint["port"] in facts["ports"]
    if ctype == "autoscaling_min":
        return facts["properties"].get("min_size", 0) >= constraint["value"]
    if ctype == "autoscaling_max":
        return facts["properties"].get("max_size", float("inf")) <= constraint["value"]
    if ctype == "attribute_exists":
        block = _resource_block_content(facts.get("_raw", ""), constraint["resource_type"])
        attr = re.escape(constraint["attribute"])
        return bool(re.search(r'\b' + attr + r'\s*=', block))
    if ctype == "attribute_equals":
        block = _resource_block_content(facts.get("_raw", ""), constraint["resource_type"])
        attr = re.escape(constraint["attribute"])
        val = re.escape(constraint["value"])
        return bool(re.search(r'\b' + attr + r'\s*=\s*"' + val + r'"', block))
    if ctype == "implicit_property":
        raw = facts.get("_raw", "")
        return any(re.search(p, raw, re.I) for p in constraint["hcl_patterns"])
    return False


def compute_ics(tf_code: str, constraints: List[Dict[str, Any]]) -> Optional[float]:
    if not constraints:
        return None
    facts = terraform_to_facts(tf_code)
    satisfied = sum(1 for c in constraints if is_satisfied(c, facts))
    return satisfied / len(constraints)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_iac_eval_intents() -> Dict[str, str]:
    """Return {scenario_id -> intent_string} for IaC-Eval."""
    entries = json.loads(IAC_EVAL_JSON.read_text(encoding="utf-8"))
    return {f"iac_eval_{i:04d}": e.get("Intent", "") for i, e in enumerate(entries)}


def load_outputs_csv(path: Path) -> List[Dict]:
    csv.field_size_limit(10 * 1024 * 1024)
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Per-prompt ICS computation across all model × strategy × dataset combinations
# ---------------------------------------------------------------------------

def build_ics_table(samples: Optional[int] = None) -> List[Dict]:
    """
    Compute ICS for every (model, strategy, dataset, prompt) combination.
    Returns a list of dicts with keys:
      scenario_id, model, prompt_type, dataset, ics, n_constraints
    """
    intents = load_iac_eval_intents()
    rows: List[Dict] = []

    for model in MODELS:
        for strategy in STRATEGIES:
            for dataset in DATASETS:
                csv_name = f"{dataset}_{model}_{strategy}.csv"
                csv_path = OUTPUTS_DIR / csv_name
                if not csv_path.exists():
                    print(f"  [skip] {csv_name} not found", file=sys.stderr)
                    continue

                records = load_outputs_csv(csv_path)
                if samples:
                    records = records[:samples]

                for rec in records:
                    sid = rec.get("scenario_id", "")
                    tf_code = rec.get("extracted_code", "").replace("\\n", "\n")

                    # Constraint extraction
                    if dataset == "iac_eval":
                        intent_str = intents.get(sid, "")
                        constraints = parse_intent_field(intent_str) if intent_str else []
                    else:
                        prompt = rec.get("prompt", "").replace("\\n", "\n")
                        # The built prompt wraps the raw request; extract it
                        # by taking text after the last "Request:\n" occurrence
                        match = re.search(r"Request:\s*\n(.*)", prompt, re.DOTALL)
                        raw_prompt = match.group(1).strip() if match else prompt
                        constraints = extract_constraints(raw_prompt)

                    if not constraints:
                        continue

                    ics = compute_ics(tf_code, constraints)
                    if ics is None:
                        continue

                    rows.append({
                        "scenario_id": sid,
                        "model": model,
                        "prompt_type": strategy,
                        "dataset": dataset,
                        "ics": ics,
                        "n_constraints": len(constraints),
                    })

    return rows


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni correction. Returns corrected p-values in input order."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = p * (n - rank)
        running_max = max(running_max, adjusted)
        corrected[orig_idx] = min(running_max, 1.0)
    return corrected


def rank_biserial_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size r = 1 - 2U / (n_x * n_y) for Mann-Whitney U."""
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0 * u_stat / (len(x) * len(y)))


def kruskal_wallis_eta_squared(h: float, k: int, n: int) -> float:
    """η² effect size for Kruskal-Wallis: (H - k + 1) / (n - k)."""
    return (h - k + 1) / (n - k) if n > k else float("nan")


def dunn_posthoc(groups: Dict[str, np.ndarray]) -> Tuple[List[Tuple], List[float]]:
    """
    Dunn's post-hoc test (z-score approximation).
    Returns list of (name_a, name_b, z, raw_p) pairs and raw p-values.
    """
    names = list(groups.keys())
    all_values = np.concatenate(list(groups.values()))
    n = len(all_values)
    # Global ranks (average for ties)
    temp_ranks = stats.rankdata(all_values)
    offset = 0
    group_ranks: Dict[str, np.ndarray] = {}
    for name in names:
        size = len(groups[name])
        group_ranks[name] = temp_ranks[offset : offset + size]
        offset += size

    # Tie correction factor
    _, counts = np.unique(all_values, return_counts=True)
    tie_correction = 1.0 - np.sum(counts**3 - counts) / (n**3 - n)

    pairs = []
    raw_p = []
    for a, b in itertools.combinations(names, 2):
        n_a, n_b = len(groups[a]), len(groups[b])
        mean_rank_a = np.mean(group_ranks[a])
        mean_rank_b = np.mean(group_ranks[b])
        se = math.sqrt(tie_correction * n * (n + 1) / 12.0 * (1.0 / n_a + 1.0 / n_b))
        if se == 0:
            z, p = 0.0, 1.0
        else:
            z = (mean_rank_a - mean_rank_b) / se
            p = float(2.0 * stats.norm.sf(abs(z)))
        pairs.append((a, b, float(z), float(p)))
        raw_p.append(float(p))

    return pairs, raw_p


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI."""
    rng = np.random.default_rng(42)
    samples = [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(B)]
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_tests(ics_rows: List[Dict]) -> Dict:
    results: Dict[str, Any] = {}

    # Organise by (dataset, model): list of ICS values (all strategies pooled)
    by_dataset_model: Dict[str, Dict[str, List[float]]] = {ds: {m: [] for m in MODELS} for ds in DATASETS}
    # Organise by (dataset, model, strategy)
    by_dms: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    for row in ics_rows:
        ds, m, st, ics = row["dataset"], row["model"], row["prompt_type"], row["ics"]
        by_dataset_model[ds][m].append(ics)
        by_dms[(ds, m, st)].append(ics)

    # ------------------------------------------------------------------
    # 1. Kruskal-Wallis across 9 models (per dataset)
    # ------------------------------------------------------------------
    kw_results: Dict[str, Any] = {}
    for ds in DATASETS:
        groups = {m: np.array(by_dataset_model[ds][m]) for m in MODELS if by_dataset_model[ds][m]}
        valid_models = [m for m in MODELS if by_dataset_model[ds][m]]
        group_arrays = [groups[m] for m in valid_models]
        if len(group_arrays) < 2:
            continue
        h_stat, p_val = stats.kruskal(*group_arrays)
        n_total = sum(len(g) for g in group_arrays)
        eta2 = kruskal_wallis_eta_squared(h_stat, len(group_arrays), n_total)
        kw_results[ds] = {
            "H": float(h_stat),
            "df": len(group_arrays) - 1,
            "p_value": float(p_val),
            "eta_squared": float(eta2),
            "n_total": n_total,
            "n_models": len(group_arrays),
        }

    results["kruskal_wallis"] = kw_results

    # ------------------------------------------------------------------
    # 2. Dunn's post-hoc pairwise tests with Holm-Bonferroni correction
    # ------------------------------------------------------------------
    dunn_results: Dict[str, Any] = {}
    for ds in DATASETS:
        groups = {m: np.array(by_dataset_model[ds][m]) for m in MODELS if by_dataset_model[ds][m]}
        pairs, raw_p = dunn_posthoc(groups)
        adj_p = holm_bonferroni(raw_p)
        pair_records = []
        for i, (a, b, z, p_raw) in enumerate(pairs):
            xa, xb = groups[a], groups[b]
            r = rank_biserial_correlation(xa, xb)
            pair_records.append({
                "model_a": a,
                "model_b": b,
                "z": round(z, 4),
                "p_raw": round(p_raw, 6),
                "p_adj_holm": round(adj_p[i], 6),
                "significant": adj_p[i] < 0.05,
                "rank_biserial_r": round(r, 4),
                "n_a": len(xa),
                "n_b": len(xb),
            })
        # Sort by adjusted p-value for readability
        pair_records.sort(key=lambda x: x["p_adj_holm"])
        dunn_results[ds] = pair_records

    results["dunn_posthoc"] = dunn_results

    # ------------------------------------------------------------------
    # 3. Bootstrap 95 % CIs for mean ICS per model per dataset
    # ------------------------------------------------------------------
    ci_results: Dict[str, Any] = {}
    for ds in DATASETS:
        ci_results[ds] = {}
        for m in MODELS:
            vals = np.array(by_dataset_model[ds][m])
            if len(vals) < 10:
                continue
            mean = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals)
            ci_results[ds][m] = {
                "mean_ics": round(mean, 4),
                "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
                "n": len(vals),
            }
    results["bootstrap_ci"] = ci_results

    # ------------------------------------------------------------------
    # 4. Friedman test for prompting-strategy effect (per dataset)
    # ------------------------------------------------------------------
    friedman_results: Dict[str, Any] = {}
    for ds in DATASETS:
        # Need balanced blocks: only scenario_ids present in all 3 strategies for all models
        # Collect ICS per (scenario_id, model) triplet across strategies
        by_sid_model: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        for row in ics_rows:
            if row["dataset"] != ds:
                continue
            key = (row["scenario_id"], row["model"])
            by_sid_model[key][row["prompt_type"]] = row["ics"]

        # Keep only triples with all 3 strategies
        zs, fs, cs = [], [], []
        for key, strats in by_sid_model.items():
            if all(st in strats for st in STRATEGIES):
                zs.append(strats["zero-shot"])
                fs.append(strats["few-shot"])
                cs.append(strats["cot"])

        if len(zs) < 10:
            friedman_results[ds] = {"note": "Insufficient balanced observations"}
            continue

        chi2, p_val = stats.friedmanchisquare(np.array(zs), np.array(fs), np.array(cs))
        n_blocks = len(zs)
        # Kendall's W from Friedman: W = chi2 / (k*(n-1)) where k=3 groups
        kendalls_w = float(chi2 / (3 * (n_blocks - 1)))

        friedman_results[ds] = {
            "chi2": round(float(chi2), 4),
            "df": 2,
            "p_value": round(float(p_val), 6),
            "kendalls_W": round(kendalls_w, 4),
            "n_blocks": n_blocks,
            "mean_ics_zero_shot": round(float(np.mean(zs)), 4),
            "mean_ics_few_shot":  round(float(np.mean(fs)), 4),
            "mean_ics_cot":       round(float(np.mean(cs)), 4),
        }

        # Post-hoc: pairwise Wilcoxon signed-rank with Holm correction
        pairs_w = [("zero-shot", "few-shot", np.array(zs), np.array(fs)),
                   ("zero-shot", "cot",      np.array(zs), np.array(cs)),
                   ("few-shot",  "cot",      np.array(fs), np.array(cs))]
        raw_ps = []
        w_stats = []
        for a, b, xa, xb in pairs_w:
            try:
                w_stat, p = stats.wilcoxon(xa, xb, alternative="two-sided")
            except ValueError:
                w_stat, p = float("nan"), 1.0
            raw_ps.append(float(p))
            w_stats.append(float(w_stat))
        adj_ps = holm_bonferroni(raw_ps)
        friedman_results[ds]["pairwise_wilcoxon"] = [
            {"a": a, "b": b, "W": round(w_stats[i], 2), "p_raw": round(raw_ps[i], 6), "p_adj_holm": round(adj_ps[i], 6), "significant": adj_ps[i] < 0.05}
            for i, (a, b, _, __) in enumerate(pairs_w)
        ]

    results["friedman"] = friedman_results

    # ------------------------------------------------------------------
    # 5. Mann-Whitney U for dataset effect (IaC-Eval vs llm-iac) per model
    # ------------------------------------------------------------------
    dataset_effect: Dict[str, Any] = {}
    for m in MODELS:
        x_ie = np.array(by_dataset_model["iac_eval"][m])
        x_li = np.array(by_dataset_model["llm_iac"][m])
        if len(x_ie) < 5 or len(x_li) < 5:
            continue
        u_stat, p_val = stats.mannwhitneyu(x_ie, x_li, alternative="two-sided")
        r = float(1.0 - 2.0 * u_stat / (len(x_ie) * len(x_li)))
        dataset_effect[m] = {
            "U": round(float(u_stat), 2),
            "p_value": round(float(p_val), 6),
            "rank_biserial_r": round(r, 4),
            "mean_ics_iac_eval": round(float(np.mean(x_ie)), 4),
            "mean_ics_llm_iac":  round(float(np.mean(x_li)), 4),
            "n_iac_eval": len(x_ie),
            "n_llm_iac":  len(x_li),
        }
    results["dataset_effect"] = dataset_effect

    return results


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def generate_latex(results: Dict, ci_results: Dict) -> str:
    lines = []

    # --- Table: Kruskal-Wallis summary ---
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Kruskal-Wallis test of ICS distributions across nine models, "
                 r"per dataset. Effect size $\eta^2 = (H - k + 1)/(n - k)$ where $k$ is the number of "
                 r"groups and $n$ the total observations. All tests are two-sided at $\alpha = 0.05$.}"
                 r"\label{tab:kruskal_wallis}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & $H$ & df & $p$-value & $\eta^2$ & Conclusion \\")
    lines.append(r"\midrule")
    for ds, label in [("iac_eval", "IaC-Eval"), ("llm_iac", "llm-iac")]:
        kw = results["kruskal_wallis"].get(ds, {})
        if not kw:
            continue
        concl = "Reject $H_0$" if kw["p_value"] < 0.05 else "Fail to reject $H_0$"
        lines.append(
            f"{label} & {kw['H']:.2f} & {kw['df']} & {fmt_p(kw['p_value'])} & "
            f"{kw['eta_squared']:.3f} & {concl} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table: Bootstrap CIs for mean ICS ---
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Bootstrap 95\,\% confidence intervals ($B = 2000$ resamples) for "
                 r"mean Intent Coverage Score per model, pooled across prompting strategies. "
                 r"IaC-Eval (left) and llm-iac (right) are evaluated independently.}"
                 r"\label{tab:ics_ci}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrlrrl}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{3}{c}{IaC-Eval} & \multicolumn{3}{c}{llm-iac} \\")
    lines.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    lines.append(r"Model & Mean & \multicolumn{2}{c}{95\,\% CI} & Mean & \multicolumn{2}{c}{95\,\% CI} \\")
    lines.append(r"\midrule")
    ie_cis = ci_results.get("iac_eval", {})
    li_cis = ci_results.get("llm_iac", {})
    for m in MODELS:
        ie = ie_cis.get(m, {})
        li = li_cis.get(m, {})
        ie_str = f"{ie['mean_ics']:.3f} & [{ie['ci95_lo']:.3f}, & {ie['ci95_hi']:.3f}]" if ie else "--- & & "
        li_str = f"{li['mean_ics']:.3f} & [{li['ci95_lo']:.3f}, & {li['ci95_hi']:.3f}]" if li else "--- & & "
        lines.append(f"{MODEL_DISPLAY[m]} & {ie_str} & {li_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table: Friedman test ---
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Friedman test for the effect of prompting strategy on ICS, "
                 r"applied separately to each dataset. $W$ is Kendall's concordance coefficient. "
                 r"Post-hoc pairwise Wilcoxon signed-rank tests use Holm-Bonferroni correction.}"
                 r"\label{tab:friedman}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & Comparison & $\chi^2$ (Friedman) & df & $p$-value & $W$ & Sig. \\")
    lines.append(r"\midrule")
    for ds, label in [("iac_eval", "IaC-Eval"), ("llm_iac", "llm-iac")]:
        fr = results["friedman"].get(ds, {})
        if "note" in fr or not fr:
            lines.append(f"{label} & --- & --- & --- & --- & --- & --- \\\\")
            continue
        lines.append(
            f"{label} & Overall & {fr['chi2']:.2f} & {fr['df']} & {fmt_p(fr['p_value'])} & "
            f"{fr['kendalls_W']:.3f} & {'$\\checkmark$' if fr['p_value'] < 0.05 else ''} \\\\"
        )
        for pw in fr.get("pairwise_wilcoxon", []):
            sig = r"$\checkmark$" if pw["significant"] else ""
            lines.append(
                f" & {pw['a']} vs.\\ {pw['b']} & {pw['W']:.1f} & 1 & {fmt_p(pw['p_adj_holm'])} & & {sig} \\\\"
            )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")
    lines.append("")

    # --- Table: Dataset effect (IaC-Eval vs llm-iac) per model ---
    lines.append(r"\begin{table}[h]")
    lines.append(r"\caption{Mann-Whitney $U$ test for the dataset effect (IaC-Eval vs.\ llm-iac) "
                 r"on ICS, per model (pooled across prompting strategies). "
                 r"$r$ is the rank-biserial correlation effect size. "
                 r"All $p$-values are Holm-Bonferroni adjusted for nine simultaneous tests.}"
                 r"\label{tab:dataset_effect}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & $\bar{\mathrm{ICS}}_{\text{IaC-Eval}}$ & $\bar{\mathrm{ICS}}_{\text{llm-iac}}$ & $U$ & $p$-value & $r$ & Sig. \\")
    lines.append(r"\midrule")
    de = results.get("dataset_effect", {})
    raw_ps_de = [de[m]["p_value"] for m in MODELS if m in de]
    adj_ps_de = holm_bonferroni(raw_ps_de)
    adj_idx = 0
    for m in MODELS:
        if m not in de:
            continue
        d = de[m]
        p_adj = adj_ps_de[adj_idx]
        adj_idx += 1
        sig = r"$\checkmark$" if p_adj < 0.05 else ""
        lines.append(
            f"{MODEL_DISPLAY[m]} & {d['mean_ics_iac_eval']:.3f} & {d['mean_ics_llm_iac']:.3f} & "
            f"{d['U']:.0f} & {fmt_p(p_adj)} & {d['rank_biserial_r']:.3f} & {sig} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=None,
                    help="Limit rows per CSV (for testing)")
    ap.add_argument("--no-cache", action="store_true",
                    help="Recompute ICS even if ics_per_prompt.csv exists")
    args = ap.parse_args()

    ics_csv = RESULTS_DIR / "ics_per_prompt.csv"
    RESULTS_DIR.mkdir(exist_ok=True)

    # Build or load per-prompt ICS table
    if ics_csv.exists() and not args.no_cache:
        print(f"Loading cached ICS table from {ics_csv}", file=sys.stderr)
        ics_rows = []
        csv.field_size_limit(10 * 1024 * 1024)
        with open(ics_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ics_rows.append({
                    "scenario_id": row["scenario_id"],
                    "model": row["model"],
                    "prompt_type": row["prompt_type"],
                    "dataset": row["dataset"],
                    "ics": float(row["ics"]),
                    "n_constraints": int(row["n_constraints"]),
                })
    else:
        print("Computing per-prompt ICS for all model × strategy × dataset combinations...", file=sys.stderr)
        ics_rows = build_ics_table(samples=args.samples)
        with open(ics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scenario_id", "model", "prompt_type", "dataset", "ics", "n_constraints"])
            writer.writeheader()
            writer.writerows(ics_rows)
        print(f"Wrote {len(ics_rows)} rows to {ics_csv}", file=sys.stderr)

    if not ics_rows:
        print("ERROR: No ICS values computed. Check that outputs/ CSVs are present.", file=sys.stderr)
        sys.exit(1)

    print(f"Running statistical tests on {len(ics_rows)} (prompt, model, strategy) observations...", file=sys.stderr)
    results = run_tests(ics_rows)

    # Write JSON
    json_path = RESULTS_DIR / "statistical_tests.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", file=sys.stderr)

    # Write LaTeX
    latex = generate_latex(results, results["bootstrap_ci"])
    tex_path = RESULTS_DIR / "tab_statistical_significance.tex"
    tex_path.write_text(latex, encoding="utf-8")
    print(f"Wrote {tex_path}", file=sys.stderr)

    # Print summary to stdout
    print("\n=== Kruskal-Wallis results ===")
    for ds in DATASETS:
        kw = results["kruskal_wallis"].get(ds, {})
        if kw:
            print(f"  {ds}: H={kw['H']:.2f}, df={kw['df']}, p={kw['p_value']:.6f}, η²={kw['eta_squared']:.4f}")

    print("\n=== Friedman (prompting strategy) ===")
    for ds in DATASETS:
        fr = results["friedman"].get(ds, {})
        if fr and "chi2" in fr:
            print(f"  {ds}: χ²={fr['chi2']:.2f}, p={fr['p_value']:.6f}, W={fr['kendalls_W']:.4f}")

    print("\n=== Dataset effect (IaC-Eval vs llm-iac) per model ===")
    for m in MODELS:
        d = results["dataset_effect"].get(m, {})
        if d:
            print(f"  {MODEL_DISPLAY[m]}: r={d['rank_biserial_r']:.3f}, p={d['p_value']:.6f}")

    print("\n=== Bootstrap 95% CI (IaC-Eval) ===")
    for m in MODELS:
        ci = results["bootstrap_ci"].get("iac_eval", {}).get(m, {})
        if ci:
            print(f"  {MODEL_DISPLAY[m]}: {ci['mean_ics']:.3f} [{ci['ci95_lo']:.3f}, {ci['ci95_hi']:.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
