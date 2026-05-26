#!/usr/bin/env python3
"""
Semantic Robustness Score (SRS) Pipeline

Flow:
  1. For each prompt, generate 1 paraphrase via OpenRouter (gpt-5.4).
     Paraphrases are cached to datasets/{dataset}_paraphrases.json.
  2. Use generate_baselines.py --paraphrases to generate Terraform for the
     paraphrased prompts (separate step, or run automatically here).
  3. Look up the original Terraform from the existing outputs/ CSVs.
  4. Look up the paraphrase Terraform from outputs/{dataset}_paraphrased_*.csv.
  5. Extract constraints from the original prompt once (rule-based).
     For iac_eval entries with an 'Intent' field, parse that directly.
  6. Compute ICS for original Terraform and paraphrase Terraform
     against those same constraints → 2 ICS values.
  7. SRS = max(0, min(1, 1 - stddev([ics_orig, ics_para])))
  8. Write results/final_results.json (idempotent, incremental saves).

Usage:
  uv run srs.py [OPTIONS]

Options:
  --api-key    OpenRouter API key (or set OPENROUTER_API_KEY env var)
  --model      Model to look up in outputs/ (default: claude-4.5-sonnet)
  --prompt-type  Prompt type to look up in outputs/ (default: few-shot)
  --samples    Limit entries per dataset
  --dry-run    Skip API calls, use dummy paraphrases
  --force      Ignore all caches and reprocess from scratch
  --paraphrases-only  Only generate paraphrases (step 1); don't compute ICS/SRS
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Retry decorator
# =============================================================================

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(f"Failed after {retries} retries: {e}")
                        raise
                    sleep = backoff_in_seconds * 2 ** x
                    logger.warning(f"Error: {e}. Retrying in {sleep}s...")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

# =============================================================================
# Rule-based ICS
# =============================================================================

RESOURCE_MAP = {
    # AWS compute
    "s3 bucket": "aws_s3_bucket",
    "s3": "aws_s3_bucket",
    "vpc": "aws_vpc",
    "subnet": "aws_subnet",
    "ec2": "aws_instance",
    "security group": "aws_security_group",
    "lambda": "aws_lambda_function",
    "load balancer": "aws_lb",
    "application load balancer": "aws_lb",
    "elb": "aws_lb",
    "alb": "aws_lb",
    "nlb": "aws_lb",
    "autoscaling": "aws_autoscaling_group",
    "auto scaling": "aws_autoscaling_group",
    # AWS data
    "rds": "aws_db_instance",
    "database": "aws_db_instance",
    "db instance": "aws_db_instance",
    "dynamodb": "aws_dynamodb_table",
    "elasticache": "aws_elasticache_cluster",
    "redshift": "aws_redshift_cluster",
    # AWS networking
    "route 53": "aws_route53_zone",
    "route53": "aws_route53_zone",
    "dns record": "aws_route53_record",
    "nat gateway": "aws_nat_gateway",
    "internet gateway": "aws_internet_gateway",
    "igw": "aws_internet_gateway",
    "route table": "aws_route_table",
    "transit gateway": "aws_ec2_transit_gateway",
    "vpn": "aws_vpn_gateway",
    "peering": "aws_vpc_peering_connection",
    # AWS monitoring/management
    "cloudwatch": "aws_cloudwatch_log_group",
    "cloudtrail": "aws_cloudtrail",
    "ssm": "aws_ssm_parameter",
    "parameter store": "aws_ssm_parameter",
    "systems manager": "aws_ssm_parameter",
    # AWS containers
    "eks": "aws_eks_cluster",
    "ecs": "aws_ecs_cluster",
    "ecr": "aws_ecr_repository",
    "fargate": "aws_ecs_cluster",
    # AWS messaging/streaming
    "sqs": "aws_sqs_queue",
    "sns": "aws_sns_topic",
    "kinesis": "aws_kinesis_stream",
    "eventbridge": "aws_cloudwatch_event_rule",
    "event bridge": "aws_cloudwatch_event_rule",
    "msk": "aws_msk_cluster",
    # AWS IAM/security
    "iam role": "aws_iam_role",
    "iam policy": "aws_iam_policy",
    "iam user": "aws_iam_user",
    "iam group": "aws_iam_group",
    "kms": "aws_kms_key",
    "waf": "aws_wafv2_web_acl",
    "secret": "aws_secretsmanager_secret",
    "secrets manager": "aws_secretsmanager_secret",
    "certificate": "aws_acm_certificate",
    # AWS application
    "elastic beanstalk": "aws_elastic_beanstalk_environment",
    "beanstalk": "aws_elastic_beanstalk_environment",
    "cloudfront": "aws_cloudfront_distribution",
    "api gateway": "aws_api_gateway_rest_api",
    "step function": "aws_sfn_state_machine",
    "sfn": "aws_sfn_state_machine",
    # AWS analytics
    "emr": "aws_emr_cluster",
    "glue": "aws_glue_job",
    "athena": "aws_athena_database",
    # AWS devops
    "codepipeline": "aws_codepipeline",
    "codecommit": "aws_codecommit_repository",
    "codebuild": "aws_codebuild_project",
    "cloudformation": "aws_cloudformation_stack",
    # GCP
    "gcp": "google_compute_instance",
    "google cloud": "google_compute_instance",
    "gke": "google_container_cluster",
    "cloud run": "google_cloud_run_service",
    "bigquery": "google_bigquery_dataset",
    "cloud storage": "google_storage_bucket",
    "bigtable": "google_bigtable_instance",
    "pub/sub": "google_pubsub_topic",
    "pubsub": "google_pubsub_topic",
    "cloud sql": "google_sql_database_instance",
    "cloud function": "google_cloudfunctions_function",
    "cloud dns": "google_dns_managed_zone",
    "cloud nat": "google_compute_router_nat",
    # Azure
    "azure": "azurerm_virtual_machine",
    "azure vm": "azurerm_virtual_machine",
    "azure storage": "azurerm_storage_account",
    "azure sql": "azurerm_sql_server",
    "azure function": "azurerm_function_app",
    "aks": "azurerm_kubernetes_cluster",
    "azure kubernetes": "azurerm_kubernetes_cluster",
    "network watcher": "azurerm_network_watcher",
    "azure firewall": "azurerm_firewall",
}

# Implicit property/behaviour terms present in prompts but not named as resource types.
# Each maps to a constraint type that verifies the property in generated Terraform.
_IMPLICIT_CONSTRAINTS = [
    # (keyword_in_prompt, constraint_dict)
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


def extract_constraints(prompt: str) -> List[Dict[str, Any]]:
    """Rule-based extraction of constraints from a free-form natural-language prompt."""
    constraints = []
    seen_resources = set()
    p_lower = prompt.lower()

    # 1. Resource existence — longest-match first prevents "s3" shadowing "s3 bucket"
    for keyword in sorted(RESOURCE_MAP, key=len, reverse=True):
        if keyword in p_lower:
            rtype = RESOURCE_MAP[keyword]
            if rtype not in seen_resources:
                seen_resources.add(rtype)
                constraints.append({"type": "resource_exists", "resource_type": rtype})

    # 2. Instance type: t3.micro, m5.large, c5.2xlarge, r5.4xlarge, g4dn.xlarge, etc.
    match = re.search(r"\b([a-z][0-9][a-z]{0,3}(?:dn|en)?\.[0-9]{0,2}(?:nano|micro|small|medium|large|xlarge|metal))", p_lower)
    if match:
        constraints.append({
            "type": "property_equals",
            "resource_type": "aws_instance",
            "property": "instance_type",
            "value": match.group(1),
        })

    # 3. Provider region (AWS, GCP, Azure region strings)
    match = re.search(r"(us|eu|ap|sa|ca|me|af|us-central|us-east|us-west|europe-west|asia-east)-[a-z]+-[0-9]", p_lower)
    if match:
        constraints.append({"type": "provider_region", "value": match.group(0)})

    # 4. Private S3 ACL
    if "private s3" in p_lower or "private bucket" in p_lower:
        constraints.append({
            "type": "property_equals",
            "resource_type": "aws_s3_bucket",
            "property": "acl",
            "value": "private",
        })

    # 5. Security ingress ports
    if "ssh" in p_lower:
        constraints.append({"type": "security_ingress", "port": 22})
    if "https" in p_lower:
        constraints.append({"type": "security_ingress", "port": 443})
    elif "http" in p_lower:
        constraints.append({"type": "security_ingress", "port": 80})

    # 6. Autoscaling bounds — "scale from N to M" or "min N, max M"
    match = re.search(r"(\d+)\s+to\s+(\d+)", p_lower)
    if match:
        constraints.append({"type": "autoscaling_min", "value": int(match.group(1))})
        constraints.append({"type": "autoscaling_max", "value": int(match.group(2))})

    # 7. Implicit property constraints (encryption, HA, monitoring, logging, etc.)
    seen_implicit = set()
    for keyword, constraint in _IMPLICIT_CONSTRAINTS:
        if keyword in p_lower and keyword not in seen_implicit:
            seen_implicit.add(keyword)
            constraints.append(dict(constraint))

    return constraints


# Patterns used to capture attribute-existence and attribute-equality constraints
# from IaC-Eval Intent "with ..." lines.  Order matters: equality checked first.
_INTENT_ATTR_EQUALS = re.compile(
    r'with\s+(?:a specified |one |multiple |an? )?\"([^\"]+)\"\s+equal\s+to\s+\"([^\"]+)\"'
)
_INTENT_ATTR_EXISTS = re.compile(
    r'with\s+(?:a specified |one |multiple |an? )?\"([^\"]+)\"'
)
# Lines that express cross-resource references or nested-block constraints —
# too complex for regex satisfaction checking; we skip them.
_INTENT_SKIP = re.compile(r"referencing|enabling|block\s+that\s+contains|block\s+with")


def parse_intent_field(intent_str: str) -> List[Dict[str, Any]]:
    """
    Parse an IaC-Eval Intent field into structured constraints.

    Captures three constraint types:
      resource_exists  — from "Has [one|multiple] \"type\" resource" lines
      attribute_equals — from "with \"attr\" equal to \"value\"" lines
      attribute_exists — from "with \"attr\"" lines (simple attribute presence)

    Lines containing cross-resource references, block-contains clauses, or
    policy-enabling clauses are not capturable by regex and are skipped.
    Falls back to extract_constraints(intent_str) when no resource lines are found.
    """
    constraints: List[Dict[str, Any]] = []
    current_resource: Optional[str] = None

    for raw_line in intent_str.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Resource existence line
        if line.startswith("Has") and "resource" in line:
            m = re.search(r'"([^"]+)"', line)
            if m:
                current_resource = m.group(1)
                constraints.append({
                    "type": "resource_exists",
                    "resource_type": current_resource,
                })
            continue

        # Property constraint line ("with ...")
        if line.startswith("with") and current_resource:
            # Skip lines that require cross-resource or nested-block semantics
            if _INTENT_SKIP.search(line):
                continue

            # attribute equals value
            m = _INTENT_ATTR_EQUALS.search(line)
            if m:
                constraints.append({
                    "type": "attribute_equals",
                    "resource_type": current_resource,
                    "attribute": m.group(1),
                    "value": m.group(2),
                })
                continue

            # attribute existence
            m = _INTENT_ATTR_EXISTS.search(line)
            if m:
                constraints.append({
                    "type": "attribute_exists",
                    "resource_type": current_resource,
                    "attribute": m.group(1),
                })

    return constraints if constraints else extract_constraints(intent_str)


def _resource_block_content(tf_code: str, resource_type: str) -> str:
    """
    Return the concatenated body text of all resource blocks whose type matches
    resource_type.  Uses a simple brace-counting approach that handles one level
    of nested blocks reliably for standard Terraform patterns.
    """
    out: List[str] = []
    pattern = re.compile(
        r'resource\s+"' + re.escape(resource_type) + r'"\s+"[^"]*"\s*\{',
    )
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
    """Extract facts from Terraform code using regex."""
    facts: Dict[str, Any] = {
        "resources": set(),
        "properties": {},
        "region": None,
        "ports": set(),
        "_raw": tf_code,  # kept for attribute_exists / implicit_property checks
    }
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
        # Check that the attribute key appears inside the correct resource block.
        block = _resource_block_content(facts.get("_raw", ""), constraint["resource_type"])
        attr = re.escape(constraint["attribute"])
        return bool(re.search(r'\b' + attr + r'\s*=', block))

    if ctype == "attribute_equals":
        block = _resource_block_content(facts.get("_raw", ""), constraint["resource_type"])
        attr = re.escape(constraint["attribute"])
        val = re.escape(constraint["value"])
        return bool(re.search(r'\b' + attr + r'\s*=\s*"' + val + r'"', block))

    if ctype == "implicit_property":
        # Any of the HCL patterns matching anywhere in the file suffices.
        raw = facts.get("_raw", "")
        return any(re.search(p, raw, re.I) for p in constraint["hcl_patterns"])

    return False


def compute_ics(tf_code: str, constraints: List[Dict[str, Any]]) -> Optional[float]:
    if not constraints:
        return None
    facts = terraform_to_facts(tf_code)
    satisfied = sum(1 for c in constraints if is_satisfied(c, facts))
    return satisfied / len(constraints)


def compute_srs(ics_values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in ics_values if v is not None]
    if len(valid) < 2:
        return None
    mean = sum(valid) / len(valid)
    variance = sum((x - mean) ** 2 for x in valid) / (len(valid) - 1)
    return max(0.0, min(1.0, 1.0 - math.sqrt(variance)))

# =============================================================================
# Paraphrase generation
# =============================================================================

@retry_with_backoff(retries=3)
def _call_openrouter(prompt: str, api_key: str) -> str:
    """Call OpenRouter to get 1 paraphrase. Returns the paraphrase string."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    system = (
        "You are a helpful assistant. Rewrite the user's prompt in a single, "
        "semantically equivalent but lexically different way. "
        "Output ONLY the rewritten prompt with no extra commentary."
    )
    data = {
        "model": "openai/gpt-5.4",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=data,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def generate_paraphrase(prompt: str, api_key: str, dry_run: bool) -> str:
    if dry_run:
        return f"[DRY RUN paraphrase of: {prompt[:50].replace(chr(10), ' ')}]"
    return _call_openrouter(prompt, api_key)

# =============================================================================
# Load original Terraform from outputs/
# =============================================================================

def load_outputs_index(outputs_dir: Path, model: str, prompt_type: str) -> Dict[str, str]:
    """
    Build a lookup {original_prompt -> extracted_code} from all relevant CSVs
    in outputs/ that match the given model and prompt_type.
    Reads both iac_eval_* and llm_iac_* files.
    """
    index: Dict[str, str] = {}
    csv.field_size_limit(10 * 1024 * 1024)
    suffix = f"_{model}_{prompt_type}.csv"
    for csv_path in outputs_dir.glob(f"*{suffix}"):
        if "paraphrased" in csv_path.name:
            continue  # skip paraphrase outputs here
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # The 'prompt' column in outputs/ CSV is the *built* prompt (includes few-shot prefix etc.)
                # We need to match against the raw base prompt.
                # The raw prompt is buried inside the built prompt; we can't easily extract it.
                # So we also store by scenario_id and match later. For now store by full prompt.
                raw_prompt = row.get("prompt", "").replace("\\n", "\n")
                code = row.get("extracted_code", "").replace("\\n", "\n")
                if raw_prompt and code:
                    index[raw_prompt] = code
    return index


def load_paraphrase_outputs_index(
    outputs_dir: Path, dataset_name: str, model: str, prompt_type: str
) -> Dict[str, str]:
    """
    Build a lookup {original_prompt -> paraphrase_extracted_code} from
    outputs/{dataset}_paraphrased_{model}_{prompt_type}.csv.
    The CSV has an 'original_prompt' column added by run_paraphrases().
    """
    index: Dict[str, str] = {}
    csv_path = outputs_dir / f"{dataset_name}_paraphrased_{model}_{prompt_type}.csv"
    if not csv_path.exists():
        return index
    csv.field_size_limit(10 * 1024 * 1024)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig = row.get("original_prompt", "").replace("\\n", "\n")
            code = row.get("extracted_code", "").replace("\\n", "\n")
            if orig and code:
                index[orig] = code
    return index

# =============================================================================
# Caching helpers
# =============================================================================

def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return default


def _save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class ParaphraseCache:
    """Stores {prompt -> paraphrase} mapping, keyed by original prompt text."""

    def __init__(self, path: Path, force: bool = False):
        if force and path.exists():
            path.unlink()
        self._path = path
        self._data: List[Dict] = _load_json(path, [])
        self._lookup: Dict[str, Dict] = {
            e["prompt"]: e for e in self._data if "prompt" in e and "paraphrase" in e
        }

    def get(self, prompt: str) -> Optional[str]:
        entry = self._lookup.get(prompt)
        return entry["paraphrase"] if entry else None

    def add(self, prompt: str, paraphrase: str, dataset: str, reference: str = ""):
        entry = {
            "prompt": prompt,
            "paraphrase": paraphrase,
            "dataset": dataset,
            "reference": reference,  # original dataset reference TF (for evaluate_outputs)
        }
        self._data.append(entry)
        self._lookup[prompt] = entry
        _save_json(self._path, self._data)


class ResultsStore:
    """Stores final SRS results, incremental saves, keyed by (dataset, prompt)."""

    def __init__(self, path: Path, force: bool = False):
        if force and path.exists():
            path.unlink()
        self._path = path
        _default = {
            "datasets": {
                "iac_eval": {"mean_srs": None, "mean_ics": None, "entries": []},
                "llm_iac": {"mean_srs": None, "mean_ics": None, "entries": []},
            },
            "overall": {"mean_srs": None, "mean_ics": None},
        }
        self.data = _load_json(path, _default)
        for ds in ("iac_eval", "llm_iac"):
            self.data["datasets"].setdefault(ds, {"mean_srs": None, "mean_ics": None, "entries": []})

        self._seen: set = set()
        for ds in ("iac_eval", "llm_iac"):
            for e in self.data["datasets"][ds].get("entries", []):
                self._seen.add((ds, e["prompt"]))

    def has(self, dataset: str, prompt: str) -> bool:
        return (dataset, prompt) in self._seen

    def add(self, dataset: str, entry: Dict):
        self.data["datasets"][dataset]["entries"].append(entry)
        self._seen.add((dataset, entry["prompt"]))
        self._recompute()
        _save_json(self._path, self.data)

    def _recompute(self):
        all_srs, all_ics = [], []
        for ds in ("iac_eval", "llm_iac"):
            entries = self.data["datasets"][ds]["entries"]
            srs_vals = [e["srs"] for e in entries if e.get("srs") is not None]
            ics_vals = [e["mean_ics"] for e in entries if e.get("mean_ics") is not None]
            self.data["datasets"][ds]["mean_srs"] = (sum(srs_vals) / len(srs_vals)) if srs_vals else None
            self.data["datasets"][ds]["mean_ics"] = (sum(ics_vals) / len(ics_vals)) if ics_vals else None
            all_srs += srs_vals
            all_ics += ics_vals
        self.data["overall"]["mean_srs"] = (sum(all_srs) / len(all_srs)) if all_srs else None
        self.data["overall"]["mean_ics"] = (sum(all_ics) / len(all_ics)) if all_ics else None

# =============================================================================
# Core pipeline
# =============================================================================

def process_dataset(
    dataset_name: str,
    records: List[Dict],  # [{"prompt", "reference"?, "intent"?}]
    paraphrase_cache: ParaphraseCache,
    results_store: ResultsStore,
    orig_tf_index: Dict[str, str],      # prompt -> original generated TF
    para_tf_index: Dict[str, str],      # original_prompt -> paraphrase generated TF
    api_key: str,
    dry_run: bool,
    samples: Optional[int],
):
    if samples:
        records = records[:samples]

    logger.info(f"=== Dataset '{dataset_name}' — {len(records)} records ===")

    for idx, record in enumerate(records):
        prompt = record["prompt"]
        prefix = f"[{idx+1}/{len(records)}]"

        if results_store.has(dataset_name, prompt):
            logger.info(f"{prefix} Already done, skipping.")
            continue

        logger.info(f"{prefix} {prompt[:80]}...")

        # 1. Extract constraints from original prompt (once)
        if dataset_name == "iac_eval" and record.get("intent"):
            constraints = parse_intent_field(record["intent"])
        else:
            constraints = extract_constraints(prompt)

        if not constraints:
            logger.warning(f"{prefix}   No constraints extracted — skipping.")
            continue

        # 2. Look up original generated Terraform
        # The outputs/ CSV prompt column contains the built prompt (with few-shot prefix etc.)
        # so we do a substring search: find the built-prompt whose suffix matches our base prompt.
        orig_tf = None
        for built_prompt, code in orig_tf_index.items():
            if prompt in built_prompt or built_prompt.endswith(prompt):
                orig_tf = code
                break

        if not orig_tf:
            logger.warning(f"{prefix}   No original Terraform found in outputs/ — skipping.")
            continue

        # 3. Generate / retrieve 1 paraphrase
        paraphrase = paraphrase_cache.get(prompt)
        if not paraphrase:
            logger.info(f"{prefix}   Generating paraphrase...")
            try:
                paraphrase = generate_paraphrase(prompt, api_key, dry_run)
                paraphrase_cache.add(prompt, paraphrase, dataset_name)
            except Exception as e:
                logger.error(f"{prefix}   Paraphrase generation failed: {e} — skipping.")
                continue

        # 4. Look up paraphrase Terraform
        para_tf = para_tf_index.get(prompt)
        if not para_tf:
            logger.warning(
                f"{prefix}   No paraphrase Terraform found in outputs/ — "
                "run: uv run generate_baselines.py --paraphrases datasets/{dataset_name}_paraphrases.json"
            )
            continue

        # 5. Compute 2 ICS values
        ics_orig = compute_ics(orig_tf, constraints)
        ics_para = compute_ics(para_tf, constraints)

        if ics_orig is None or ics_para is None:
            logger.warning(f"{prefix}   ICS computation failed (None) — skipping.")
            continue

        ics_values = [ics_orig, ics_para]
        mean_ics = sum(ics_values) / 2
        mean_sq = sum((v - mean_ics) ** 2 for v in ics_values)
        ics_stddev = math.sqrt(mean_sq / 1)  # ddof=1, n=2
        srs = compute_srs(ics_values)

        # 6. Store result
        entry = {
            "prompt": prompt,
            "paraphrase": paraphrase,
            "ics_values": ics_values,
            "mean_ics": mean_ics,
            "ics_stddev": ics_stddev,
            "srs": srs,
        }
        results_store.add(dataset_name, entry)
        logger.info(f"{prefix}   ICS=[{ics_orig:.3f}, {ics_para:.3f}]  SRS={srs:.3f}")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="Compute Semantic Robustness Score (SRS)")
    p.add_argument("--api-key", help="OpenRouter API key (overrides OPENROUTER_API_KEY env var)")
    p.add_argument("--model", default="claude-4.5-sonnet",
                   help="Model name to look up in outputs/ (default: claude-4.5-sonnet)")
    p.add_argument("--prompt-type", default="few-shot",
                   choices=["zero-shot", "few-shot", "cot"],
                   help="Prompt type to look up in outputs/ (default: few-shot)")
    p.add_argument("--samples", type=int, default=None,
                   help="Limit entries per dataset")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip API calls, use dummy paraphrases")
    p.add_argument("--force", action="store_true",
                   help="Ignore all caches and reprocess from scratch")
    p.add_argument("--paraphrases-only", action="store_true",
                   help="Only generate and cache paraphrases; skip ICS/SRS computation")
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("OPENROUTER_API_KEY is required (or use --api-key / --dry-run).")
        sys.exit(1)

    root = Path(__file__).resolve().parent
    datasets_dir = root / "datasets"
    outputs_dir = root / "outputs"
    results_dir = root / "results"

    iac_eval_para_path = datasets_dir / "iac_eval_paraphrases.json"
    llm_iac_para_path = datasets_dir / "llm_iac_paraphrases.json"
    final_results_path = results_dir / "final_results.json"

    iac_para_cache = ParaphraseCache(iac_eval_para_path, args.force)
    llm_para_cache = ParaphraseCache(llm_iac_para_path, args.force)
    results_store = ResultsStore(final_results_path, args.force)

    # Load original Terraform outputs lookup
    orig_tf_index = load_outputs_index(outputs_dir, args.model, args.prompt_type)
    logger.info(f"Loaded {len(orig_tf_index)} original Terraform entries from outputs/")

    # --- iac_eval ---
    iac_eval_path = datasets_dir / "iac_eval_dataset.json"
    if iac_eval_path.exists():
        logger.info("Loading iac_eval_dataset.json")
        raw = json.loads(iac_eval_path.read_text(encoding="utf-8"))
        iac_eval_records = [
            {
                "prompt": r["Prompt"],
                "intent": r.get("Intent"),
                "reference": r.get("Reference output", ""),
            }
            for r in raw
            if "Prompt" in r
        ]

        # Phase 1: paraphrases
        _paraphrase_phase(
            "iac_eval", iac_eval_records, iac_para_cache, api_key, args.dry_run, args.samples
        )

        if not args.paraphrases_only:
            para_tf_index = load_paraphrase_outputs_index(
                outputs_dir, "iac_eval", args.model, args.prompt_type
            )
            logger.info(f"Loaded {len(para_tf_index)} paraphrase Terraform entries (iac_eval)")
            process_dataset(
                "iac_eval", iac_eval_records,
                iac_para_cache, results_store,
                orig_tf_index, para_tf_index,
                api_key, args.dry_run, args.samples,
            )
    else:
        logger.warning(f"{iac_eval_path} not found.")

    # --- llm_iac ---
    llm_iac_path = datasets_dir / "llm-iac.csv"
    if llm_iac_path.exists():
        logger.info("Loading llm-iac.csv")
        csv.field_size_limit(10 * 1024 * 1024)
        llm_iac_records = []
        with open(llm_iac_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if "User_Query" in row:
                    llm_iac_records.append({
                        "prompt": row["User_Query"],
                        "reference": row.get("Terraform_Code", ""),
                    })

        _paraphrase_phase(
            "llm_iac", llm_iac_records, llm_para_cache, api_key, args.dry_run, args.samples
        )

        if not args.paraphrases_only:
            para_tf_index = load_paraphrase_outputs_index(
                outputs_dir, "llm_iac", args.model, args.prompt_type
            )
            logger.info(f"Loaded {len(para_tf_index)} paraphrase Terraform entries (llm_iac)")
            process_dataset(
                "llm_iac", llm_iac_records,
                llm_para_cache, results_store,
                orig_tf_index, para_tf_index,
                api_key, args.dry_run, args.samples,
            )
    else:
        logger.warning(f"{llm_iac_path} not found.")

    if args.paraphrases_only:
        logger.info(
            "Paraphrase phase complete.\n"
            "Next step — generate Terraform for paraphrases:\n"
            f"  uv run generate_baselines.py --paraphrases datasets/iac_eval_paraphrases.json "
            f"--models {args.model} --prompt-type {args.prompt_type}\n"
            f"  uv run generate_baselines.py --paraphrases datasets/llm_iac_paraphrases.json "
            f"--models {args.model} --prompt-type {args.prompt_type}"
        )
    else:
        logger.info(f"Pipeline complete. Results → {final_results_path}")


def _paraphrase_phase(
    dataset_name: str,
    records: List[Dict],
    cache: ParaphraseCache,
    api_key: Optional[str],
    dry_run: bool,
    samples: Optional[int],
):
    """Generate and cache 1 paraphrase per prompt (idempotent)."""
    if samples:
        records = records[:samples]
    logger.info(f"[{dataset_name}] Paraphrase phase — {len(records)} records")
    for idx, record in enumerate(records):
        prompt = record["prompt"]
        if cache.get(prompt):
            continue
        try:
            para = generate_paraphrase(prompt, api_key, dry_run)
            cache.add(
                prompt, para, dataset_name,
                reference=record.get("reference", ""),  # pass through for evaluate_outputs
            )
            logger.info(f"  [{idx+1}/{len(records)}] Paraphrase cached.")
        except Exception as e:
            logger.error(f"  [{idx+1}/{len(records)}] Paraphrase failed: {e}")


if __name__ == "__main__":
    main()
