#!/usr/bin/env python3
"""
Generate IaC Baselines Script

Calls Claude 4.5 Sonnet, Grok 4.1 Fast, and Gemini 3 Flash APIs to generate
Terraform configurations from the IaC-Eval datasets.

Based on prompts from:
- IaC-Eval (NeurIPS 2024): https://huggingface.co/datasets/autoiac-project/iac-eval
- llm-iac.csv: Infrastructure as Code prompts dataset
"""

import argparse
import csv
import json
import os
import re
import time

from pathlib import Path
from typing import Literal
from dotenv import load_dotenv

import anthropic
import openai
from google import genai

load_dotenv(".env")

# =============================================================================
# Prompt Templates
# =============================================================================

# Enhanced system prompt for best-practice Terraform generation
SYSTEM_PROMPT = """You are TerraformAI, an expert AI assistant specialising in generating Terraform Infrastructure as Code. 
Your role is to generate Terraform configurations for cloud infrastructure across AWS, GCP, and Azure based on user queries.

IMPORTANT: Output exactly ONE code block containing all the Terraform/HCL code. Do not split the code into multiple code blocks.
"""

# Few-shot examples from IaC-Eval paper
# Source: https://github.com/autoiac-project/iac-eval/blob/main/evaluation/prompt-templates/few-shot.txt
FEW_SHOT_PROMPT = """Here are a few examples:

Example prompt 1: Create an AWS RDS instance with randomly generated id and password
Example output 1: 
```hcl
resource "random_id" "suffix" {
  byte_length = 4
}

resource "random_password" "db" {
  length  = 16
  special = false
}

resource "aws_db_instance" "test" {
  identifier          = "metricbeat-test-${random_id.suffix.hex}"
  allocated_storage   = 20 // Gigabytes
  engine              = "mysql"
  instance_class      = "db.t2.micro"
  db_name                = "metricbeattest"
  username            = "foo"
  password            = random_password.db.result
  skip_final_snapshot = true // Required for cleanup
}
```

Example prompt 2: Create an 20GB MySQL instance on aws with randomly generated id and password
Example output 2: 
```hcl
resource "random_id" "suffix" {
  byte_length = 4
}

resource "random_password" "db" {
  length  = 16
  special = false
}

resource "aws_db_instance" "test" {
  identifier          = "metricbeat-test-${random_id.suffix.hex}"
  allocated_storage   = 20 // Gigabytes
  engine              = "mysql"
  instance_class      = "db.t2.micro"
  db_name                = "metricbeattest"
  username            = "foo"
  password            = random_password.db.result
  skip_final_snapshot = true // Required for cleanup
}
```

Example prompt 3: create a AWS EFS, and create a replica of an this created EFS file system using regional storage in us-west-3
Example output 3: 
```hcl
resource "aws_efs_file_system" "example" {}

resource "aws_efs_replication_configuration" "example" {
  source_file_system_id = aws_efs_file_system.example.id

  destination {
    availability_zone_name = "us-west-2b"
    kms_key_id             = "1234abcd-12ab-34cd-56ef-1234567890ab"
  }
}
```

Here is the actual prompt to answer:
"""

# Chain-of-Thought prompt suffix from IaC-Eval paper
# Source: https://www.promptingguide.ai/techniques/cot#zero-shot-cot-prompting
COT_SUFFIX = "\n\nLet's think step by step."


# Zero-shot prompt prefix to provide context and formatting instructions
ZERO_SHOT_PREFIX = """Please generate the Terraform configuration for the following request. 
Provide the code in HCL format within a Markdown code block (using ```hcl).
Do not include any additional explanation unless necessary for understanding the code.

Request:
"""


# =============================================================================
# API Clients
# =============================================================================

class ClaudeClient:
    """Claude API client using Anthropic SDK."""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-5-20250929"
        self.name = "claude-4.5-sonnet"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text


class GrokClient:
    """Grok API client using OpenAI-compatible API."""
    
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.model = "grok-4-1-fast-non-reasoning"
        self.name = "grok-4.1-fast"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Grok."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content


class GeminiClient:
    """Gemini API client using Google GenAI SDK."""
    
    def __init__(self):
        self.client = genai.Client()
        self.model = "gemini-3-flash-preview"
        self.name = "gemini-3-flash"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Gemini."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                {"role": "user", "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
            ],
        )
        return response.text


class KimiClient:
    """Kimi API client using OpenAI-compatible API."""
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.ai/v1",
        )
        self.model = "kimi-k2.5"
        self.name = "kimi-k2.5"

    def generate(self, system_prompt: str, user_prompt: str):
        """Generate a response from Kimi."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            extra_body={
                "thinking": {"type": "disabled"}
            },
            max_tokens=32768
        )
        return response.choices[0].message.content


class GLMClient:
    """GLM API client using OpenAI-compatible API (Z.AI)."""
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=os.getenv("ZAI_API_KEY"),
            base_url="https://api.z.ai/api/paas/v4/",
        )
        self.model = "glm-4.7"
        self.name = "glm-4.7"

    def generate(self, system_prompt: str, user_prompt: str):
        """Generate a response from GLM."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.choices[0].message.content


# =============================================================================
# Prompt Builders
# =============================================================================

PromptType = Literal["zero-shot", "few-shot", "cot"]


def build_prompt(user_prompt: str, prompt_type: PromptType) -> str:
    """Build the final user prompt based on the prompt type."""
    if prompt_type == "zero-shot":
        return ZERO_SHOT_PREFIX + user_prompt
    elif prompt_type == "few-shot":
        return FEW_SHOT_PROMPT + user_prompt
    elif prompt_type == "cot":
        return user_prompt + COT_SUFFIX
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


# =============================================================================
# Code Extraction
# =============================================================================

def extract_terraform_code(response: str) -> str:
    """Extract Terraform/HCL code from a response."""
    # Try to find code in markdown code blocks
    patterns = [
        r"```(?:hcl|terraform|tf)\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()  # Return the first code block
    
    # If no code blocks, return the full response
    return response


# =============================================================================
# Dataset Loading
# =============================================================================

def load_iac_eval_dataset(path: Path) -> list[dict]:
    """Load the IaC-Eval dataset."""
    with open(path) as f:
        return json.load(f)


def load_llm_iac_dataset(path: Path) -> list[dict]:
    """Load the llm-iac.csv dataset."""
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append({
                'id': row['ID'],
                'category': row['Category'],
                'cloud_provider': row['Cloud_Provider'],
                'user_query': row['User_Query'],
                'terraform_code': row['Terraform_Code']
            })
    return dataset


# =============================================================================
# Result Saving
# =============================================================================

def get_output_filename(
    dataset_name: str,
    model_name: str,
    prompt_type: str,
) -> str:
    """Generate the output filename for a dataset/model/prompt combination."""
    return f"{dataset_name}_{model_name}_{prompt_type}.csv"


def load_existing_results(output_path: Path) -> list[dict]:
    """Load existing results from a JSON file if it exists."""
    if output_path.exists():
        results = []
        with open(output_path, "r", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Unescape newlines when reading back into memory
                processed_row = {k: unescape_newlines(v) for k, v in row.items()}
                results.append(processed_row)
        return results
    return []


def get_completed_scenario_ids(results: list[dict]) -> set[str]:
    """Extract scenario IDs that have already been processed."""
    return {r["scenario_id"] for r in results}


def escape_newlines(value):
    """Escape newlines in strings for CSV output."""
    if isinstance(value, str):
        return value.replace('\n', '\\n').replace('\r', '')
    return value


def unescape_newlines(value):
    """Unescape newlines in strings from CSV input."""
    if isinstance(value, str):
        return value.replace('\\n', '\n')
    return value


def save_results(output_path: Path, results: list[dict]):
    """Save all results to a CSV file."""
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # distinct keys from all records
    keys = set()
    for entry in results:
        keys.update(entry.keys())
    fieldnames = sorted(list(keys))
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for entry in results:
            processed_entry = {k: escape_newlines(v) for k, v in entry.items()}
            writer.writerow(processed_entry)


# =============================================================================
# Main Generation Logic
# =============================================================================

def format_eta(seconds: float) -> str:
    """Format seconds into a human-readable ETA string."""
    if seconds < 0:
        return "--"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def run_iac_eval(
    dataset: list[dict],
    clients: list,
    prompt_type: PromptType,
    output_dir: Path,
    samples: int | None = None,
):
    """Run generation on IaC-Eval dataset."""
    if samples:
        dataset = dataset[:samples]
    
    print(f"\n{'='*60}")
    print(f"Processing IaC-Eval dataset ({len(dataset)} scenarios)")
    print(f"Prompt type: {prompt_type}")
    print(f"{'='*60}\n")
    
    for client in clients:
        # Load existing results for this client
        output_filename = get_output_filename("iac_eval", client.name, prompt_type)
        output_path = output_dir / output_filename
        results = load_existing_results(output_path)
        completed_ids = get_completed_scenario_ids(results)
        
        print(f"\n--- {client.name} ({len(completed_ids)} already completed) ---")
        
        # ETA tracking
        client_start_time = time.time()
        processed_count = 0
        
        for idx, scenario in enumerate(dataset):
            scenario_id = f"iac_eval_{idx:04d}"
            
            # Skip if already completed
            if scenario_id in completed_ids:
                print(f"[{idx+1}/{len(dataset)}] Skipping {scenario_id} (already exists)")
                continue
            
            base_prompt = scenario["Prompt"]
            reference = scenario.get("Reference output")
            difficulty = scenario.get("Difficulty", "unknown")
            
            # Calculate ETA
            remaining = len(dataset) - idx - 1 - (len(completed_ids) - processed_count)
            if processed_count > 0:
                elapsed = time.time() - client_start_time
                avg_time = elapsed / processed_count
                eta_seconds = remaining * avg_time
                eta_str = f" | ETA: {format_eta(eta_seconds)}"
            else:
                eta_str = " | ETA: calculating..."
            
            print(f"[{idx+1}/{len(dataset)}] Scenario: {scenario_id} (difficulty: {difficulty}){eta_str}")
            print(f"  Prompt: {base_prompt[:80]}...")
            
            user_prompt = build_prompt(base_prompt, prompt_type)
            
            print(f"  Generating with {client.name}...")
            try:
                start_time = time.time()
                response = client.generate(SYSTEM_PROMPT, user_prompt)
                duration = round(time.time() - start_time, 2)
                extracted_code = extract_terraform_code(response)
                
                result = {
                    "dataset": "iac_eval",
                    "scenario_id": scenario_id,
                    "model": client.name,
                    "prompt_type": prompt_type,
                    "prompt": user_prompt,
                    "response": response,
                    "extracted_code": extracted_code,
                    "reference": reference,
                    "generation_time": duration,
                }

                results.append(result)
                processed_count += 1
                
                # Save after each successful generation for resume safety
                save_results(output_path, results)
                print(f"  Saved to: {output_path}")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        total_time = time.time() - client_start_time
        print(f"Completed {client.name}: {len(results)} total results in {output_path} (took {format_eta(total_time)})")


def run_llm_iac(
    dataset: list[dict],
    clients: list,
    prompt_type: PromptType,
    output_dir: Path,
    samples: int | None = None,
):
    """Run generation on llm-iac.csv dataset."""
    if samples:
        dataset = dataset[:samples]
    
    print(f"\n{'='*60}")
    print(f"Processing llm-iac dataset ({len(dataset)} scenarios)")
    print(f"Prompt type: {prompt_type}")
    print(f"{'='*60}\n")
    
    for client in clients:
        # Load existing results for this client
        output_filename = get_output_filename("llm_iac", client.name, prompt_type)
        output_path = output_dir / output_filename
        results = load_existing_results(output_path)
        completed_ids = get_completed_scenario_ids(results)
        
        print(f"\n--- {client.name} ({len(completed_ids)} already completed) ---")
        
        # ETA tracking
        client_start_time = time.time()
        processed_count = 0
        
        for idx, scenario in enumerate(dataset):
            scenario_id = f"llm_iac_{scenario['id']}"
            
            # Skip if already completed
            if scenario_id in completed_ids:
                print(f"[{idx+1}/{len(dataset)}] Skipping {scenario_id} (already exists)")
                continue
            
            base_prompt = scenario['user_query']
            reference = scenario['terraform_code']
            category = scenario.get('category', 'unknown')
            cloud_provider = scenario.get('cloud_provider', 'unknown')
            
            # Calculate ETA
            remaining = len(dataset) - idx - 1 - (len(completed_ids) - processed_count)
            if processed_count > 0:
                elapsed = time.time() - client_start_time
                avg_time = elapsed / processed_count
                eta_seconds = remaining * avg_time
                eta_str = f" | ETA: {format_eta(eta_seconds)}"
            else:
                eta_str = " | ETA: calculating..."
            
            print(f"[{idx+1}/{len(dataset)}] Scenario: {scenario_id} ({cloud_provider} - {category}){eta_str}")
            print(f"  Prompt: {base_prompt[:80]}...")
            
            user_prompt = build_prompt(base_prompt, prompt_type)
            
            print(f"  Generating with {client.name}...")
            try:
                start_time = time.time()
                response = client.generate(SYSTEM_PROMPT, user_prompt)
                duration = round(time.time() - start_time, 2)
                extracted_code = extract_terraform_code(response)
                
                result = {
                    "dataset": "llm_iac",
                    "scenario_id": scenario_id,
                    "model": client.name,
                    "prompt_type": prompt_type,
                    "prompt": user_prompt,
                    "response": response,
                    "extracted_code": extracted_code,
                    "reference": reference,
                    "generation_time": duration,
                }

                results.append(result)
                processed_count += 1
                
                # Save after each successful generation for resume safety
                save_results(output_path, results)
                print(f"  Saved to: {output_path}")
                
            except Exception as e:
                print(f"    Error: {e}")
        
        total_time = time.time() - client_start_time
        print(f"Completed {client.name}: {len(results)} total results in {output_path} (took {format_eta(total_time)})")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate IaC baselines using Claude, Grok, and Gemini APIs"
    )
    parser.add_argument(
        "--prompt-type",
        choices=["zero-shot", "few-shot", "cot"],
        default="zero-shot",
        help="Type of prompting strategy to use (default: zero-shot)"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="claude,grok,gemini,kimi,glm",
        help="Comma-separated list of models to use (default: claude,grok,gemini,kimi,glm)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process per dataset (default: all)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path("outputs")
    
    # Initialize clients based on selected models
    model_list = [m.strip().lower() for m in args.models.split(",")]
    clients = []
    
    if "claude" in model_list:
        print("Initializing Claude client...")
        clients.append(ClaudeClient())
    
    if "grok" in model_list:
        print("Initializing Grok client...")
        clients.append(GrokClient())
    
    if "gemini" in model_list:
        print("Initializing Gemini client...")
        clients.append(GeminiClient())
    
    if "kimi" in model_list:
        print("Initializing Kimi client...")
        clients.append(KimiClient())
    
    if "glm" in model_list:
        print("Initializing GLM client...")
        clients.append(GLMClient())
    
    if not clients:
        print("Error: No valid models specified")
        return
    
    # Load datasets
    iac_eval_path = Path("datasets/iac_eval_dataset.json")
    llm_iac_path = Path("datasets/llm-iac.csv")
    
    if iac_eval_path.exists():
        iac_eval_dataset = load_iac_eval_dataset(iac_eval_path)
        run_iac_eval(
            dataset=iac_eval_dataset,
            clients=clients,
            prompt_type=args.prompt_type,
            output_dir=output_dir,
            samples=args.samples,
        )
    else:
        print(f"Warning: {iac_eval_path} not found, skipping IaC-Eval")
    
    if llm_iac_path.exists():
        llm_iac_dataset = load_llm_iac_dataset(llm_iac_path)
        run_llm_iac(
            dataset=llm_iac_dataset,
            clients=clients,
            prompt_type=args.prompt_type,
            output_dir=output_dir,
            samples=args.samples,
        )
    else:
        print(f"Warning: {llm_iac_path} not found, skipping llm-iac")
    
    print("\n" + "="*60)
    print("Generation complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
