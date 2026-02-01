#!/usr/bin/env python3
"""
Generate IaC Baselines Script

Calls Claude 4.5 Sonnet, Grok 4.1 Fast, and Gemini 3 Pro APIs to generate
Terraform configurations from the IaC-Eval datasets.

Based on prompts from:
- IaC-Eval (NeurIPS 2024): https://huggingface.co/datasets/autoiac-project/iac-eval
- Multi-IaC-Eval: https://huggingface.co/datasets/AmazonScience/Multi-IaC-Eval
"""

import argparse
import json
import os
import re
from datetime import datetime
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

# System prompt from IaC-Eval paper
# Source: https://github.com/autoiac-project/iac-eval/blob/main/evaluation/prompt-templates/system-prompt.txt
SYSTEM_PROMPT = """You are TerraformAI, an AI agent that builds and deploys Cloud Infrastructure written in Terraform HCL. Generate a description of the Terraform program you will define, followed by a single Terraform HCL program in response"""

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
        self.model = "gemini-3-pro-preview"
        self.name = "gemini-3-pro"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Gemini."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )
        return response.text


# =============================================================================
# Prompt Builders
# =============================================================================

PromptType = Literal["zero-shot", "few-shot", "cot"]


def build_prompt(user_prompt: str, prompt_type: PromptType) -> str:
    """Build the final user prompt based on the prompt type."""
    if prompt_type == "zero-shot":
        return user_prompt
    elif prompt_type == "few-shot":
        return FEW_SHOT_PROMPT + user_prompt
    elif prompt_type == "cot":
        return user_prompt + COT_SUFFIX
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def build_multi_iac_prompt(initial_template: str, utterance: str, template_type: str) -> str:
    """Build prompt for Multi-IaC-Eval modification tasks."""
    return f"""You are given an existing {template_type} template and a modification request.
Apply the requested changes to the template and return the complete updated template.

## Current Template:
```
{initial_template}
```

## Modification Request:
{utterance}

## Updated Template:
"""


# =============================================================================
# Code Extraction
# =============================================================================

def extract_terraform_code(response: str) -> str:
    """Extract Terraform/HCL code from a response."""
    # Try to find code in markdown code blocks
    patterns = [
        r"```(?:hcl|terraform|tf)\n(.*?)```",
        r"```\n(.*?)```",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[-1].strip()  # Return the last code block
    
    # If no code blocks, return the full response
    return response


# =============================================================================
# Dataset Loading
# =============================================================================

def load_iac_eval_dataset(path: Path) -> list[dict]:
    """Load the IaC-Eval dataset."""
    with open(path) as f:
        return json.load(f)


def load_multi_iac_eval_dataset(path: Path) -> list[dict]:
    """Load the Multi-IaC-Eval dataset."""
    with open(path) as f:
        return json.load(f)


# =============================================================================
# Result Saving
# =============================================================================

def save_result(
    output_dir: Path,
    dataset_name: str,
    scenario_id: str,
    model_name: str,
    prompt_type: str,
    prompt: str,
    response: str,
    extracted_code: str,
    reference: str | None = None,
):
    """Save a generation result to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "dataset": dataset_name,
        "scenario_id": scenario_id,
        "model": model_name,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "response": response,
        "extracted_code": extracted_code,
        "reference": reference,
        "timestamp": datetime.now().isoformat(),
    }
    
    filename = f"{dataset_name}_{scenario_id}_{model_name}_{prompt_type}.json"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"  Saved: {output_path}")


# =============================================================================
# Main Generation Logic
# =============================================================================

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
    
    for idx, scenario in enumerate(dataset):
        scenario_id = f"iac_eval_{idx:04d}"
        base_prompt = scenario["Prompt"]
        reference = scenario.get("Reference output")
        difficulty = scenario.get("Difficulty", "unknown")
        
        print(f"[{idx+1}/{len(dataset)}] Scenario: {scenario_id} (difficulty: {difficulty})")
        print(f"  Prompt: {base_prompt[:80]}...")
        
        user_prompt = build_prompt(base_prompt, prompt_type)
        
        for client in clients:
            print(f"  Generating with {client.name}...")
            try:
                response = client.generate(SYSTEM_PROMPT, user_prompt)
                extracted_code = extract_terraform_code(response)
                
                save_result(
                    output_dir=output_dir,
                    dataset_name="iac_eval",
                    scenario_id=scenario_id,
                    model_name=client.name,
                    prompt_type=prompt_type,
                    prompt=user_prompt,
                    response=response,
                    extracted_code=extracted_code,
                    reference=reference,
                )
            except Exception as e:
                print(f"    Error: {e}")


def run_multi_iac_eval(
    dataset: list[dict],
    clients: list,
    prompt_type: PromptType,
    output_dir: Path,
    samples: int | None = None,
):
    """Run generation on Multi-IaC-Eval dataset."""
    # Filter for Terraform tasks only
    terraform_tasks = [d for d in dataset if d.get("type") == "terraform"]
    
    if samples:
        terraform_tasks = terraform_tasks[:samples]
    
    print(f"\n{'='*60}")
    print(f"Processing Multi-IaC-Eval dataset ({len(terraform_tasks)} Terraform scenarios)")
    print(f"Prompt type: {prompt_type}")
    print(f"{'='*60}\n")
    
    for idx, scenario in enumerate(terraform_tasks):
        scenario_id = f"multi_iac_{scenario.get('source', idx)}"
        initial = scenario["initial"]
        utterance = scenario["utterance"]
        expected = scenario["expected"]
        template_type = scenario.get("type", "terraform")
        
        print(f"[{idx+1}/{len(terraform_tasks)}] Scenario: {scenario_id}")
        print(f"  Utterance: {utterance[:80]}...")
        
        base_prompt = build_multi_iac_prompt(initial, utterance, template_type)
        user_prompt = build_prompt(base_prompt, prompt_type)
        
        for client in clients:
            print(f"  Generating with {client.name}...")
            try:
                response = client.generate(SYSTEM_PROMPT, user_prompt)
                extracted_code = extract_terraform_code(response)
                
                save_result(
                    output_dir=output_dir,
                    dataset_name="multi_iac_eval",
                    scenario_id=scenario_id,
                    model_name=client.name,
                    prompt_type=prompt_type,
                    prompt=user_prompt,
                    response=response,
                    extracted_code=extracted_code,
                    reference=expected,
                )
            except Exception as e:
                print(f"    Error: {e}")


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
        default="claude,grok,gemini",
        help="Comma-separated list of models to use (default: claude,grok,gemini)"
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
    
    if not clients:
        print("Error: No valid models specified")
        return
    
    # Load datasets
    iac_eval_path = Path("iac_eval_dataset.json")
    multi_iac_eval_path = Path("multi_iac_eval_dataset.json")
    
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
    
    if multi_iac_eval_path.exists():
        multi_iac_eval_dataset = load_multi_iac_eval_dataset(multi_iac_eval_path)
        run_multi_iac_eval(
            dataset=multi_iac_eval_dataset,
            clients=clients,
            prompt_type=args.prompt_type,
            output_dir=output_dir,
            samples=args.samples,
        )
    else:
        print(f"Warning: {multi_iac_eval_path} not found, skipping Multi-IaC-Eval")
    
    print("\n" + "="*60)
    print("Generation complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
