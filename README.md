# IaC-RL

## Generating Baselines
Generate Terraform configurations from the IaC-Eval datasets using LLM APIs.

### Setup

```bash
# Install dependencies
uv sync
```

Set API keys in `.env` (copy from `example.env`).

### Usage

```bash
uv run generate_baselines.py [OPTIONS]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--prompt-type` | `zero-shot`, `few-shot`, `cot`, `all` | `zero-shot` | Prompting strategy. Use `all` to run zero-shot, few-shot, and cot sequentially |
| `--models` | Comma-separated list | `claude,grok,gemini,kimi,glm,qwen,phi4,ministral,gemma3` | Models to use |
| `--samples` | Integer | All | Number of samples per dataset |

### Examples

```bash
# Run all models with zero-shot prompting
uv run generate_baselines.py

# Run with few-shot prompting (3 examples)
uv run generate_baselines.py --prompt-type few-shot

# Run with Chain-of-Thought prompting
uv run generate_baselines.py --prompt-type cot

# Test with 5 samples using only Claude
uv run generate_baselines.py --samples 5 --models claude
```

### Datasets

The script processes both datasets automatically from the `datasets/` folder:
- **iac_eval_dataset.json**: IaC-Eval dataset from NeurIPS 2024
  - Contains Terraform generation prompts with difficulty ratings and reference outputs
- **llm-iac.csv**: Infrastructure as Code prompts dataset
  - Columns: ID, Category, Cloud_Provider, User_Query, Terraform_Code
  - Covers AWS, GCP, Azure and other cloud providers

### Output

Results saved to `outputs/` as CSV files containing:
- Input prompt
- Model response
- Extracted Terraform code
- Reference solution

## Evaluating Outputs

Run [TerraMetrics](https://github.com/stilab-ets/terametrics) on all generated outputs to extract quality metrics (structural, complexity, documentation, risk).

Requires Java.

### Usage

```bash
uv run evaluate_outputs.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | All CSVs in `outputs/` | Path to a specific CSV to evaluate |
| `--samples` | All | Only process first N rows per CSV |
| `--workers` | `4` | Number of parallel workers |
| `--jar` | `terraform_metrics-1.0.jar` | Path to TerraMetrics JAR |
| `--summary-only` | — | Regenerate summary from existing results |

### Examples

```bash
# Full run
uv run evaluate_outputs.py

# Single CSV
uv run evaluate_outputs.py --csv outputs/iac_eval_claude-4.5-sonnet_zero-shot.csv

# Test with 5 rows, 8 workers
uv run evaluate_outputs.py --samples 5 --workers 8

# Regenerate summary only
uv run evaluate_outputs.py --summary-only
```

### Output

Results saved to `results/`:
- **`*_metrics.csv`** — one per input CSV, with `gen_*` and `ref_*` metric columns for both generated and reference code
- **`summary.csv`** — aggregated means per model/prompt_type/dataset

Resume is automatic for both generation and evaluation: re-running skips any `scenario_id` already in the result CSV.
