# IaC-RL

## Generating Baselines
Generate Terraform configurations from the IaC-Eval datasets using Claude, Grok, and Gemini APIs.

### Setup

```bash
# Install dependencies
uv sync

# Set API keys
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...
export GOOGLE_API_KEY=...
```

### Usage

```bash
uv run generate_baselines.py [OPTIONS]
```

### Options

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--prompt-type` | `zero-shot`, `few-shot`, `cot` | `zero-shot` | Prompting strategy |
| `--models` | Comma-separated list | `claude,grok,gemini,kimi,glm` | Models to use |
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

Results saved to `outputs/` as JSON files containing:
- Input prompt
- Model response
- Extracted Terraform code
- Reference solution
