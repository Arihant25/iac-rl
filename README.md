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

## Semantic Robustness Score (SRS)

Measures how consistently a model's Terraform output satisfies the original intent across semantically equivalent prompt phrasings. Uses rule-based **Intent Coverage Score (ICS)**:

```
SRS = max(0, min(1, 1 − stddev([ics_original, ics_paraphrase])))
```

Requires `OPENROUTER_API_KEY` (or `--api-key`) for paraphrase generation via GPT-5.4.

### Setup

```bash
export OPENROUTER_API_KEY=...
```

### Workflow (3 steps)

**Step 1 — Generate paraphrases** (calls OpenRouter; outputs cached to `datasets/`):

```bash
uv run srs.py --paraphrases-only [--samples N] [--dry-run]
```

**Step 2 — Generate Terraform for paraphrases** (extends `generate_baselines.py`):

```bash
uv run generate_baselines.py --paraphrases datasets/iac_eval_paraphrases.json --models claude --prompt-type few-shot
uv run generate_baselines.py --paraphrases datasets/llm_iac_paraphrases.json  --models claude --prompt-type few-shot
```

Outputs land in `outputs/{dataset}_paraphrased_{model}_{prompt_type}.csv`.

**Step 3 — Compute ICS & SRS** (reads from `outputs/`; writes `results/final_results.json`):

```bash
uv run srs.py --model claude-4.5-sonnet --prompt-type few-shot [--samples N]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--model` | `claude-4.5-sonnet` | Model name to look up in `outputs/` |
| `--prompt-type` | `few-shot` | Prompt type to look up in `outputs/` |
| `--samples` | All | Limit entries per dataset |
| `--dry-run` | — | Skip API calls, use dummy paraphrases |
| `--force` | — | Ignore all caches and reprocess from scratch |
| `--paraphrases-only` | — | Run only Step 1 (paraphrase generation) |

### How it works

1. **Constraint extraction** — rule-based keyword/regex matching on the original prompt only (resource types, instance types, regions, ports, etc.). For `iac_eval_dataset.json`, the `Intent` field is used directly.
2. **Paraphrase generation** — 1 paraphrase per prompt via OpenRouter (`openai/gpt-5.4`).
3. **Terraform generation** — paraphrase prompts are fed through `generate_baselines.py` to produce new Terraform code.
4. **ICS computation** — both the original Terraform (from `outputs/`) and the paraphrase Terraform are evaluated against the same constraints → 2 ICS values.
5. **SRS computation** — `1 − stddev([ics_orig, ics_para])`, clamped to `[0, 1]`.

### Output

- **`datasets/iac_eval_paraphrases.json`** — cached paraphrases for `iac_eval`
- **`datasets/llm_iac_paraphrases.json`** — cached paraphrases for `llm_iac`
- **`outputs/{dataset}_paraphrased_{model}_{prompt_type}.csv`** — Terraform generated for paraphrases
- **`results/final_results.json`** — per-entry and aggregate SRS/ICS results

All three steps are fully **idempotent** — re-running resumes from where it left off.
