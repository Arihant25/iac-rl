#!/usr/bin/env bash
# run_all.sh
# Complete pipeline to reproduce the IaC-RL study.

set -e

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Configuration
MODELS="claude,grok,gemini,kimi,glm,qwen,phi4,ministral,gemma3"
PROMPT_TYPES="zero-shot few-shot cot"

# Optional: Add --samples N during testing to limit data size
EXTRA_ARGS=""
# EXTRA_ARGS="--samples 5"

echo "=========================================================="
echo " Starting Full Evaluation Pipeline"
echo " Models: $MODELS"
echo " Prompts: $PROMPT_TYPES"
echo "=========================================================="

# ---------------------------------------------------------
# Phase 1: Baseline Generation (Original Prompts)
# ---------------------------------------------------------
echo -e "\n[Phase 1] Generate Terraform for Original Prompts"
for pt in $PROMPT_TYPES; do
    echo "Running original generation: --models $MODELS --prompt-type $pt"
    uv run generate_baselines.py --models "$MODELS" --prompt-type "$pt" $EXTRA_ARGS
done

# ---------------------------------------------------------
# Phase 2: Paraphrase Generation (Original Prompts -> Paraphrases)
# ---------------------------------------------------------
echo -e "\n[Phase 2] Generate Paraphrased Prompts (via GPT-5.4 on OpenRouter)"
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Warning: OPENROUTER_API_KEY environment variable is not set."
    echo "Phase 2 and 4 will fail without an API key."
fi

# We run this once per dataset
uv run srs.py --paraphrases-only $EXTRA_ARGS

# ---------------------------------------------------------
# Phase 3: Baseline Generation (Paraphrased Prompts)
# ---------------------------------------------------------
echo -e "\n[Phase 3] Generate Terraform for Paraphrased Prompts"
for dataset in iac_eval_paraphrases llm_iac_paraphrases; do
    # Ensure the paraphrase file exists before running
    if [ -f "datasets/${dataset}.json" ]; then
        for pt in $PROMPT_TYPES; do
            echo "Running paraphrase generation: $dataset --models $MODELS --prompt-type $pt"
            uv run generate_baselines.py \
              --paraphrases "datasets/${dataset}.json" \
              --models "$MODELS" \
              --prompt-type "$pt" \
              $EXTRA_ARGS
        done
    else
        echo "Error: datasets/${dataset}.json not found. Did Phase 2 fail?"
        exit 1
    fi
done

# ---------------------------------------------------------
# Phase 4: Semantic Robustness Score (SRS)
# ---------------------------------------------------------
echo -e "\n[Phase 4] Compute Semantic Robustness Scores (SRS/ICS)"
# Map short model names to the exact model identifiers used in `outputs/`
declare -A MODEL_MAP=(
    [claude]=claude-4.5-sonnet
    [grok]=grok-4.1-fast
    [gemini]=gemini-3-flash
    [kimi]=kimi-k2.5
    [glm]=glm-4.7
    [qwen]=qwen3-235b
    [phi4]=phi-4
    [ministral]=ministral-8b
    [gemma3]=gemma-3-27b
)

for model in ${MODELS//,/ }; do
        # Use mapped full model name when available (fallback to the original)
        full_model="${MODEL_MAP[$model]:-$model}"
        for pt in $PROMPT_TYPES; do
                echo "Computing SRS for: $full_model ($pt)"
                # srs.py expects the exact model name as used in the filenames under outputs/
                uv run srs.py \
                    --model "$full_model" \
                    --prompt-type "$pt" \
                    $EXTRA_ARGS
        done
done

# ---------------------------------------------------------
# Phase 5: TerraMetrics Evaluation 
# ---------------------------------------------------------
echo -e "\n[Phase 5] Run TerraMetrics Evaluation"
echo "Evaluating all generated Terraform configurations (original + paraphrased)..."
uv run evaluate_outputs.py $EXTRA_ARGS

echo "=========================================================="
echo " Pipeline Complete."
echo " Summary statistics located in: results/summary.csv"
echo " SRS results located in: results/final_results.json"
echo "=========================================================="
