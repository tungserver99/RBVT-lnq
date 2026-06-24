#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RBVT_DEVICE="${RBVT_DEVICE:-cuda:0}"
NUM_GROUPS="${NUM_GROUPS:-4}"
RBVT_MODE="${RBVT_MODE:-naive}"

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2.5-7B"
)

BITS_LIST=(
  "3"
  "4"
)

LM_EVAL_TASKS=(
  "arc_easy"
  "arc_challenge"
  "hellaswag"
  "piqa"
  "winogrande"
  "boolq"
  "rte"
  "openbookqa"
  "lambada_openai"
  "mmlu"
  "gsm8k"
)

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s//-/_}"
  echo "$s"
}

for model in "${MODELS[@]}"; do
  model_slug="$(slugify "$model")"
  echo "=== LNQ + RBVT assignment-last (${RBVT_MODE}) runs for ${model} ==="

  for bits in "${BITS_LIST[@]}"; do
    python main.py \
      --model-path "$model" \
      --bits "$bits" \
      --device "$RBVT_DEVICE" \
      --output-root "./outputs/${model_slug}_lnq_rbvt_assignment_last_${RBVT_MODE}_${bits}bit" \
      --dataset c4 \
      --seq-len 2048 \
      --num-examples 128 \
      --num-groups "$NUM_GROUPS" \
      --rbvt-calib-dataset c4 \
      --rbvt-n-calib 128 \
      --rbvt-max-length 2048 \
      --eval-max-length 2048 \
      --include-lm-eval \
      --lm-eval-tasks "${LM_EVAL_TASKS[@]}" \
      --rbvt-position assignment_last \
      --rbvt-mode "${RBVT_MODE}"
  done
done
  done
done
