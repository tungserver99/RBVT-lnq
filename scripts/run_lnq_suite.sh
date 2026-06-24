#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-cuda:0}"
NUM_GROUPS="${NUM_GROUPS:-4}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
RBVT_CALIB_DATASET="${RBVT_CALIB_DATASET:-c4}"
RBVT_N_CALIB="${RBVT_N_CALIB:-128}"
RBVT_MAX_LENGTH="${RBVT_MAX_LENGTH:-2048}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
INCLUDE_LM_EVAL=1

DEFAULT_MODELS=(
  "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-v0.3"
  "Qwen/Qwen2.5-7B"
)

DEFAULT_BITS=(
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

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_lnq_suite.sh [options]

Options:
  --mode <value>            One of: lnq, codebook_last, assignment_last, all
  --rbvt-mode <value>       One of: naive, lnq_aware, all
  --model <hf_ref>          Repeatable. Default: llama3.1-8b, mistral-7b-v0.3, qwen2.5-7b
  --bits <n>                Repeatable. Default: 3 and 4
  --device <value>          Override DEVICE
  --num-groups <n>          Override NUM_GROUPS
  --output-root <path>      Root output directory. Default: ./outputs/suite
  --skip-lm-eval            Disable lm-eval and run perplexity only
  --dry-run                 Print commands without executing
  --help                    Show this message

Examples:
  bash scripts/run_lnq_suite.sh --mode lnq
  bash scripts/run_lnq_suite.sh --mode codebook_last --rbvt-mode lnq_aware
  bash scripts/run_lnq_suite.sh --mode assignment_last --rbvt-mode naive --model Qwen/Qwen2.5-7B --bits 4
  bash scripts/run_lnq_suite.sh --mode all --rbvt-mode all
EOF
}

MODE="all"
RBVT_MODE="all"
OUTPUT_ROOT="./outputs/suite"
DRY_RUN=0
MODELS=()
BITS_LIST=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --rbvt-mode)
      RBVT_MODE="$2"
      shift 2
      ;;
    --model)
      MODELS+=("$2")
      shift 2
      ;;
    --bits)
      BITS_LIST+=("$2")
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-groups)
      NUM_GROUPS="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --skip-lm-eval)
      INCLUDE_LM_EVAL=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("${DEFAULT_MODELS[@]}")
fi

if [[ ${#BITS_LIST[@]} -eq 0 ]]; then
  BITS_LIST=("${DEFAULT_BITS[@]}")
fi

case "$MODE" in
  lnq|codebook_last|assignment_last|all) ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    exit 1
    ;;
esac

case "$RBVT_MODE" in
  naive|lnq_aware|all) ;;
  *)
    echo "Invalid --rbvt-mode: $RBVT_MODE" >&2
    exit 1
    ;;
esac

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s//-/_}"
  echo "$s"
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_one() {
  local model="$1"
  local bits="$2"
  local mode="$3"
  local rbvt_mode="$4"
  local model_slug output_dir
  local lm_eval_args=()

  model_slug="$(slugify "$model")"

  if [[ "$INCLUDE_LM_EVAL" == "1" ]]; then
    lm_eval_args=(--include-lm-eval --lm-eval-tasks "${LM_EVAL_TASKS[@]}")
  fi

  if [[ "$mode" == "lnq" ]]; then
    output_dir="${OUTPUT_ROOT}/${model_slug}_lnq_${bits}bit"
    echo "=== LNQ | model=${model} | bits=${bits} ==="
    run_cmd python main.py \
      --model-path "$model" \
      --bits "$bits" \
      --device "$DEVICE" \
      --output-root "$output_dir" \
      --dataset "$DATASET" \
      --seq-len "$SEQ_LEN" \
      --num-examples "$NUM_EXAMPLES" \
      --num-groups "$NUM_GROUPS" \
      --rbvt-calib-dataset "$RBVT_CALIB_DATASET" \
      --rbvt-n-calib "$RBVT_N_CALIB" \
      --rbvt-max-length "$RBVT_MAX_LENGTH" \
      --eval-max-length "$EVAL_MAX_LENGTH" \
      "${lm_eval_args[@]}" \
      --skip-rbvt
    return
  fi

  output_dir="${OUTPUT_ROOT}/${model_slug}_lnq_rbvt_${mode}_${rbvt_mode}_${bits}bit"
  echo "=== LNQ + RBVT | position=${mode} | target=${rbvt_mode} | model=${model} | bits=${bits} ==="
  run_cmd python main.py \
    --model-path "$model" \
    --bits "$bits" \
    --device "$DEVICE" \
    --output-root "$output_dir" \
    --dataset "$DATASET" \
    --seq-len "$SEQ_LEN" \
    --num-examples "$NUM_EXAMPLES" \
    --num-groups "$NUM_GROUPS" \
    --rbvt-calib-dataset "$RBVT_CALIB_DATASET" \
    --rbvt-n-calib "$RBVT_N_CALIB" \
    --rbvt-max-length "$RBVT_MAX_LENGTH" \
    --eval-max-length "$EVAL_MAX_LENGTH" \
    "${lm_eval_args[@]}" \
    --rbvt-position "$mode" \
    --rbvt-mode "$rbvt_mode"
}

expand_modes() {
  if [[ "$MODE" == "all" ]]; then
    printf '%s\n' "lnq" "codebook_last" "assignment_last"
  else
    printf '%s\n' "$MODE"
  fi
}

expand_rbvt_modes() {
  if [[ "$RBVT_MODE" == "all" ]]; then
    printf '%s\n' "naive" "lnq_aware"
  else
    printf '%s\n' "$RBVT_MODE"
  fi
}

while IFS= read -r mode; do
  for model in "${MODELS[@]}"; do
    for bits in "${BITS_LIST[@]}"; do
      if [[ "$mode" == "lnq" ]]; then
        run_one "$model" "$bits" "$mode" "na"
      else
        while IFS= read -r rbvt_mode; do
          run_one "$model" "$bits" "$mode" "$rbvt_mode"
        done < <(expand_rbvt_modes)
      fi
    done
  done
done < <(expand_modes)
