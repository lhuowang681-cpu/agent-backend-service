#!/usr/bin/env bash
set -euo pipefail

# Set FIT_VERDICT_DRY_RUN=1 to inspect the command without starting training.
: "${FIT_VERDICT_MODEL:=fixtures/models/fixture-model}"
: "${FIT_VERDICT_DATASET:=data/synthetic/fit_verdict_train.jsonl}"
: "${FIT_VERDICT_OUTPUT_DIR:=output/fit_verdict_lora}"
: "${FIT_VERDICT_EPOCHS:=1}"
: "${FIT_VERDICT_BATCH_SIZE:=1}"
: "${FIT_VERDICT_GRAD_ACCUM:=8}"
: "${FIT_VERDICT_LR:=1e-4}"
: "${FIT_VERDICT_MAX_LENGTH:=2048}"
: "${FIT_VERDICT_DRY_RUN:=0}"

train_cmd=(
  swift sft
  --model "${FIT_VERDICT_MODEL}"
  --tuner_type lora
  --dataset "${FIT_VERDICT_DATASET}"
  --torch_dtype bfloat16
  --num_train_epochs "${FIT_VERDICT_EPOCHS}"
  --per_device_train_batch_size "${FIT_VERDICT_BATCH_SIZE}"
  --per_device_eval_batch_size "${FIT_VERDICT_BATCH_SIZE}"
  --learning_rate "${FIT_VERDICT_LR}"
  --lora_rank 8
  --lora_alpha 16
  --target_modules all-linear
  --gradient_accumulation_steps "${FIT_VERDICT_GRAD_ACCUM}"
  --eval_steps 50
  --save_steps 50
  --save_total_limit 2
  --logging_steps 5
  --max_length "${FIT_VERDICT_MAX_LENGTH}"
  --output_dir "${FIT_VERDICT_OUTPUT_DIR}"
)

if [[ "${FIT_VERDICT_DRY_RUN}" == "1" ]]; then
  printf '%q ' "${train_cmd[@]}"
  printf '\n'
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"${train_cmd[@]}"
