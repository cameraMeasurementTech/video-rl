#!/usr/bin/env bash
# LoRA SFT on multi-turn SWE trajectories (VERL fsdp_sft_trainer + MultiTurnSFTDataset).
#
# Prerequisites:
#   - Parquet from: python -m data.build_sft_parquet ...  (train.parquet / val.parquet with `messages`)
#   - VERL_ROOT pointing at repo containing verl/ (same as GRPO)
#   - torchrun, FlashAttention-friendly env (optional)
#
# Usage (from swe_rl_training/):
#   export VERL_ROOT=/path/to/klearreasoner_or_verl
#   export BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
#   export SFT_TRAIN_PARQUET=$PWD/data/parquet_sft/train.parquet
#   export SFT_VAL_PARQUET=$PWD/data/parquet_sft/val.parquet
#   bash train/scripts/run_lora_sft_swe.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/.env"
  set +a
fi

VERL_ROOT="${VERL_ROOT:-}"
if [[ -z "${VERL_ROOT}" ]] && [[ -d "${PROJECT_ROOT}/../KlearReasoner/verl" ]]; then
  VERL_ROOT="$(cd "${PROJECT_ROOT}/../KlearReasoner" && pwd)"
fi
if [[ -z "${VERL_ROOT}" ]] || [[ ! -d "${VERL_ROOT}/verl" ]]; then
  echo "Set VERL_ROOT to a directory containing the verl package."
  exit 1
fi
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SFT_TRAIN_PARQUET="${SFT_TRAIN_PARQUET:-${PROJECT_ROOT}/data/parquet_sft/train.parquet}"
SFT_VAL_PARQUET="${SFT_VAL_PARQUET:-${PROJECT_ROOT}/data/parquet_sft/val.parquet}"

N_GPUS="${N_GPUS:-2}"
TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH="${SFT_MICRO_BATCH_PER_GPU:-2}"
MAX_LENGTH="${SFT_MAX_LENGTH:-8192}"
TOTAL_EPOCHS="${SFT_EPOCHS:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LR="${SFT_LR:-2e-4}"

DATE=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${SFT_CHECKPOINT_DIR:-${PROJECT_ROOT}/train/artifacts/sft_lora/${DATE}}"
mkdir -p "${OUT_DIR}"

export WANDB_PROJECT="${WANDB_PROJECT:-swe-sft-lora}"
export WANDB_EXP="${WANDB_EXP:-swe-lora-${DATE}}"

torchrun --nnodes=1 --nproc_per_node="${N_GPUS}" \
  -m verl.trainer.fsdp_sft_trainer \
  hydra.run.dir="${PROJECT_ROOT}/train/outputs_sft" \
  data.train_files="${SFT_TRAIN_PARQUET}" \
  data.val_files="${SFT_VAL_PARQUET}" \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${MICRO_BATCH}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=left \
  model.partial_pretrain="${BASE_MODEL}" \
  model.lora_rank="${LORA_RANK}" \
  model.lora_alpha="${LORA_ALPHA}" \
  model.target_modules=all-linear \
  model.enable_gradient_checkpointing=true \
  model.trust_remote_code=true \
  optim.lr="${LR}" \
  optim.betas="[0.9, 0.95]" \
  optim.weight_decay=0.01 \
  optim.warmup_steps_ratio=0.03 \
  optim.clip_grad=1.0 \
  optim.lr_scheduler=cosine \
  trainer.default_local_dir="${OUT_DIR}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${WANDB_EXP}" \
  trainer.logger='["console","wandb"]' \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"

echo "SFT checkpoints under: ${OUT_DIR}"
echo "Merge LoRA into full weights, then GRPO:"
echo "  python -m train.tools.merge_peft_adapter --base-model ${BASE_MODEL} --adapter-path ${OUT_DIR}/global_step_<N> --out train/artifacts/merged_for_grpo"
