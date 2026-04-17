#!/usr/bin/env bash
# GRPO RL training for SWE trajectories using VERL (main_ppo).
#
# Prerequisites:
#   - Install VERL (pip install verl @ git+... OR set VERL_ROOT to a checkout containing ``verl/``).
#   - Parquet from data/build_rl_parquet.py with column ``messages`` (see data.prompt_key below).
#   - HF_TOKEN for model download if needed.
#
# Usage (from repo root ``swe_rl_training/``):
#   export VERL_ROOT=/path/to/verl/repo   # parent folder that contains verl/ package
#   export PYTHONPATH="${VERL_ROOT}:${PWD}"
#   bash train/scripts/run_grpo_swe.sh
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
if [[ -z "${VERL_ROOT}" ]]; then
  # Default: sibling KlearReasoner checkout in this workspace layout
  if [[ -d "${PROJECT_ROOT}/../KlearReasoner/verl" ]]; then
    VERL_ROOT="$(cd "${PROJECT_ROOT}/../KlearReasoner" && pwd)"
  fi
fi
if [[ -z "${VERL_ROOT}" ]] || [[ ! -d "${VERL_ROOT}/verl" ]]; then
  echo "Set VERL_ROOT to a directory containing the verl package (e.g. KlearReasoner or volcengine/verl checkout)."
  exit 1
fi

export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

REWARD_FN_PATH="${PROJECT_ROOT}/train/tools/reward_swe.py"
TRAIN_FILES="${TRAIN_FILES:-${PROJECT_ROOT}/data/parquet/train.parquet}"
VAL_FILES="${VAL_FILES:-${PROJECT_ROOT}/data/parquet/val.parquet}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"

# Optional: HF token
: "${HF_TOKEN:=}"

gpu_count="${N_GPUS:-4}"
train_batch_size="${TRAIN_BATCH_SIZE:-32}"
rollout_n="${ROLLOUT_N:-8}"
policy_lr="${POLICY_LR:-6e-6}"
total_steps="${TOTAL_STEPS:-1000}"
max_prompt_len="${MAX_PROMPT_LEN:-8192}"
max_response_len="${MAX_RESPONSE_LEN:-2048}"

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
  hydra.run.dir="${PROJECT_ROOT}/train/outputs" \
  algorithm.adv_estimator=grpo \
  data.filter_overlong_prompts=True \
  data.train_files="[\"${TRAIN_FILES}\"]" \
  data.val_files="[\"${VAL_FILES}\"]" \
  data.train_batch_size="${train_batch_size}" \
  data.max_prompt_length="${max_prompt_len}" \
  data.max_response_length="${max_response_len}" \
  data.dataloader_num_workers="${DATALOADER_WORKERS:-4}" \
  data.prompt_key=messages \
  data.truncation=left \
  data.return_raw_chat=True \
  data.reward_fn_key=data_source \
  data.shuffle=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  custom_reward_function.path="${REWARD_FN_PATH}" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.actor.optim.lr="${policy_lr}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH:-16}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_PER_GPU:-2}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${rollout_n}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-2}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.85}" \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.default_local_dir="${PROJECT_ROOT}/train/artifacts/checkpoints" \
  trainer.n_gpus_per_node="${gpu_count}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-50}" \
  trainer.total_training_steps="${total_steps}" \
  trainer.val_before_train=False \
  "$@"
