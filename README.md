# SWE trajectories → LoRA SFT + full GRPO

End-to-end pipeline: **multi-turn trajectory Parquet** → manifest join → **cheap LoRA SFT** (VERL `fsdp_sft_trainer`) → **merge adapters** → **full-weight GRPO** (VERL `main_ppo`) with a **real reward** when you wire `SWE_REWARD_SCRIPT`.

```text
Raw shards (47× parquet …)     Optional manifest (instance_id → repo/commit/patch…)
  only: instance_id +            (never in the trajectory table — separate JSONL
  messages                       from SWE-Bench / HF export, if you need it)
           │                                    │
           └────────────┬───────────────────────┘
                        ▼
              build_sft_parquet / build_rl_parquet
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   data/parquet_sft/              data/parquet/
   train.parquet                  train.parquet
   (messages multiturn)           (messages = RL prompt)
         │                             │
         ▼                             │
   LoRA SFT (torchrun)                  │
         │                             │
         ▼                             │
   merge_peft_adapter                  │
         │                             │
         └──────────► MODEL_PATH ◄─────┘
                         │
                         ▼
                   GRPO (full FT)
```

---

## Prerequisites

1. **Python ≥ 3.10**, CUDA machine for training.
2. **VERL** checkout: set `VERL_ROOT` to the repo root that contains the `verl` package (e.g. KlearReasoner or volcengine/verl). Same code is used for SFT and GRPO.
3. **Hugging Face** access for the base model (`HF_TOKEN` if needed).
4. **Optional manifest** JSONL (only if you join benchmark metadata): one object per line with `instance_id` plus any fields your reward needs (`repo`, `base_commit`, `patch`, …). **Your trajectory Parquet does not need these columns** — see below.

Install this repo:

```bash
cd swe_rl_training
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# For merging LoRA after SFT:
pip install -e ".[merge]"
```

---

## Step 1 — Prepare data

### What your trajectory files contain (normal case)

Typical columns are **only**:

| Column | Content |
|--------|---------|
| `instance_id` | Stable id (e.g. SWE-Bench instance id) |
| `messages` | Multi-turn chat list `[{role, content}, …]` |

Some exports use **`message`** (singular) instead of `messages` — pass `--messages-column message`.

There are **no** `repo`, `base_commit`, or `patch` columns in the trajectory table. Those live in a **separate manifest** you build from the benchmark (e.g. export SWE-Bench metadata by `instance_id`), **or** your `SWE_REWARD_SCRIPT` resolves `instance_id` alone (Docker image, checkout, tests).

- **LoRA SFT on trajectories only:** use **`--skip-manifest`** (imitation learning; no benchmark columns required).
- **GRPO with real test rewards:** add a manifest **or** a reward script that maps `instance_id` → eval environment.

### 1a) Multi-turn SFT parquet (for LoRA SFT)

Produces `data/parquet_sft/train.parquet` and `val.parquet` with a **`messages`** column (full conversations with at least one **assistant** turn). VERL applies loss only on assistant tokens (`MultiTurnSFTDataset`).

**With optional manifest join** (extra metadata in `extra_info`):

```bash
python -m data.build_sft_parquet \
  --input /path/to/your_parquet_shards/ \
  --manifest /path/to/swe_manifest.jsonl \
  --out-dir data/parquet_sft \
  --val-ratio 0.02 \
  --recursive
```

**Trajectory-only** (only `instance_id` + `messages` in data — no manifest file):

```bash
python -m data.build_sft_parquet \
  --input /path/to/your_parquet_shards/ \
  --skip-manifest \
  --out-dir data/parquet_sft \
  --val-ratio 0.02 \
  --recursive
```

Options:

- `--recursive` — include `**/*.parquet` under a directory.
- `--instance-id-column` / `--messages-column` — if your columns differ (e.g. `--messages-column message`).
- `--skip-manifest` — no `--manifest`; benchmark fields are **not** in the table and not joined.
- `--keep-missing-manifest` — when using `--manifest`, keep rows without a manifest hit (usually **omit**).

### 1b) RL parquet (for GRPO prompts)

Expands trajectories into RL rows (`prefix_next_assistant`, etc.). **With manifest** (recommended when `extra_info` should carry repo/commit for rewards):

```bash
python -m data.build_rl_parquet \
  --input /path/to/your_parquet_shards/ \
  --manifest /path/to/swe_manifest.jsonl \
  --out-dir data/parquet_rl \
  --strategy prefix_next_assistant \
  --val-ratio 0.02 \
  --recursive
```

**Without manifest** (stub `extra_info`; use when `SWE_REWARD_SCRIPT` only needs `instance_id`):

```bash
python -m data.build_rl_parquet \
  --input /path/to/your_parquet_shards/ \
  --skip-manifest \
  --out-dir data/parquet_rl \
  --strategy prefix_next_assistant \
  --val-ratio 0.02 \
  --recursive
```

Use **`data/parquet_rl/train.parquet`** as `TRAIN_FILES` for GRPO (after merge in step 4).

---

## Step 2 — LoRA SFT (cheap)

Uses VERL **`fsdp_sft_trainer`** with **`data.multiturn.enable=true`**.

```bash
export VERL_ROOT=/path/to/repo_containing_verl
export BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
export SFT_TRAIN_PARQUET=$PWD/data/parquet_sft/train.parquet
export SFT_VAL_PARQUET=$PWD/data/parquet_sft/val.parquet
export N_GPUS=2

bash train/scripts/run_lora_sft_swe.sh
```

Important environment variables (see script for full list):

| Variable | Meaning |
|----------|---------|
| `BASE_MODEL` | HF id or local path to base LM |
| `SFT_TRAIN_PARQUET` / `SFT_VAL_PARQUET` | Outputs of `build_sft_parquet` |
| `N_GPUS` | `torchrun` `--nproc_per_node` |
| `LORA_RANK` / `LORA_ALPHA` | Default 64 / 16 |
| `SFT_MAX_LENGTH` | Default 8192 (raise if OOM with care) |
| `SFT_CHECKPOINT_DIR` | Where checkpoints are written |

Checkpoints appear under `train/artifacts/sft_lora/<timestamp>/` (or your `SFT_CHECKPOINT_DIR`), typically `global_step_<N>/`.

---

## Step 3 — Merge LoRA into a full model (for GRPO)

GRPO in this stack trains **full** weights; vLLM rollout expects a **dense** HF model directory.

Pick the best `global_step_*` folder from SFT, then:

```bash
pip install -e ".[merge]"   # transformers, peft, torch

python -m train.tools.merge_peft_adapter \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter-path train/artifacts/sft_lora/<run>/global_step_<N> \
  --out train/artifacts/merged_for_grpo \
  --dtype bfloat16
```

If there is **no** `adapter_config.json` in the checkpoint (already a full model), the tool prints a hint: point **`MODEL_PATH`** directly at that directory and skip merging.

---

## Step 4 — GRPO (full fine-tune) on merged weights

Point **`MODEL_PATH`** at the **merged** directory (or full checkpoint). Use **RL** parquet from step 1b.

```bash
export VERL_ROOT=/path/to/repo_containing_verl
export MODEL_PATH=$PWD/train/artifacts/merged_for_grpo
export TRAIN_FILES=$PWD/data/parquet_rl/train.parquet
export VAL_FILES=$PWD/data/parquet_rl/val.parquet

# Real reward (recommended): executable script — see below
export SWE_REWARD_MODE=subprocess
export SWE_REWARD_SCRIPT=$PWD/train/tools/example_reward_stub.sh   # replace with your harness

export N_GPUS=4
bash train/scripts/run_grpo_swe.sh
```

`run_grpo_swe.sh` sets `data.prompt_key=messages`, GRPO, vLLM rollout, and `custom_reward_function` → `train/tools/reward_swe.py`.

---

## Reward modes (`train/tools/reward_swe.py`)

| `SWE_REWARD_MODE` | Use case |
|-------------------|----------|
| `dry_run` | Pipeline test only; **not** for real SWE quality. |
| `heuristic` | Cheap proxy from patch overlap; **weak** vs real tests. |
| `subprocess` | **Recommended:** set **`SWE_REWARD_SCRIPT`** to an executable that you implement (Docker / SWE-Bench / internal CI). |
| `harness` | Placeholder; use **`subprocess`** unless you integrate `swebench` yourself. |

### Subprocess contract (`SWE_REWARD_MODE=subprocess`)

- Env: **`SWE_REWARD_SCRIPT`** = path to executable.
- Invocation: `SWE_REWARD_SCRIPT <instance_id> <path_to_prediction.txt>`  
  The file contains the **decoded model response** (full text).
- Exit code 0: read **last line of stdout** as a float in **[0, 1]** (pass rate, binary success, etc.).
- Non-zero exit or empty stdout → **0** reward.

Optional: **`SWE_REWARD_TIMEOUT`** (seconds, default 600).

Implement this script to call your SWE-Bench Docker flow or internal test runner using `instance_id` + manifest you already joined into parquet `extra_info`.

---

## Layout (repo)

| Path | Role |
|------|------|
| `data/trajectory_io.py` | Shared Parquet/JSONL loaders |
| `data/build_sft_parquet.py` | SFT multiturn parquet |
| `data/build_rl_parquet.py` | RL parquet with expansion strategies |
| `data/enrich_instance.py` | Manifest join |
| `train/scripts/run_lora_sft_swe.sh` | LoRA SFT (`fsdp_sft_trainer`) |
| `train/tools/merge_peft_adapter.py` | LoRA → full HF weights |
| `train/scripts/run_grpo_swe.sh` | GRPO (`main_ppo`) |
| `train/tools/reward_swe.py` | `compute_score` |

---

## Tests

```bash
pytest -q
```

---

## Troubleshooting

- **OOM during SFT**: Lower `SFT_MAX_LENGTH`, `SFT_TRAIN_BATCH_SIZE`, or `LORA_RANK`.
- **OOM during GRPO**: Lower `TRAIN_BATCH_SIZE`, `ROLLOUT_N`, `MAX_PROMPT_LEN`, or TP size in `run_grpo_swe.sh`.
- **No learning**: Switch from `dry_run` / `heuristic` to **`subprocess`** with a real test-based script.
- **VERL errors**: Ensure `PYTHONPATH` includes `VERL_ROOT` and versions match the fork you use.
