#!/bin/bash
# Transcriptome SFT with reasoning models (QwQ-32B / Qwen3-32B)
# Uses FSDP backend — no Megatron weight conversion needed.
#
# Usage:
#   ./examples/transcriptome_sft/run_transcriptome_sft.sh
#
# Environment variables:
#   SLIME_SCRIPT_MODEL_PATH  — HF model directory (default: /root/models/QwQ-32B)
#   SLIME_SCRIPT_DATASET     — training parquet   (default: /root/data/transcriptome_sft_train.parquet)
#   SLIME_SCRIPT_NUM_GPUS    — number of GPUs     (default: 8)
#   WANDB_API_KEY            — set to enable W&B logging

MODEL_PATH=${SLIME_SCRIPT_MODEL_PATH:-"/root/models/QwQ-32B"}
DATASET=${SLIME_SCRIPT_DATASET:-"/root/data/transcriptome_sft_train.parquet"}
NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}

# Cleanup stale processes
pkill -9 sglang 2>/dev/null; sleep 1
ray stop --force 2>/dev/null; sleep 1

set -ex

export PYTHONBUFFERED=16

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --load "${MODEL_PATH}"
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${DATASET}"
   --input-key messages
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 3
   --rollout-batch-size 32
   --global-batch-size 32

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

MULTIMODAL_KEYS='{"transcriptome": "transcriptome"}'

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

if [ -n "$WANDB_API_KEY" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-transcriptome-sft
        --wandb-key "${WANDB_API_KEY}"
    )
else
    WANDB_ARGS=()
fi

BACKEND_ARGS=(
   --train-backend fsdp
   --gradient-checkpointing
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

# Start Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${BACKEND_ARGS[@]}"
