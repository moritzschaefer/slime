# Transcriptome SFT with Reasoning Models

Supervised fine-tuning (SFT) of reasoning LLMs on transcriptome (single-cell) data using pre-computed cell foundation model embeddings.
This guide covers training **Qwen QwQ-32B** and **GPT-o3-mini** (open-weight 20B-class reasoning models) on datasets where each sample pairs a cell embedding with a question and a text answer that includes a reasoning trace.

## Overview

The pipeline has three stages:

1. **Embed cells** — run a cell foundation model (e.g. Geneformer V2) offline to produce a fixed-length embedding per cell.
2. **Prepare the dataset** — combine the embedding, a text question, and a target answer (with optional `<think>…</think>` reasoning) into the format expected by slime.
3. **Run SFT** — launch `train_async.py` with the SFT rollout, pointing at the prepared dataset.

Only the adapter layer that projects cell embeddings into the LLM's token space is new; the rest uses standard slime SFT machinery.

---

## 1. Prerequisites

| Requirement | Details |
|---|---|
| **Hardware** | ≥ 4 × H100/H200 80 GB (QwQ-32B); 2 × H100 suffice for 7B-class models |
| **Software** | `slimerl/slime:latest` Docker image (includes Ray, SGLang, Megatron, FSDP) |
| **Models** | `Qwen/QwQ-32B` or `Qwen/Qwen3-32B` from Hugging Face |
| **Dataset** | Pre-computed cell embeddings (`.npz`) + question/answer pairs |

```bash
# Pull and start Docker
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash

# Install slime (already in image, update to latest)
cd /root/slime && git pull && pip install -e . --no-deps
```

### Download model weights

```bash
mkdir -p /root/models

# Option A: Qwen QwQ-32B (reasoning model)
hf download Qwen/QwQ-32B --local-dir /root/models/QwQ-32B

# Option B: Qwen3-32B (general model, same architecture)
hf download Qwen/Qwen3-32B --local-dir /root/models/Qwen3-32B
```

> **Note on GPT-o3-mini / "gpt-oss-20B":** If you have access to an
> open-weight ~20B reasoning model (e.g. a community reproduction), download
> it similarly and adjust paths below.  The SFT procedure is identical for
> any causal-LM that HuggingFace Transformers supports.  For models with a
> different architecture (e.g. Llama-based), use the corresponding
> `scripts/models/*.sh` config or create one.

---

## 2. Data Preparation

### 2.1 Source data

You need two files:

| File | Description |
|---|---|
| `embeddings.npz` | Pre-computed cell embeddings from Geneformer V2 (or any cell foundation model). Shape `(N, D)`, dtype `float32`. |
| `annotations.jsonl` | One JSON object per cell with a question and answer. |

**Example `annotations.jsonl`:**
```jsonl
{"cell_idx": 0, "question": "What cell type is this?", "answer": "<think>\nThe expression profile shows high CD3D and CD8A, which are markers for cytotoxic T cells.\n</think>\nCD8+ cytotoxic T cell"}
{"cell_idx": 1, "question": "Is this cell from a tumor or healthy tissue?", "answer": "<think>\nElevated expression of proliferation markers and MKI67 suggests active division, typical of tumor-infiltrating cells.\n</think>\nTumor tissue"}
```

The answer field contains:
- A **reasoning trace** inside `<think>…</think>` tags (the model learns to produce chain-of-thought)
- A **final answer** after the closing `</think>` tag

### 2.2 Build the training dataset

The script below merges embeddings and annotations into a single Parquet file that slime expects.

```python
#!/usr/bin/env python3
"""prepare_dataset.py — build an SFT-ready Parquet file."""

import json
import numpy as np
import pandas as pd

# --- Paths (adjust as needed) -------------------------------------------
EMBEDDINGS_PATH = "/root/data/embeddings.npz"
ANNOTATIONS_PATH = "/root/data/annotations.jsonl"
OUTPUT_PATH = "/root/data/transcriptome_sft_train.parquet"

# --- Load embeddings -----------------------------------------------------
embeddings = np.load(EMBEDDINGS_PATH)["cell_embeddings"]  # (N, 1152)
print(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

# --- Load annotations ----------------------------------------------------
annotations = []
with open(ANNOTATIONS_PATH) as f:
    for line in f:
        annotations.append(json.loads(line))

# --- Build messages in OpenAI format -------------------------------------
records = []
for ann in annotations:
    idx = ann["cell_idx"]
    emb = embeddings[idx].tolist()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "transcriptome", "transcriptome": emb},
                {"type": "text", "text": ann["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": ann["answer"],
        },
    ]
    records.append({"messages": messages})

df = pd.DataFrame(records)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Wrote {len(df)} samples to {OUTPUT_PATH}")
```

> **Reasoning format:** The assistant's answer contains `<think>…</think>` followed by the final output.
> During SFT the model learns to produce both the reasoning trace and the answer.
> The loss is computed on the **entire** assistant turn (reasoning + answer) via `--loss-type sft_loss`.

---

## 3. Training

### 3.1 Choose a backend

| Backend | When to use |
|---|---|
| **FSDP** | Simpler setup, good for ≤ 8 GPUs, native HuggingFace model loading |
| **Megatron** | Better performance at scale (tensor/pipeline parallelism), requires weight conversion |

### 3.2 FSDP backend (recommended for getting started)

```bash
cd /root/slime
bash examples/transcriptome_sft/run_transcriptome_sft.sh
```

Or manually:

```bash
# Start Ray
export MASTER_ADDR=127.0.0.1
ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
  -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    \
    --hf-checkpoint /root/models/QwQ-32B \
    --load /root/models/QwQ-32B \
    \
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
    --prompt-data /root/data/transcriptome_sft_train.parquet \
    --input-key messages \
    --apply-chat-template \
    --multimodal-keys '{"transcriptome": "transcriptome"}' \
    --rollout-shuffle \
    --num-epoch 3 \
    --rollout-batch-size 32 \
    --global-batch-size 32 \
    \
    --loss-type sft_loss \
    --calculate-per-token-loss \
    --disable-compute-advantages-and-returns \
    --debug-train-only \
    \
    --train-backend fsdp \
    --gradient-checkpointing \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 4096 \
    \
    --optimizer adam \
    --lr 1e-5 \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95
```

### 3.3 Megatron backend

For larger-scale training with tensor parallelism:

```bash
# 1. Convert HF weights to Megatron format
cd /root/slime
source scripts/models/qwen3-32B.sh   # QwQ-32B uses the same architecture as Qwen3-32B

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/QwQ-32B \
    --save /root/models/QwQ-32B_torch_dist

# 2. Run training
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM/","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
  -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/QwQ-32B \
    --load /root/models/QwQ-32B_torch_dist \
    --save /root/models/QwQ-32B_slime \
    --save-interval 500 \
    \
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
    --prompt-data /root/data/transcriptome_sft_train.parquet \
    --input-key messages \
    --apply-chat-template \
    --multimodal-keys '{"transcriptome": "transcriptome"}' \
    --rollout-shuffle \
    --num-epoch 3 \
    --rollout-batch-size 128 \
    --global-batch-size 128 \
    \
    --loss-type sft_loss \
    --calculate-per-token-loss \
    --disable-compute-advantages-and-returns \
    --debug-train-only \
    \
    --tensor-model-parallel-size 8 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 4096 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend flash \
    \
    --optimizer adam \
    --lr 1e-5 \
    --lr-decay-style cosine \
    --min-lr 1e-6 \
    --lr-warmup-fraction 0.1 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95
```

---

## 4. Key Arguments Explained

| Argument | Purpose |
|---|---|
| `--rollout-function-path slime.rollout.sft_rollout.generate_rollout` | Use the SFT rollout (supervised, no RL sampling) |
| `--loss-type sft_loss` | Cross-entropy loss on assistant tokens |
| `--calculate-per-token-loss` | Per-token loss (recommended for variable-length sequences) |
| `--disable-compute-advantages-and-returns` | Skip RL advantage computation (not needed for SFT) |
| `--debug-train-only` | No inference engine — SFT only needs the training backend |
| `--apply-chat-template` | Apply the model's chat template to messages |
| `--multimodal-keys '{"transcriptome": "transcriptome"}'` | Map the `transcriptome` placeholder to the embedding column in the dataset |
| `--input-key messages` | Column name in the Parquet file containing the conversation |

---

## 5. Adapting for Different Models

### Qwen QwQ-32B

QwQ-32B uses the same dense Transformer architecture as Qwen3-32B (64 layers, 5120 hidden, 64 heads, 8 KV heads).
Use `scripts/models/qwen3-32B.sh` for Megatron model args.

```bash
source scripts/models/qwen3-32B.sh
```

### Other Qwen3 models (4B, 8B, 14B)

Replace the model config and checkpoint path:

```bash
# Example: Qwen3-8B
source scripts/models/qwen3-8B.sh
--hf-checkpoint /root/models/Qwen3-8B
```

### GPT-class open-weight reasoning models

For models using a Llama-style architecture, use the corresponding config:

```bash
source scripts/models/llama3.1-8B-Instruct.sh
--hf-checkpoint /root/models/your-model
```

For a new architecture, create a model config in `scripts/models/` following the existing examples (specify `--num-layers`, `--hidden-size`, `--ffn-hidden-size`, `--num-attention-heads`, etc.).

---

## 6. Reasoning Data Format

For reasoning models, the training data should contain a **chain-of-thought** in the assistant response.
The model learns to produce the full response including the reasoning trace.

### Format A: Think tags (recommended for QwQ / Qwen3-Thinking)

```json
{
  "role": "assistant",
  "content": "<think>\nStep 1: The cell shows high CD3D expression...\nStep 2: Combined with CD8A...\n</think>\nCD8+ cytotoxic T cell"
}
```

### Format B: Boxed answer (for math-style tasks)

```json
{
  "role": "assistant",
  "content": "The expression pattern suggests a myeloid lineage...\n\nAnswer: \\boxed{Monocyte}"
}
```

### Format C: Plain text (simplest)

```json
{
  "role": "assistant",
  "content": "This is a CD8+ cytotoxic T cell based on the expression of CD3D and CD8A markers."
}
```

The SFT loss is computed on the **entire** assistant response.
All text tokens in the assistant turns contribute to the loss (the `MultiTurnLossMaskGenerator` handles masking user turns automatically).

---

## 7. Converting Checkpoints After Training

After SFT training, convert the saved checkpoint back to HuggingFace format for inference:

```bash
# Megatron → HuggingFace
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /root/models/QwQ-32B_slime/iter_XXX/ \
  --output-dir /root/models/QwQ-32B-SFT \
  --origin-hf-dir /root/models/QwQ-32B
```

For FSDP, checkpoints are already in HuggingFace format and can be loaded directly.

---

## 8. GPU Memory Guide

| Model | Backend | GPUs | TP | Notes |
|---|---|---|---|---|
| Qwen3-8B | FSDP | 2 × H100 | — | `--gradient-checkpointing` |
| QwQ-32B | FSDP | 4 × H100 | — | `--gradient-checkpointing` |
| QwQ-32B | Megatron | 8 × H100 | 8 | Best throughput |
| QwQ-32B | Megatron | 4 × H100 | 4 | `--optimizer-cpu-offload` |

Reduce `--max-tokens-per-gpu` if you run out of memory.
