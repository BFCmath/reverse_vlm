# Complete Setup Guide for REVERSE-VLM Training

This guide walks you through everything needed to replicate the training results for REVERSE-VLM.

## 📋 Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Download Base Models](#2-download-base-models)
3. [Download Training Data](#3-download-training-data)
4. [Download Image Datasets](#4-download-image-datasets)
5. [Prepare Models with Special Tokens](#5-prepare-models-with-special-tokens)
6. [Training](#6-training)
7. [Merge LoRA Weights](#7-merge-lora-weights-llava-only)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Environment Setup

### For LLaVA Models (LLaVA-v1.5-7B and LLaVA-MORE-8B):

```bash
# Create conda environment
conda create -n reverse python=3.10 -y
conda activate reverse

# Install dependencies
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

### For Qwen2.5-VL Models:

Follow the installation guide from [Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune):

```bash
conda create -n qwen_reverse python=3.10 -y
conda activate qwen_reverse

# Install required packages
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
# Additional Qwen-specific dependencies
pip install qwen-vl-utils
```

---

## 2. Download Base Models

You need to download the base models and pretrained projectors:

### Option A: LLaVA-v1.5-7B (Vicuna backbone)

```bash
# Download base LLM
cd /path/to/reverse_vlm
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ./vicuna-7b-v1.5

# Download pretrained projector
mkdir -p checkpoints/llava_v15_pretraining
huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 \
    --include "mm_projector.bin" \
    --local-dir checkpoints/llava_v15_pretraining
```

### Option B: LLaVA-MORE-8B (Llama-3.1 backbone)

```bash
# Download base LLM (requires HF token if gated)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-instruct

# Download pretrained projector
mkdir -p checkpoints/llava_v31_pretraining
huggingface-cli download aimagelab/LLaVA_MORE-llama_3_1-8B-pretrain \
    --include "mm_projector.bin" \
    --local-dir checkpoints/llava_v31_pretraining
```

### Option C: Qwen2.5-VL-3B

```bash
# Download the full model
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen2.5-vl-3b-instruct
```

---

## 3. Download Training Data

Download the REVERSE training dataset from HuggingFace:

```bash
# Download the dataset JSON file
huggingface-cli download tsunghanwu/reverse-instruct-1.3m \
    --repo-type dataset \
    --include "*.json" \
    --local-dir ./data_json

# Move to project root
mv ./data_json/*.json ./final_dataset.json
```

### For Qwen2.5-VL (Subsample to 100k):

```bash
# Subsample the dataset
python scripts/subsample_data.py
# This creates: ./final_dataset_subsample.json
```

---

## 4. Download Image Datasets

All training data uses images from existing VLM datasets. Set up the image folder structure:

```bash
mkdir -p playground/data
cd playground/data
```

### Download Required Datasets:

#### COCO (2017)
```bash
# Download and extract
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d coco/
unzip val2017.zip -d coco/
unzip test2017.zip -d coco/
unzip annotations_trainval2017.zip -d coco/
```

#### GQA
```bash
# Download from official source
wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
unzip images.zip -d gqa/
```

#### OCR-VQA
```bash
# Download from the official dataset page
# https://ocr-vqa.github.io/
# Place images in: playground/data/ocr_vqa/images/
```

#### TextVQA
```bash
# Download from https://textvqa.org/dataset
# Place train_images in: playground/data/textvqa/train_images/
```

#### Visual Genome (VG)
```bash
# Download VG_100K parts 1 and 2
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip images.zip.1 -d vg/
unzip images2.zip -d vg/
```

### Expected Data Structure:

After downloading, your structure should look like:

```
playground/data/
├── coco
│   ├── annotations/
│   ├── test2017/
│   ├── train2017/
│   └── val2017/
├── gqa
│   └── images/
├── ocr_vqa
│   └── images/
├── textvqa
│   └── train_images/
└── vg
    ├── VG_100K/
    └── VG_100K_2/
```

---

## 5. Prepare Models with Special Tokens

REVERSE uses three special tokens: `<SPAN>`, `</CN>`, `</UN>`. Add them to your base models:

**⚠️ IMPORTANT:** Activate your training environment BEFORE running these scripts!

```bash
# Activate the environment first
conda activate reverse  # or qwen_reverse for Qwen models
```

### For LLaVA (Vicuna-7b-v1.5):

```bash
python scripts/add_new_token_to_llava.py \
    --model-name ./vicuna-7b-v1.5 \
    --output-dir ./vicuna1.5_7b_with_new_tokens
```

### For LLaVA-MORE (Llama-3.1-8B):

```bash
python scripts/add_new_token_to_llava.py \
    --model-name ./llama-3.1-8b-instruct \
    --output-dir ./llama3.1_instruct_with_new_tokens
```

### For Qwen2.5-VL:

```bash
python scripts/add_new_token_to_qwen.py \
    --model-name ./qwen2.5-vl-3b-instruct \
    --output-dir ./qwen25_vlm_3b_with_new_tokens
```

**Verify it worked:**
```bash
python -c "from transformers import AutoTokenizer; \
    tok = AutoTokenizer.from_pretrained('./vicuna1.5_7b_with_new_tokens'); \
    print('✓ Success! Vocab size:', tok.vocab_size); \
    print('✓ Has <SPAN>:', '<SPAN>' in tok.get_vocab())"
```

---

## 6. Training

### Configure GPU Settings

**First, check how many GPUs you have:**

```bash
nvidia-smi --list-gpus | wc -l
```

**Important:** The default scripts are configured for **8 GPUs**. If you have fewer GPUs, you MUST modify the settings:

#### For Single GPU (1 GPU):
Use the dedicated single-GPU scripts (already configured):
```bash
# Single GPU scripts available:
scripts/train/reverse_llavav15_7b_sft_single_gpu.sh
```

Or manually edit the training script:
```bash
# In scripts/train/*.sh, modify:
GPU_SETTINGS="localhost:0"  # Single GPU
--per_device_train_batch_size 1  # Reduce from 8
--gradient_accumulation_steps 32  # Increase from 4 to maintain effective batch size
--deepspeed ./scripts/train/zero2.json  # Use zero2 instead of zero3 for single GPU
```

#### For Multiple GPUs (2-7 GPUs):
```bash
# Example for 4 GPUs:
GPU_SETTINGS="localhost:0,1,2,3"
--per_device_train_batch_size 4  # Adjust based on GPU memory
--gradient_accumulation_steps 8  # Adjust to maintain global batch size

# Example for 2 GPUs:
GPU_SETTINGS="localhost:0,1"
--per_device_train_batch_size 2
--gradient_accumulation_steps 16
```

#### For 8 GPUs (Original Configuration):
No changes needed - use the default scripts as-is.

**Port Configuration:**
```bash
MASTER_PORT="19487"  # Change if port is in use
```

### Train LLaVA-v1.5-7B:

**For 8 GPUs (original):**
```bash
bash scripts/train/reverse_llavav15_7b_sft.sh
```

**For 1 GPU:**
```bash
bash scripts/train/reverse_llavav15_7b_sft_single_gpu.sh
```

**Key settings:**
- Uses LoRA (r=128, alpha=256)
- Original: 8 GPUs with per_device_batch_size=8, grad_accum=4 (global batch: 256)
- Single GPU: 1 GPU with per_device_batch_size=1, grad_accum=32 (global batch: 32)
- Learning rate: 2e-4
- Training: 1 epoch on full 1.3M dataset
- Output: `checkpoints/reverse_v15_7b/` (or `reverse_v15_7b_single_gpu/`)
- **Single GPU training time: ~7-10 days** (vs 20-24 hours on 8 GPUs)

### Train LLaVA-MORE-8B:

```bash
bash scripts/train/reverse_llavamore_8b_sft.sh
```

**Key settings:**
- Uses LoRA (r=128, alpha=256)
- 8 GPUs with per_device_batch_size=8
- Learning rate: 1e-4 (lower than v1.5)
- Training: 1 epoch on full 1.3M dataset
- Output: `checkpoints/reverse_v31_8b/`

### Train Qwen2.5-VL-3B:

```bash
bash scripts/train/reverse_qwen25vl_3b_sft.sh
```

**Key settings:**
- Full model finetuning (no LoRA)
- 4 GPUs with per_device_batch_size=4
- Learning rate: 5e-5
- Training: 1 epoch on **100k subset** (fair comparison)
- Output: `output/reverse_qwen25vl_3b/`

### Training Monitoring

All scripts use WandB for logging. Make sure you:

```bash
# Login to wandb
wandb login

# Or disable it by editing the scripts:
# Change: --report_to wandb
# To:     --report_to none
```

---

## 7. Merge LoRA Weights (LLaVA Only)

After training LLaVA models, merge LoRA adapters with the base model:

### For LLaVA-v1.5:

```bash
python llava/merge_lora_weights.py \
    --model-path checkpoints/reverse_v15_7b \
    --model-base ./vicuna1.5_7b_with_new_tokens \
    --save-model-path checkpoints/reverse_v15_7b_merged
```

### For LLaVA-MORE:

```bash
python llava/merge_lora_weights.py \
    --model-path checkpoints/reverse_v31_8b \
    --model-base ./llama3.1_instruct_with_new_tokens \
    --save-model-path checkpoints/reverse_v31_8b_merged
```

**Important Naming Convention:**
- LoRA checkpoint directory must contain `llava_lora` in the path
- Final merged model path must contain `llava`
- This is required for LLaVA's internal model loading logic

### Qwen Models

No merging needed - Qwen models are directly fine-tuned (no LoRA).

---

## 8. Troubleshooting

### Common Issues:

#### Out of Memory (OOM)
```bash
# Reduce batch size in training scripts
--per_device_train_batch_size 4  # instead of 8
--gradient_accumulation_steps 8  # increase to maintain effective batch size
```

#### Port Already in Use
```bash
# Change MASTER_PORT in training scripts
MASTER_PORT="29500"  # or any free port
```

#### NCCL Errors with Multi-GPU
```bash
# Try disabling P2P/IB in training scripts:
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

#### Missing Images
Check that image paths in `final_dataset.json` match your folder structure:
```bash
# Image paths should be relative to playground/data/
# e.g., "coco/train2017/000000123456.jpg"
```

#### DeepSpeed Compilation Issues
```bash
# Reinstall with specific CUDA version
pip uninstall deepspeed
DS_BUILD_OPS=1 pip install deepspeed --no-cache-dir
```

---

## 9. Expected Training Times

Approximate training times on 8x A100 (80GB):

- **LLaVA-v1.5-7B**: ~20-24 hours (1.3M samples, 1 epoch)
- **LLaVA-MORE-8B**: ~24-28 hours (1.3M samples, 1 epoch)
- **Qwen2.5-VL-3B** (4 GPUs): ~3-4 hours (100k samples, 1 epoch)

---

## 10. Verification

After training, verify your checkpoints:

```bash
# Check checkpoint structure
ls -lh checkpoints/reverse_v15_7b/
# Should contain: adapter_config.json, adapter_model.bin, mm_projector.bin, etc.

# Test loading (simple check)
python -c "from llava.model.builder import load_pretrained_model; \
    load_pretrained_model('checkpoints/reverse_v15_7b_merged', None, 'llava-v1.5-7b')"
```

---

## 11. Next Steps

After training:
1. **Evaluation**: Use scripts in `scripts/eval/` to evaluate on benchmarks
2. **Download eval data**: See README - download from [Google Drive](https://drive.google.com/file/d/1gdGFNFUAe09dAObVK3Riyr-4ejxYqMSt/view?usp=sharing)
3. **Inference**: Load your trained model for inference

---

## Quick Start Checklist

- [ ] Environment installed (`conda env`, dependencies)
- [ ] Base models downloaded (Vicuna/Llama/Qwen)
- [ ] Projectors downloaded (for LLaVA models)
- [ ] Training data JSON downloaded (`final_dataset.json`)
- [ ] Image datasets downloaded and organized
- [ ] Special tokens added to models
- [ ] GPU settings configured in training scripts
- [ ] WandB configured (or disabled)
- [ ] Training launched
- [ ] (LLaVA only) LoRA weights merged

Good luck with your training! 🚀
