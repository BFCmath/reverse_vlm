#!/bin/bash
# Complete download script for REVERSE-VLM training
# This script downloads all required models and datasets
# Modify paths and choices based on your needs

set -e  # Exit on error

echo "=========================================="
echo "REVERSE-VLM Complete Setup Script"
echo "=========================================="
echo ""

# Configuration - Change these as needed
WORKSPACE_ROOT=$(pwd)
MODEL_CHOICE="llava_v15"  # Options: llava_v15, llava_more, qwen25vl

echo "Workspace: $WORKSPACE_ROOT"
echo "Model choice: $MODEL_CHOICE"
echo ""

# ==========================================
# 1. Download Training Data
# ==========================================
echo "Step 1: Downloading training data..."

if [ ! -f "$WORKSPACE_ROOT/final_dataset.json" ]; then
    echo "Downloading REVERSE dataset (1.3M samples)..."
    huggingface-cli download tsunghanwu/reverse-instruct-1.3m \
        --repo-type dataset \
        --include "*.json" \
        --local-dir ./temp_data
    
    mv ./temp_data/*.json ./final_dataset.json
    rm -rf ./temp_data
    echo "✓ Dataset downloaded: final_dataset.json"
else
    echo "✓ Dataset already exists: final_dataset.json"
fi

# ==========================================
# 2. Download Base Models
# ==========================================
echo ""
echo "Step 2: Downloading base models..."

if [ "$MODEL_CHOICE" == "llava_v15" ]; then
    # LLaVA-v1.5-7B setup
    if [ ! -d "$WORKSPACE_ROOT/vicuna-7b-v1.5" ]; then
        echo "Downloading Vicuna-7B-v1.5..."
        huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir ./vicuna-7b-v1.5
        echo "✓ Vicuna-7B-v1.5 downloaded"
    else
        echo "✓ Vicuna-7B-v1.5 already exists"
    fi
    
    # Download pretrained projector
    if [ ! -f "$WORKSPACE_ROOT/checkpoints/llava_v15_pretraining/mm_projector.bin" ]; then
        echo "Downloading LLaVA-v1.5 pretrained projector..."
        mkdir -p checkpoints/llava_v15_pretraining
        huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 \
            --include "mm_projector.bin" \
            --local-dir checkpoints/llava_v15_pretraining
        echo "✓ Projector downloaded"
    else
        echo "✓ Projector already exists"
    fi

elif [ "$MODEL_CHOICE" == "llava_more" ]; then
    # LLaVA-MORE-8B setup
    if [ ! -d "$WORKSPACE_ROOT/llama-3.1-8b-instruct" ]; then
        echo "Downloading Llama-3.1-8B-Instruct..."
        echo "Note: This may require HF authentication for gated models"
        huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ./llama-3.1-8b-instruct
        echo "✓ Llama-3.1-8B downloaded"
    else
        echo "✓ Llama-3.1-8B already exists"
    fi
    
    # Download pretrained projector
    if [ ! -f "$WORKSPACE_ROOT/checkpoints/llava_v31_pretraining/mm_projector.bin" ]; then
        echo "Downloading LLaVA-MORE pretrained projector..."
        mkdir -p checkpoints/llava_v31_pretraining
        huggingface-cli download aimagelab/LLaVA_MORE-llama_3_1-8B-pretrain \
            --include "mm_projector.bin" \
            --local-dir checkpoints/llava_v31_pretraining
        echo "✓ Projector downloaded"
    else
        echo "✓ Projector already exists"
    fi

elif [ "$MODEL_CHOICE" == "qwen25vl" ]; then
    # Qwen2.5-VL-3B setup
    if [ ! -d "$WORKSPACE_ROOT/qwen2.5-vl-3b-instruct" ]; then
        echo "Downloading Qwen2.5-VL-3B-Instruct..."
        huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./qwen2.5-vl-3b-instruct
        echo "✓ Qwen2.5-VL-3B downloaded"
    else
        echo "✓ Qwen2.5-VL-3B already exists"
    fi
    
    # Create subsampled dataset
    if [ ! -f "$WORKSPACE_ROOT/final_dataset_subsample.json" ]; then
        echo "Creating 100k subsample for Qwen..."
        python scripts/subsample_data.py
        echo "✓ Subsampled dataset created"
    else
        echo "✓ Subsampled dataset already exists"
    fi
else
    echo "Error: Invalid MODEL_CHOICE. Use: llava_v15, llava_more, or qwen25vl"
    exit 1
fi

# ==========================================
# 3. Download Image Datasets
# ==========================================
echo ""
echo "Step 3: Downloading image datasets..."
echo "This will take significant time and storage (~200GB+)"
echo ""

mkdir -p playground/data
cd playground/data

# COCO 2017
if [ ! -d "coco/train2017" ]; then
    echo "Downloading COCO 2017..."
    mkdir -p coco
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    
    unzip -q train2017.zip -d coco/
    unzip -q val2017.zip -d coco/
    unzip -q annotations_trainval2017.zip -d coco/
    
    rm train2017.zip val2017.zip annotations_trainval2017.zip
    echo "✓ COCO downloaded"
else
    echo "✓ COCO already exists"
fi

# GQA
if [ ! -d "gqa/images" ]; then
    echo "Downloading GQA..."
    mkdir -p gqa
    wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
    unzip -q images.zip
    mv images gqa/
    rm images.zip
    echo "✓ GQA downloaded"
else
    echo "✓ GQA already exists"
fi

# Visual Genome
if [ ! -d "vg/VG_100K" ]; then
    echo "Downloading Visual Genome..."
    mkdir -p vg
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
    
    unzip -q images.zip -d vg/
    unzip -q images2.zip -d vg/
    
    rm images.zip images2.zip
    echo "✓ Visual Genome downloaded"
else
    echo "✓ Visual Genome already exists"
fi

# Note for other datasets
echo ""
echo "Note: OCR-VQA and TextVQA require manual download:"
echo "  - OCR-VQA: https://ocr-vqa.github.io/"
echo "  - TextVQA: https://textvqa.org/dataset"
echo "  Place them in:"
echo "    playground/data/ocr_vqa/images/"
echo "    playground/data/textvqa/train_images/"
echo ""

cd $WORKSPACE_ROOT

# ==========================================
# 4. Add Special Tokens
# ==========================================
echo ""
echo "Step 4: Adding special tokens to models..."

if [ "$MODEL_CHOICE" == "llava_v15" ]; then
    if [ ! -d "$WORKSPACE_ROOT/vicuna1.5_7b_with_new_tokens" ]; then
        echo "Adding tokens to Vicuna-7B-v1.5..."
        python scripts/add_new_token_to_llava.py \
            --model-name ./vicuna-7b-v1.5 \
            --output-dir ./vicuna1.5_7b_with_new_tokens
        echo "✓ Tokens added to Vicuna model"
    else
        echo "✓ Vicuna model with tokens already exists"
    fi

elif [ "$MODEL_CHOICE" == "llava_more" ]; then
    if [ ! -d "$WORKSPACE_ROOT/llama3.1_instruct_with_new_tokens" ]; then
        echo "Adding tokens to Llama-3.1-8B..."
        python scripts/add_new_token_to_llava.py \
            --model-name ./llama-3.1-8b-instruct \
            --output-dir ./llama3.1_instruct_with_new_tokens
        echo "✓ Tokens added to Llama model"
    else
        echo "✓ Llama model with tokens already exists"
    fi

elif [ "$MODEL_CHOICE" == "qwen25vl" ]; then
    if [ ! -d "$WORKSPACE_ROOT/qwen25_vlm_3b_with_new_tokens" ]; then
        echo "Adding tokens to Qwen2.5-VL-3B..."
        python scripts/add_new_token_to_qwen.py \
            --model-name ./qwen2.5-vl-3b-instruct \
            --output-dir ./qwen25_vlm_3b_with_new_tokens
        echo "✓ Tokens added to Qwen model"
    else
        echo "✓ Qwen model with tokens already exists"
    fi
fi

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify image datasets are complete (especially OCR-VQA and TextVQA)"
echo "2. Configure GPU settings in scripts/train/*.sh"
echo "3. Start training:"

if [ "$MODEL_CHOICE" == "llava_v15" ]; then
    echo "   bash scripts/train/reverse_llavav15_7b_sft.sh"
elif [ "$MODEL_CHOICE" == "llava_more" ]; then
    echo "   bash scripts/train/reverse_llavamore_8b_sft.sh"
elif [ "$MODEL_CHOICE" == "qwen25vl" ]; then
    echo "   bash scripts/train/reverse_qwen25vl_3b_sft.sh"
fi

echo ""
echo "For detailed instructions, see: SETUP_GUIDE.md"
echo ""
