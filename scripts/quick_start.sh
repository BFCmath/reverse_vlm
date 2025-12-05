#!/bin/bash
# Quick Start Script for REVERSE-VLM
# This script helps you interactively set up your training environment

set -e

echo "=========================================="
echo "   REVERSE-VLM Quick Start Setup"
echo "=========================================="
echo ""
echo "This script will guide you through the setup process."
echo ""

# Step 1: Choose model
echo "Step 1: Choose your model architecture"
echo "---------------------------------------"
echo "1) LLaVA-v1.5-7B (Vicuna backbone) - Requires 8 GPUs"
echo "2) LLaVA-MORE-8B (Llama 3.1 backbone) - Requires 8 GPUs"
echo "3) Qwen2.5-VL-3B - Requires 4 GPUs"
echo ""
read -p "Enter your choice (1-3): " model_choice

case $model_choice in
    1)
        MODEL="llava_v15"
        MODEL_NAME="LLaVA-v1.5-7B"
        ;;
    2)
        MODEL="llava_more"
        MODEL_NAME="LLaVA-MORE-8B"
        ;;
    3)
        MODEL="qwen25vl"
        MODEL_NAME="Qwen2.5-VL-3B"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Selected: $MODEL_NAME"
echo ""

# Step 2: Check environment
echo "Step 2: Checking environment"
echo "---------------------------------------"

if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda/Miniconda first."
    exit 1
else
    echo "✓ Conda found"
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. CUDA may not be available."
    exit 1
else
    echo "✓ NVIDIA GPU detected"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "  Available GPUs: $GPU_COUNT"
fi

# Step 3: Environment setup
echo ""
echo "Step 3: Environment Setup"
echo "---------------------------------------"

read -p "Create/activate conda environment 'reverse'? (y/n): " create_env

if [ "$create_env" = "y" ]; then
    if conda env list | grep -q "^reverse "; then
        echo "Environment 'reverse' already exists."
        read -p "Recreate it? (y/n): " recreate
        if [ "$recreate" = "y" ]; then
            conda env remove -n reverse -y
            conda create -n reverse python=3.10 -y
        fi
    else
        conda create -n reverse python=3.10 -y
    fi
    
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate reverse"
    echo ""
    echo "Then install dependencies:"
    echo "  pip install --upgrade pip"
    echo "  pip install -e ."
    echo "  pip install -e \".[train]\""
    
    if [ "$MODEL" != "qwen25vl" ]; then
        echo "  pip install flash-attn --no-build-isolation --no-cache-dir"
    else
        echo "  pip install qwen-vl-utils"
    fi
fi

# Step 4: Download models
echo ""
echo "Step 4: Download Base Models"
echo "---------------------------------------"

if [ "$MODEL" = "llava_v15" ]; then
    echo "Need to download:"
    echo "  1. lmsys/vicuna-7b-v1.5 (~13GB)"
    echo "  2. LLaVA-v1.5 pretrained projector (~1GB)"
    
elif [ "$MODEL" = "llava_more" ]; then
    echo "Need to download:"
    echo "  1. meta-llama/Llama-3.1-8B-Instruct (~16GB, requires HF auth)"
    echo "  2. LLaVA-MORE pretrained projector (~1GB)"
    
elif [ "$MODEL" = "qwen25vl" ]; then
    echo "Need to download:"
    echo "  1. Qwen/Qwen2.5-VL-3B-Instruct (~6GB)"
fi

echo ""
read -p "Download models now? (y/n): " download_models

if [ "$download_models" = "y" ]; then
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Installing huggingface-cli..."
        pip install -U "huggingface_hub[cli]"
    fi
    
    bash scripts/download_all.sh
fi

# Step 5: Download dataset
echo ""
echo "Step 5: Download Training Dataset"
echo "---------------------------------------"
echo "REVERSE dataset: 1.3M samples (~1GB JSON)"
echo ""

read -p "Download training data now? (y/n): " download_data

if [ "$download_data" = "y" ]; then
    if [ ! -f "final_dataset.json" ]; then
        echo "Downloading REVERSE dataset..."
        huggingface-cli download tsunghanwu/reverse-instruct-1.3m \
            --repo-type dataset \
            --include "*.json" \
            --local-dir ./temp_data
        mv ./temp_data/*.json ./final_dataset.json
        rm -rf ./temp_data
        echo "✓ Dataset downloaded"
    else
        echo "✓ Dataset already exists"
    fi
    
    if [ "$MODEL" = "qwen25vl" ]; then
        if [ ! -f "final_dataset_subsample.json" ]; then
            echo "Creating 100k subsample for Qwen..."
            python scripts/subsample_data.py
            echo "✓ Subsample created"
        fi
    fi
fi

# Step 6: Image datasets
echo ""
echo "Step 6: Image Datasets"
echo "---------------------------------------"
echo "Required datasets (~200GB total):"
echo "  - COCO 2017"
echo "  - GQA"
echo "  - Visual Genome"
echo "  - OCR-VQA (manual download)"
echo "  - TextVQA (manual download)"
echo ""
echo "This will take several hours to download."
echo ""

read -p "Download image datasets now? (y/n): " download_images

if [ "$download_images" = "y" ]; then
    echo "Downloading image datasets..."
    echo "You can stop this and resume later if needed (Ctrl+C)"
    sleep 3
    
    mkdir -p playground/data
    cd playground/data
    
    # COCO
    if [ ! -d "coco/train2017" ]; then
        echo "Downloading COCO..."
        mkdir -p coco
        wget http://images.cocodataset.org/zips/train2017.zip
        wget http://images.cocodataset.org/zips/val2017.zip
        wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q train2017.zip -d coco/
        unzip -q val2017.zip -d coco/
        unzip -q annotations_trainval2017.zip -d coco/
        rm *.zip
    fi
    
    # GQA
    if [ ! -d "gqa/images" ]; then
        echo "Downloading GQA..."
        wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
        unzip -q images.zip
        mkdir -p gqa
        mv images gqa/
        rm images.zip
    fi
    
    # Visual Genome
    if [ ! -d "vg/VG_100K" ]; then
        echo "Downloading Visual Genome..."
        mkdir -p vg
        wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
        wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
        unzip -q images.zip -d vg/
        unzip -q images2.zip -d vg/
        rm *.zip
    fi
    
    cd ../..
fi

# Step 7: Add special tokens
echo ""
echo "Step 7: Add Special Tokens to Model"
echo "---------------------------------------"

read -p "Add special tokens (<SPAN>, </CN>, </UN>) to model? (y/n): " add_tokens

if [ "$add_tokens" = "y" ]; then
    if [ "$MODEL" = "llava_v15" ]; then
        python scripts/add_new_token_to_llava.py \
            --model-name ./vicuna-7b-v1.5 \
            --output-dir ./vicuna1.5_7b_with_new_tokens
    elif [ "$MODEL" = "llava_more" ]; then
        python scripts/add_new_token_to_llava.py \
            --model-name ./llama-3.1-8b-instruct \
            --output-dir ./llama3.1_instruct_with_new_tokens
    elif [ "$MODEL" = "qwen25vl" ]; then
        python scripts/add_new_token_to_qwen.py \
            --model-name ./qwen2.5-vl-3b-instruct \
            --output-dir ./qwen25_vlm_3b_with_new_tokens
    fi
fi

# Step 8: Configure training
echo ""
echo "Step 8: Configure Training Script"
echo "---------------------------------------"

read -p "How many GPUs will you use for training? " num_gpus

if [ "$MODEL" = "llava_v15" ] || [ "$MODEL" = "llava_more" ]; then
    if [ "$num_gpus" -lt 8 ]; then
        echo "⚠ Warning: LLaVA models are tested with 8 GPUs."
        echo "  You may need to adjust batch size settings."
    fi
fi

echo ""
echo "GPU IDs to use (e.g., 0,1,2,3 or 4,5,6,7):"
read -p "Enter GPU IDs: " gpu_ids

echo ""
echo "Creating custom training script..."

if [ "$MODEL" = "llava_v15" ]; then
    SCRIPT="scripts/train/my_reverse_llavav15_7b_sft.sh"
    cp scripts/train/reverse_llavav15_7b_sft.sh $SCRIPT
    sed -i "s/GPU_SETTINGS=\"localhost:.*\"/GPU_SETTINGS=\"localhost:$gpu_ids\"/" $SCRIPT
    echo "✓ Created: $SCRIPT"
    
elif [ "$MODEL" = "llava_more" ]; then
    SCRIPT="scripts/train/my_reverse_llavamore_8b_sft.sh"
    cp scripts/train/reverse_llavamore_8b_sft.sh $SCRIPT
    sed -i "s/GPU_SETTINGS=\"localhost:.*\"/GPU_SETTINGS=\"localhost:$gpu_ids\"/" $SCRIPT
    echo "✓ Created: $SCRIPT"
    
elif [ "$MODEL" = "qwen25vl" ]; then
    SCRIPT="scripts/train/my_reverse_qwen25vl_3b_sft.sh"
    cp scripts/train/reverse_qwen25vl_3b_sft.sh $SCRIPT
    sed -i "s/GPU_SETTINGS=\"localhost:.*\"/GPU_SETTINGS=\"localhost:$gpu_ids\"/" $SCRIPT
    echo "✓ Created: $SCRIPT"
fi

# Summary
echo ""
echo "=========================================="
echo "   Setup Summary"
echo "=========================================="
echo ""
echo "Model: $MODEL_NAME"
echo "GPUs: $num_gpus ($gpu_ids)"
echo "Training script: $SCRIPT"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   conda activate reverse"
echo ""
echo "2. Verify all data is downloaded:"
echo "   - Training data: final_dataset.json"
echo "   - Images: playground/data/"
echo "   - Model with tokens: *_with_new_tokens/"
echo ""
echo "3. (Optional) Configure WandB:"
echo "   wandb login"
echo ""
echo "4. Start training:"
echo "   bash $SCRIPT"
echo ""
echo "5. Monitor training:"
echo "   - Check WandB dashboard"
echo "   - Watch terminal output"
echo "   - Check checkpoints/ directory"
echo ""
echo "For detailed guidance, see:"
echo "   - SETUP_GUIDE.md (complete setup instructions)"
echo "   - TRAINING_CHECKLIST.md (tracking progress)"
echo ""
echo "Good luck! 🚀"
echo ""
