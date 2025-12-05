#!/bin/bash
# Pre-training verification script
# Run this before starting training to ensure everything is ready

echo "=========================================="
echo "  REVERSE-VLM Setup Verification"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Conda environment
echo "1. Checking conda environment..."
if conda env list | grep -q "^reverse "; then
    echo -e "${GREEN}✓${NC} Conda environment 'reverse' exists"
else
    echo -e "${RED}✗${NC} Conda environment 'reverse' not found"
    echo "   Run: conda create -n reverse python=3.10 -y"
    ((ERRORS++))
fi

# Check 2: GPU availability
echo ""
echo "2. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓${NC} Found $GPU_COUNT GPU(s)"
    if [ "$GPU_COUNT" -lt 8 ]; then
        echo -e "${YELLOW}⚠${NC}  You have fewer than 8 GPUs. Use single-GPU or multi-GPU scripts."
        ((WARNINGS++))
    fi
else
    echo -e "${RED}✗${NC} nvidia-smi not found - no GPU detected"
    ((ERRORS++))
fi

# Check 3: Base model with tokens
echo ""
echo "3. Checking model with special tokens..."
if [ -d "./vicuna1.5_7b_with_new_tokens" ]; then
    echo -e "${GREEN}✓${NC} vicuna1.5_7b_with_new_tokens exists"
    
    # Check if tokenizer works
    if conda run -n reverse python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('./vicuna1.5_7b_with_new_tokens')" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Tokenizer loads successfully"
    else
        echo -e "${RED}✗${NC} Tokenizer fails to load - may be corrupted"
        echo "   Run: rm -rf ./vicuna1.5_7b_with_new_tokens"
        echo "   Then: conda activate reverse && python scripts/add_new_token_to_llava.py --model-name ./vicuna-7b-v1.5 --output-dir ./vicuna1.5_7b_with_new_tokens"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗${NC} vicuna1.5_7b_with_new_tokens not found"
    echo "   Run: conda activate reverse && python scripts/add_new_token_to_llava.py --model-name ./vicuna-7b-v1.5 --output-dir ./vicuna1.5_7b_with_new_tokens"
    ((ERRORS++))
fi

# Check 4: Pretrained projector
echo ""
echo "4. Checking pretrained projector..."
if [ -f "checkpoints/llava_v15_pretraining/mm_projector.bin" ]; then
    SIZE=$(ls -lh checkpoints/llava_v15_pretraining/mm_projector.bin | awk '{print $5}')
    echo -e "${GREEN}✓${NC} mm_projector.bin exists ($SIZE)"
else
    echo -e "${RED}✗${NC} mm_projector.bin not found"
    echo "   Run: huggingface-cli download liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5 --include 'mm_projector.bin' --local-dir checkpoints/llava_v15_pretraining"
    ((ERRORS++))
fi

# Check 5: Training data
echo ""
echo "5. Checking training data..."
if [ -f "./final_dataset.json" ]; then
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('./final_dataset.json'))))" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓${NC} final_dataset.json exists ($SAMPLE_COUNT samples)"
    if [ "$SAMPLE_COUNT" -eq 0 ]; then
        echo -e "${RED}✗${NC} Dataset appears to be empty or invalid"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗${NC} final_dataset.json not found"
    echo "   Run: huggingface-cli download tsunghanwu/reverse-instruct-1.3m --repo-type dataset --include '*.json' --local-dir ./data_json && mv ./data_json/*.json ./final_dataset.json"
    ((ERRORS++))
fi

# Check 6: Image datasets
echo ""
echo "6. Checking image datasets..."
DATASETS=("coco/train2017" "gqa/images" "vg/VG_100K" "textvqa/train_images" "ocr_vqa/images")
for ds in "${DATASETS[@]}"; do
    if [ -d "playground/data/$ds" ]; then
        IMG_COUNT=$(find "playground/data/$ds" -type f \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l)
        echo -e "${GREEN}✓${NC} $ds exists ($IMG_COUNT images)"
    else
        echo -e "${YELLOW}⚠${NC}  $ds not found"
        ((WARNINGS++))
    fi
done

if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠${NC}  Some image datasets missing - training may fail if they're needed"
fi

# Check 7: DeepSpeed config
echo ""
echo "7. Checking DeepSpeed config..."
if [ -f "./scripts/train/zero2.json" ]; then
    echo -e "${GREEN}✓${NC} zero2.json exists"
else
    echo -e "${RED}✗${NC} zero2.json not found"
    ((ERRORS++))
fi

if [ -f "./scripts/train/zero3.json" ]; then
    echo -e "${GREEN}✓${NC} zero3.json exists"
else
    echo -e "${RED}✗${NC} zero3.json not found"
    ((ERRORS++))
fi

# Check 8: Disk space
echo ""
echo "8. Checking disk space..."
AVAILABLE=$(df -h . | tail -1 | awk '{print $4}')
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "   Available: $AVAILABLE"
if [ "$AVAILABLE_GB" -lt 100 ]; then
    echo -e "${YELLOW}⚠${NC}  Low disk space - recommend at least 100GB free"
    ((WARNINGS++))
else
    echo -e "${GREEN}✓${NC} Sufficient disk space"
fi

# Summary
echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You're ready to train. Run:"
    echo "  bash scripts/train/reverse_llavav15_7b_sft_single_gpu.sh"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    echo ""
    echo "You can proceed with training, but some features may not work."
    echo ""
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    fi
    echo ""
    echo "Please fix the errors above before training."
    echo "See SETUP_GUIDE.md for detailed instructions."
    echo ""
    exit 1
fi
