import argparse
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    # Load tokenizer from base model
    print(f"Loading tokenizer from {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    
    # Load the base model config but update it with LLaVA config from LoRA checkpoint
    print(f"Loading config from {args.model_path}")
    lora_cfg = AutoConfig.from_pretrained(args.model_path)
    
    # Load base LLM
    print(f"Loading base model from {args.model_base}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_base, 
        config=lora_cfg
    )
    
    # Load and merge LoRA weights
    print(f"Loading LoRA weights from {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    
    # Load non-LoRA trainables (mm_projector weights)
    non_lora_path = os.path.join(args.model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        print(f"Loading mm_projector weights from {non_lora_path}")
        non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
        model.load_state_dict(non_lora_trainables, strict=False)
        print("Loaded mm_projector weights successfully")
    else:
        print(f"Warning: {non_lora_path} not found!")
    
    # Save the merged model
    print(f"Saving merged model to {args.save_model_path}")
    os.makedirs(args.save_model_path, exist_ok=True)
    
    # Convert to bfloat16 (same as training) for numerical stability
    print("Converting to bfloat16 (same dtype as training)...")
    model = model.to(torch.bfloat16)
    
    # Move to CPU before saving
    model = model.cpu()
    
    model.save_pretrained(args.save_model_path, max_shard_size="2GB")
    tokenizer.save_pretrained(args.save_model_path)
    
    print(f"\n✓ Merge completed successfully!")
    print(f"✓ Merged model saved to: {args.save_model_path}")
    print(f"✓ This model is ready for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)
