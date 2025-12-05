#!/usr/bin/env python3
"""
Split the dataset into train and eval sets.
Default: 99% train, 1% eval (roughly 12k eval samples)
"""

import json
import random
import os

def split_dataset(input_file, train_file, eval_file, eval_ratio=0.01, seed=42):
    """Split dataset into train and eval."""
    
    print(f"Loading dataset from: {input_file}")
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    print(f"Total samples: {len(dataset):,}")
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Calculate split point
    eval_size = int(len(dataset) * eval_ratio)
    train_size = len(dataset) - eval_size
    
    # Split
    train_data = dataset[:train_size]
    eval_data = dataset[train_size:]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_data):,} samples ({(1-eval_ratio)*100:.1f}%)")
    print(f"  Eval:  {len(eval_data):,} samples ({eval_ratio*100:.1f}%)")
    
    # Save train set
    print(f"\nSaving train set to: {train_file}")
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    
    # Save eval set
    print(f"Saving eval set to: {eval_file}")
    with open(eval_file, 'w') as f:
        json.dump(eval_data, f)
    
    print("\n✓ Dataset split complete!")
    print(f"\nUpdate your training script:")
    print(f'  DATA_PATH="./{train_file}"')
    print(f'  EVAL_DATA_PATH="./{eval_file}"')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="final_dataset.json",
                       help="Input dataset file")
    parser.add_argument("--train-output", type=str, default="final_dataset_train.json",
                       help="Output train file")
    parser.add_argument("--eval-output", type=str, default="final_dataset_eval.json",
                       help="Output eval file")
    parser.add_argument("--eval-ratio", type=float, default=0.01,
                       help="Ratio of data for eval (default: 0.01 = 1%)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        exit(1)
    
    split_dataset(args.input, args.train_output, args.eval_output, 
                  args.eval_ratio, args.seed)
