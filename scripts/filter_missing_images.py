#!/usr/bin/env python3
"""
Filter out samples with missing images from the dataset.
Creates a new dataset file with only samples that have all images present.
"""

import json
import os
from pathlib import Path

def filter_dataset(dataset_path, image_folder, output_path):
    """Filter dataset to remove samples with missing images."""
    
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Checking images in: {image_folder}")
    print("")
    
    filtered_dataset = []
    skipped_samples = []
    
    for idx, sample in enumerate(dataset):
        if idx % 10000 == 0:
            print(f"Progress: {idx}/{len(dataset)} samples...", end='\r')
        
        # Get image path(s)
        image_paths = []
        if 'image' in sample:
            if isinstance(sample['image'], str):
                image_paths = [sample['image']]
            elif isinstance(sample['image'], list):
                image_paths = sample['image']
        
        # Check if all images exist
        all_exist = True
        for img_path in image_paths:
            full_path = os.path.join(image_folder, img_path)
            if not os.path.exists(full_path):
                all_exist = False
                break
        
        # Add to filtered dataset if all images exist
        if all_exist:
            filtered_dataset.append(sample)
        else:
            skipped_samples.append({
                'id': sample.get('id', f'sample_{idx}'),
                'image': sample.get('image'),
                'missing_images': [img for img in image_paths 
                                  if not os.path.exists(os.path.join(image_folder, img))]
            })
    
    print(f"\nProgress: {len(dataset)}/{len(dataset)} samples... Done!")
    print("")
    
    # Save filtered dataset
    print(f"Saving filtered dataset to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(filtered_dataset, f)
    
    # Save skipped samples report
    skipped_report_path = output_path.replace('.json', '_skipped.json')
    with open(skipped_report_path, 'w') as f:
        json.dump(skipped_samples, f, indent=2)
    
    print("")
    print("=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print(f"Original samples:  {len(dataset):,}")
    print(f"Filtered samples:  {len(filtered_dataset):,}")
    print(f"Skipped samples:   {len(skipped_samples):,}")
    print(f"Retention rate:    {len(filtered_dataset)/len(dataset)*100:.2f}%")
    print("")
    print(f"✓ Filtered dataset saved to:     {output_path}")
    print(f"✓ Skipped samples report saved to: {skipped_report_path}")
    print("=" * 80)
    print("")
    print("To use the filtered dataset, update your training script:")
    print(f"  Change: DATA_PATH=\"./final_dataset.json\"")
    print(f"  To:     DATA_PATH=\"./{output_path}\"")
    print("")

def main():
    dataset_path = "final_dataset.json"
    image_folder = "playground/data"
    output_path = "final_dataset_filtered.json"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return
    
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        return
    
    filter_dataset(dataset_path, image_folder, output_path)

if __name__ == "__main__":
    main()
