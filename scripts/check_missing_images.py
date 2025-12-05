#!/usr/bin/env python3
"""
Check which images referenced in final_dataset.json are missing from playground/data
and generate a report with statistics and missing file list.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import sys

def check_missing_images(dataset_path, image_folder):
    """Check for missing images and generate statistics."""
    
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"Total samples in dataset: {len(dataset)}")
    print(f"Checking images in: {image_folder}")
    print("")
    
    # Statistics
    stats = defaultdict(lambda: {"total": 0, "missing": 0, "found": 0})
    missing_images = []
    missing_by_dataset = defaultdict(list)
    
    # Check each sample
    for idx, sample in enumerate(dataset):
        if idx % 10000 == 0:
            print(f"Progress: {idx}/{len(dataset)} samples checked...", end='\r')
        
        # Get image path(s)
        image_paths = []
        if 'image' in sample:
            if isinstance(sample['image'], str):
                image_paths = [sample['image']]
            elif isinstance(sample['image'], list):
                image_paths = sample['image']
        
        # Check each image
        for img_path in image_paths:
            # Determine which dataset this image belongs to
            dataset_name = img_path.split('/')[0] if '/' in img_path else 'unknown'
            stats[dataset_name]["total"] += 1
            
            # Full path to image
            full_path = os.path.join(image_folder, img_path)
            
            if not os.path.exists(full_path):
                stats[dataset_name]["missing"] += 1
                missing_images.append({
                    "image_path": img_path,
                    "dataset": dataset_name,
                    "sample_id": sample.get('id', f'sample_{idx}'),
                    "full_path": full_path
                })
                missing_by_dataset[dataset_name].append(img_path)
            else:
                stats[dataset_name]["found"] += 1
    
    print(f"\nProgress: {len(dataset)}/{len(dataset)} samples checked... Done!")
    print("")
    
    # Print statistics
    print("=" * 80)
    print("IMAGE STATISTICS BY DATASET")
    print("=" * 80)
    
    total_images = 0
    total_missing = 0
    total_found = 0
    
    for dataset_name in sorted(stats.keys()):
        total = stats[dataset_name]["total"]
        missing = stats[dataset_name]["missing"]
        found = stats[dataset_name]["found"]
        missing_pct = (missing / total * 100) if total > 0 else 0
        
        total_images += total
        total_missing += missing
        total_found += found
        
        status = "❌ MISSING DATA" if missing_pct > 50 else "⚠️  PARTIAL" if missing > 0 else "✓ OK"
        
        print(f"\n{dataset_name.upper()}:")
        print(f"  Total images:   {total:,}")
        print(f"  Found:          {found:,}")
        print(f"  Missing:        {missing:,} ({missing_pct:.1f}%)")
        print(f"  Status:         {status}")
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL SUMMARY:")
    print(f"  Total images:   {total_images:,}")
    print(f"  Found:          {total_found:,}")
    print(f"  Missing:        {total_missing:,} ({total_missing/total_images*100:.1f}%)")
    print(f"{'=' * 80}\n")
    
    return missing_images, missing_by_dataset, stats

def save_reports(missing_images, missing_by_dataset, stats):
    """Save detailed reports to JSON files."""
    
    # Save complete missing images list
    output_file = "missing_images_report.json"
    with open(output_file, 'w') as f:
        json.dump({
            "total_missing": len(missing_images),
            "missing_images": missing_images,
            "statistics": dict(stats)
        }, f, indent=2)
    print(f"✓ Detailed report saved to: {output_file}")
    
    # Save missing images grouped by dataset
    output_file_by_dataset = "missing_images_by_dataset.json"
    with open(output_file_by_dataset, 'w') as f:
        json.dump({
            dataset: {
                "count": len(images),
                "images": images
            } for dataset, images in missing_by_dataset.items()
        }, f, indent=2)
    print(f"✓ Missing images by dataset saved to: {output_file_by_dataset}")
    
    # Save simple list of missing paths (for easy downloading)
    output_file_list = "missing_images_paths.txt"
    with open(output_file_list, 'w') as f:
        for img in missing_images:
            f.write(img['image_path'] + '\n')
    print(f"✓ Simple path list saved to: {output_file_list}")
    
    # Save download instructions by dataset
    output_instructions = "download_instructions.txt"
    with open(output_instructions, 'w') as f:
        f.write("MISSING DATASETS AND DOWNLOAD INSTRUCTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        for dataset_name in sorted(missing_by_dataset.keys()):
            count = len(missing_by_dataset[dataset_name])
            total = stats[dataset_name]["total"]
            f.write(f"{dataset_name.upper()}: {count}/{total} images missing\n")
            
            # Provide download instructions
            if dataset_name == 'coco':
                f.write("  Download: http://cocodataset.org/#download\n")
                f.write("  Commands:\n")
                f.write("    wget http://images.cocodataset.org/zips/train2017.zip\n")
                f.write("    wget http://images.cocodataset.org/zips/val2017.zip\n")
                f.write("    unzip train2017.zip -d playground/data/coco/\n")
                f.write("    unzip val2017.zip -d playground/data/coco/\n")
            
            elif dataset_name == 'gqa':
                f.write("  Download: https://cs.stanford.edu/people/dorarad/gqa/\n")
                f.write("  Commands:\n")
                f.write("    wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip\n")
                f.write("    unzip images.zip\n")
                f.write("    mv images playground/data/gqa/\n")
            
            elif dataset_name == 'vg':
                f.write("  Download: https://visualgenome.org/api/v0/api_home.html\n")
                f.write("  Commands:\n")
                f.write("    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip\n")
                f.write("    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip\n")
                f.write("    unzip images.zip -d playground/data/vg/\n")
                f.write("    unzip images2.zip -d playground/data/vg/\n")
            
            elif dataset_name == 'textvqa':
                f.write("  Download: https://textvqa.org/dataset\n")
                f.write("  Place in: playground/data/textvqa/train_images/\n")
            
            elif dataset_name == 'ocr_vqa':
                f.write("  Download: https://ocr-vqa.github.io/\n")
                f.write("  Place in: playground/data/ocr_vqa/images/\n")
            
            elif dataset_name == 'share_textvqa':
                f.write("  Download: Same as TextVQA dataset\n")
                f.write("  Place in: playground/data/share_textvqa/images/\n")
            
            f.write("\n")
    
    print(f"✓ Download instructions saved to: {output_instructions}")

def main():
    dataset_path = "final_dataset.json"
    image_folder = "playground/data"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        print("Make sure you're running this from the reverse_vlm directory")
        sys.exit(1)
    
    # Check if image folder exists
    if not os.path.exists(image_folder):
        print(f"Warning: Image folder not found: {image_folder}")
        print("Creating directory...")
        os.makedirs(image_folder, exist_ok=True)
    
    # Run the check
    missing_images, missing_by_dataset, stats = check_missing_images(dataset_path, image_folder)
    
    # Save reports
    if missing_images:
        print("")
        save_reports(missing_images, missing_by_dataset, stats)
        print("")
        print("=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Check 'download_instructions.txt' for dataset download commands")
        print("2. Download missing datasets using the provided commands")
        print("3. Re-run this script to verify all images are present")
        print("")
        print(f"Total missing: {len(missing_images):,} images")
        print("=" * 80)
    else:
        print("")
        print("=" * 80)
        print("✓ SUCCESS: All images are present!")
        print("=" * 80)

if __name__ == "__main__":
    main()
