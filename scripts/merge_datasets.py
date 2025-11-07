"""
Merge Datasets
===============
Combines WhatsApp and Instagram datasets into a single training file.

Usage:
    python scripts/merge_datasets.py
"""

import json
import sys
from pathlib import Path
# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import WHATSAPP_OUTPUT, INSTAGRAM_OUTPUT, COMBINED_OUTPUT


def merge_jsonl_files(*input_files, output_file):
    """
    Merge multiple JSONL files into one.
    
    Args:
        *input_files: Variable number of input file paths
        output_file: Output file path
    """
    total_lines = 0
    file_stats = {}
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            if not input_file.exists():
                print(f"⚠️  Skipping {input_file.name} (not found)")
                continue
            
            line_count = 0
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line)
                        line_count += 1
                        total_lines += 1
            
            file_stats[input_file.name] = line_count
            print(f"  ✓ {input_file.name}: {line_count:,} conversations")
    
    return total_lines, file_stats


def main():
    """Main merge pipeline."""
    print("=" * 70)
    print("Merge Datasets")
    print("=" * 70)
    print(f"\nMerging:")
    print(f"  - {WHATSAPP_OUTPUT}")
    print(f"  - {INSTAGRAM_OUTPUT}")
    print(f"\nOutput:")
    print(f"  - {COMBINED_OUTPUT}\n")
    
    # Merge files
    total, stats = merge_jsonl_files(
        WHATSAPP_OUTPUT,
        INSTAGRAM_OUTPUT,
        output_file=COMBINED_OUTPUT
    )
    
    if total == 0:
        print("\n❌ No data to merge!")
        print("   Please run preprocessing scripts first:")
        print("   1. python scripts/1_whatsapp_preprocessor.py")
        print("   2. python scripts/2_instagram_preprocessor.py")
        return
    
    print("\n" + "=" * 70)
    print("📊 Merge Statistics:")
    print("=" * 70)
    for filename, count in stats.items():
        print(f"  {filename}: {count:,} conversations")
    print(f"\n  Total: {total:,} conversations")
    print(f"\n✅ Merge complete!")
    print(f"   Output saved to: {COMBINED_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()