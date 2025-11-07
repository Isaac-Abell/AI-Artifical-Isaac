"""
Llama to Qwen Format Converter
================================
Converts Llama 3 chat format to Qwen 2.5 chat format.

Llama 3 format:
    <start_header_id>user<end_header_id>Hello<|eot_id|>
    
Qwen 2.5 format:
    <|im_start|>user
    Hello<|im_end|>

Usage:
    python scripts/llama_to_qwen_converter.py
"""

import json
import re
import sys
from pathlib import Path
from tqdm.auto import tqdm
# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import COMBINED_OUTPUT, QWEN_OUTPUT


def convert_llama_to_qwen(text: str) -> str:
    """
    Convert Llama-formatted text to Qwen format.
    
    Args:
        text: Text in Llama format
        
    Returns:
        Text in Qwen format
    """
    # Replace header tags
    converted = re.sub(
        r'<start_header_id>user<end_header_id>',
        '<|im_start|>user\n',
        text
    )
    
    # Replace system with assistant (for your responses)
    converted = re.sub(
        r'<start_header_id>system<end_header_id>',
        '<|im_start|>assistant\n',
        converted
    )
    
    # Replace assistant tags (if any)
    converted = re.sub(
        r'<start_header_id>assistant<end_header_id>',
        '<|im_start|>assistant\n',
        converted
    )
    
    # Replace end of turn tokens
    converted = re.sub(
        r'<\|eot_id\|>',
        '<|im_end|>\n',
        converted
    )
    
    # Add final end token if not present
    if not converted.endswith('<|im_end|>') and not converted.endswith('<|im_end|>\n'):
        converted += '<|im_end|>'
    
    return converted


def main():
    """Main conversion pipeline."""
    print("=" * 70)
    print("Llama → Qwen Format Converter")
    print("=" * 70)
    print(f"\nInput:  {COMBINED_OUTPUT}")
    print(f"Output: {QWEN_OUTPUT}\n")
    
    if not COMBINED_OUTPUT.exists():
        print(f"❌ Input file not found: {COMBINED_OUTPUT}")
        print("   Please run the merge script first:")
        print("   python scripts/3_merge_datasets.py")
        return
    
    # Count total lines for progress bar
    with open(COMBINED_OUTPUT, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Converting {total_lines:,} conversations...\n")
    
    # Convert
    converted_count = 0
    with open(COMBINED_OUTPUT, 'r', encoding='utf-8') as f_in, \
         open(QWEN_OUTPUT, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Converting"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            original_input = data['input']
            
            # Convert format
            converted = convert_llama_to_qwen(original_input)
            
            # Write converted data
            json.dump({'input': converted}, f_out, ensure_ascii=False)
            f_out.write('\n')
            converted_count += 1
    
    print("\n" + "=" * 70)
    print("📊 Conversion Statistics:")
    print("=" * 70)
    print(f"  Conversations converted: {converted_count:,}")
    print(f"\n✅ Conversion complete!")
    print(f"   Output saved to: {QWEN_OUTPUT}")
    print("=" * 70)
    
    # Show example
    print("\n📝 Example Conversion:\n")
    with open(QWEN_OUTPUT, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        example = json.loads(first_line)['input']
        # Show first 500 characters
        print(example[:500])
        if len(example) > 500:
            print("...")


if __name__ == "__main__":
    main()