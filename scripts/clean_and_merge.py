"""
Clean and Merge Conversations
===============================
Merges consecutive same-role messages and validates format.
This produces the final training-ready dataset.

Usage:
    python scripts/clean_and_merge.py
"""

import json
import sys
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import QWEN_OUTPUT, CLEANED_OUTPUT, BASE_MODEL_ID, CONVO_MIN_TOKENS


def merge_consecutive_messages(conversation_text: str) -> str:
    """
    Takes a raw Qwen-style conversation string and merges consecutive
    same-role messages.
    
    Args:
        conversation_text: Conversation in Qwen format
        
    Returns:
        Cleaned conversation with merged messages
    """
    lines = conversation_text.strip().split("<|im_start|>")
    messages = []

    # Parse into (role, content) pairs
    for block in lines:
        if not block.strip():
            continue
        role_split = block.split("\n", 1)
        if len(role_split) < 2:
            continue
        role = role_split[0].strip()
        content = role_split[1].replace("<|im_end|>", "").strip()
        messages.append((role, content))

    # Merge consecutive same-role messages
    merged = []
    prev_role, buffer = None, []
    
    for role, msg in messages:
        if role == prev_role:
            buffer.append(msg)
        else:
            if buffer:
                merged.append((prev_role, "\n".join(buffer)))
            prev_role, buffer = role, [msg]
    
    if buffer:
        merged.append((prev_role, "\n".join(buffer)))

    # Rebuild cleaned conversation
    parts = []
    for role, msg in merged:
        parts.append(f"<|im_start|>{role}\n{msg}\n<|im_end|>")
    
    return "\n".join(parts)


def main():
    """Main cleaning pipeline."""
    print("=" * 70)
    print("Clean and Merge Conversations")
    print("=" * 70)
    print(f"\nInput:  {QWEN_OUTPUT}")
    print(f"Output: {CLEANED_OUTPUT}\n")
    
    if not QWEN_OUTPUT.exists():
        print(f"❌ Input file not found: {QWEN_OUTPUT}")
        print("   Please run the converter script first:")
        print("   python scripts/4_llama_to_qwen_converter.py")
        return
    
    # Load tokenizer for token counting
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False, trust_remote_code=True)
    
    # Process conversations
    total_tokens = 0
    total_conversations = 0
    filtered_out = 0
    
    with open(QWEN_OUTPUT, "r", encoding="utf-8") as fin, \
         open(CLEANED_OUTPUT, "w", encoding="utf-8") as fout:
        
        # Count lines for progress bar
        lines = fin.readlines()
        fin.seek(0)
        
        print(f"Processing {len(lines):,} conversations...\n")
        
        for line in tqdm(lines, desc="Cleaning"):
            if not line.strip():
                continue
            
            data = json.loads(line)
            conv = data.get("text") or data.get("input")
            if not conv:
                continue

            # Merge consecutive messages
            cleaned = merge_consecutive_messages(conv)
            
            # Count tokens
            tokens = tokenizer(cleaned, add_special_tokens=False, return_length=True)["length"]
            
            # Filter by minimum length
            if tokens < CONVO_MIN_TOKENS:
                filtered_out += 1
                continue

            total_tokens += tokens
            total_conversations += 1

            # Write to output
            fout.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
    
    if total_conversations == 0:
        print("\n❌ No conversations passed filtering!")
        return
    
    # Statistics
    avg_tokens = total_tokens / total_conversations
    
    print("\n" + "=" * 70)
    print("📊 Cleaning Statistics:")
    print("=" * 70)
    print(f"  Input conversations:       {len(lines):,}")
    print(f"  Filtered out (too short):  {filtered_out:,}")
    print(f"  Output conversations:      {total_conversations:,}")
    print(f"  Total tokens:              {total_tokens:,}")
    print(f"  Average tokens/conv:       {avg_tokens:.1f}")
    print(f"\n✅ Cleaning complete!")
    print(f"   Output saved to: {CLEANED_OUTPUT}")
    print("=" * 70)
    
    # Show example
    print("\n📝 Example Cleaned Conversation:\n")
    with open(CLEANED_OUTPUT, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        example = json.loads(first_line)['text']
        # Show first 500 characters
        print(example[:500])
        if len(example) > 500:
            print("...")


if __name__ == "__main__":
    main()