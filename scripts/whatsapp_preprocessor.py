"""
WhatsApp Chat Preprocessor
===========================
Converts WhatsApp chat exports to training-ready JSONL format.

Usage:
    python scripts/whatsapp_preprocessor.py
"""

import zipfile
import json
import sys
from pathlib import Path
import pandas as pd
from whatstk import WhatsAppChat
from transformers import AutoTokenizer
from tqdm.auto import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    WHATSAPP_DIR, WHATSAPP_OUTPUT, CHAT_OWNER, ROLE,
    SAME_CONVO_THRESHOLD_SECONDS, SAME_USER_THRESHOLD_SECONDS,
    HISTORY_MAX_TOKENS, CONVO_MIN_TOKENS, TOKENIZER_ID
)


def load_tokenizer():
    """Load tokenizer for token counting."""
    return AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        trust_remote_code=True,
        use_fast=True
    )


def collapse_messages(df: pd.DataFrame, delta_threshold: int = SAME_USER_THRESHOLD_SECONDS) -> pd.DataFrame:
    """
    Merge consecutive messages from the same sender within time threshold.
    
    Args:
        df: DataFrame with chat messages
        delta_threshold: Maximum seconds between messages to merge
        
    Returns:
        DataFrame with collapsed messages
    """
    if len(df) == 0:
        return df
    
    new_data = []
    df_temp = df.copy()
    current_row = df_temp.iloc[0]
    current_role = current_row["chat_message"][0]

    for _, row in df_temp[1:].iterrows():
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        if row_role == current_role and row["time_delta"] < delta_threshold:
            # Merge with previous message
            current_row["chat_message"] = (
                current_row["chat_message"][0],
                current_row["chat_message"][1] + "\n" + row_message
            )
        else:
            # Save previous and start new
            new_data.append(current_row.to_dict())
            current_row = row
            current_role = row_role
    
    # Don't forget the last message
    new_data.append(current_row.to_dict())
    return pd.DataFrame(new_data)


def preprocess_conversation(
    input_path: Path,
    chat_owner: str,
    role: str,
    encoder: AutoTokenizer
) -> list:
    """
    Convert WhatsApp chat export to conversation segments.
    
    Args:
        input_path: Path to WhatsApp .txt file
        chat_owner: Your name in the chat
        role: Role tag for other participants
        encoder: Tokenizer for token counting
        
    Returns:
        List of conversation segments
    """
    # Parse WhatsApp chat
    chat = WhatsAppChat.from_source(filepath=str(input_path))
    df = chat.df

    # Calculate time differences between messages
    df["date_previous"] = df["date"].shift(periods=1)
    df["time_delta"] = (df["date"] - df["date_previous"]).dt.total_seconds()

    # Assign roles: you are "system", others are "user"
    df["chat_message"] = df.apply(
        lambda x: ("system" if x["username"] == chat_owner else role, x["message"]),
        axis=1
    )

    # Merge consecutive messages from same sender
    df = collapse_messages(df)

    # Group into conversations based on time gaps
    conversations = []
    current_conversation = []
    token_count = 0

    for _, row in df.iterrows():
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        # Skip media-only messages
        if row_message == "<Media omitted>":
            continue

        # Format message
        formatted = f"<start_header_id>{row_role}<end_header_id>{row_message}"
        message_tokens = len(encoder.encode(formatted))

        # Check if we should start a new conversation
        if (row["time_delta"] < SAME_CONVO_THRESHOLD_SECONDS and 
            token_count + message_tokens < HISTORY_MAX_TOKENS):
            # Add to current conversation
            current_conversation.append(formatted)
            token_count += message_tokens
        else:
            # Save current conversation and start new one
            if current_conversation:
                conversations.append(current_conversation)
            current_conversation = [formatted]
            token_count = message_tokens

    # Don't forget the last conversation
    if current_conversation:
        conversations.append(current_conversation)

    return conversations


def main():
    """Main preprocessing pipeline for WhatsApp data."""
    print("=" * 70)
    print("WhatsApp Chat Preprocessor")
    print("=" * 70)
    print(f"\nChat owner: {CHAT_OWNER}")
    print(f"Input directory: {WHATSAPP_DIR}")
    print(f"Output file: {WHATSAPP_OUTPUT}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    encoder = load_tokenizer()
    
    # Extract any zip files
    for zip_path in WHATSAPP_DIR.glob("*.zip"):
        print(f"📦 Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(WHATSAPP_DIR)
    
    # Process all .txt files
    all_conversations = []
    txt_files = list(WHATSAPP_DIR.glob("*.txt"))
    
    if not txt_files:
        print("❌ No WhatsApp chat files found!")
        print(f"   Please place .txt exports in: {WHATSAPP_DIR}")
        return
    
    print(f"Found {len(txt_files)} WhatsApp chat file(s)\n")
    
    for txt_file in txt_files:
        print(f"Processing: {txt_file.name}")
        try:
            conversations = preprocess_conversation(
                txt_file, CHAT_OWNER, ROLE, encoder
            )
            all_conversations.extend(conversations)
            print(f"  ✓ Extracted {len(conversations)} conversation segments")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            continue
    
    if not all_conversations:
        print("\n❌ No conversations extracted!")
        return
    
    # Convert to JSONL format
    print("\nConverting to JSONL format...")
    df_model = pd.DataFrame({"query": all_conversations})
    df_model["query_str"] = df_model["query"].apply(lambda x: "<|eot_id|>".join(x))
    df_model["query_len"] = df_model["query_str"].apply(lambda x: len(encoder.encode(x)))
    
    # Filter by minimum length
    df_filtered = df_model[df_model["query_len"] > CONVO_MIN_TOKENS]
    
    # Write output
    WHATSAPP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(WHATSAPP_OUTPUT, "w", encoding="utf-8") as f:
        for _, row in df_filtered.iterrows():
            f.write(json.dumps({"input": row["query_str"]}, ensure_ascii=False) + "\n")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("📊 Statistics:")
    print("=" * 70)
    print(f"  Chat files processed:     {len(txt_files)}")
    print(f"  Total segments:           {len(all_conversations)}")
    print(f"  After filtering (>{CONVO_MIN_TOKENS} tokens): {len(df_filtered)}")
    print(f"  Average tokens/segment:   {df_filtered['query_len'].mean():.0f}")
    print(f"  Total tokens:             {df_filtered['query_len'].sum():,}")
    print(f"\n✅ Preprocessing complete!")
    print(f"   Output saved to: {WHATSAPP_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()