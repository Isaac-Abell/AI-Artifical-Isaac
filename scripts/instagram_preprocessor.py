"""
Instagram DM Preprocessor
==========================
Converts Instagram Direct Message exports to training-ready JSONL format.

Usage:
    python scripts/instagram_preprocessor.py
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    INSTAGRAM_DIR, INSTAGRAM_OUTPUT, CHAT_OWNER, ROLE,
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


def load_instagram_messages(folder_path: Path) -> tuple:
    """
    Load all message_*.json files from an Instagram conversation folder.
    
    Args:
        folder_path: Path to conversation folder
        
    Returns:
        (messages_list, is_group_chat)
    """
    all_messages = []
    participants = None
    
    for filename in sorted(os.listdir(folder_path)):
        if not (filename.startswith("message_") and filename.endswith(".json")):
            continue
            
        file_path = folder_path / filename
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            messages = data.get("messages", [])
            
            # Get participants from first file
            if participants is None:
                participants = data.get("participants", [])
            
            all_messages.extend(messages)
    
    # Check if group chat (more than 2 participants)
    is_group = len(participants) > 2 if participants else False
    
    return all_messages, is_group


def is_valid_message(content: str) -> bool:
    """
    Check if message content is valid and not a system message.
    
    Args:
        content: Message text
        
    Returns:
        True if message should be included
    """
    if not content:
        return False
    
    content_lower = content.lower()
    
    # Filter out system/action messages
    invalid_patterns = [
        "started a video chat",
        "liked a message",
        "sent an attachment",
        "reacted",
        "sent a photo",
        "sent a video",
        "sent a voice message",
        "started a call",
        "missed a call",
        "ended the call"
    ]
    
    for pattern in invalid_patterns:
        if pattern in content_lower:
            return False
    
    # Check for valid text content
    try:
        content.encode('utf-8')
        # Filter messages that are mostly emojis/special characters
        text_chars = sum(c.isalnum() or c.isspace() for c in content)
        if len(content) > 0 and text_chars / len(content) < 0.3:
            return False
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    
    return True


def messages_to_dataframe(messages: list, chat_owner: str) -> pd.DataFrame:
    """
    Convert Instagram messages to DataFrame for processing.
    
    Args:
        messages: List of message dicts from Instagram JSON
        chat_owner: Your Instagram username
        
    Returns:
        DataFrame with processed messages
    """
    if not messages:
        return pd.DataFrame()
    
    # Sort by timestamp (oldest first - Instagram exports are reverse)
    messages.sort(key=lambda m: m.get("timestamp_ms", 0))
    
    rows = []
    for msg in messages:
        sender = msg.get("sender_name")
        content = msg.get("content", "").strip()
        timestamp_ms = msg.get("timestamp_ms", 0)
        
        # Skip invalid messages
        if not is_valid_message(content):
            continue
        
        rows.append({
            "username": sender,
            "message": content,
            "timestamp_ms": timestamp_ms
        })
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Calculate time deltas
    df["timestamp_s"] = df["timestamp_ms"] / 1000.0
    df["timestamp_previous"] = df["timestamp_s"].shift(periods=1)
    df["time_delta"] = df["timestamp_s"] - df["timestamp_previous"]
    df["time_delta"] = df["time_delta"].fillna(0)
    
    # Assign roles
    df["chat_message"] = df.apply(
        lambda x: ("system" if x["username"] == chat_owner else ROLE, x["message"]),
        axis=1
    )
    
    return df


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
            current_row["chat_message"] = (
                current_row["chat_message"][0],
                current_row["chat_message"][1] + "\n" + row_message
            )
        else:
            new_data.append(current_row.to_dict())
            current_row = row
            current_role = row_role
    
    new_data.append(current_row.to_dict())
    return pd.DataFrame(new_data)


def process_conversation(folder_path: Path, chat_owner: str, encoder: AutoTokenizer) -> list:
    """
    Process a single Instagram conversation folder.
    
    Args:
        folder_path: Path to conversation folder
        chat_owner: Your Instagram username
        encoder: Tokenizer for token counting
        
    Returns:
        List of conversation segments
    """
    messages, is_group = load_instagram_messages(folder_path)
    
    # Skip group chats
    if is_group:
        return []
    
    if not messages:
        return []
    
    # Convert to DataFrame
    df = messages_to_dataframe(messages, chat_owner)
    
    if len(df) == 0:
        return []
    
    # Merge consecutive messages from same sender
    df = collapse_messages(df)
    
    # Group into conversations based on time gaps
    conversations = []
    current_conversation = []
    token_count = 0

    for _, row in df.iterrows():
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        formatted = f"<start_header_id>{row_role}<end_header_id>{row_message}"
        message_tokens = len(encoder.encode(formatted))

        if (row["time_delta"] < SAME_CONVO_THRESHOLD_SECONDS and 
            token_count + message_tokens < HISTORY_MAX_TOKENS):
            current_conversation.append(formatted)
            token_count += message_tokens
        else:
            if current_conversation:
                conversations.append(current_conversation)
            current_conversation = [formatted]
            token_count = message_tokens

    if current_conversation:
        conversations.append(current_conversation)
    
    return conversations


def main():
    """Main preprocessing pipeline for Instagram data."""
    print("=" * 70)
    print("Instagram DM Preprocessor")
    print("=" * 70)
    print(f"\nChat owner: {CHAT_OWNER}")
    print(f"Input directory: {INSTAGRAM_DIR}")
    print(f"Output file: {INSTAGRAM_OUTPUT}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    encoder = load_tokenizer()
    
    # Check if inbox directory exists
    if not INSTAGRAM_DIR.exists():
        print(f"❌ Instagram inbox directory not found!")
        print(f"   Expected location: {INSTAGRAM_DIR}")
        print(f"\n   To get your Instagram data:")
        print(f"   1. Instagram → Settings → Privacy and security")
        print(f"   2. Download your information → Request download")
        print(f"   3. Format: JSON")
        print(f"   4. Extract to: {INSTAGRAM_DIR.parent}")
        return
    
    # Process each conversation folder
    all_conversations = []
    total_folders = 0
    skipped_group_chats = 0
    processed_folders = 0
    
    folder_list = [f for f in INSTAGRAM_DIR.iterdir() if f.is_dir()]
    
    if not folder_list:
        print(f"❌ No conversation folders found in {INSTAGRAM_DIR}")
        return
    
    print(f"Found {len(folder_list)} conversation folders\n")
    
    for folder in tqdm(sorted(folder_list), desc="Processing conversations"):
        total_folders += 1
        
        conversations = process_conversation(folder, CHAT_OWNER, encoder)
        
        if conversations:
            all_conversations.extend(conversations)
            processed_folders += 1
        else:
            # Check if it was a group chat
            _, is_group = load_instagram_messages(folder)
            if is_group:
                skipped_group_chats += 1
    
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
    INSTAGRAM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(INSTAGRAM_OUTPUT, "w", encoding="utf-8") as f:
        for _, row in df_filtered.iterrows():
            f.write(json.dumps({"input": row["query_str"]}, ensure_ascii=False) + "\n")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("📊 Statistics:")
    print("=" * 70)
    print(f"  Total conversation folders:    {total_folders}")
    print(f"  Skipped (group chats):         {skipped_group_chats}")
    print(f"  Processed (1-on-1):            {processed_folders}")
    print(f"  Total segments:                {len(all_conversations)}")
    print(f"  After filtering (>{CONVO_MIN_TOKENS} tokens):  {len(df_filtered)}")
    print(f"  Average tokens/segment:        {df_filtered['query_len'].mean():.0f}")
    print(f"  Total tokens:                  {df_filtered['query_len'].sum():,}")
    print(f"\n✅ Preprocessing complete!")
    print(f"   Output saved to: {INSTAGRAM_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()