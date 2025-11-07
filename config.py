"""
Global configuration for AI: Artificial Isaac
Edit these values to customize your training pipeline.
"""

from pathlib import Path

# ==============================
# IDENTITY & PATHS
# ==============================

# Replace with your own name as it appears in WhatsApp/Instagram
CHAT_OWNER = "Isaac Abell"

# Data paths
DATA_DIR = Path("data")
WHATSAPP_DIR = DATA_DIR / "whatsapp"
INSTAGRAM_DIR = DATA_DIR / "instagram" / "inbox"

# Output paths
TRAINING_DATA_DIR = Path("training_data")
WHATSAPP_OUTPUT = TRAINING_DATA_DIR / "whatsapp_finetune.jsonl"
INSTAGRAM_OUTPUT = TRAINING_DATA_DIR / "instagram_finetune.jsonl"
COMBINED_OUTPUT = TRAINING_DATA_DIR / "dataset_combined.jsonl"
QWEN_OUTPUT = TRAINING_DATA_DIR / "dataset_qwen.jsonl"
CLEANED_OUTPUT = TRAINING_DATA_DIR / "dataset_qwen_cleaned.jsonl"

# Model paths
MODEL_OUTPUT_DIR = Path("qwen2.5_7b_finetuned")
CHROMA_DB_DIR = Path("chroma_db")
RAG_DATA_DIR = Path("rag_data")

# ==============================
# DATA PROCESSING
# ==============================

# Conversation grouping
SAME_CONVO_THRESHOLD_SECONDS = 1800  # 30 minutes - start new conversation
SAME_USER_THRESHOLD_SECONDS = 600    # 10 minutes - merge consecutive messages

# Token limits
HISTORY_MAX_TOKENS = 3000  # Maximum tokens per conversation
CONVO_MIN_TOKENS = 75      # Minimum tokens to include conversation

# Message role
ROLE = "user"  # Role for other participants (you are "system" / "assistant")

# ==============================
# MODEL & TRAINING
# ==============================

# Base model
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOKENIZER_ID = "NousResearch/Meta-Llama-3-8B"  # For preprocessing only

# LoRA configuration
LORA_R = 16                 # Rank of update matrices
LORA_ALPHA = 32            # Scaling factor
LORA_DROPOUT = 0.05        # Dropout probability

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
MAX_LENGTH = 3072          # Maximum sequence length
USE_BF16 = True           # Use bfloat16 precision

# Quantization
USE_4BIT = True           # Enable 4-bit quantization
QUANT_TYPE = "nf4"        # Quantization type

# ==============================
# RAG CONFIGURATION
# ==============================

RAG_COLLECTION_NAME = "rag_data"
MAX_RAG_CONTEXT_TOKENS = 1024  # Max tokens for retrieved contexts
RAG_N_RESULTS = 3          # Number of contexts to retrieve

# ==============================
# INFERENCE
# ==============================

INFERENCE_MAX_TOKENS = 512
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_P = 0.9

# ==============================
# LOGGING
# ==============================

LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")

# Create directories
for directory in [
    DATA_DIR, WHATSAPP_DIR, INSTAGRAM_DIR,
    TRAINING_DATA_DIR, MODEL_OUTPUT_DIR,
    CHROMA_DB_DIR, RAG_DATA_DIR,
    LOG_DIR, RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)