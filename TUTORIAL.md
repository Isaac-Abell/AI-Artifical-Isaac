# Complete Tutorial: Train Your Personal AI

This guide walks you through every step of creating your personal AI chatbot.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Collection](#data-collection)
3. [Environment Setup](#environment-setup)
4. [Data Processing](#data-processing)
5. [Model Training](#model-training)
6. [RAG Setup](#rag-setup)
7. [Testing & Evaluation](#testing--evaluation)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Recommended:**
- GPU: NVIDIA RTX 4080 or better (16GB+ VRAM)
- RAM: 64GB system RAM

### Software Requirements

- **Python 3.11**: (could work on other versions but this is the one I used)
- **CUDA 13.0 or higher**: (could work on other versions but this is the one I used)
- **Git**: For cloning the repository

### Check Your Setup

```bash
# Check CUDA version
nvidia-smi

# Check Python version
python --version

# Verify CUDA is accessible to Python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Data Collection

### WhatsApp Export

1. **Open WhatsApp** on your phone
2. **Select a chat** you want to include
3. Tap **⋮** (three dots) → **More** → **Export chat**
4. Choose **"Without Media"** (media not needed for text training)
5. Save or email the `.txt` file to yourself
6. **Repeat** for all chats you want to include

**Tips:**
- Export individual 1-on-1 chats (not group chats)
- More data = better results (aim for 5+ active chats)
- Quality over quantity (export chats where you're most active)

### Instagram Export

1. **Open Instagram** → **Settings**
2. **Privacy and security** → **Download your information**
3. Click **Request download**
4. **Format**: Select **JSON** (not HTML!)
5. **Date range**: All time
6. **Media quality**: Not needed (we only use text)
7. Wait for email (can take 24-48 hours)
8. Download and **extract the ZIP file**

**What you'll get:**
```
instagram-export/
├── messages/
│   └── inbox/
│       ├── conversation1_abc123/
│       │   ├── message_1.json
│       │   └── message_2.json
│       └── conversation2_def456/
│           └── message_1.json
```

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Isaac-Abell/AI-Artifical-Isaac.git
cd AI-Artifical-Isaac
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "
import torch
import transformers
import chromadb
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
```

### 5. Set Up Directory Structure

```bash
# The scripts will create these, but you can do it manually:
mkdir -p data/whatsapp
mkdir -p data/instagram/inbox
mkdir -p training_data
mkdir -p rag_data/{core,professional,projects,interests,worldview,life}
mkdir -p results
mkdir -p logs
```

---

## Data Processing

### Step 1: Copy Your Data

```bash
# Copy WhatsApp .txt files
cp /path/to/your/whatsapp/exports/*.txt data/whatsapp/

# Copy Instagram inbox folder
cp -r /path/to/instagram-export/messages/inbox/* data/instagram/inbox/
```

### Step 2: Configure Your Name

Edit `config.py`:

```python
CHAT_OWNER = "Your Full Name"  # Exactly as it appears in chats
```

**Important:** Use your exact name from the exports:
- WhatsApp: Check the first line of any .txt file
- Instagram: Check the "sender_name" in message JSON files

### Step 3: Process WhatsApp Data

```bash
python scripts/whatsapp_preprocessor.py
```

**Expected output:**
```
====================================================================
WhatsApp Chat Preprocessor
====================================================================

Chat owner: Your Name
Found 5 WhatsApp chat file(s)

Processing: Chat with Friend 1.txt
  ✓ Extracted 234 conversation segments
Processing: Chat with Friend 2.txt
  ✓ Extracted 156 conversation segments
...

====================================================================
📊 Statistics:
====================================================================
  Chat files processed:     5
  Total segments:           1,234
  After filtering (>75 tokens):  1,180
  Average tokens/segment:   287
  Total tokens:             338,860

✅ Preprocessing complete!
   Output saved to: training_data/whatsapp_finetune.jsonl
====================================================================
```

### Step 4: Process Instagram Data

```bash
python scripts/instagram_preprocessor.py
```

**Expected output:**
```
====================================================================
Instagram DM Preprocessor
====================================================================

Found 45 conversation folders

Processing conversations: 100%|████████| 45/45 [00:12<00:00]

====================================================================
📊 Statistics:
====================================================================
  Total conversation folders:    45
  Skipped (group chats):         8
  Processed (1-on-1):            37
  Total segments:                1,613
  After filtering (>75 tokens):  1,580
  Average tokens/segment:        245
  Total tokens:                  387,100

✅ Preprocessing complete!
   Output saved to: training_data/instagram_finetune.jsonl
====================================================================
```

### Step 5: Merge Datasets

```bash
python scripts/merge_datasets.py
```

**Output:**
```
====================================================================
Merge Datasets
====================================================================

  ✓ whatsapp_finetune.jsonl: 1,180 conversations
  ✓ instagram_finetune.jsonl: 1,580 conversations

====================================================================
📊 Merge Statistics:
====================================================================
  Total: 2,760 conversations

✅ Merge complete!
   Output saved to: training_data/dataset_combined.jsonl
====================================================================
```

### Step 6: Convert to Qwen Format

```bash
python scripts/llama_to_qwen_converter.py
```

### Step 7: Clean and Merge Messages

```bash
python scripts/clean_and_merge.py
```

**Final output:**
```
====================================================================
📊 Cleaning Statistics:
====================================================================
  Input conversations:       2,760
  Filtered out (too short):  87
  Output conversations:      2,673
  Total tokens:              725,987
  Average tokens/conv:       271.7

✅ Cleaning complete!
   Output saved to: training_data/dataset_qwen_cleaned.jsonl
====================================================================
```

**What just happened?**
- Consecutive messages from same person were merged
- Very short conversations (<75 tokens) were removed
- Format was validated and cleaned

---

## Model Training

### Understanding the Training Process

Training will:
1. Load Qwen 2.5 7B model (in 4-bit precision)
2. Add LoRA adapters (trainable parameters)
3. Train for 5 epochs (~2-6 hours depending on GPU)
4. Save checkpoints after each epoch

### Start Training

```bash
python scripts/train_qwen.py
```

**What you'll see:**
```
Loading dataset...
Loading Qwen2.5-7B-Instruct model and tokenizer...
Loading checkpoint shards: 100%|████████| 4/4 [00:15<00:00]

Tokenizing dataset...
Map: 100%|████████| 2673/2673 [00:08<00:00]

Preparing model for k-bit training...
Trainable params: 41,943,040 || All params: 7,615,617,024 || Trainable%: 0.55%

Starting training...

Epoch 1/5:  25%|██▌       | 168/632 [23:45<1:05:32, 8.47s/it, loss=2.34]
```

### Monitor Training

**GPU Usage:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

Look for:
- GPU Utilization: Should be 95-100%
- Memory Usage: Should be ~11-16GB
- Temperature: Should stay below 85°C

**Training Loss:**
- Should decrease over time
- Typical final loss: 2.0-2.5
- If loss doesn't decrease: learning rate might be too high/low

### Checkpoint Selection

Training saves checkpoints after each epoch:
```
qwen2.5_7b_finetuned/
├── checkpoint-158/   # Epoch 1
├── checkpoint-316/   # Epoch 2
├── checkpoint-474/   # Epoch 3
├── checkpoint-632/   # Epoch 4
└── checkpoint-790/   # Epoch 5 (final)
```

### 🧭 Which Epoch Should You Use?

The ideal number of epochs depends on your dataset size.
As a general rule of thumb for datasets **under ~500k tokens**:

| **Epoch** | **Characteristics**                            | **Notes**                                       |
| :-------- | :--------------------------------------------- | :---------------------------------------------- |
| **2–3**   | More assistant-like and conservative           | Best for stability and generalization           |
| **4**     | 🟢 **Balanced personality and safety**         | ✅ **Recommended**                               |
| **5+**    | Strong personality, higher risk of overfitting | Use cautiously — especially on smaller datasets |

> ⚠️ **Note:** Results will vary depending on dataset size, diversity, and task complexity.

---

## RAG Setup

### Why Use RAG?

RAG (Retrieval-Augmented Generation) adds factual accuracy:
- ✓ Accurate personal information
- ✓ Up-to-date knowledge
- ✓ Prevents hallucinations
- ✓ Citations for claims

### Create Your Knowledge Base

Edit files in `rag_data/` with your information:

**Example: `rag_data/core/technical_profile.json`**
```json
{
  "content": [
    {
      "type": "skill",
      "title": "Python",
      "details": "Expert level, 5+ years experience. Primary language for ML, data science, and backend development. Proficient with PyTorch, TensorFlow, FastAPI, and Django."
    },
    {
      "type": "skill",
      "title": "Machine Learning",
      "details": "3 years experience in NLP and computer vision. Specialized in fine-tuning transformers, RAG systems, and LLM applications."
    }
  ]
}
```

**Example: `rag_data/professional/work_history.json`**
```json
{
  "content": [
    {
      "type": "job",
      "title": "ML Engineer at TechCorp",
      "details": "January 2022 - Present. Built production ML pipelines, fine-tuned LLMs for customer service, reduced inference latency by 40%."
    }
  ]
}
```

### Index Your Data

```bash
python scripts/setup_rag.py
```

**Output:**
```
Indexing RAG Data
=================

  Indexed 12 chunks from core/technical_profile.json
  Indexed 8 chunks from professional/work_history.json
  Indexed 15 chunks from projects/projects_index.json
  ...

✓ Total: Indexed 87 semantic chunks into ChromaDB
```

---

## Testing & Evaluation

### Interactive Testing

Best for casual exploration:

```bash
python scripts/inference.py --mode interactive
```

```
====================================================================
INTERACTIVE TESTING MODE
====================================================================
Commands:
  - Type your question/prompt and press Enter
  - Type 'quit' or 'exit' to end
  - Type 'clear' to clear conversation history
====================================================================

You: What programming languages do you know?
Assistant: I'm most proficient in Python - been using it for about 5 years now. 
I also know JavaScript pretty well, especially for frontend work with React. 
I've dabbled in Rust recently and really enjoying it...

You: Tell me about your latest project
Assistant: [response]

You: quit
Goodbye!
```
---

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptom:** `CUDA out of memory` error during training

**Solutions:**
1. Reduce batch size in `config.py`:
   ```python
   BATCH_SIZE = 1
   GRADIENT_ACCUMULATION_STEPS = 4  # Reduce from 8
   ```

2. Reduce max sequence length:
   ```python
   MAX_LENGTH = 2048  # Reduce from 3072
   ```

3. Enable more aggressive quantization:
   ```python
   USE_4BIT = True  # Make sure this is enabled
   ```

### Model Not Learning

**Symptom:** Loss stays high or doesn't decrease

**Solutions:**
1. Check your data quality
2. Increase learning rate slightly:
   ```python
   LEARNING_RATE = 3e-5  # Increase from 2e-5
   ```
3. Train for more epochs

### Model Sounds Too Robotic

**Symptom:** Responses are accurate but lack personality

**Solutions:**
- Train for more epochs
- Make sure you have enough diverse training data (1000+ conversations)
- Check that your name in `config.py` matches exactly

### Model Repeats Training Data

**Symptom:** Model directly quotes your old messages

**Solutions:**
- This is overfitting - use an earlier checkpoint (epoch 2-3 instead of 4)
- Reduce number of epochs to 2-3
- Increase dropout:
  ```python
  LORA_DROPOUT = 0.1  # Increase from 0.05
  ```

### ChromaDB Errors

**Symptom:** RAG indexing fails

**Solutions:**
```bash
# Clear database and start fresh
rm -rf chroma_db/
python scripts/setup_rag.py
```

### Slow Training

**Normal:**
- 8-10 hours on RTX 4080

**If unusually slow:**
1. Check GPU utilization: `nvidia-smi`
2. Make sure no other processes are using GPU
3. Enable bf16 if supported:
   ```python
   USE_BF16 = True
   ```

---

## Next Steps

After successfully training your model:

1. **Experiment with prompts** in interactive mode
2. **Test different checkpoints** to find your favorite
3. **Expand your RAG knowledge base** with more personal info
4. **Fine-tune hyperparameters** for your specific use case
5. **Share your results** (but not your data!)

---

## Getting Help

If you encounter issues:

1. Check the [FAQ](#troubleshooting)

---

**Happy training! 🚀**