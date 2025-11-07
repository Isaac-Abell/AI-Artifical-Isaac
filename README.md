# 🤖 AI: Artificial Isaac

**Fine-tune a Large Language Model to speak like you using your WhatsApp and Instagram chat history.**

This project demonstrates how to create a personalized AI chatbot that mimics your communication style by fine-tuning Qwen 2.5 (7B) on your messaging data, enhanced with RAG (Retrieval-Augmented Generation) for accurate personal information retrieval.

## 🛠️ Technologies Used

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54&style=plastic)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white&style=plastic)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange.svg)](https://huggingface.co/transformers/)
[![ChromaDB](https://img.shields.io/badge/💾-ChromaDB-green.svg)](https://www.trychroma.com/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Overview](#-project-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Detailed Workflow](#-detailed-workflow)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Results & Evaluation](#-results--evaluation)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

- **Multi-Platform Data Processing**: Import and process WhatsApp and Instagram chat exports
- **Intelligent Message Merging**: Automatically combines consecutive messages from the same sender
- **Qwen 2.5 Fine-tuning**: Uses 7B parameter model with LoRA (Low-Rank Adaptation)
- **4-bit Quantization**: Efficient training on consumer GPUs (16GB+ VRAM)
- **RAG Integration**: Semantic search over personal knowledge base for accurate information retrieval
- **Privacy-First**: All processing happens locally—your data never leaves your machine

---

## 🎯 Project Overview

This project follows a complete ML pipeline:

```
1. Data Collection     →  Export chats from WhatsApp/Instagram
2. Data Processing     →  Parse, clean, and format messages
3. Format Conversion   →  Convert to Qwen chat format
4. Message Merging     →  Combine consecutive same-role messages
5. Model Fine-tuning   →  Train Qwen 2.5 with LoRA
6. RAG Setup           →  Index personal knowledge in ChromaDB
7. Evaluation          →  Test and compare model outputs
8. Deployment          →  Interactive chatbot
```

### Why This Works

- **Communication Style**: The model learns your vocabulary, sentence structure, and conversational patterns
- **Personal Context**: RAG retrieval ensures factual accuracy about your life, work, and interests
- **Efficient Training**: LoRA + quantization makes training feasible on consumer hardware
- **Conversation Dynamics**: Preserves natural dialogue flow and turn-taking

---

## 🚀 Installation

### Prerequisites

- **Python 3.11**
- **CUDA-capable GPU** with 16GB+ VRAM (recommended: RTX RTX 4080, or better)

### Setup

```bash
# Clone the repository
git clone https://github.com/Isaac-Abell/AI-Artifical-Isaac.git
cd AI-Artifical-Isaac

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Visit https://pytorch.org/get-started/locally/ for your specific version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Requirements

The project uses these key libraries (see `requirements.txt` for full list):

- `transformers>=4.40.0` - Hugging Face model library
- `torch` - PyTorch for training
- `peft` - Parameter-Efficient Fine-Tuning
- `bitsandbytes` - Quantization support
- `chromadb` - Vector database for RAG
- `whatstk` - WhatsApp chat parser
- `pandas`, `tqdm` - Data processing utilities

---

## ⚡ Instructions

### See [TUTORIAL.md](./TUTORIAL.md)

## 📁 Project Structure

```
AI-Artifical-Isaac/
│
├── data/                          # Raw data (gitignored)
│   ├── whatsapp/                  # WhatsApp .txt exports
│   └── instagram/inbox/           # Instagram JSON folders
│
├── scripts/                       # All executable scripts
│   ├── whatsapp_preprocessor.py
│   ├── instagram_preprocessor.py
│   ├── merge_datasets.py
│   ├── llama_to_qwen_converter.py
│   ├── clean_and_merge.py
│   ├── train_qwen.py
│   ├── setup_rag.py
│   └── inference.py
│
├── rag_data/                      # Personal knowledge base
│   ├── core/
│   ├── professional/
│   ├── projects/
│   ├── interests/
│   ├── worldview/
│   └── life/
│
├── training_data/                 # Processed datasets
│   ├── whatsapp_finetune.jsonl
│   ├── instagram_finetune.jsonl
│   ├── dataset_combined.jsonl
│   ├── dataset_qwen.jsonl
│   └── dataset_qwen_cleaned.jsonl
│
├── qwen2.5_7b_finetuned/         # Model checkpoints
│   ├── checkpoint-xxx/
│   ├── checkpoint-xxx/
│   └── checkpoint-xxx/
│
├── chroma_db/                     # RAG vector database
│
├── requirements.txt
├── README.md
├── TUTORIAL.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Configuration

### Global Settings

Edit `scripts/config.py` to customize parameters:

### Per-Script Configuration

Each script has a `CONFIG` section at the top for easy customization.

---

## 📊 Results & Evaluation

### Training Metrics

Example from a real training run:

```
Dataset: 1287 conversations (cleaned)
Total tokens: ~375k
Training time: ~7.5 hours (RTX 4080)
GPU memory: 15.8GB peak
Final loss: 2.23
```
---

## 🔬 Advanced Usage

### Custom RAG Data

Create new categories in `rag_data/`:

```json
{
  "content": [
    {
      "type": "skill",
      "title": "Python",
      "details": "Expert level, 5 years experience..."
    }
  ]
}
```

Or use nested JSON:

```json
{
  "content": {
    "primary_languages": [
      {
        "name": "Python",
        "proficiency": "Expert",
        "years_experience": 5
      }
    ]
  }
}
```

## 💬 Example Conversation

Here’s a sample interaction with **Artificial Isaac** after fine-tuning and RAG setup:

```
You: Hello, who are you?
Artificial Isaac: I'm Isaac Abell, a 21 year old dual citizen of Canada and the US. I was born in NYC but live in Toronto now.

You: Tell me about some programming projects you have made?
Artificial Isaac: I built a chatbot that uses RAG to answer questions

You: lmao you are crazy
Artificial Isaac: Lol And a website to visualize algorithms

You: Do you like to mountain bike
Artificial Isaac: Yes

You: what bike do you have
Artificial Isaac: A Propain Spindrift 4
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hugging Face** for transformers and PEFT libraries
- **Qwen Team** for the excellent base models
- **ChromaDB** for semantic search infrastructure
- **whatstk** for WhatsApp parsing utilities
- The open-source ML community for making this accessible

---
