"""
Merge Datasets
===============
Combines WhatsApp and Instagram datasets into a single training file.

Usage:
    python scripts/setup_rag.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from rag.rag_helper import GranularRAGHelper
from config import (
    RAG_COLLECTION_NAME
)

def main():
    rag_helper = GranularRAGHelper(persist_directory="./chroma_db", collection_name="personal_rag")
    try:
        rag_helper.index_directory(RAG_COLLECTION_NAME)
        print("✓ Indexing successful")
        
        # Check collection stats
        count = rag_helper.collection.count()
        print(f"✓ Total indexed chunks: {count}")
        
        return True
    except Exception as e:
        print(f"✗ Indexing failed: {e}")
        return False


if __name__ == "__main__":
    main()