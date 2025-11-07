import chromadb
from chromadb.config import Settings
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib


class GranularRAGHelper:
    """
    Helper class for indexing and querying granular JSON/MD content in ChromaDB.
    Supports two JSON styles:
      1) List-of-details style: [{ "type": "...", "title": "...", "details": "..." }, ...]
      2) Nested structured JSON (previous behavior) - creates semantic chunks.
    """

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "personal_rag"):
        """
        Initialize the RAG helper with ChromaDB.

        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection to use
        """
        # Use PersistentClient (same API as before)
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _generate_id(self, source: str, key_path: str) -> str:
        """Generate a stable unique ID for a document chunk."""
        content = f"{source}::{key_path}"
        return hashlib.md5(content.encode()).hexdigest()

    def _dict_to_text(self, data: Dict, indent: int = 0) -> str:
        """
        Convert a dictionary to natural language text that preserves context.

        Args:
            data: Dictionary to convert
            indent: Indentation level for nested items

        Returns:
            Natural language representation
        """
        lines = []
        prefix = "  " * indent

        for key, value in data.items():
            formatted_key = key.replace('_', ' ').title()

            if isinstance(value, dict):
                lines.append(f"{prefix}{formatted_key}:")
                lines.append(self._dict_to_text(value, indent + 1))
            elif isinstance(value, list):
                if not value:
                    continue
                if all(isinstance(v, str) for v in value):
                    lines.append(f"{prefix}{formatted_key}: {', '.join(str(v) for v in value)}")
                else:
                    lines.append(f"{prefix}{formatted_key}: [see detailed entries]")
            else:
                lines.append(f"{prefix}{formatted_key}: {value}")

        return '\n'.join(lines)

    def _add_context_prefix(self, key: str, category: str, file_name: str) -> str:
        """
        Add contextual prefix to help with semantic search.
        """
        context_map = {
            'tech_stack': 'Current Technical Skills and Experience',
            'skills_strengths_weaknesses': 'Current Skills, Strengths, and Weaknesses',
            'future_goals': 'Future Goals and Skills to Learn',
            'hobbies_sports': 'Hobbies, Sports, and Personal Interests',
            'work_history': 'Work Experience and Employment History',
            'projects_index': 'Projects and Portfolio',
        }

        prefix = context_map.get(file_name, '')

        if 'primary_languages' in key:
            prefix = 'Programming Languages I Know'
        elif 'frameworks_libraries' in key:
            prefix = 'Frameworks and Libraries I Use'
        elif 'cloud_platforms' in key:
            prefix = 'Cloud Platforms I Have Experience With'
        elif 'want_to_learn' in key or 'future' in key.lower():
            prefix = 'Skills and Topics I Want to Learn in the Future'
        elif 'disliked' in key.lower():
            prefix = 'Technologies I Dislike'

        return prefix

    def _create_semantic_chunks(self, data: Any, parent_key: str = '', category: str = '', file_name: str = '') -> List[Tuple[str, str, Dict]]:
        """
        Create semantically meaningful chunks from JSON data.
        Returns list of (key_path, text_content, metadata) tuples.
        """
        chunks = []

        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}.{key}" if parent_key else key

                # Skip metadata fields
                if key == 'metadata':
                    continue

                if isinstance(value, dict):
                    has_nested_structures = any(isinstance(v, (dict, list)) for v in value.values())

                    if not has_nested_structures:
                        context_prefix = self._add_context_prefix(new_key, category, file_name)
                        text = f"{context_prefix}\n\n" if context_prefix else ""
                        text += f"{key.replace('_', ' ').title()}\n"
                        text += self._dict_to_text(value)
                        chunks.append((new_key, text, {'chunk_type': 'object'}))
                    else:
                        nested_chunks = self._create_semantic_chunks(value, new_key, category, file_name)
                        chunks.extend(nested_chunks)

                        if not parent_key and key in ['primary_languages', 'frameworks_libraries', 'cloud_platforms',
                                                      'databases', 'primary_hobbies', 'technical_strengths']:
                            summary_text = self._create_summary_chunk(key, value, category, file_name)
                            if summary_text:
                                chunks.append((f"{new_key}_summary", summary_text, {'chunk_type': 'summary'}))

                elif isinstance(value, list):
                    if not value:
                        continue

                    if isinstance(value[0], dict):
                        for i, item in enumerate(value):
                            item_name = item.get('name', item.get('title', item.get('skill', f'Item {i+1}')))

                            context_prefix = self._add_context_prefix(new_key, category, file_name)
                            text = f"{context_prefix}\n\n" if context_prefix else ""
                            text += f"{key.replace('_', ' ').title()}: {item_name}\n"
                            text += self._dict_to_text(item)

                            item_key = f"{new_key}[{i}]"
                            chunks.append((item_key, text, {
                                'chunk_type': 'array_item',
                                'array_name': key,
                                'item_index': i,
                                'item_name': item_name
                            }))
                    else:
                        context_prefix = self._add_context_prefix(new_key, category, file_name)
                        text = f"{context_prefix}\n\n" if context_prefix else ""
                        text += f"{key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}"
                        chunks.append((new_key, text, {'chunk_type': 'array_simple'}))

                else:
                    if len(str(value)) > 20 or not parent_key:
                        context_prefix = self._add_context_prefix(new_key, category, file_name)
                        text = f"{context_prefix}\n\n" if context_prefix else ""
                        text += f"{key.replace('_', ' ').title()}: {value}"
                        chunks.append((new_key, text, {'chunk_type': 'simple'}))

        return chunks

    def _create_summary_chunk(self, key: str, value: Any, category: str, file_name: str) -> str:
        """Create a summary chunk for a complex section."""
        context_prefix = self._add_context_prefix(key, category, file_name)

        if key == 'primary_languages' and isinstance(value, list):
            langs = [item.get('name', '') for item in value if isinstance(item, dict)]
            summary = f"{context_prefix}\n\n"
            summary += f"Programming Languages: {', '.join(langs)}\n"
            for item in value:
                if isinstance(item, dict):
                    name = item.get('name', '')
                    prof = item.get('proficiency', '')
                    years = item.get('years_experience', '')
                    summary += f"- {name}: {prof} ({years} years)\n"
            return summary

        if key == 'frameworks_libraries' and isinstance(value, dict):
            summary = f"{context_prefix}\n\n"
            for category_name, items in value.items():
                if isinstance(items, list):
                    if items and isinstance(items[0], dict):
                        names = [item.get('name', '') for item in items]
                        summary += f"{category_name.replace('_', ' ').title()}: {', '.join(names)}\n"
                    else:
                        summary += f"{category_name.replace('_', ' ').title()}: {', '.join(str(i) for i in items)}\n"
            return summary

        return ""

    def _process_json_file(self, file_path: str, category: str) -> List[Dict[str, Any]]:
        """
        Process a JSON file and extract semantic chunks for indexing.

        - If the content is a list of {type,title,details}, index each item as a single chunk.
        - Otherwise, fall back to the semantic chunker for nested JSON.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_name = Path(file_path).stem

        # Get content section if it exists
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        file_name = Path(file_path).stem

        # Determine content
        if isinstance(data, dict):
            content = data.get('content', data)
        elif isinstance(data, list):
            content = data
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}")


        documents = []

        # If content is a list and matches your "details" style -> handle directly
        if isinstance(content, list) and content and all(isinstance(item, dict) for item in content):
            # Heuristic: treat items that have 'details' field as the details-style entries
            is_details_style = all('details' in item and 'title' in item for item in content)

            if is_details_style:
                for i, item in enumerate(content):
                    item_type = item.get('type', 'unknown')
                    item_title = item.get('title', f'item_{i}')
                    item_details = item.get('details', '')

                    key_path = f"{file_name}[{i}]"
                    doc_id = self._generate_id(f"{category}/{file_name}", key_path)

                    metadata = {
                        'source_file': file_name,
                        'category': category,
                        'key_path': key_path,
                        'file_path': file_path,
                        'item_type': item_type,
                        'item_title': item_title,
                        'chunk_type': 'details_item'
                    }

                    documents.append({
                        'id': doc_id,
                        'text': item_details,
                        'metadata': metadata
                    })

                return documents

        # Fallback: create semantic chunks from structured JSON
        chunks = self._create_semantic_chunks(content, '', category, file_name)

        for key_path, text, extra_metadata in chunks:
            doc_id = self._generate_id(f"{category}/{file_name}", key_path)

            metadata = {
                'source_file': file_name,
                'category': category,
                'key_path': key_path,
                'file_path': file_path,
                **extra_metadata
            }

            documents.append({
                'id': doc_id,
                'text': text,
                'metadata': metadata
            })

        return documents

    def _process_markdown_file(self, file_path: str, category: str) -> List[Dict[str, Any]]:
        """
        Process a Markdown file by splitting it into semantic sections.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_name = Path(file_path).stem

        # Split by headers
        sections = []
        current_section = []
        current_header = "Introduction"
        header_level = 0

        for line in content.split('\n'):
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:
                        sections.append((current_header, header_level, section_text))

                # Start new section
                header_level = len(line) - len(line.lstrip('#'))
                current_header = line.lstrip('#').strip()
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append((current_header, header_level, section_text))

        documents = []
        for i, (header, level, text) in enumerate(sections):
            doc_id = self._generate_id(f"{category}/{file_name}", f"section_{i}_{header}")

            documents.append({
                'id': doc_id,
                'text': text,
                'metadata': {
                    'source_file': file_name,
                    'category': category,
                    'section_header': header,
                    'section_index': i,
                    'heading_level': level,
                    'file_path': file_path,
                    'chunk_type': 'markdown_section'
                }
            })

        return documents

    def index_directory(self, rag_data_path: str):
        """
        Index all JSON and MD files from the RAG data directory.
        """
        all_documents = []

        for category_dir in Path(rag_data_path).iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            for file_path in category_dir.iterdir():
                if file_path.suffix == '.json':
                    docs = self._process_json_file(str(file_path), category)
                    all_documents.extend(docs)
                    print(f"  Indexed {len(docs)} chunks from {category}/{file_path.name}")
                elif file_path.suffix == '.md':
                    docs = self._process_markdown_file(str(file_path), category)
                    all_documents.extend(docs)
                    print(f"  Indexed {len(docs)} chunks from {category}/{file_path.name}")

        # Add to ChromaDB
        if all_documents:
            self.collection.add(
                ids=[doc['id'] for doc in all_documents],
                documents=[doc['text'] for doc in all_documents],
                metadatas=[doc['metadata'] for doc in all_documents]
            )

            print(f"\n✓ Total: Indexed {len(all_documents)} semantic chunks")

    def query_context(self, prompt: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the RAG system for the most relevant context.
        """
        results = self.collection.query(
            query_texts=[prompt],
            n_results=n_results
        )

        contexts = []
        # results contains lists for each query (we send one)
        ids_list = results.get('ids', [[]])[0]
        docs_list = results.get('documents', [[]])[0]
        metas_list = results.get('metadatas', [[]])[0]
        dists_list = results.get('distances', [[]])[0] if 'distances' in results else [None] * len(ids_list)

        for i in range(len(ids_list)):
            contexts.append({
                'id': ids_list[i],
                'text': docs_list[i],
                'metadata': metas_list[i],
                'distance': dists_list[i]
            })

        return contexts

    def format_context_for_prompt(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format retrieved contexts into a string suitable for prompt injection.
        """
        if not contexts:
            return ""

        formatted = "=== RELEVANT CONTEXT ===\n\n"

        for i, ctx in enumerate(contexts, 1):
            meta = ctx['metadata']
            src = f"{meta.get('category', 'unknown')}/{meta.get('source_file', 'unknown')}"
            title = meta.get('item_title') or meta.get('item_name') or meta.get('section_header') or meta.get('key_path')
            formatted += f"[{i}] From {src} - {title}\n"
            formatted += f"{ctx['text']}\n\n"

        return formatted

    def clear_collection(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Collection cleared")


# Convenience function
def get_context_for_prompt(prompt: str, rag_helper: GranularRAGHelper, n_results: int = 5) -> str:
    """
    Convenience function to get formatted context for a prompt.
    """
    contexts = rag_helper.query_context(prompt, n_results=n_results)
    return rag_helper.format_context_for_prompt(contexts)
