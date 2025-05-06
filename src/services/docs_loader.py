# docs_loader.py

import os
from pathlib import Path
from typing import List, Dict
from src.config import DOCS_FOLDER
from .sectioner import split_into_sections


def load_documents(folder: str = DOCS_FOLDER) -> List[Dict[str, str]]:
    """
    Recursively read all .txt files under `folder`, split into sections,
    and return a list of dicts:
      {
        'file_path': relative path from `folder`,
        'section': section name (e.g., 'INTRODUCTION', 'METHODS', 'FULL_TEXT'),
        'text': section text
      }
    """
    documents: List[Dict[str, str]] = []
    base = Path(folder)

    for txt_file in base.rglob("*.txt"):
        try:
            raw_text = txt_file.read_text(encoding="utf-8")
            rel_path = txt_file.relative_to(base)
            # Split into sections (fallback to FULL_TEXT if no headings)
            sections = split_into_sections(raw_text)
            for section_name, section_text in sections:
                if section_text:
                    documents.append({
                        "file_path": str(rel_path),
                        "section": section_name,
                        "text": section_text
                    })
        except Exception as e:
            print(f"⚠️ Could not process {txt_file}: {e}")

    return documents


# if __name__ == "__main__":
#     docs = load_documents()
#     print(f"Loaded {len(docs)} sections:")
#     for d in docs[:5]:
#         print(f" • {d['file_path']} - {d['section']} ({len(d['text'])} chars)")
