# chunker.py

from typing import List
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits a long text string into overlapping chunks.

    :param text: The full text to chunk.
    :param size: Maximum characters per chunk.
    :param overlap: Number of overlapping characters between chunks.
    :return: List of text chunks.
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must be non-negative and less than chunk size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        # move start forward by (size - overlap)
        start += size - overlap

    return chunks

# if __name__ == "__main__":
#     # Quick test
#     sample = "Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 10
#     result = chunk_text(sample)
#     print(f"Generated {len(result)} chunks, each up to {CHUNK_SIZE} chars with {CHUNK_OVERLAP} overlap.")
