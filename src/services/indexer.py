# indexer.py

import os
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, BulkIndexError
from tqdm import tqdm
from src.config import (
    ELASTIC_CONNECTION_URL,
    ELASTIC_USERNAME,
    ELASTIC_PASSWORD,
    INDEX_NAME,
    EMBEDDING_DIMS,
    DOCS_FOLDER
)
from .docs_loader import load_documents
from src.agents.chunker import chunk_text
from src.agents.embedder import embed_texts

# Batch size to avoid OOM: embed this many chunks at a time
BATCH_SIZE = 16

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def create_index(es: Elasticsearch):
    """
    Create the Elasticsearch index with mapping for dense vectors and metadata.
    """
    mapping = {
        "mappings": {
            "properties": {
                "file_path": {"type": "keyword"},
                "section":   {"type": "keyword"},
                "chunk_id":  {"type": "integer"},
                "text":      {"type": "text"},
                "vector":    {"type": "dense_vector", "dims": EMBEDDING_DIMS}
            }
        }
    }
    if es.indices.exists(index=INDEX_NAME):
        logger.info(f"Index '{INDEX_NAME}' already exists.")
    else:
        es.indices.create(index=INDEX_NAME, body=mapping)
        logger.info(f"Created index '{INDEX_NAME}'.")


def index_documents(es: Elasticsearch):
    """
    Load documents, chunk and embed them in small batches, then index with robust error handling.
    """
    docs = load_documents(DOCS_FOLDER)
    total_docs = len(docs)
    logger.info(f"Starting indexing of {total_docs} document sections...")

    for idx, doc in enumerate(tqdm(docs, desc="Indexing documents"), start=1):
        file_path = doc["file_path"]
        section = doc["section"]
        text = doc["text"]
        chunks = chunk_text(text)
        logger.info(f"[{idx}/{total_docs}] '{file_path}' - section '{section}' with {len(chunks)} chunks")

        # Prepare actions per document
        actions = []
        for start in range(0, len(chunks), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(chunks))
            batch_chunks = chunks[start:end]
            logger.info(f"  Embedding batch {start}:{end} on device")
            try:
                vectors = embed_texts(batch_chunks)
            except Exception as e:
                logger.error(f"Embedding failed for {file_path} section {section} batch {start}-{end}: {e}")
                continue

            for offset, (chunk_text_item, vector) in enumerate(zip(batch_chunks, vectors), start=start):
                actions.append({
                    "_index": INDEX_NAME,
                    "_source": {
                        "file_path": file_path,
                        "section": section,
                        "chunk_id": offset,
                        "text": chunk_text_item,
                        "vector": vector
                    }
                })

        if not actions:
            logger.warning(f"No chunks to index for {file_path} section {section}")
            continue

        # Try bulk indexing
        try:
            logger.info(f"  Bulk indexing {len(actions)} chunks for this section")
            success, failed = bulk(es, actions, stats_only=True)
            logger.info(f"  Bulk indexed {success} docs, {failed} failures")
            if failed > 0:
                raise BulkIndexError(f"{failed} docs failed to index", [])
        except BulkIndexError as bulk_err:
            logger.error(f"BulkIndexError in '{file_path}' section '{section}': {bulk_err}")
            # Fallback: index one by one
            for action in actions:
                try:
                    es.index(index=action["_index"], document=action["_source"])
                except Exception as e2:
                    src = action["_source"]
                    logger.error(
                        f"Failed to index chunk {src['chunk_id']} of '{src['file_path']}' section '{src['section']}': {e2}"
                    )

    logger.info("Indexing process complete.")


def main():
    es = Elasticsearch(
        [ELASTIC_CONNECTION_URL],
        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
        verify_certs=False
    )

    create_index(es)
    index_documents(es)


if __name__ == "__main__":
    main()
