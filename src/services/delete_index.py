# deleteindex.py

from elasticsearch import Elasticsearch
from src.config import (
    ELASTIC_CONNECTION_URL,
    ELASTIC_USERNAME,
    ELASTIC_PASSWORD,
    INDEX_NAME
)

def delete_index():
    # Connect to Elasticsearch
    es = Elasticsearch(
        [ELASTIC_CONNECTION_URL],
        http_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
        verify_certs=False
    )

    # Delete the index if it exists
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"Deleted index '{INDEX_NAME}'.")
    else:
        print(f"Index '{INDEX_NAME}' does not exist.")

if __name__ == "__main__":
    delete_index()
