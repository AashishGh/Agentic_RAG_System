# deleteindex.py
import sys, os, time, logging
# When run as a script, set package so relative imports work
if __name__ == "__main__" and __package__ is None:
    __package__ = "src.services"
# Add project root (parent of src/) to sys.path so 'src' is recognized
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

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
