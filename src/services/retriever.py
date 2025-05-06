# retriever.py
"""
Module for vector + hybrid retrieval from Elasticsearch for Agentic RAG.
"""

from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from src.config import (
    ELASTIC_CONNECTION_URL,
    ELASTIC_USERNAME,
    ELASTIC_PASSWORD,
    INDEX_NAME,
    top_k
)
from src.agents.embedder import embed_text


def init_es_client() -> Elasticsearch:
    """
    Initialize and return an Elasticsearch client with basic authentication.
    """
    es = Elasticsearch(
        [ELASTIC_CONNECTION_URL],
        basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD),
        verify_certs=False
    )
    return es

def build_advanced_hybrid_query(
    query_text: str,
    query_vector: List[float],
    top_k: int,
    min_score: float = 1.2,
    bm25_boost: float = 1.0,
    knn_boost: float = 2.0,
    knn_candidates_factor: int = 20,
) -> Dict[str, Any]:
    """
    Build an ES query that:
      1. Combines BM25 match + kNN in a bool SHOULD
      2. Wraps that in a script_score (so final score = _score)
      3. Applies a min_score threshold
    """
    # 1) the inner hybrid
    hybrid_bool = {
        "bool": {
            "should": [
                {"match": {"text": {"query": query_text, "boost": bm25_boost}}},
                {
                    "knn": {
                        "field": "vector",
                        "query_vector": query_vector,
                        "num_candidates": top_k * knn_candidates_factor,
                        "boost": knn_boost
                    }
                }
            ]
        }
    }

    # 2) wrap in script_score so we can also use min_score
    script_scored = {
        "script_score": {
            "query": hybrid_bool,
            "script": {
                # simply use the combined _score from the hybrid
                "source": "_score"
            }
        }
    }

    return {
        "size": top_k,
        "query": script_scored,
        # drop anything whose hybrid score < min_score
        "min_score": min_score,
        # only return the fields you need
        "_source": ["file_path", "section", "chunk_id", "text"]
    }

# def build_hybrid_query(query_text: str, query_vector: List[float],
#                        top_k: int) -> Dict[str, Any]:
#     """
#     Build a hybrid BM25 + k-NN Elasticsearch query body.

#     :param query_text: The original user query string.
#     :param query_vector: The embedding vector of the query.
#     :param top_k: Number of results to return.
#     :return: Query body dict.
#     """
#     return {
#         "size": top_k,
#         "query": {
#             "bool": {
#                 "should": [
#                     {
#                         "match": {
#                             "text": {
#                                 "query": query_text,
#                                 "boost": 1.0
#                             }
#                         }
#                     },
#                     {
#                         "knn": {
#                             "field": "vector",
#                             "query_vector": query_vector,
#                             "num_candidates": top_k * 20,
#                             "boost": 2.0
#                             }

#                     }
#                 ]
#             }
#         }
#     }


def retrieve(query_text: str, es: Elasticsearch, top_k: int) -> List[Dict[str, Any]]:
    """
    Retrieve top_k document chunks for a query using hybrid search.

    :param query_text: The user query string.
    :param es: Initialized Elasticsearch client.
    :param top_k: Number of results to return.
    :return: List of hits with metadata and text.
    """
    # 1) Embed the query
    query_vector = embed_text(query_text)

    # 2) Build the hybrid query body
    body = build_advanced_hybrid_query(query_text, query_vector, top_k)

    # 3) Execute search
    resp = es.search(index=INDEX_NAME, body=body)

    # 4) Parse hits
    results = []
    for hit in resp.get("hits", {}).get("hits", []):
        src = hit.get("_source", {})
        results.append({
            "file_path": src.get("file_path"),
            "section": src.get("section"),
            "chunk_id": src.get("chunk_id"),
            "text": src.get("text"),
            "score": hit.get("_score")
        })
    return results


# def main():
#     """
#     Simple CLI for testing retrieval.
#     """
#     es = init_es_client()
#     # Example usage
#     query = input("Enter query: ")
#     hits = retrieve(query, es, top_k)
#     print(f"\nTop {top_k} results for: '{query}'\n")
#     for r in hits:
#         print(f"→ {r['file_path']} [{r['section']} - chunk {r['chunk_id']}] (score: {r['score']:.3f})")
#         print(f"   {r['text'][:200]}...\n")


# if __name__ == "__main__":
#     main()
