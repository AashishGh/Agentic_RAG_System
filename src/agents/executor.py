# executor.py
from .planner import plan
from src.services.retriever import init_es_client, retrieve
from .reasoner import reason
from src.config import top_k
from src.logger import logger


def execute(query: str) -> str:
    """
    Orchestrate the full Agentic RAG: plan → retrieve → reason.
    """
    logger.info(f"Starting execution for query: {query}")
    es = init_es_client()

    subqueries = plan(query)
    all_context = []
    for sq in subqueries:
        logger.info(f"Retrieving for subquery: {sq}")
        hits = retrieve(sq, es, top_k)
        chunks = [h["text"] for h in hits]
        all_context.extend(chunks)

    logger.info(f"Reasoning with {len(all_context)} context chunks")
    answer = reason(query, all_context)
    logger.info("Execution complete")
    return answer
