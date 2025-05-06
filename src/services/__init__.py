# src/services/__init__.py

# document ingestion & partitioning
from .docs_loader  import load_documents
from .sectioner    import split_into_sections

# external‑system wrappers
from .indexer      import index_documents
from .retriever    import retrieve
from .delete_index import delete_index

__all__ = [
    "load_documents",
    "split_into_sections",
    "index_documents",
    "retrieve",
    "delete_index",
]
