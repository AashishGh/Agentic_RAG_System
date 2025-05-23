import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Elasticsearch configuration
ELASTIC_CONNECTION_URL = os.getenv("ELASTIC_CONNECTION_URL")  # e.g. "https://localhost:9200"
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

# Documents directory
# DOCS_FOLDER = os.getenv("DOCS_FOLDER", "./documents")  # Folder where txt files are stored
# Get the absolute path to the project root (one level above `src`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_FOLDER =  str(PROJECT_ROOT / "documents")


# Elasticsearch index name
INDEX_NAME = os.getenv("INDEX_NAME", "rag_docs")

# Embedding configuration
# Model to use for embedding locally (no external API key required)

MODEL_NAME    = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", 4096))  # Must match your model’s hidden size
# …


# Text chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))        # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))  # overlap between chunks

# LLM (OpenAI) configuration (if used for reasoning)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Retrieval parameters
top_k = int(os.getenv("TOP_K", 5))  # number of chunks to retrieve per query
