"""
Configuration management for OpenAI Gateway
"""
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)

# Backend server configuration
GENERATIVE_URL = os.getenv("GENERATIVE_URL", "http://localhost:8710")
EMBEDDING_URL = os.getenv("EMBEDDING_URL", "http://localhost:8711")
RERANKER_URL = os.getenv("RERANKER_URL", "http://localhost:8712")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "Welcome!123")

# Qdrant configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "Welcome!123")

# RAG configuration
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
RAG_RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "3"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "knowledge_base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "2560"))  # Qwen3-Embedding-4B default

# Multi-collection search configuration
SEARCH_ALL_COLLECTIONS = os.getenv("SEARCH_ALL_COLLECTIONS", "true").lower() == "true"
SEARCH_COLLECTIONS_OVERRIDE = os.getenv("SEARCH_COLLECTIONS", "").split(",") if os.getenv("SEARCH_COLLECTIONS") else []
SEARCH_COLLECTIONS_OVERRIDE = [c.strip() for c in SEARCH_COLLECTIONS_OVERRIDE if c.strip()]

# Will be populated at startup
SEARCH_COLLECTIONS = []

# Data directory
DATA_DIR = os.getenv("DATA_DIR", "data")

# Files configuration
FILES_DIR = os.getenv("FILES_DIR", "data/files")
FILES_METADATA_FILE = os.path.join(FILES_DIR, "metadata.json")

# Assistants configuration
ASSISTANTS_DIR = os.getenv("ASSISTANTS_DIR", "data/assistants")
ASSISTANTS_METADATA_FILE = os.path.join(ASSISTANTS_DIR, "metadata.json")

# Threads configuration
THREADS_DIR = os.getenv("THREADS_DIR", "data/threads")
THREADS_METADATA_FILE = os.path.join(THREADS_DIR, "metadata.json")

# Runs configuration
RUNS_DIR = os.getenv("RUNS_DIR", "data/runs")
RUNS_METADATA_FILE = os.path.join(RUNS_DIR, "metadata.json")

# Ensure data, files, assistants, threads, and runs directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(ASSISTANTS_DIR, exist_ok=True)
os.makedirs(THREADS_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0
)

def initialize_collections():
    """
    Initialize Qdrant collections and set up SEARCH_COLLECTIONS
    """
    global SEARCH_COLLECTIONS

    # Ensure collection exists
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        logger.info(f"Connected to existing collection: {COLLECTION_NAME}")
    except:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Created new collection: {COLLECTION_NAME}")

    # Auto-discover collections for search
    if SEARCH_COLLECTIONS_OVERRIDE:
        # Use explicit override
        SEARCH_COLLECTIONS = SEARCH_COLLECTIONS_OVERRIDE
        logger.info(f"Using explicit collection list: {SEARCH_COLLECTIONS}")
    elif SEARCH_ALL_COLLECTIONS:
        # Auto-discover all collections (supports dimension reduction)
        try:
            all_collections = qdrant_client.get_collections()
            for col in all_collections.collections:
                try:
                    col_info = qdrant_client.get_collection(col.name)
                    # Check if single vector (not named vectors)
                    if hasattr(col_info.config.params.vectors, 'size'):
                        col_dim = col_info.config.params.vectors.size
                        # Include all collections - we'll reduce dimensions as needed
                        if col_dim <= VECTOR_SIZE:
                            SEARCH_COLLECTIONS.append(col.name)
                            logger.info(f"Auto-discovered: {col.name} ({col_dim}d, {col_info.points_count} points)")
                        else:
                            logger.warning(f"Skipping {col.name}: requires {col_dim}d > our max {VECTOR_SIZE}d")
                    else:
                        logger.info(f"Skipping {col.name}: named vectors not supported yet")
                except Exception as e:
                    logger.warning(f"Error checking collection {col.name}: {e}")
            logger.info(f"Auto-discovery complete. Will search {len(SEARCH_COLLECTIONS)} collections: {SEARCH_COLLECTIONS}")
        except Exception as e:
            logger.warning(f"Collection auto-discovery failed: {e}, using primary collection only")
            SEARCH_COLLECTIONS = [COLLECTION_NAME]
    else:
        # Just use primary collection
        SEARCH_COLLECTIONS = [COLLECTION_NAME]
        logger.info(f"Searching primary collection only: {COLLECTION_NAME}")