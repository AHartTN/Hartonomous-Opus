"""
Hartonomous Cache Loader

Loads vocabulary, bigrams, and attention from PostgreSQL into C++ engine caches.
This is the "model loading" step - but the model IS the database.
"""
import logging
from typing import Optional
import psycopg2

from ..config import POSTGRES_URL, USE_OPUS_DB
from .hartonomous_client import get_hartonomous_client

logger = logging.getLogger(__name__)

_caches_loaded = False


def initialize_hartonomous_caches(force_reload: bool = False):
    """
    Load all Hartonomous caches from database

    This is the equivalent of "loading a model" in traditional AI frameworks.
    But here, we're just populating in-memory caches from the database substrate.

    Args:
        force_reload: If True, reload even if already loaded
    """
    global _caches_loaded

    if _caches_loaded and not force_reload:
        logger.info("Hartonomous caches already loaded")
        return

    if not USE_OPUS_DB:
        logger.warning("USE_OPUS_DB=false, skipping Hartonomous cache loading")
        return

    logger.info("Initializing Hartonomous caches from database...")

    try:
        client = get_hartonomous_client()

        # Connect to PostgreSQL
        conn = psycopg2.connect(POSTGRES_URL)
        cursor = conn.cursor()

        # Load vocabulary (required)
        logger.info("Loading vocabulary cache...")
        client.load_vocabulary_from_db(cursor)

        # Load bigrams (optional but recommended)
        logger.info("Loading bigram (PMI) cache...")
        try:
            client.load_bigrams_from_db(cursor, min_rating=1000.0)
        except Exception as e:
            logger.warning(f"Failed to load bigrams: {e}")

        # Load attention (optional)
        logger.info("Loading attention cache...")
        try:
            client.load_attention_from_db(cursor, min_rating=1200.0)
        except Exception as e:
            logger.warning(f"Failed to load attention: {e}")

        cursor.close()
        conn.close()

        _caches_loaded = True
        logger.info("Hartonomous caches initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize Hartonomous caches: {e}", exc_info=True)
        raise


def get_cache_stats() -> dict:
    """Get statistics about loaded caches"""
    if not _caches_loaded:
        return {
            "loaded": False,
            "message": "Caches not loaded"
        }

    try:
        from .hartonomous_client import _generative_lib

        if not _generative_lib:
            return {
                "loaded": False,
                "message": "generative_c DLL not available"
            }

        return {
            "loaded": True,
            "vocab_size": _generative_lib.gen_vocab_count(),
            "bigram_count": _generative_lib.gen_bigram_count(),
            "attention_edges": _generative_lib.gen_attention_count(),
        }
    except Exception as e:
        return {
            "loaded": False,
            "error": str(e)
        }


def reload_caches():
    """Force reload all caches from database"""
    global _caches_loaded
    _caches_loaded = False
    initialize_hartonomous_caches(force_reload=True)
