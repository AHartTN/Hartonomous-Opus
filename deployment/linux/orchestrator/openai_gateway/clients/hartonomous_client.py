"""
Hartonomous C API Python Bridge

Direct interface to Hartonomous DLLs for:
- Embedding operations (SIMD-accelerated vector operations)
- Text generation (database-native inference)
- Hypercube geometry (4D coordinate mapping)

Replaces llama.cpp dependencies - the database IS the model.
"""
import ctypes
import os
import platform
from typing import List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DLL Loading
# =============================================================================

def _find_dll(dll_name: str) -> Optional[Path]:
    """Find Hartonomous DLL in various possible locations"""
    repo_root = Path(__file__).parent.parent.parent.parent

    # Possible locations
    search_paths = [
        repo_root / "cpp" / "build" / "bin" / "Release",
        repo_root / "cpp" / "build" / "bin" / "Debug",
        repo_root / "cpp" / "build" / "Release",
        repo_root / "cpp" / "build" / "Debug",
        repo_root / "cppbuild" / "bin" / "Release",
        repo_root / "cppbuild" / "bin" / "Debug",
    ]

    system = platform.system()
    if system == "Windows":
        dll_filename = f"{dll_name}.dll"
    elif system == "Darwin":
        dll_filename = f"lib{dll_name}.dylib"
    else:
        dll_filename = f"lib{dll_name}.so"

    for path in search_paths:
        dll_path = path / dll_filename
        if dll_path.exists():
            logger.info(f"Found {dll_name} at {dll_path}")
            return dll_path

    logger.warning(f"Could not find {dll_name} in any search path")
    return None


# Load DLLs
_embedding_dll_path = _find_dll("embedding_c")
_generative_dll_path = _find_dll("generative_c")
_hypercube_dll_path = _find_dll("hypercube_c")

# On Windows, add DLL directory to PATH so dependencies can be found
if platform.system() == "Windows" and _hypercube_dll_path:
    dll_dir = str(_hypercube_dll_path.parent)
    if dll_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Added {dll_dir} to PATH for DLL dependencies")

    # Also use add_dll_directory for Python 3.8+
    try:
        os.add_dll_directory(dll_dir)
        logger.info(f"Added {dll_dir} to DLL search directories")
    except AttributeError:
        pass  # Python < 3.8

if _embedding_dll_path:
    _embedding_lib = ctypes.CDLL(str(_embedding_dll_path))
else:
    _embedding_lib = None
    logger.error("embedding_c DLL not found - embedding operations unavailable")

if _generative_dll_path:
    _generative_lib = ctypes.CDLL(str(_generative_dll_path))
else:
    _generative_lib = None
    logger.error("generative_c DLL not found - text generation unavailable")

if _hypercube_dll_path:
    _hypercube_lib = ctypes.CDLL(str(_hypercube_dll_path))
else:
    _hypercube_lib = None
    logger.error("hypercube_c DLL not found - geometry operations unavailable")


# =============================================================================
# C Structure Definitions
# =============================================================================

class EmbeddingSimilarityResult(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_size_t),
        ("similarity", ctypes.c_double),
    ]


class GenTokenResult(ctypes.Structure):
    _fields_ = [
        ("token_index", ctypes.c_size_t),
        ("score_centroid", ctypes.c_double),
        ("score_pmi", ctypes.c_double),
        ("score_attn", ctypes.c_double),
        ("score_global", ctypes.c_double),
        ("score_total", ctypes.c_double),
    ]


class GeomPoint4D(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint64),
        ("y", ctypes.c_uint64),
        ("z", ctypes.c_uint64),
        ("m", ctypes.c_uint64),
    ]


# =============================================================================
# Embedding Operations API
# =============================================================================

if _embedding_lib:
    # embedding_c_cosine_similarity
    _embedding_lib.embedding_c_cosine_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t
    ]
    _embedding_lib.embedding_c_cosine_similarity.restype = ctypes.c_double

    # embedding_c_find_top_k
    _embedding_lib.embedding_c_find_top_k.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # query
        ctypes.POINTER(ctypes.c_float),  # embeddings
        ctypes.c_size_t,                 # n_embeddings
        ctypes.c_size_t,                 # dim
        ctypes.c_size_t,                 # k
        ctypes.POINTER(EmbeddingSimilarityResult)
    ]
    _embedding_lib.embedding_c_find_top_k.restype = ctypes.c_size_t

    # embedding_c_cache_*
    _embedding_lib.embedding_c_cache_init.argtypes = []
    _embedding_lib.embedding_c_cache_init.restype = ctypes.c_int

    _embedding_lib.embedding_c_cache_clear.argtypes = []
    _embedding_lib.embedding_c_cache_clear.restype = None

    _embedding_lib.embedding_c_cache_add.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # id
        ctypes.c_size_t,                 # id_len
        ctypes.c_char_p,                 # label
        ctypes.POINTER(ctypes.c_float),  # embedding
        ctypes.c_size_t                  # dim
    ]
    _embedding_lib.embedding_c_cache_add.restype = ctypes.c_int64

    _embedding_lib.embedding_c_simd_level.argtypes = []
    _embedding_lib.embedding_c_simd_level.restype = ctypes.c_char_p


# =============================================================================
# Generative Engine API
# =============================================================================

if _generative_lib:
    # gen_vocab_*
    _generative_lib.gen_vocab_clear.argtypes = []
    _generative_lib.gen_vocab_clear.restype = None

    _generative_lib.gen_vocab_add.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # id (32 bytes)
        ctypes.c_char_p,                 # label
        ctypes.c_int,                    # depth
        ctypes.c_double,                 # frequency
        ctypes.c_double                  # hilbert
    ]
    _generative_lib.gen_vocab_add.restype = ctypes.c_int64

    _generative_lib.gen_vocab_set_centroid.argtypes = [
        ctypes.c_size_t,    # idx
        ctypes.c_double,    # x
        ctypes.c_double,    # y
        ctypes.c_double,    # z
        ctypes.c_double     # m
    ]
    _generative_lib.gen_vocab_set_centroid.restype = ctypes.c_int

    _generative_lib.gen_vocab_count.argtypes = []
    _generative_lib.gen_vocab_count.restype = ctypes.c_size_t

    _generative_lib.gen_vocab_find_label.argtypes = [ctypes.c_char_p]
    _generative_lib.gen_vocab_find_label.restype = ctypes.c_int64

    _generative_lib.gen_vocab_get_label.argtypes = [ctypes.c_size_t]
    _generative_lib.gen_vocab_get_label.restype = ctypes.c_char_p

    # gen_bigram_*
    _generative_lib.gen_bigram_clear.argtypes = []
    _generative_lib.gen_bigram_clear.restype = None

    _generative_lib.gen_bigram_add.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # left_id
        ctypes.POINTER(ctypes.c_uint8),  # right_id
        ctypes.c_double                  # score
    ]
    _generative_lib.gen_bigram_add.restype = None

    _generative_lib.gen_bigram_count.argtypes = []
    _generative_lib.gen_bigram_count.restype = ctypes.c_size_t

    # gen_attention_*
    _generative_lib.gen_attention_clear.argtypes = []
    _generative_lib.gen_attention_clear.restype = None

    _generative_lib.gen_attention_add.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # source_id
        ctypes.POINTER(ctypes.c_uint8),  # target_id
        ctypes.c_double                  # weight
    ]
    _generative_lib.gen_attention_add.restype = None

    _generative_lib.gen_attention_count.argtypes = []
    _generative_lib.gen_attention_count.restype = ctypes.c_size_t

    # gen_config_*
    _generative_lib.gen_config_set_weights.argtypes = [
        ctypes.c_double,  # w_centroid
        ctypes.c_double,  # w_pmi
        ctypes.c_double,  # w_attn
        ctypes.c_double   # w_global
    ]
    _generative_lib.gen_config_set_weights.restype = None

    _generative_lib.gen_config_set_policy.argtypes = [
        ctypes.c_int,     # greedy
        ctypes.c_double   # temperature
    ]
    _generative_lib.gen_config_set_policy.restype = None

    # gen_generate
    _generative_lib.gen_generate.argtypes = [
        ctypes.c_char_p,                    # start_label
        ctypes.c_size_t,                    # max_tokens
        ctypes.POINTER(GenTokenResult)      # results
    ]
    _generative_lib.gen_generate.restype = ctypes.c_size_t

    # gen_score_candidates
    _generative_lib.gen_score_candidates.argtypes = [
        ctypes.c_char_p,                    # current_label
        ctypes.c_size_t,                    # top_k
        ctypes.POINTER(GenTokenResult)      # results
    ]
    _generative_lib.gen_score_candidates.restype = ctypes.c_size_t


# =============================================================================
# High-Level Python API
# =============================================================================

class HartonomousClient:
    """
    Hartonomous database-native AI client

    Replaces llama.cpp - the database IS the model.
    Ingestion IS training. Queries ARE inference.
    """

    def __init__(self):
        self.vocab_loaded = False
        self.bigrams_loaded = False
        self.attention_loaded = False

        # Initialize embedding cache
        if _embedding_lib:
            _embedding_lib.embedding_c_cache_init()
            simd_level = _embedding_lib.embedding_c_simd_level()
            logger.info(f"Hartonomous embedding engine initialized (SIMD: {simd_level.decode()})")

        # Set default generation config
        if _generative_lib:
            # Weights: centroid=0.4, pmi=0.3, attention=0.2, global=0.1
            _generative_lib.gen_config_set_weights(0.4, 0.3, 0.2, 0.1)
            # Stochastic sampling with temperature=0.7
            _generative_lib.gen_config_set_policy(0, 0.7)
            logger.info("Hartonomous generative engine initialized")

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors (SIMD-accelerated)"""
        if not _embedding_lib:
            raise RuntimeError("embedding_c DLL not loaded")

        if len(a) != len(b):
            raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

        dim = len(a)
        a_arr = (ctypes.c_float * dim)(*a)
        b_arr = (ctypes.c_float * dim)(*b)

        return _embedding_lib.embedding_c_cosine_similarity(a_arr, b_arr, dim)

    def find_top_k(
        self,
        query: List[float],
        embeddings: List[List[float]],
        k: int
    ) -> List[Tuple[int, float]]:
        """
        Find top-k most similar embeddings (SIMD-accelerated batch comparison)

        Returns list of (index, similarity) tuples
        """
        if not _embedding_lib:
            raise RuntimeError("embedding_c DLL not loaded")

        dim = len(query)
        n_embeddings = len(embeddings)

        # Flatten embeddings to C array
        flat_embeddings = []
        for emb in embeddings:
            if len(emb) != dim:
                raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {len(emb)}")
            flat_embeddings.extend(emb)

        query_arr = (ctypes.c_float * dim)(*query)
        embeddings_arr = (ctypes.c_float * (n_embeddings * dim))(*flat_embeddings)
        results_arr = (EmbeddingSimilarityResult * k)()

        n_results = _embedding_lib.embedding_c_find_top_k(
            query_arr,
            embeddings_arr,
            n_embeddings,
            dim,
            k,
            results_arr
        )

        return [(results_arr[i].index, results_arr[i].similarity) for i in range(n_results)]

    def load_vocabulary_from_db(self, db_cursor):
        """
        Load vocabulary cache from PostgreSQL composition table

        Populates the generative engine with tokens, centroids, and metadata.
        """
        if not _generative_lib:
            raise RuntimeError("generative_c DLL not loaded")

        logger.info("Loading vocabulary from database...")
        _generative_lib.gen_vocab_clear()

        # Query compositions with text metadata
        db_cursor.execute("""
            SELECT
                id,
                metadata->>'text' AS label,
                layer AS depth,
                COALESCE((metadata->>'frequency')::double precision, 1.0) AS frequency,
                COALESCE((metadata->>'hilbert')::double precision, 0.5) AS hilbert,
                COALESCE((metadata->>'centroid_x')::double precision, 0.0) AS cx,
                COALESCE((metadata->>'centroid_y')::double precision, 0.0) AS cy,
                COALESCE((metadata->>'centroid_z')::double precision, 0.0) AS cz,
                COALESCE((metadata->>'centroid_m')::double precision, 0.0) AS cm
            FROM composition
            WHERE metadata->>'text' IS NOT NULL
            ORDER BY layer, id
        """)

        count = 0
        for row in db_cursor.fetchall():
            comp_id, label, depth, frequency, hilbert, cx, cy, cz, cm = row

            # Convert hex ID to bytes
            if isinstance(comp_id, str):
                id_bytes = bytes.fromhex(comp_id)
            else:
                id_bytes = bytes(comp_id)

            # Add vocab entry
            id_arr = (ctypes.c_uint8 * 32)(*id_bytes[:32])
            idx = _generative_lib.gen_vocab_add(
                id_arr,
                label.encode('utf-8'),
                depth,
                frequency,
                hilbert
            )

            # Set centroid
            if idx >= 0:
                _generative_lib.gen_vocab_set_centroid(idx, cx, cy, cz, cm)
                count += 1

        vocab_size = _generative_lib.gen_vocab_count()
        logger.info(f"Loaded {count} vocabulary entries (vocab size: {vocab_size})")
        self.vocab_loaded = True

    def load_bigrams_from_db(self, db_cursor, min_rating: float = 1000.0):
        """
        Load bigram (PMI) cache from relation_evidence table

        Uses ELO-rated relations as statistical co-occurrence scores.
        """
        if not _generative_lib:
            raise RuntimeError("generative_c DLL not loaded")

        logger.info(f"Loading bigrams from database (min rating: {min_rating})...")
        _generative_lib.gen_bigram_clear()

        # Query high-quality relations
        db_cursor.execute("""
            SELECT
                source_id,
                target_id,
                rating / 1000.0 AS pmi_score
            FROM relation_evidence
            WHERE rating >= %s
            ORDER BY rating DESC
            LIMIT 100000
        """, (min_rating,))

        count = 0
        for row in db_cursor.fetchall():
            source_id, target_id, pmi_score = row

            # Convert IDs to bytes
            source_bytes = bytes.fromhex(source_id) if isinstance(source_id, str) else bytes(source_id)
            target_bytes = bytes.fromhex(target_id) if isinstance(target_id, str) else bytes(target_id)

            source_arr = (ctypes.c_uint8 * 32)(*source_bytes[:32])
            target_arr = (ctypes.c_uint8 * 32)(*target_bytes[:32])

            _generative_lib.gen_bigram_add(source_arr, target_arr, pmi_score)
            count += 1

        bigram_count = _generative_lib.gen_bigram_count()
        logger.info(f"Loaded {count} bigram entries (cache size: {bigram_count})")
        self.bigrams_loaded = True

    def load_attention_from_db(self, db_cursor, min_rating: float = 1200.0):
        """
        Load attention cache from high-quality relations

        Attention edges represent learned associations between tokens.
        """
        if not _generative_lib:
            raise RuntimeError("generative_c DLL not loaded")

        logger.info(f"Loading attention from database (min rating: {min_rating})...")
        _generative_lib.gen_attention_clear()

        db_cursor.execute("""
            SELECT
                source_id,
                target_id,
                rating / 1500.0 AS weight
            FROM relation_evidence
            WHERE rating >= %s
            ORDER BY rating DESC
            LIMIT 50000
        """, (min_rating,))

        count = 0
        for row in db_cursor.fetchall():
            source_id, target_id, weight = row

            source_bytes = bytes.fromhex(source_id) if isinstance(source_id, str) else bytes(source_id)
            target_bytes = bytes.fromhex(target_id) if isinstance(target_id, str) else bytes(target_id)

            source_arr = (ctypes.c_uint8 * 32)(*source_bytes[:32])
            target_arr = (ctypes.c_uint8 * 32)(*target_bytes[:32])

            _generative_lib.gen_attention_add(source_arr, target_arr, weight)
            count += 1

        attention_count = _generative_lib.gen_attention_count()
        logger.info(f"Loaded {count} attention entries (cache size: {attention_count})")
        self.attention_loaded = True

    def generate_text(self, start_text: str, max_tokens: int = 50) -> str:
        """
        Generate text starting from a seed

        This is the core "inference" - database queries replace forward pass.
        No neural network evaluation, just geometric/statistical scoring.
        """
        if not _generative_lib:
            raise RuntimeError("generative_c DLL not loaded")

        if not self.vocab_loaded:
            raise RuntimeError("Vocabulary not loaded - call load_vocabulary_from_db() first")

        logger.info(f"Generating text from: '{start_text}' (max {max_tokens} tokens)")

        results_arr = (GenTokenResult * max_tokens)()
        n_tokens = _generative_lib.gen_generate(
            start_text.encode('utf-8'),
            max_tokens,
            results_arr
        )

        # Reconstruct text from token indices
        generated_tokens = []
        for i in range(n_tokens):
            token_idx = results_arr[i].token_index
            label = _generative_lib.gen_vocab_get_label(token_idx)
            if label:
                generated_tokens.append(label.decode('utf-8'))

        generated_text = ' '.join(generated_tokens)
        logger.info(f"Generated {n_tokens} tokens: '{generated_text[:100]}...'")

        return generated_text


# Global client instance
_client = None


def get_hartonomous_client() -> HartonomousClient:
    """Get singleton Hartonomous client instance"""
    global _client
    if _client is None:
        _client = HartonomousClient()
    return _client
