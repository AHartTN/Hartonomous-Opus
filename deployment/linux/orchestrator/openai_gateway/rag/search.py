"""
RAG search orchestration and ranking algorithms
"""
from typing import List, Dict, Any, Optional
import logging

from ..config import SEARCH_COLLECTIONS, RAG_TOP_K, RAG_RERANK_TOP_N, USE_OPUS_DB, RAG_MIN_RATING, RAG_MAX_HOPS

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(results_list: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Merge multiple search result lists using Reciprocal Rank Fusion (RRF)

    Args:
        results_list: List of result lists, each containing dicts with 'document' and 'score'
        k: RRF constant (default 60)

    Returns:
        Merged and re-ranked results
    """
    fusion_scores = {}

    for results in results_list:
        for rank, item in enumerate(results, start=1):
            doc = item["document"]
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)

            if doc in fusion_scores:
                fusion_scores[doc]["rrf_score"] += rrf_score
                fusion_scores[doc]["collections"].append(item.get("collection", "unknown"))
            else:
                fusion_scores[doc] = {
                    "document": doc,
                    "rrf_score": rrf_score,
                    "original_score": item.get("score", 0),
                    "collections": [item.get("collection", "unknown")],
                    "metadata": item.get("metadata", {})
                }

    # Sort by RRF score
    merged = sorted(fusion_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged


async def get_embedding_with_dimension(text: str, target_dim: Optional[int] = None) -> List[float]:
    """Get embedding with optional dimension reduction"""
    embedding = await llamacpp_client.get_embedding(text)

    if target_dim and target_dim < len(embedding):
        # Truncate to target dimension (simple but effective)
        return embedding[:target_dim]

    return embedding


async def rag_search_opus(query: str, top_k: int = RAG_TOP_K, rerank_top_n: int = RAG_RERANK_TOP_N) -> List[str]:
    """
    Perform RAG search using Hartonomous (pure database-native):
    1. Get query embedding (from Hartonomous, not llama.cpp!)
    2. Semantic search in composition table
    3. Optional: Expand via relations (multi-hop)
    4. Rerank results (using Hartonomous cosine similarity, not llama.cpp!)
    5. Extract text content

    NO llama.cpp dependencies - Hartonomous IS the model.
    """
    try:
        from ..clients.opus_postgres_client import get_opus_client
        from ..clients.hartonomous_client import get_hartonomous_client

        opus_client = get_opus_client()
        hartonomous_client = get_hartonomous_client()
        logger.info(f"Starting Hartonomous RAG search for query: {query[:100]}...")

        # Step 1: Get query embedding from Hartonomous
        # TODO: Implement proper query embedding from composition lookup or encoding
        # For now, use semantic search directly which handles embedding internally
        logger.info("Using Hartonomous database-native search (no external embedding needed)")

        # Step 2: Semantic search in Hartonomous database
        # The database handles embedding via pgvector - no external service needed
        search_results = opus_client.semantic_search_by_text(
            query_text=query,
            top_k=top_k * 2  # Get more candidates for reranking
        )

        if not search_results:
            logger.info("No results from Hartonomous semantic search")
            return []

        logger.info(f"Hartonomous semantic search returned {len(search_results)} results")

        # Step 3: Optional multi-hop expansion via relations
        if RAG_MAX_HOPS > 0:
            logger.info(f"Expanding via relations (max {RAG_MAX_HOPS} hops)")

            # Get top result IDs for expansion
            top_ids = [r['id'] for r in search_results[:5]]  # Expand from top 5

            expanded_ids = set()
            for comp_id in top_ids:
                related = opus_client.get_related_compositions(
                    source_id=comp_id,
                    min_rating=RAG_MIN_RATING,
                    max_results=10
                )
                for rel in related:
                    expanded_ids.add(rel['target_id'])

            if expanded_ids:
                # Get context for expanded compositions
                expanded_results = opus_client.get_composition_context(list(expanded_ids))
                # Merge with original results (deduplicate by ID)
                existing_ids = {r['id'] for r in search_results}
                for exp in expanded_results:
                    if exp['id'] not in existing_ids:
                        search_results.append(exp)

                logger.info(f"Expanded to {len(search_results)} total results via relations")

        # Step 4: Get text content from metadata
        text_results = opus_client.get_text_content(
            [r['id'] for r in search_results]
        )

        # Extract documents for reranking
        documents = []
        doc_metadata = {}

        for text_res in text_results:
            text = text_res.get('text', '')
            if text:
                documents.append(text)
                doc_metadata[text] = {
                    'id': text_res['id'],
                    'model': text_res['model'],
                    'layer': text_res['layer'],
                    'source': text_res.get('source', 'unknown')
                }

        if not documents:
            logger.info("No text content found in results")
            return []

        logger.info(f"Extracted {len(documents)} documents for reranking")

        # Step 5: Rerank using Hartonomous cosine similarity (NOT llama.cpp!)
        # TODO: Implement proper reranking using Hartonomous embedding operations
        # For now, just return top N documents by distance
        final_docs = documents[:rerank_top_n]
        logger.info(f"Selected top {len(final_docs)} documents (Hartonomous reranking)")

        return final_docs

    except Exception as e:
        logger.error(f"Hartonomous RAG search error: {e}", exc_info=True)
        return []


async def rag_search(query: str, top_k: int = RAG_TOP_K, rerank_top_n: int = RAG_RERANK_TOP_N) -> List[str]:
    """
    Perform RAG search - routes to either Opus PostgreSQL or Qdrant based on config
    """
    # Route to appropriate backend
    if USE_OPUS_DB:
        logger.info("Using Opus PostgreSQL for RAG search")
        return await rag_search_opus(query, top_k, rerank_top_n)

    # Qdrant fallback/legacy path
    logger.info("Using Qdrant for RAG search")
    try:
        all_results = []

        # Search across all configured collections
        for collection_name in SEARCH_COLLECTIONS:
            try:
                logger.info(f"Starting search in collection: {collection_name}")

                # Get collection info to determine vector dimension
                col_info = qdrant_vector_client.get_collection_info(collection_name)

                # Get the dimension of this collection
                if col_info and hasattr(col_info.config.params.vectors, 'size'):
                    collection_dim = col_info.config.params.vectors.size
                    logger.info(f"Collection {collection_name} dimension: {collection_dim}")
                else:
                    logger.warning(f"Skipping {collection_name}: unable to get dimension info")
                    continue

                # Get embedding at the right dimension for this collection
                query_embedding = await get_embedding_with_dimension(query, collection_dim)
                logger.info(f"Got embedding for {collection_name}: {len(query_embedding)}d")

                if len(query_embedding) != collection_dim:
                    logger.warning(f"Skipping {collection_name}: can't reduce {len(query_embedding)}d to {collection_dim}d")
                    continue

                # Search this collection
                search_results = qdrant_vector_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=top_k
                )

                logger.info(f"Collection {collection_name}: searched with {len(query_embedding)}d vector, found {len(search_results)} results")

                if search_results:
                    results_with_meta = [
                        {
                            "document": hit.payload.get("text", hit.payload.get("content", str(hit.payload))),
                            "score": hit.score,
                            "collection": collection_name,
                            "metadata": hit.payload
                        }
                        for hit in search_results
                    ]
                    all_results.append(results_with_meta)
                    logger.info(f"Found {len(search_results)} results in {collection_name}")

            except Exception as e:
                logger.warning(f"Error searching collection {collection_name}: {e}")
                continue

        if not all_results:
            logger.info("No documents found in any collection")
            return []

        # Merge results using RRF if multiple collections
        if len(all_results) > 1:
            logger.info(f"Merging results from {len(all_results)} collections using RRF")
            merged_results = reciprocal_rank_fusion(all_results)
            documents = [r["document"] for r in merged_results[:top_k * 2]]  # Take more for reranking
        else:
            documents = [r["document"] for r in all_results[0]]

        # Rerank the top documents
        reranked = await llamacpp_client.rerank_documents(query, documents, rerank_top_n)

        return [r["document"] for r in reranked]

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return []