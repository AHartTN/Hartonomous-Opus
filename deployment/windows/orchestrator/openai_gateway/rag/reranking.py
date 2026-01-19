"""
Document reranking logic and strategies
"""
from typing import List, Dict, Any, Optional

from ..clients.llamacpp_client import llamacpp_client


class Reranker:
    """Document reranking service"""

    def __init__(self):
        self.client = llamacpp_client

    async def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query

        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Number of top results to return (None for all)

        Returns:
            Reranked documents with scores
        """
        return await self.client.rerank_documents(query, documents, top_n)


# Global instance
reranker = Reranker()