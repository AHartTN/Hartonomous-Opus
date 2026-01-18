"""
Qdrant client wrapper for vector database operations
"""
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import models
from ..config import qdrant_client, COLLECTION_NAME, VECTOR_SIZE

logger = logging.getLogger(__name__)


class QdrantVectorClient:
    """Client for Qdrant vector database operations"""

    def __init__(self):
        self.client = qdrant_client

    def search(self, collection_name: str, query_vector: List[float], limit: int = 10,
               query_filter: Optional[models.Filter] = None) -> List[models.ScoredPoint]:
        """Search for similar vectors in a collection"""
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )

    def multi_search(self, collection_names: List[str], query_vector: List[float], limit: int = 10) -> Dict[str, List[models.ScoredPoint]]:
        """Search across multiple collections"""
        results = {}
        for collection_name in collection_names:
            try:
                results[collection_name] = self.search(collection_name, query_vector, limit)
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                results[collection_name] = []
        return results

    def upsert_points(self, collection_name: str, points: List[models.PointStruct]) -> bool:
        """Add or update points in a collection"""
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"Error upserting points to {collection_name}: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[models.CollectionInfo]:
        """Get information about a collection"""
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None

    def list_collections(self) -> List[models.CollectionDescription]:
        """List all collections"""
        try:
            return self.client.get_collections().collections
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def create_collection(self, collection_name: str, vector_size: int = VECTOR_SIZE) -> bool:
        """Create a new collection"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            return True
        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    def recreate_collection(self, collection_name: str, vector_size: int = VECTOR_SIZE) -> bool:
        """Recreate a collection (delete and create)"""
        if self.delete_collection(collection_name):
            return self.create_collection(collection_name, vector_size)
        return False


# Global instance
qdrant_vector_client = QdrantVectorClient()