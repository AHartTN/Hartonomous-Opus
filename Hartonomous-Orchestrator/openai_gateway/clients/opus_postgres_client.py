"""
PostgreSQL client for Opus database - semantic search and relation queries
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OpusPostgresClient:
    """Client for querying Opus PostgreSQL database for semantic operations"""

    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL client for Opus database

        Args:
            connection_string: PostgreSQL connection string
                e.g., "postgresql://user:pass@localhost:5432/hypercube"
        """
        self.connection_string = connection_string
        self.conn = None
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
            logger.info("Connected to Opus PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to Opus database: {e}")
            raise

    def _ensure_connection(self):
        """Ensure database connection is alive"""
        if self.conn is None or self.conn.closed:
            logger.warning("Database connection lost, reconnecting...")
            self._connect()

    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        model_filter: Optional[str] = None,
        layer_filter: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector similarity

        Args:
            query_embedding: Query vector (from embedding model)
            top_k: Number of results to return
            model_filter: Optional model name filter (e.g., 'minilm')
            layer_filter: Optional layer number filter

        Returns:
            List of dicts with keys: id, model, layer, component, distance, metadata
        """
        self._ensure_connection()

        # Convert embedding to PostgreSQL array format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Build query with optional filters
        query = """
            SELECT
                ENCODE(id, 'hex') as id,
                model,
                layer,
                component,
                embedding <-> %s::vector AS distance,
                metadata
            FROM composition
            WHERE embedding IS NOT NULL
        """
        params = [embedding_str]

        if model_filter:
            query += " AND model = %s"
            params.append(model_filter)

        if layer_filter is not None:
            query += " AND layer = %s"
            params.append(layer_filter)

        query += """
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        params.extend([embedding_str, top_k])

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()

                logger.info(f"Semantic search returned {len(results)} results")
                return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            self.conn.rollback()
            return []

    def get_composition_context(
        self,
        composition_ids: List[str],
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get full context for composition IDs

        Args:
            composition_ids: List of hex-encoded composition IDs
            include_metadata: Whether to include metadata field

        Returns:
            List of composition records
        """
        self._ensure_connection()

        if not composition_ids:
            return []

        # Convert hex IDs to bytea
        id_params = [bytes.fromhex(cid) for cid in composition_ids]

        query = """
            SELECT
                ENCODE(id, 'hex') as id,
                model,
                layer,
                component,
                position_x,
                position_y,
                position_z,
                position_w
        """

        if include_metadata:
            query += ", metadata"

        query += """
            FROM composition
            WHERE id = ANY(%s)
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, [id_params])
                results = cur.fetchall()
                return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get composition context: {e}")
            self.conn.rollback()
            return []

    def get_related_compositions(
        self,
        source_id: str,
        relation_types: Optional[List[str]] = None,
        min_rating: float = 1000.0,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get compositions related to a source composition via relation_evidence

        Args:
            source_id: Hex-encoded source composition ID
            relation_types: Optional list of relation types ('E', 'T', 'S', 'M')
            min_rating: Minimum ELO rating threshold
            max_results: Maximum number of results

        Returns:
            List of related compositions with relation metadata
        """
        self._ensure_connection()

        source_bytes = bytes.fromhex(source_id)

        query = """
            SELECT
                ENCODE(r.target_id, 'hex') as target_id,
                r.relation_type,
                r.rating,
                r.raw_weight,
                r.normalized_weight,
                r.observation_count,
                ENCODE(c.id, 'hex') as composition_id,
                c.model,
                c.layer,
                c.component,
                c.metadata
            FROM relation_evidence r
            JOIN composition c ON r.target_id = c.id
            WHERE r.source_id = %s
              AND r.rating >= %s
        """
        params = [source_bytes, min_rating]

        if relation_types:
            query += " AND r.relation_type = ANY(%s)"
            params.append(relation_types)

        query += """
            ORDER BY r.rating DESC
            LIMIT %s
        """
        params.append(max_results)

        try:
            with self.conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()

                logger.info(f"Found {len(results)} related compositions for {source_id}")
                return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get related compositions: {e}")
            self.conn.rollback()
            return []

    def multi_hop_search(
        self,
        start_embedding: List[float],
        max_hops: int = 3,
        top_k_per_hop: int = 5,
        min_rating: float = 1200.0
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop semantic search: start from embedding, traverse relations

        Args:
            start_embedding: Initial query embedding
            max_hops: Maximum number of relation hops
            top_k_per_hop: Number of results to expand per hop
            min_rating: Minimum ELO rating for relations

        Returns:
            List of compositions reached through traversal
        """
        # Start with semantic search
        initial_results = self.semantic_search(start_embedding, top_k=top_k_per_hop)

        if not initial_results:
            return []

        visited = set()
        current_frontier = [r['id'] for r in initial_results]
        all_results = initial_results.copy()

        for hop in range(max_hops):
            next_frontier = []

            for comp_id in current_frontier:
                if comp_id in visited:
                    continue

                visited.add(comp_id)

                # Get related compositions
                related = self.get_related_compositions(
                    comp_id,
                    min_rating=min_rating,
                    max_results=top_k_per_hop
                )

                for rel in related:
                    target_id = rel['target_id']
                    if target_id not in visited:
                        next_frontier.append(target_id)
                        all_results.append(rel)

            if not next_frontier:
                break

            current_frontier = next_frontier[:top_k_per_hop]

            logger.info(f"Hop {hop + 1}: {len(current_frontier)} new nodes")

        return all_results

    def get_text_content(
        self,
        composition_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get text content associated with compositions from metadata

        Args:
            composition_ids: List of hex-encoded composition IDs

        Returns:
            List of dicts with id, text, source, etc.
        """
        compositions = self.get_composition_context(composition_ids, include_metadata=True)

        results = []
        for comp in compositions:
            # Extract text from metadata if available
            metadata = comp.get('metadata', {})

            text_content = {
                'id': comp['id'],
                'model': comp['model'],
                'layer': comp['layer'],
                'text': metadata.get('text', metadata.get('content', '')),
                'source': metadata.get('source', 'unknown'),
                'metadata': metadata
            }
            results.append(text_content)

        return results

    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            self._ensure_connection()
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        self._ensure_connection()

        stats = {}

        try:
            with self.conn.cursor() as cur:
                # Count compositions
                cur.execute("SELECT COUNT(*) as count FROM composition")
                stats['composition_count'] = cur.fetchone()['count']

                # Count relations
                cur.execute("SELECT COUNT(*) as count FROM relation_evidence")
                stats['relation_count'] = cur.fetchone()['count']

                # Count by model
                cur.execute("""
                    SELECT model, COUNT(*) as count
                    FROM composition
                    GROUP BY model
                """)
                stats['models'] = {row['model']: row['count'] for row in cur.fetchall()}

                # Average ELO rating
                cur.execute("SELECT AVG(rating) as avg_rating FROM relation_evidence")
                stats['avg_relation_rating'] = float(cur.fetchone()['avg_rating'] or 0)

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return stats

    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Closed Opus database connection")


# Global client instance (initialized on startup)
opus_db_client: Optional[OpusPostgresClient] = None


def initialize_opus_client(connection_string: str):
    """Initialize the global Opus database client"""
    global opus_db_client
    opus_db_client = OpusPostgresClient(connection_string)
    logger.info("Opus database client initialized")


def get_opus_client() -> OpusPostgresClient:
    """Get the global Opus database client"""
    if opus_db_client is None:
        raise RuntimeError("Opus database client not initialized. Call initialize_opus_client() first.")
    return opus_db_client
