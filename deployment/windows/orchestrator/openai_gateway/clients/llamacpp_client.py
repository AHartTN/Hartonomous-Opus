"""
Client for interacting with llama.cpp servers (generative, embedding, reranker)
"""
import httpx
from typing import List, Dict, Any, AsyncGenerator, Optional
from fastapi import HTTPException
import logging

from ..config import GENERATIVE_URL, EMBEDDING_URL, RERANKER_URL, BACKEND_API_KEY

logger = logging.getLogger(__name__)


class LlamaCppClient:
    """Unified client for llama.cpp server interactions"""

    def __init__(self):
        self.generative_url = GENERATIVE_URL
        self.embedding_url = EMBEDDING_URL
        self.reranker_url = RERANKER_URL
        self.api_key = BACKEND_API_KEY
        self.timeout = 300.0  # 5 minutes for long generations

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        return {"Authorization": f"Bearer {self.api_key}"}

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from llama.cpp embedding server"""
        headers = self._get_headers()
        payload = {"content": text}

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.embedding_url}/embedding",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()

                # Handle various llama.cpp embedding response formats:
                # 1. Direct list: [0.1, 0.2, ...]
                # 2. Dict with embedding key: {"embedding": [0.1, 0.2, ...]}
                # 3. List of dicts (batch): [{"index": 0, "embedding": [0.1, ...]}]

                if isinstance(result, list):
                    if len(result) > 0 and isinstance(result[0], dict) and "embedding" in result[0]:
                        # Format 3: Extract embedding from first result
                        return result[0]["embedding"]
                    elif len(result) > 0 and isinstance(result[0], (int, float)):
                        # Format 1: Direct list of numbers
                        return result
                    else:
                        logger.error(f"Unexpected list format in embedding response: {result[:3] if len(result) > 0 else result}")
                        return []
                elif isinstance(result, dict):
                    # Format 2: Dict with embedding key
                    return result.get("embedding", [])
                else:
                    logger.error(f"Unexpected embedding response type: {type(result)}")
                    return []
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

    async def rerank_documents(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank documents using llama.cpp reranker"""
        headers = self._get_headers()
        results = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for idx, doc in enumerate(documents):
                try:
                    payload = {"query": query, "document": doc}
                    response = await client.post(
                        f"{self.reranker_url}/rerank",
                        json=payload,
                        headers=headers
                    )
                    response.raise_for_status()
                    result = response.json()
                    results.append({
                        "index": idx,
                        "document": doc,
                        "score": result.get("score", 0.0)
                    })
                except Exception as e:
                    logger.warning(f"Reranking error for doc {idx}: {e}")
                    results.append({
                        "index": idx,
                        "document": doc,
                        "score": 0.0
                    })

        # Sort by score and return top_n
        results.sort(key=lambda x: x["score"], reverse=True)
        if top_n:
            return results[:top_n]
        return results

    async def generate_completion(self, payload: Dict[str, Any], stream: bool = False) -> Any:
        """Generate completion using llama.cpp generative server"""
        headers = self._get_headers()
        url = f"{self.generative_url}/completion"
        logger.info(f"Proxying to llama.cpp: {url}, stream={stream}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                if stream:
                    async def generate():
                        async with client.stream("POST", url, json=payload, headers=headers) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_bytes():
                                yield chunk
                    return generate()
                else:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Backend error: {e}")
                raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    async def health_check(self, service: str) -> bool:
        """Check health of a specific service"""
        urls = {
            "generative": self.generative_url,
            "embedding": self.embedding_url,
            "reranker": self.reranker_url
        }

        if service not in urls:
            return False

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{urls[service]}/health",
                    headers=self._get_headers()
                )
                return response.status_code == 200
        except:
            return False


# Global instance
llamacpp_client = LlamaCppClient()