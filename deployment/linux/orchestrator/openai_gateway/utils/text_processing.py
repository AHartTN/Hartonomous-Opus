"""
Text processing utilities for OpenAI Gateway
"""
import logging
import tiktoken
from typing import List, Dict, Any, Union
from ..models import ChatMessage

logger = logging.getLogger(__name__)

# Tokenizer for chunking
try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception as e:
    logger.warning(f"Failed to load tiktoken encoder, using character count fallback: {e}")
    tokenizer = None


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Chunk text into smaller pieces with overlap"""
    if tokenizer:
        tokens = tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(tokenizer.decode(chunk_tokens))
        return chunks
    else:
        # Character-based fallback
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks


def convert_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert OpenAI messages format to llama.cpp prompt"""
    prompt_parts = []

    def extract_text_content(content) -> str:
        """Extract text from content, handling both string and list formats"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle OpenAI content blocks format
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    # Skip image blocks or other non-text content
                else:
                    text_parts.append(str(block))
            return "".join(text_parts)
        else:
            return str(content)

    for msg in messages:
        content_text = extract_text_content(msg.content)
        if msg.role == "system":
            prompt_parts.append(f"System: {content_text}")
        elif msg.role == "user":
            prompt_parts.append(f"User: {content_text}")
        elif msg.role == "assistant":
            prompt_parts.append(f"Assistant: {content_text}")

    prompt_parts.append("Assistant:")
    return "\n".join(prompt_parts)