"""
RAG prompt construction utilities
"""
from typing import List

from ..models import ChatMessage


def build_rag_prompt(messages: List[ChatMessage], context_docs: List[str]) -> List[ChatMessage]:
    """
    Inject RAG context into messages

    Args:
        messages: Original chat messages
        context_docs: Retrieved context documents

    Returns:
        Enhanced messages with RAG context
    """
    if not context_docs:
        return messages

    context_text = "\n\n---\n\n".join([f"Context {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])

    rag_system_msg = ChatMessage(
        role="system",
        content=f"You are a helpful assistant. Use the following context to answer the user's question accurately.\n\n{context_text}"
    )

    # Insert RAG context before user messages
    enhanced_messages = [rag_system_msg]
    for msg in messages:
        if msg.role != "system":  # Skip existing system messages to avoid duplication
            enhanced_messages.append(msg)

    return enhanced_messages


class PromptBuilder:
    """RAG prompt construction service"""

    def build_prompt(self, messages: List[ChatMessage], context_docs: List[str]) -> List[ChatMessage]:
        """Build RAG-enhanced prompt"""
        return build_rag_prompt(messages, context_docs)


# Global instance
prompt_builder = PromptBuilder()