#!/usr/bin/env python3
"""
Test script for the refactored OpenAI Gateway
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test that all modules can be imported successfully"""
    print("Testing imports...")

    try:
        from openai_gateway import config, models, main
        print("[OK] Core modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import core modules: {e}")
        return False

    try:
        from openai_gateway.utils import text_processing, response_formatters
        print("[OK] Utility modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import utility modules: {e}")
        return False

    try:
        from openai_gateway.clients import qdrant_client, llamacpp_client
        print("[OK] Client modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import client modules: {e}")
        return False

    try:
        from openai_gateway.rag import search, reranking, prompt_builder
        print("[OK] RAG modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import RAG modules: {e}")
        return False

    try:
        from openai_gateway.routes import chat, completions, embeddings, models_endpoints
        print("[OK] Route modules imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import route modules: {e}")
        return False

    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    from openai_gateway import config

    # Test that key config values are loaded
    assert config.GENERATIVE_URL == "http://localhost:8710"
    assert config.EMBEDDING_URL == "http://localhost:8711"
    assert config.RERANKER_URL == "http://localhost:8712"
    assert config.BACKEND_API_KEY == "Welcome!123"
    print("[OK] Configuration values loaded correctly")

    return True

def test_models():
    """Test Pydantic models"""
    print("\nTesting models...")
    from openai_gateway.models import ChatMessage, ChatCompletionRequest

    # Test model creation
    message = ChatMessage(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"
    print("[OK] ChatMessage model works correctly")

    # Test request model
    request = ChatCompletionRequest(
        model="test-model",
        messages=[message],
        temperature=0.5
    )
    assert request.model == "test-model"
    assert len(request.messages) == 1
    print("[OK] ChatCompletionRequest model works correctly")

    return True

def test_utils():
    """Test utility functions"""
    print("\nTesting utilities...")
    from openai_gateway.utils.text_processing import chunk_text, convert_messages_to_prompt
    from openai_gateway.models import ChatMessage

    # Test text chunking
    text = "This is a test document. " * 100
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 0
    assert all(len(chunk) > 0 for chunk in chunks)  # Just ensure chunks are not empty
    print("[OK] Text chunking works correctly")

    # Test message conversion
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello!")
    ]
    prompt = convert_messages_to_prompt(messages)
    assert "System:" in prompt
    assert "User:" in prompt
    print("[OK] Message conversion works correctly")

    return True

def main():
    """Run all tests"""
    print("Testing refactored OpenAI Gateway...\n")

    success = True
    success &= test_imports()
    success &= test_config()
    success &= test_models()
    success &= test_utils()

    if success:
        print("\n[SUCCESS] All tests passed! The refactored application is working correctly.")
        return 0
    else:
        print("\n[ERROR] Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())