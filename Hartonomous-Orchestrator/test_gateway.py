"""
Comprehensive test script for RAG gateway
"""
import httpx
import json

BASE_URL = "http://localhost:8700"

def test_health():
    print("=" * 60)
    print("Testing health endpoint...")
    print("=" * 60)
    response = httpx.get(f"{BASE_URL}/health", timeout=10.0)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_models():
    print("=" * 60)
    print("Testing models endpoint...")
    print("=" * 60)
    response = httpx.get(f"{BASE_URL}/v1/models", timeout=10.0)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_collection_stats():
    print("=" * 60)
    print("Testing collection stats...")
    print("=" * 60)
    response = httpx.get(f"{BASE_URL}/v1/collection/stats", timeout=10.0)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()

def test_ingestion():
    print("=" * 60)
    print("Testing document ingestion...")
    print("=" * 60)

    docs = [
        "Python is a high-level, interpreted programming language known for its readability and versatility.",
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Neural networks are computing systems inspired by biological neural networks in animal brains.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation."
    ]

    response = httpx.post(f"{BASE_URL}/v1/ingest", json={
        "documents": docs,
        "metadata": [{"source": "test", "topic": "AI"}] * len(docs),
        "chunk": True
    }, timeout=60.0)

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Documents ingested: {result.get('documents_ingested')}")
    print(f"Chunks created: {result.get('chunks_created')}")
    print(f"Collection size: {result.get('collection_size')}")
    print()

def test_search():
    print("=" * 60)
    print("Testing vector search...")
    print("=" * 60)

    response = httpx.post(f"{BASE_URL}/v1/search", json={
        "query": "What is machine learning?",
        "top_k": 3
    }, timeout=30.0)

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Results found: {result.get('total')}")
    for i, r in enumerate(result.get('results', [])[:3]):
        print(f"\n  Result {i+1}:")
        print(f"    Score: {r.get('score', 0):.4f}")
        print(f"    Text: {r.get('document', '')[:100]}...")
    print()

def test_embeddings():
    print("=" * 60)
    print("Testing embeddings endpoint...")
    print("=" * 60)
    response = httpx.post(f"{BASE_URL}/v1/embeddings", json={
        "model": "qwen3-embedding-4b",
        "input": "Hello, world!"
    }, timeout=30.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    if "data" in result and len(result["data"]) > 0:
        embedding = result["data"][0]["embedding"]
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    print()

def test_rerank():
    print("=" * 60)
    print("Testing rerank endpoint...")
    print("=" * 60)
    response = httpx.post(f"{BASE_URL}/v1/rerank", json={
        "model": "qwen3-reranker-4b",
        "query": "What is the fastest programming language?",
        "documents": [
            "Python is known for readability and ease of use",
            "Rust is a systems programming language focused on speed and safety",
            "JavaScript runs in web browsers and on servers"
        ],
        "top_n": 2
    }, timeout=30.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    if "results" in result:
        for r in result["results"]:
            print(f"  Score: {r['score']:.4f} - {r['document']}")
    print()

def test_chat_no_rag():
    print("=" * 60)
    print("Testing chat completions (RAG disabled)...")
    print("=" * 60)
    response = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "qwen3-coder-30b",
        "messages": [
            {"role": "user", "content": "Say 'hello' in one word"}
        ],
        "max_tokens": 10,
        "stream": False,
        "rag_enabled": False
    }, timeout=60.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    if "choices" in result:
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 0)}")
    print()

def test_chat_with_rag():
    print("=" * 60)
    print("Testing chat completions (RAG enabled)...")
    print("=" * 60)
    response = httpx.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": "qwen3-coder-30b",
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ],
        "max_tokens": 100,
        "stream": False,
        "rag_enabled": True,
        "rag_top_k": 5,
        "rag_rerank_top_n": 2
    }, timeout=60.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    if "choices" in result:
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 0)}")
    print()

def test_streaming():
    print("=" * 60)
    print("Testing streaming chat completions...")
    print("=" * 60)

    try:
        with httpx.stream("POST", f"{BASE_URL}/v1/chat/completions", json={
            "model": "qwen3-coder-30b",
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "max_tokens": 50,
            "stream": True,
            "rag_enabled": False
        }, timeout=None) as response:
            print(f"Status: {response.status_code}")
            print("Stream output: ", end="", flush=True)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        chunk_data = json.loads(line[6:])
                        content = chunk_data["choices"][0]["delta"].get("content", "")
                        print(content, end="", flush=True)
                    except:
                        pass
            print()
    except Exception as e:
        print(f"Streaming error: {e}")
    print()

def test_completions():
    print("=" * 60)
    print("Testing text completions endpoint...")
    print("=" * 60)
    response = httpx.post(f"{BASE_URL}/v1/completions", json={
        "model": "qwen3-coder-30b",
        "prompt": "Write a haiku about programming:",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
    }, timeout=60.0)
    print(f"Status: {response.status_code}")
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        print(f"Response: {result['choices'][0]['text']}")
        print(f"Tokens used: {result.get('usage', {}).get('total_tokens', 0)}")
    print()

def test_completions_streaming():
    print("=" * 60)
    print("Testing streaming text completions...")
    print("=" * 60)

    try:
        with httpx.stream("POST", f"{BASE_URL}/v1/completions", json={
            "model": "qwen3-coder-30b",
            "prompt": "Count from 1 to 3:",
            "max_tokens": 20,
            "stream": True
        }, timeout=None) as response:
            print(f"Status: {response.status_code}")
            print("Stream output: ", end="", flush=True)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    if line.strip() == "data: [DONE]":
                        break
                    try:
                        chunk_data = json.loads(line[6:])
                        content = chunk_data["choices"][0].get("text", "")
                        print(content, end="", flush=True)
                    except:
                        pass
            print()
    except Exception as e:
        print(f"Streaming error: {e}")
    print()

def test_assistants():
    print("=" * 60)
    print("Testing assistants endpoints...")
    print("=" * 60)

    # Test creating an assistant
    print("Creating assistant...")
    response = httpx.post(f"{BASE_URL}/v1/assistants", json={
        "model": "qwen3-coder-30b",
        "name": "Test Assistant",
        "description": "A test assistant",
        "instructions": "You are a helpful assistant.",
        "tools": [
            {"type": "code_interpreter"},
            {"type": "file_search"}
        ]
    }, timeout=10.0)
    print(f"Create status: {response.status_code}")
    if response.status_code == 200:
        assistant = response.json()
        assistant_id = assistant["id"]
        print(f"Assistant created: {assistant_id}")
        print(json.dumps(assistant, indent=2))

        # Test getting the assistant
        print("\nRetrieving assistant...")
        response = httpx.get(f"{BASE_URL}/v1/assistants/{assistant_id}", timeout=10.0)
        print(f"Get status: {response.status_code}")
        if response.status_code == 200:
            print("Assistant retrieved successfully")

        # Test listing assistants
        print("\nListing assistants...")
        response = httpx.get(f"{BASE_URL}/v1/assistants", timeout=10.0)
        print(f"List status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Assistants found: {len(result['data'])}")

        # Test updating assistant
        print("\nUpdating assistant...")
        response = httpx.post(f"{BASE_URL}/v1/assistants/{assistant_id}", json={
            "name": "Updated Test Assistant"
        }, timeout=10.0)
        print(f"Update status: {response.status_code}")

        # Test deleting assistant
        print("\nDeleting assistant...")
        response = httpx.delete(f"{BASE_URL}/v1/assistants/{assistant_id}", timeout=10.0)
        print(f"Delete status: {response.status_code}")

    else:
        print(f"Failed to create assistant: {response.text}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("RAG Gateway Test Suite")
    print("=" * 60)
    print()

    try:
        # Basic tests
        test_health()
        test_models()
        test_collection_stats()

        # RAG functionality tests
        print("\n" + "=" * 60)
        print("RAG FUNCTIONALITY TESTS")
        print("=" * 60 + "\n")

        test_ingestion()
        test_search()
        test_rerank()

        # Chat tests
        print("\n" + "=" * 60)
        print("CHAT COMPLETION TESTS")
        print("=" * 60 + "\n")

        test_embeddings()
        test_chat_no_rag()
        test_chat_with_rag()
        test_streaming()

        # Completion tests
        print("\n" + "=" * 60)
        print("TEXT COMPLETION TESTS")
        print("=" * 60 + "\n")

        test_completions()
        test_completions_streaming()

        # Assistant tests
        print("\n" + "=" * 60)
        print("ASSISTANT TESTS")
        print("=" * 60 + "\n")

        test_assistants()

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Gateway is running (docker-compose up)")
        print("2. Llama.cpp servers are running on ports 8710, 8711, 8712")
        print("3. Qdrant is running on port 6333")
