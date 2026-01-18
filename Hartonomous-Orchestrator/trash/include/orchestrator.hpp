#ifndef ORCHESTRATOR_HPP
#define ORCHESTRATOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <boost/json.hpp>
#include "embedding_client.hpp"
#include "reranking_client.hpp"
#include "generative_client.hpp"
#include "qdrant_client.hpp"
#include "http_client.hpp"
#include "logging.hpp"
#include "metrics.hpp"

// Structure for RAG query result
struct RAGResult {
    std::string query_id;
    std::string answer;
    std::vector<std::string> relevant_document_ids;
    std::vector<float> relevance_scores;
    boost::json::object metadata;
};

// Structure for document processing
struct Document {
    std::string id;
    std::string content;
    std::vector<float> embedding;
    boost::json::object metadata;
};

class Orchestrator {
public:
    Orchestrator(
        std::shared_ptr<HttpClient> http_client,
        std::shared_ptr<EmbeddingClient> embedding_client,
        std::shared_ptr<RerankingClient> reranking_client,
        std::shared_ptr<GenerativeClient> generative_client,
        std::shared_ptr<QdrantClient> qdrant_client);
    
    virtual ~Orchestrator() = default;
    
    // RAG Ingest Pipeline
    bool ingest_document(const std::string& document_id, const std::string& content, const boost::json::object& metadata = {});
    bool ingest_documents(const std::vector<std::string>& document_ids, const std::vector<std::string>& contents, const std::vector<boost::json::object>& metadata = {});
    
    // RAG Query Pipeline
    RAGResult query(const std::string& query, const std::string& collection_name = "default");
    std::vector<RAGResult> batch_query(const std::vector<std::string>& queries, const std::string& collection_name = "default");
    
    // RAG Operations
    bool create_collection(const std::string& collection_name, size_t vector_size = 1536);
    bool delete_collection(const std::string& collection_name);
    bool collection_exists(const std::string& collection_name);
    
    // Health check
    bool is_healthy() const;
    
private:
    std::shared_ptr<HttpClient> http_client_;
    std::shared_ptr<EmbeddingClient> embedding_client_;
    std::shared_ptr<RerankingClient> reranking_client_;
    std::shared_ptr<GenerativeClient> generative_client_;
    std::shared_ptr<QdrantClient> qdrant_client_;
    
    std::string default_collection_;
    bool is_healthy_;
    
    // Internal helper methods
    std::vector<float> generate_embedding(const std::string& text);
    std::vector<std::vector<float>> generate_embeddings_batch(const std::vector<std::string>& texts);
    
    std::vector<std::string> search_relevant_documents(const std::vector<float>& query_embedding, 
                                                      const std::string& collection_name, 
                                                      size_t limit = 10);
    
    std::vector<std::string> rerank_documents(const std::string& query, 
                                            const std::vector<std::string>& document_ids);
    
    std::string generate_answer(const std::string& query, 
                                const std::vector<std::string>& relevant_documents);
    
    std::string generate_answer_with_context(const std::string& query, 
                                             const std::vector<std::string>& relevant_documents);
    
    std::vector<std::string> extract_document_content(const std::vector<std::string>& document_ids, 
                                                    const std::string& collection_name);
    
    std::vector<std::string> process_query(const std::string& query);
    
    // Metrics tracking
    void record_query_metrics(const std::string& query_id, const std::vector<std::string>& document_ids, 
                              double latency_seconds, const std::string& model_used);
};

#endif // ORCHESTRATOR_HPP