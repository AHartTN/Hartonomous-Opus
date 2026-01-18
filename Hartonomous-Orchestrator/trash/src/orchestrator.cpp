#include "orchestrator.hpp"
#include "logging.hpp"
#include "metrics.hpp"
#include <boost/json.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>

Orchestrator::Orchestrator(
    std::shared_ptr<HttpClient> http_client,
    std::shared_ptr<EmbeddingClient> embedding_client,
    std::shared_ptr<RerankingClient> reranking_client,
    std::shared_ptr<GenerativeClient> generative_client,
    std::shared_ptr<QdrantClient> qdrant_client)
    : http_client_(http_client),
      embedding_client_(embedding_client),
      reranking_client_(reranking_client),
      generative_client_(generative_client),
      qdrant_client_(qdrant_client),
      default_collection_("default"),
      is_healthy_(true) {
    // Initialize orchestrator with all service clients
}

bool Orchestrator::ingest_document(const std::string& document_id, const std::string& content, const boost::json::object& metadata) {
    try {
        LOG_DEBUG("Ingesting document with ID: " + document_id);
        
        // Generate embedding for the content
        auto embedding = generate_embedding(content);
        
        // Create VectorData structure
        VectorData vector_data;
        vector_data.id = document_id;
        vector_data.vector = embedding;
        vector_data.payload = metadata;
        
        // Store in Qdrant
        std::vector<VectorData> vectors = {vector_data};
        bool success = qdrant_client_->upsert_vectors(vectors, default_collection_);
        
        LOG_DEBUG("Document ingested successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to ingest document: " + std::string(e.what()));
        return false;
    }
}

bool Orchestrator::ingest_documents(const std::vector<std::string>& document_ids, const std::vector<std::string>& contents, const std::vector<boost::json::object>& metadata) {
    try {
        LOG_DEBUG("Ingesting " + std::to_string(document_ids.size()) + " documents");
        
        // Generate embeddings for all documents
        auto embeddings = generate_embeddings_batch(contents);
        
        // Create VectorData structures
        std::vector<VectorData> vectors;
        vectors.reserve(document_ids.size());
        
        for (size_t i = 0; i < document_ids.size(); ++i) {
            VectorData vector_data;
            vector_data.id = document_ids[i];
            vector_data.vector = embeddings[i];
            vector_data.payload = metadata.empty() ? boost::json::object{} : metadata[i];
            vectors.push_back(vector_data);
        }
        
        // Store in Qdrant
        bool success = qdrant_client_->upsert_vectors(vectors, default_collection_);
        
        LOG_DEBUG("Batch documents ingested successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to ingest batch documents: " + std::string(e.what()));
        return false;
    }
}

RAGResult Orchestrator::query(const std::string& query, const std::string& collection_name) {
    try {
        LOG_DEBUG("Processing query: " + query.substr(0, 50) + "...");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Generate query embedding
        auto query_embedding = generate_embedding(query);
        
        // Search relevant documents
        auto document_ids = search_relevant_documents(query_embedding, collection_name, 10);
        
        // Rerank documents
        auto reranked_ids = rerank_documents(query, document_ids);
        
        // Extract document content
        auto relevant_docs = extract_document_content(reranked_ids, collection_name);
        
        // Generate answer
        auto answer = generate_answer_with_context(query, relevant_docs);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // Record metrics
        record_query_metrics(query, document_ids, latency, "gpt-3.5-turbo");
        
        // Create result
        RAGResult result;
        result.query_id = query;
        result.answer = answer;
        result.relevant_document_ids = reranked_ids;
        result.metadata = boost::json::object{};
        
        LOG_DEBUG("Query completed successfully");
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to process query: " + std::string(e.what()));
        throw;
    }
}

std::vector<RAGResult> Orchestrator::batch_query(const std::vector<std::string>& queries, const std::string& collection_name) {
    try {
        LOG_DEBUG("Processing batch query with " + std::to_string(queries.size()) + " queries");
        
        std::vector<RAGResult> results;
        results.reserve(queries.size());
        
        for (const auto& query : queries) {
            auto result = query(query, collection_name);
            results.push_back(result);
        }
        
        LOG_DEBUG("Batch query completed successfully");
        return results;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to process batch query: " + std::string(e.what()));
        throw;
    }
}

bool Orchestrator::create_collection(const std::string& collection_name, size_t vector_size) {
    try {
        LOG_DEBUG("Creating collection: " + collection_name);
        
        bool success = qdrant_client_->create_collection(collection_name, vector_size);
        
        LOG_DEBUG("Collection created successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create collection: " + std::string(e.what()));
        return false;
    }
}

bool Orchestrator::delete_collection(const std::string& collection_name) {
    try {
        LOG_DEBUG("Deleting collection: " + collection_name);
        
        bool success = qdrant_client_->delete_collection(collection_name);
        
        LOG_DEBUG("Collection deleted successfully");
        return success;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to delete collection: " + std::string(e.what()));
        return false;
    }
}

bool Orchestrator::collection_exists(const std::string& collection_name) {
    try {
        LOG_DEBUG("Checking if collection exists: " + collection_name);
        
        bool exists = qdrant_client_->collection_exists(collection_name);
        
        LOG_DEBUG("Collection exists check completed");
        return exists;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to check collection exists: " + std::string(e.what()));
        return false;
    }
}

bool Orchestrator::is_healthy() const {
    return is_healthy_ && 
           embedding_client_->is_healthy() &&
           reranking_client_->is_healthy() &&
           generative_client_->is_healthy() &&
           qdrant_client_->is_healthy();
}

std::vector<float> Orchestrator::generate_embedding(const std::string& text) {
    return embedding_client_->generate_embeddings(text);
}

std::vector<std::vector<float>> Orchestrator::generate_embeddings_batch(const std::vector<std::string>& texts) {
    return embedding_client_->generate_embeddings_batch(texts);
}

std::vector<std::string> Orchestrator::search_relevant_documents(const std::vector<float>& query_embedding, 
                                                                  const std::string& collection_name, 
                                                                  size_t limit) {
    auto results = qdrant_client_->search_vectors(query_embedding, collection_name, limit);
    std::vector<std::string> document_ids;
    document_ids.reserve(results.size());
    
    for (const auto& result : results) {
        document_ids.push_back(result.id);
    }
    
    return document_ids;
}

std::vector<std::string> Orchestrator::rerank_documents(const std::string& query, 
                                                       const std::vector<std::string>& document_ids) {
    // In a real implementation, this would use the reranking service
    // For now, we'll return documents in the same order (no actual reranking)
    return document_ids;
}

std::string Orchestrator::generate_answer(const std::string& query, 
                                          const std::vector<std::string>& relevant_documents) {
    // Create a prompt for the generative service
    std::string prompt = "Answer this query: \"" + query + "\"\n\n";
    prompt += "Context:\n";
    
    for (const auto& doc : relevant_documents) {
        prompt += "- " + doc + "\n";
    }
    
    // Generate response using generative client
    // For simplicity, we'll return a placeholder
    return "Generated answer for query: " + query;
}

std::string Orchestrator::generate_answer_with_context(const std::string& query, 
                                                     const std::vector<std::string>& relevant_documents) {
    // Build context from relevant documents
    std::string context = "Context:\n";
    for (const auto& doc : relevant_documents) {
        context += "- " + doc + "\n";
    }
    
    // Create prompt for generative service
    std::string prompt = "Based on the following context, please answer the question:\n\n";
    prompt += context + "\n";
    prompt += "Question: " + query + "\n";
    prompt += "Answer:";
    
    // In a real implementation, this would call the generative client
    // For now, return a placeholder response
    return "This is a response generated by the RAG system based on the retrieved context.";
}

std::vector<std::string> Orchestrator::extract_document_content(const std::vector<std::string>& document_ids, 
                                                                 const std::string& collection_name) {
    // In a real implementation, this would retrieve the actual document contents
    // For now, return placeholder content
    std::vector<std::string> content;
    content.reserve(document_ids.size());
    
    for (const auto& id : document_ids) {
        content.push_back("Content of document " + id);
    }
    
    return content;
}

std::vector<std::string> Orchestrator::process_query(const std::string& query) {
    // Preprocess query if needed
    return {query}; // Return as is for now
}

void Orchestrator::record_query_metrics(const std::string& query_id, const std::vector<std::string>& document_ids, 
                                      double latency_seconds, const std::string& model_used) {
    // Record query metrics for monitoring and analytics
    
    // Record latency
    Metrics::record_query_latency(latency_seconds);
    
    // Record number of documents retrieved
    Metrics::record_documents_retrieved(document_ids.size());
    
    // Record model used
    Metrics::record_model_used(model_used);
    
    // Record query count
    Metrics::increment_query_count();
}