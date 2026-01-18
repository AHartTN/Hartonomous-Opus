#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <string>
#include <vector>
#include <boost/json.hpp>
#include "orchestrator.hpp"

// Mock classes for all dependencies
class MockHttpClient : public HttpClient {
public:
    MOCK_METHOD(bool, send_request,
        (const std::string&, const std::string&, const std::string&, std::string&,
         const std::string&, uint16_t), (override));
};

class MockEmbeddingClient : public EmbeddingClient {
public:
    MOCK_METHOD(std::vector<float>, generate_embeddings,
        (const std::string&, const std::string&), (override));
    MOCK_METHOD(std::vector<std::vector<float>>, generate_embeddings_batch,
        (const std::vector<std::string>&, const std::string&), (override));
    MOCK_METHOD(bool, is_healthy, (), (const, override));
};

class MockRerankingClient : public RerankingClient {
public:
    MOCK_METHOD(std::vector<std::pair<int, float>>, rerank_documents,
        (const std::string&, const std::vector<std::string>&, const std::string&), (override));
    MOCK_METHOD(bool, is_healthy, (), (const, override));
};

class MockGenerativeClient : public GenerativeClient {
public:
    MOCK_METHOD(ChatCompletion, create_chat_completion,
        (const std::vector<ChatMessage>&, const std::string&, double, int), (override));
    MOCK_METHOD(bool, is_healthy, (), (const, override));
};

class MockQdrantClient : public QdrantClient {
public:
    MOCK_METHOD(bool, upsert_vectors,
        (const std::vector<VectorData>&, const std::string&), (override));
    MOCK_METHOD(std::vector<SearchResult>, search_vectors,
        (const std::vector<float>&, const std::string&, size_t), (override));
    MOCK_METHOD(bool, create_collection,
        (const std::string&, size_t), (override));
    MOCK_METHOD(bool, delete_collection,
        (const std::string&), (override));
    MOCK_METHOD(bool, collection_exists,
        (const std::string&), (override));
    MOCK_METHOD(bool, is_healthy, (), (const, override));
};

class OrchestratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_http_client_ = std::make_shared<MockHttpClient>();
        mock_embedding_client_ = std::make_shared<MockEmbeddingClient>();
        mock_reranking_client_ = std::make_shared<MockRerankingClient>();
        mock_generative_client_ = std::make_shared<MockGenerativeClient>();
        mock_qdrant_client_ = std::make_shared<MockQdrantClient>();

        orchestrator_ = std::make_shared<Orchestrator>(
            mock_http_client_,
            mock_embedding_client_,
            mock_reranking_client_,
            mock_generative_client_,
            mock_qdrant_client_
        );
    }

    std::shared_ptr<MockHttpClient> mock_http_client_;
    std::shared_ptr<MockEmbeddingClient> mock_embedding_client_;
    std::shared_ptr<MockRerankingClient> mock_reranking_client_;
    std::shared_ptr<MockGenerativeClient> mock_generative_client_;
    std::shared_ptr<MockQdrantClient> mock_qdrant_client_;
    std::shared_ptr<Orchestrator> orchestrator_;
};

TEST_F(OrchestratorTest, IngestDocument_Success) {
    // Setup mock expectations
    std::vector<float> mock_embedding = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    EXPECT_CALL(*mock_embedding_client_, generate_embeddings("test content", testing::_))
        .WillOnce(testing::Return(mock_embedding));

    EXPECT_CALL(*mock_qdrant_client_, upsert_vectors(testing::_, "default"))
        .WillOnce(testing::Return(true));

    // Test
    boost::json::object metadata = {{"key", "value"}};
    bool result = orchestrator_->ingest_document("doc1", "test content", metadata);

    EXPECT_TRUE(result);
}

TEST_F(OrchestratorTest, IngestDocument_EmbeddingFailure) {
    EXPECT_CALL(*mock_embedding_client_, generate_embeddings("test content", testing::_))
        .WillOnce(testing::Throw(std::runtime_error("Embedding failed")));

    boost::json::object metadata = {{"key", "value"}};
    bool result = orchestrator_->ingest_document("doc1", "test content", metadata);

    EXPECT_FALSE(result);
}

TEST_F(OrchestratorTest, IngestDocumentsBatch_Success) {
    // Setup mock expectations
    std::vector<std::string> doc_ids = {"doc1", "doc2"};
    std::vector<std::string> contents = {"content1", "content2"};
    std::vector<boost::json::object> metadata = {{{"key", "value1"}}, {{"key", "value2"}}};

    std::vector<std::vector<float>> mock_embeddings = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };

    EXPECT_CALL(*mock_embedding_client_, generate_embeddings_batch(contents, testing::_))
        .WillOnce(testing::Return(mock_embeddings));

    EXPECT_CALL(*mock_qdrant_client_, upsert_vectors(testing::_, "default"))
        .WillOnce(testing::Return(true));

    // Test
    bool result = orchestrator_->ingest_documents(doc_ids, contents, metadata);

    EXPECT_TRUE(result);
}

TEST_F(OrchestratorTest, Query_Success) {
    // Setup mock expectations
    std::string query = "test query";
    std::vector<float> query_embedding = {0.1f, 0.2f, 0.3f};
    std::vector<SearchResult> search_results = {
        {"doc1", 0.9f, boost::json::object{}},
        {"doc2", 0.8f, boost::json::object{}}
    };

    EXPECT_CALL(*mock_embedding_client_, generate_embeddings(query, testing::_))
        .WillOnce(testing::Return(query_embedding));

    EXPECT_CALL(*mock_qdrant_client_, search_vectors(query_embedding, "default", 10))
        .WillOnce(testing::Return(search_results));

    // Test
    RAGResult result = orchestrator_->query(query, "default");

    EXPECT_EQ(result.query_id, query);
    EXPECT_FALSE(result.answer.empty());
    EXPECT_EQ(result.relevant_document_ids.size(), 2);
    EXPECT_EQ(result.relevant_document_ids[0], "doc1");
    EXPECT_EQ(result.relevant_document_ids[1], "doc2");
}

TEST_F(OrchestratorTest, Query_EmbeddingFailure) {
    EXPECT_CALL(*mock_embedding_client_, generate_embeddings("test query", testing::_))
        .WillOnce(testing::Throw(std::runtime_error("Embedding failed")));

    EXPECT_THROW(
        orchestrator_->query("test query", "default"),
        std::runtime_error
    );
}

TEST_F(OrchestratorTest, BatchQuery_Success) {
    std::vector<std::string> queries = {"query1", "query2"};

    // Setup expectations for each query
    EXPECT_CALL(*mock_embedding_client_, generate_embeddings(testing::_, testing::_))
        .Times(2)
        .WillRepeatedly(testing::Return(std::vector<float>{0.1f, 0.2f, 0.3f}));

    EXPECT_CALL(*mock_qdrant_client_, search_vectors(testing::_, "default", 10))
        .Times(2)
        .WillRepeatedly(testing::Return(std::vector<SearchResult>{
            {"doc1", 0.9f, boost::json::object{}}
        }));

    // Test
    std::vector<RAGResult> results = orchestrator_->batch_query(queries, "default");

    EXPECT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].query_id, "query1");
    EXPECT_EQ(results[1].query_id, "query2");
}

TEST_F(OrchestratorTest, CreateCollection_Success) {
    EXPECT_CALL(*mock_qdrant_client_, create_collection("test_collection", 1536))
        .WillOnce(testing::Return(true));

    bool result = orchestrator_->create_collection("test_collection", 1536);

    EXPECT_TRUE(result);
}

TEST_F(OrchestratorTest, DeleteCollection_Success) {
    EXPECT_CALL(*mock_qdrant_client_, delete_collection("test_collection"))
        .WillOnce(testing::Return(true));

    bool result = orchestrator_->delete_collection("test_collection");

    EXPECT_TRUE(result);
}

TEST_F(OrchestratorTest, CollectionExists_Success) {
    EXPECT_CALL(*mock_qdrant_client_, collection_exists("test_collection"))
        .WillOnce(testing::Return(true));

    bool result = orchestrator_->collection_exists("test_collection");

    EXPECT_TRUE(result);
}

TEST_F(OrchestratorTest, IsHealthy_AllHealthy) {
    EXPECT_CALL(*mock_embedding_client_, is_healthy())
        .WillOnce(testing::Return(true));
    EXPECT_CALL(*mock_reranking_client_, is_healthy())
        .WillOnce(testing::Return(true));
    EXPECT_CALL(*mock_generative_client_, is_healthy())
        .WillOnce(testing::Return(true));
    EXPECT_CALL(*mock_qdrant_client_, is_healthy())
        .WillOnce(testing::Return(true));

    bool result = orchestrator_->is_healthy();

    EXPECT_TRUE(result);
}

TEST_F(OrchestratorTest, IsHealthy_OneUnhealthy) {
    EXPECT_CALL(*mock_embedding_client_, is_healthy())
        .WillOnce(testing::Return(true));
    EXPECT_CALL(*mock_reranking_client_, is_healthy())
        .WillOnce(testing::Return(false)); // This one is unhealthy
    EXPECT_CALL(*mock_generative_client_, is_healthy())
        .WillOnce(testing::Return(true));
    EXPECT_CALL(*mock_qdrant_client_, is_healthy())
        .WillOnce(testing::Return(true));

    bool result = orchestrator_->is_healthy();

    EXPECT_FALSE(result);
}

TEST_F(OrchestratorTest, GenerateAnswerWithContext) {
    // Test the answer generation method indirectly through query
    std::vector<float> query_embedding = {0.1f, 0.2f, 0.3f};
    std::vector<SearchResult> search_results = {
        {"doc1", 0.9f, boost::json::object{}}
    };

    EXPECT_CALL(*mock_embedding_client_, generate_embeddings("test query", testing::_))
        .WillOnce(testing::Return(query_embedding));

    EXPECT_CALL(*mock_qdrant_client_, search_vectors(query_embedding, "default", 10))
        .WillOnce(testing::Return(search_results));

    RAGResult result = orchestrator_->query("test query", "default");

    // The answer should contain some generated response
    EXPECT_FALSE(result.answer.empty());
    EXPECT_NE(result.answer.find("response"), std::string::npos);
}

TEST_F(OrchestratorTest, ExtractDocumentContent) {
    // Test document content extraction indirectly through query flow
    std::vector<float> query_embedding = {0.1f, 0.2f, 0.3f};
    std::vector<SearchResult> search_results = {
        {"doc1", 0.9f, boost::json::object{}},
        {"doc2", 0.8f, boost::json::object{}}
    };

    EXPECT_CALL(*mock_embedding_client_, generate_embeddings("test query", testing::_))
        .WillOnce(testing::Return(query_embedding));

    EXPECT_CALL(*mock_qdrant_client_, search_vectors(query_embedding, "default", 10))
        .WillOnce(testing::Return(search_results));

    RAGResult result = orchestrator_->query("test query", "default");

    // Should have retrieved 2 documents
    EXPECT_EQ(result.relevant_document_ids.size(), 2);
}