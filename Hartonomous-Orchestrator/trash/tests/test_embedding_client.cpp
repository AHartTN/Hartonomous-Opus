#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <string>
#include <vector>
#include "embedding_client.hpp"

// Mock HTTP client for testing
class MockHttpClient : public HttpClient {
public:
    MOCK_METHOD(bool, send_request,
        (const std::string&, const std::string&, const std::string&, std::string&,
         const std::string&, uint16_t), (override));
    MOCK_METHOD(bool, send_request,
        (const std::string&, const std::string&, const boost::json::object&, std::string&,
         const std::string&, uint16_t), (override));
    MOCK_METHOD(void, send_async_request,
        (const std::string&, const std::string&, const std::string&,
         const std::string&, uint16_t), (override));
};

class EmbeddingClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_http_client_ = std::make_shared<MockHttpClient>();
        embedding_client_ = std::make_shared<EmbeddingClient>(mock_http_client_);
    }

    void TearDown() override {
        // Clean up
    }

    std::shared_ptr<MockHttpClient> mock_http_client_;
    std::shared_ptr<EmbeddingClient> embedding_client_;
};

TEST_F(EmbeddingClientTest, GenerateEmbeddings_Success) {
    // Sample response from embedding service
    std::string mock_response = R"({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "index": 0
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5
        }
    })";

    EXPECT_CALL(*mock_http_client_, send_request(
        testing::StrEq("POST"),
        testing::StrEq("/embeddings"),
        testing::_,
        testing::_,
        testing::StrEq("localhost"),
        testing::Eq(8711)))
        .WillOnce(testing::DoAll(
            testing::SetArgReferee<3>(mock_response),
            testing::Return(true)
        ));

    std::vector<float> result = embedding_client_->generate_embeddings("test text", "text-embedding-3-small");

    EXPECT_EQ(result.size(), 5);
    EXPECT_FLOAT_EQ(result[0], 0.1f);
    EXPECT_FLOAT_EQ(result[1], 0.2f);
    EXPECT_FLOAT_EQ(result[2], 0.3f);
    EXPECT_FLOAT_EQ(result[3], 0.4f);
    EXPECT_FLOAT_EQ(result[4], 0.5f);
}

TEST_F(EmbeddingClientTest, GenerateEmbeddings_Failure) {
    EXPECT_CALL(*mock_http_client_, send_request(
        testing::StrEq("POST"),
        testing::StrEq("/embeddings"),
        testing::_,
        testing::_,
        testing::StrEq("localhost"),
        testing::Eq(8711)))
        .WillOnce(testing::Return(false));

    EXPECT_THROW(
        embedding_client_->generate_embeddings("test text", "text-embedding-3-small"),
        std::runtime_error
    );
}

TEST_F(EmbeddingClientTest, GenerateEmbeddingsBatch_Success) {
    // Sample batch response
    std::string mock_response = R"({
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [0.1, 0.2, 0.3],
                "index": 0
            },
            {
                "object": "embedding",
                "embedding": [0.4, 0.5, 0.6],
                "index": 1
            }
        ],
        "model": "text-embedding-3-small",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    })";

    EXPECT_CALL(*mock_http_client_, send_request(
        testing::StrEq("POST"),
        testing::StrEq("/embeddings"),
        testing::_,
        testing::_,
        testing::StrEq("localhost"),
        testing::Eq(8711)))
        .WillOnce(testing::DoAll(
            testing::SetArgReferee<3>(mock_response),
            testing::Return(true)
        ));

    std::vector<std::string> texts = {"first text", "second text"};
    std::vector<std::vector<float>> result = embedding_client_->generate_embeddings_batch(texts, "text-embedding-3-small");

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].size(), 3);
    EXPECT_EQ(result[1].size(), 3);

    EXPECT_FLOAT_EQ(result[0][0], 0.1f);
    EXPECT_FLOAT_EQ(result[0][1], 0.2f);
    EXPECT_FLOAT_EQ(result[0][2], 0.3f);

    EXPECT_FLOAT_EQ(result[1][0], 0.4f);
    EXPECT_FLOAT_EQ(result[1][1], 0.5f);
    EXPECT_FLOAT_EQ(result[1][2], 0.6f);
}

TEST_F(EmbeddingClientTest, GenerateEmbeddingsBatch_Failure) {
    EXPECT_CALL(*mock_http_client_, send_request(
        testing::StrEq("POST"),
        testing::StrEq("/embeddings"),
        testing::_,
        testing::_,
        testing::StrEq("localhost"),
        testing::Eq(8711)))
        .WillOnce(testing::Return(false));

    std::vector<std::string> texts = {"first text", "second text"};

    EXPECT_THROW(
        embedding_client_->generate_embeddings_batch(texts, "text-embedding-3-small"),
        std::runtime_error
    );
}

TEST_F(EmbeddingClientTest, HealthCheck) {
    // Initially should be healthy
    EXPECT_TRUE(embedding_client_->is_healthy());
}

TEST_F(EmbeddingClientTest, BuildEmbeddingRequest) {
    // Test that we can create an embedding client and call private method via reflection/testing
    // This tests the request building logic indirectly through the public API

    std::string mock_response = R"({
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": [0.1, 0.2, 0.3],
            "index": 0
        }],
        "model": "test-model"
    })";

    EXPECT_CALL(*mock_http_client_, send_request)
        .WillOnce(testing::DoAll(
            testing::SetArgReferee<3>(mock_response),
            testing::Return(true)
        ));

    // The request building is tested indirectly through successful API calls
    std::vector<float> result = embedding_client_->generate_embeddings("test", "test-model");
    EXPECT_EQ(result.size(), 3);
}

TEST_F(EmbeddingClientTest, ParseEmbeddingResponse_EmptyData) {
    std::string empty_response = R"({
        "object": "list",
        "data": [],
        "model": "test-model"
    })";

    EXPECT_CALL(*mock_http_client_, send_request)
        .WillOnce(testing::DoAll(
            testing::SetArgReferee<3>(empty_response),
            testing::Return(true)
        ));

    EXPECT_THROW(
        embedding_client_->generate_embeddings("test", "test-model"),
        std::runtime_error
    );
}

TEST_F(EmbeddingClientTest, ParseEmbeddingResponse_InvalidJson) {
    std::string invalid_response = "invalid json";

    EXPECT_CALL(*mock_http_client_, send_request)
        .WillOnce(testing::DoAll(
            testing::SetArgReferee<3>(invalid_response),
            testing::Return(true)
        ));

    EXPECT_THROW(
        embedding_client_->generate_embeddings("test", "test-model"),
        std::runtime_error
    );
}