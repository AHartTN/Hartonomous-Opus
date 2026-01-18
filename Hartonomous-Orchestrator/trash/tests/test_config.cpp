#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include "config.hpp"

class ConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary config file for testing
        temp_config_path_ = "test_config.yaml";
        createTestConfigFile();
    }

    void TearDown() override {
        // Clean up temporary config file
        if (std::filesystem::exists(temp_config_path_)) {
            std::filesystem::remove(temp_config_path_);
        }
    }

    void createTestConfigFile() {
        std::ofstream config_file(temp_config_path_);
        config_file << R"(
server:
  host: "127.0.0.1"
  port: 9090
  use_ssl: true
  ssl_cert_file: "test_cert.pem"
  ssl_key_file: "test_key.pem"

services:
  embedding:
    endpoint: "http://localhost:8711"
    api_key: "test-embedding-key"
    model: "text-embedding-3-small"
    timeout: 15
    max_retries: 2
    retry_backoff: 1.0

  reranking:
    endpoint: "http://localhost:8712"
    api_key: "test-reranking-key"
    model: "rerank-v1"
    timeout: 20
    max_retries: 2
    retry_backoff: 1.0

  generative:
    endpoint: "http://localhost:8710"
    api_key: "test-generative-key"
    model: "gpt-3.5-turbo"
    timeout: 45
    max_retries: 3
    retry_backoff: 1.5

  vector_db:
    endpoint: "http://localhost:6333"
    api_key: "test-qdrant-key"
    timeout: 20
    max_retries: 2
    retry_backoff: 1.0

logging:
  level: "debug"
  file: "test.log"
  max_size: 5242880
  max_files: 3
  format: "text"
  console_output: true

metrics:
  enabled: true
  port: 9091
  export_format: "prometheus"
  collection_interval: 15
)";
        config_file.close();
    }

    std::string temp_config_path_;
};

TEST_F(ConfigTest, LoadValidConfig) {
    Config config = load_config(temp_config_path_);

    // Test server config
    EXPECT_EQ(config.server.host, "127.0.0.1");
    EXPECT_EQ(config.server.port, 9090);
    EXPECT_TRUE(config.server.use_ssl);
    EXPECT_EQ(config.server.ssl_cert_file, "test_cert.pem");
    EXPECT_EQ(config.server.ssl_key_file, "test_key.pem");

    // Test service configs
    EXPECT_EQ(config.embedding_service.host, "localhost");
    EXPECT_EQ(config.embedding_service.port, 8711);
    EXPECT_EQ(config.embedding_service.api_key, "test-embedding-key");

    EXPECT_EQ(config.reranking_service.host, "localhost");
    EXPECT_EQ(config.reranking_service.port, 8712);
    EXPECT_EQ(config.reranking_service.api_key, "test-reranking-key");

    EXPECT_EQ(config.generative_service.host, "localhost");
    EXPECT_EQ(config.generative_service.port, 8710);
    EXPECT_EQ(config.generative_service.api_key, "test-generative-key");

    EXPECT_EQ(config.qdrant_service.host, "localhost");
    EXPECT_EQ(config.qdrant_service.port, 6333);
    EXPECT_EQ(config.qdrant_service.api_key, "test-qdrant-key");

    // Test logging config
    EXPECT_EQ(config.log_level, "debug");
    EXPECT_EQ(config.log_file, "test.log");

    // Test metrics config (if implemented)
    // EXPECT_TRUE(config.metrics.enabled);
    // EXPECT_EQ(config.metrics.port, 9091);
}

TEST_F(ConfigTest, LoadNonExistentConfig) {
    Config config = load_config("nonexistent.yaml");

    // Should load defaults
    EXPECT_EQ(config.server.host, "0.0.0.0");
    EXPECT_EQ(config.server.port, 8080);
    EXPECT_FALSE(config.server.use_ssl);
}

TEST_F(ConfigTest, LoadEmptyConfigPath) {
    Config config = load_config("");

    // Should load defaults
    EXPECT_EQ(config.server.host, "0.0.0.0");
    EXPECT_EQ(config.server.port, 8080);
}

TEST_F(ConfigTest, ConfigFilePersistence) {
    // Test that we can create and load a config file
    std::string test_path = "temp_test_config.yaml";

    // Create a simple config
    std::ofstream config_file(test_path);
    config_file << "server:\n  port: 9999\n";
    config_file.close();

    Config config = load_config(test_path);
    EXPECT_EQ(config.server.port, 9999);

    // Clean up
    std::filesystem::remove(test_path);
}