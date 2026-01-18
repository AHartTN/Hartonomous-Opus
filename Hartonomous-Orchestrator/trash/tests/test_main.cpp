#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <boost/beast.hpp>
#include "logging.hpp"

// Test fixture for common setup/teardown
class HartonomousTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logging for tests
        Logging::initialize_logging("info", "", false);

        // Suppress logs during tests unless explicitly needed
        spdlog::set_level(spdlog::level::err);
    }

    void TearDown() override {
        // Clean up after each test
    }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Set up test environment
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    return RUN_ALL_TESTS();
}