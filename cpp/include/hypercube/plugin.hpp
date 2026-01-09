/**
 * Plugin Architecture for Hartonomous-Opus
 * =======================================
 *
 * Extensible plugin system allowing:
 * - Custom geometric operations (distance metrics, similarity functions)
 * - Custom ingestion processors (model formats, tokenizers)
 * - Custom analysis algorithms (clustering, dimensionality reduction)
 * - Custom generation strategies (decoding algorithms, scoring functions)
 *
 * Plugin types are loaded dynamically at runtime and registered with the
 * PluginRegistry. Each plugin implements a specific interface and can be
 * discovered and invoked through the composition-based system.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include "types.hpp"

namespace hypercube {

// Forward declarations
struct PluginContext;
class PluginRegistry;

/**
 * Plugin Interface - Base class for all plugins
 */
class Plugin {
public:
    virtual ~Plugin() = default;

    /**
     * Plugin metadata
     */
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_description() const = 0;
    virtual std::vector<std::string> get_dependencies() const { return {}; }

    /**
     * Lifecycle methods
     */
    virtual bool initialize(const PluginContext& context) = 0;
    virtual void shutdown() = 0;

    /**
     * Plugin type identifier
     */
    virtual std::string get_type() const = 0;
};

/**
 * Plugin Context - Provides access to core system resources
 */
struct PluginContext {
    // Database connection (if needed)
    void* db_connection = nullptr;

    // Configuration parameters
    std::unordered_map<std::string, std::string> config;

    // Access to plugin registry for cross-plugin communication
    PluginRegistry* registry = nullptr;

    // Logging function
    std::function<void(const std::string&)> log_info;
    std::function<void(const std::string&)> log_warning;
    std::function<void(const std::string&)> log_error;
};

/**
 * Geometric Operation Plugin Interface
 * ===================================
 * Plugins that implement custom geometric operations in 4D space
 */
class GeometricOperationPlugin : public Plugin {
public:
    virtual std::string get_type() const override { return "geometric_operation"; }

    /**
     * Compute distance between two 4D points
     * @param a First point
     * @param b Second point
     * @return Distance value
     */
    virtual double compute_distance(const Point4D& a, const Point4D& b) const = 0;

    /**
     * Compute similarity between two compositions
     * @param composition_a First composition ID
     * @param composition_b Second composition ID
     * @return Similarity score (0-1)
     */
    virtual double compute_similarity(const Blake3Hash& composition_a,
                                    const Blake3Hash& composition_b) const = 0;

    /**
     * Find k-nearest neighbors in geometric space
     * @param query_point Query point
     * @param k Number of neighbors to find
     * @return Vector of (composition_id, distance) pairs
     */
    virtual std::vector<std::pair<Blake3Hash, double>>
        find_neighbors(const Point4D& query_point, size_t k) const = 0;
};

/**
 * Ingestion Processor Plugin Interface
 * ===================================
 * Plugins that handle custom model/data ingestion formats
 */
class IngestionProcessorPlugin : public Plugin {
public:
    virtual std::string get_type() const override { return "ingestion_processor"; }

    /**
     * Check if this plugin can handle the given file/path
     * @param path File or directory path
     * @return True if this plugin can process it
     */
    virtual bool can_handle(const std::string& path) const = 0;

    /**
     * Supported file extensions/formats
     * @return Vector of supported extensions (e.g., ".safetensors", ".onnx")
     */
    virtual std::vector<std::string> get_supported_formats() const = 0;

    /**
     * Process the file/directory and extract compositions/relations
     * @param path Input path
     * @param context Plugin context for database access
     * @return Success status
     */
    virtual bool process(const std::string& path, const PluginContext& context) = 0;

    /**
     * Extract metadata about the processed content
     * @param path Input path
     * @return Metadata key-value pairs
     */
    virtual std::unordered_map<std::string, std::string>
        extract_metadata(const std::string& path) const = 0;
};

/**
 * Analysis Algorithm Plugin Interface
 * ==================================
 * Plugins that implement custom analysis algorithms
 */
class AnalysisAlgorithmPlugin : public Plugin {
public:
    virtual std::string get_type() const override { return "analysis_algorithm"; }

    /**
     * Supported analysis types
     * @return Vector of analysis names this plugin supports
     */
    virtual std::vector<std::string> get_supported_analyses() const = 0;

    /**
     * Execute analysis on a set of compositions
     * @param analysis_type Type of analysis to perform
     * @param composition_ids Input composition IDs
     * @param parameters Analysis parameters
     * @param context Plugin context
     * @return Analysis results as JSON-like structure
     */
    virtual std::unordered_map<std::string, std::string>
        execute_analysis(const std::string& analysis_type,
                        const std::vector<Blake3Hash>& composition_ids,
                        const std::unordered_map<std::string, std::string>& parameters,
                        const PluginContext& context) = 0;
};

/**
 * Generation Context - Information about current generation state
 */
struct GenerationContext {
    Blake3Hash current_token;                    // Current token being generated from
    std::vector<Blake3Hash> context_tokens;      // Recent tokens in context
    std::unordered_map<std::string, double> weights; // Current scoring weights
    size_t max_tokens;                         // Maximum tokens to generate
    double temperature;                        // Sampling temperature
};

/**
 * Generation Strategy Plugin Interface
 * ===================================
 * Plugins that implement custom token generation strategies
 */
class GenerationStrategyPlugin : public Plugin {
public:
    virtual std::string get_type() const override { return "generation_strategy"; }

    /**
     * Score candidate tokens for generation
     * @param current_context Current generation context
     * @param candidates Candidate tokens to score
     * @return Scored candidates with ranking
     */
    virtual std::vector<std::pair<Blake3Hash, double>>
        score_candidates(const GenerationContext& current_context,
                        const std::vector<Blake3Hash>& candidates) const = 0;

    /**
     * Update strategy based on feedback/outcome
     * @param feedback Feedback data from generation outcome
     */
    virtual void update_from_feedback(const std::unordered_map<std::string, std::string>& feedback) = 0;
};

/**
 * Plugin Registry - Manages plugin loading, registration, and discovery
 */
class PluginRegistry {
public:
    static PluginRegistry& get_instance();

    /**
     * Load plugins from a directory
     * @param plugin_dir Directory containing plugin libraries
     * @return Number of plugins loaded
     */
    size_t load_plugins(const std::string& plugin_dir);

    /**
     * Register a plugin instance directly
     * @param plugin Plugin instance to register
     */
    void register_plugin(std::unique_ptr<Plugin> plugin);

    /**
     * Get plugin by name and type
     * @param name Plugin name
     * @param type Plugin type
     * @return Plugin instance or nullptr if not found
     */
    Plugin* get_plugin(const std::string& name, const std::string& type);

    /**
     * Get all plugins of a specific type
     * @param type Plugin type
     * @return Vector of plugins of that type
     */
    std::vector<Plugin*> get_plugins_by_type(const std::string& type);

    /**
     * Initialize all loaded plugins
     * @param context Plugin context
     * @return Number of plugins successfully initialized
     */
    size_t initialize_plugins(const PluginContext& context);

    /**
     * Shutdown all plugins
     */
    void shutdown_plugins();

    /**
     * List all registered plugins
     * @return Plugin information
     */
    std::vector<std::unordered_map<std::string, std::string>> list_plugins() const;

private:
    PluginRegistry() = default;

    std::unordered_map<std::string, std::unique_ptr<Plugin>> plugins_;
    std::unordered_map<std::string, std::vector<Plugin*>> plugins_by_type_;
};

/**
 * Plugin Factory Function Type
 * ===========================
 * Function signature for plugin creation functions
 */
using PluginFactory = std::unique_ptr<Plugin>(*)();

/**
 * Plugin Registration Macros
 * ==========================
 * Macros to simplify plugin creation and registration
 */

#define PLUGIN_EXPORT extern "C"

#define DEFINE_PLUGIN(PluginClass) \
    PLUGIN_EXPORT std::unique_ptr<hypercube::Plugin> create_plugin() { \
        return std::make_unique<PluginClass>(); \
    }

#define REGISTER_PLUGIN(PluginClass) \
    DEFINE_PLUGIN(PluginClass)

} // namespace hypercube