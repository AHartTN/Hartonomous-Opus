/**
 * Plugin Registry Implementation
 * =============================
 *
 * Manages loading, registration, and lifecycle of plugins
 * Cross-platform: Windows DLL + Unix .so/.dylib
 */

#include "hypercube/plugin.hpp"
#include "hypercube/logging.hpp"

#include <filesystem>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

// Platform-specific dynamic loading
#ifdef _WIN32
#include <windows.h>
#define DLL_HANDLE HMODULE
#define DLL_LOAD(path) LoadLibraryA(path)
#define DLL_SYMBOL(handle, symbol) GetProcAddress(handle, symbol)
#define DLL_UNLOAD(handle) FreeLibrary(handle)
#define DLL_ERROR() "GetLastError: " + std::to_string(GetLastError())
#else
#include <dlfcn.h>
#define DLL_HANDLE void*
#define DLL_LOAD(path) dlopen(path, RTLD_LAZY)
#define DLL_SYMBOL(handle, symbol) dlsym(handle, symbol)
#define DLL_UNLOAD(handle) dlclose(handle)
#define DLL_ERROR() dlerror()
#endif

namespace fs = std::filesystem;

namespace hypercube {

// Plugin handle storage for proper cleanup
struct PluginHandle {
    DLL_HANDLE dll_handle;
    std::string path;
};

// Store handles for cleanup
static std::unordered_map<std::string, PluginHandle> plugin_handles_;

// Plugin Registry Singleton
PluginRegistry& PluginRegistry::get_instance() {
    static PluginRegistry instance;
    return instance;
}

size_t PluginRegistry::load_plugins(const std::string& plugin_dir) {
    size_t loaded_count = 0;

    if (!fs::exists(plugin_dir)) {
        std::cerr << "[PLUGIN] Plugin directory not found: " << plugin_dir << std::endl;
        return 0;
    }

    std::cerr << "[PLUGIN] Scanning plugin directory: " << plugin_dir << std::endl;

    try {
        for (const auto& entry : fs::directory_iterator(plugin_dir)) {
            if (!entry.is_regular_file()) continue;

            std::string filename = entry.path().filename().string();
            std::string extension = entry.path().extension().string();

            // Platform-specific library extensions
#ifdef _WIN32
            if (extension != ".dll") continue;
#else
            if (extension != ".so" && extension != ".dylib") continue;
#endif

            std::string plugin_path = entry.path().string();

            // Load the plugin library (cross-platform)
            DLL_HANDLE handle = DLL_LOAD(plugin_path.c_str());
            if (!handle) {
                std::cerr << "[PLUGIN] Failed to load " << filename << ": " << DLL_ERROR() << std::endl;
                continue;
            }

            // Find the factory function
            using FactoryFunc = std::unique_ptr<Plugin>(*)();
            FactoryFunc factory = reinterpret_cast<FactoryFunc>(DLL_SYMBOL(handle, "create_plugin"));

            if (!factory) {
                std::cerr << "[PLUGIN] No factory function in " << filename << ": " << DLL_ERROR() << std::endl;
                DLL_UNLOAD(handle);
                continue;
            }

            // Create the plugin instance
            try {
                auto plugin = factory();
                if (plugin) {
                    std::string plugin_name = plugin->get_name();
                    std::string plugin_type = plugin->get_type();

                    // Store handle for cleanup
                    std::string plugin_key = plugin_type + ":" + plugin_name;
                    plugin_handles_[plugin_key] = {handle, plugin_path};

                    // Register the plugin
                    register_plugin(std::move(plugin));

                    std::cerr << "[PLUGIN] Loaded " << plugin_type << " plugin: " << plugin_name << std::endl;
                    loaded_count++;
                } else {
                    DLL_UNLOAD(handle);
                }
            } catch (const std::exception& e) {
                std::cerr << "[PLUGIN] Exception creating plugin from " << filename << ": " << e.what() << std::endl;
                DLL_UNLOAD(handle);
                continue;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[PLUGIN] Error scanning plugins: " << e.what() << std::endl;
    }

    std::cerr << "[PLUGIN] Loaded " << loaded_count << " plugins total" << std::endl;
    return loaded_count;
}

void PluginRegistry::register_plugin(std::unique_ptr<Plugin> plugin) {
    if (!plugin) return;

    std::string name = plugin->get_name();
    std::string type = plugin->get_type();

    // Create unique key
    std::string key = type + ":" + name;

    plugins_[key] = std::move(plugin);
    plugins_by_type_[type].push_back(plugins_[key].get());
}

Plugin* PluginRegistry::get_plugin(const std::string& name, const std::string& type) {
    std::string key = type + ":" + name;
    auto it = plugins_.find(key);
    return it != plugins_.end() ? it->second.get() : nullptr;
}

std::vector<Plugin*> PluginRegistry::get_plugins_by_type(const std::string& type) {
    auto it = plugins_by_type_.find(type);
    return it != plugins_by_type_.end() ? it->second : std::vector<Plugin*>{};
}

size_t PluginRegistry::initialize_plugins(const PluginContext& context) {
    size_t initialized_count = 0;

    for (auto& [key, plugin] : plugins_) {
        try {
            if (plugin->initialize(context)) {
                initialized_count++;
            } else {
                std::cerr << "[PLUGIN] Failed to initialize plugin: " << plugin->get_name() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[PLUGIN] Exception initializing plugin " << plugin->get_name()
                      << ": " << e.what() << std::endl;
        }
    }

    std::cerr << "[PLUGIN] Initialized " << initialized_count << "/" << plugins_.size() << " plugins" << std::endl;
    return initialized_count;
}

void PluginRegistry::shutdown_plugins() {
    // Shutdown plugins first
    for (auto& [key, plugin] : plugins_) {
        try {
            plugin->shutdown();
        } catch (const std::exception& e) {
            std::cerr << "[PLUGIN] Exception shutting down plugin " << plugin->get_name()
                      << ": " << e.what() << std::endl;
        }
    }

    plugins_.clear();
    plugins_by_type_.clear();

    // Unload DLLs
    for (auto& [key, handle_info] : plugin_handles_) {
        try {
            if (DLL_UNLOAD(handle_info.dll_handle) != 0) {
                std::cerr << "[PLUGIN] Warning: Failed to unload " << handle_info.path << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[PLUGIN] Exception unloading " << handle_info.path << ": " << e.what() << std::endl;
        }
    }
    plugin_handles_.clear();

    std::cerr << "[PLUGIN] All plugins shut down and unloaded" << std::endl;
}

std::vector<std::unordered_map<std::string, std::string>> PluginRegistry::list_plugins() const {
    std::vector<std::unordered_map<std::string, std::string>> result;

    for (const auto& [key, plugin] : plugins_) {
        std::unordered_map<std::string, std::string> info;
        info["name"] = plugin->get_name();
        info["type"] = plugin->get_type();
        info["version"] = plugin->get_version();
        info["description"] = plugin->get_description();

        auto deps = plugin->get_dependencies();
        if (!deps.empty()) {
            std::string dep_str;
            for (size_t i = 0; i < deps.size(); ++i) {
                if (i > 0) dep_str += ",";
                dep_str += deps[i];
            }
            info["dependencies"] = dep_str;
        }

        result.push_back(std::move(info));
    }

    return result;
}

} // namespace hypercube