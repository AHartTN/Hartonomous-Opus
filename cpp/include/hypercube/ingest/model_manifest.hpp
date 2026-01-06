#pragma once
// =============================================================================
// MODEL MANIFEST - Config-Driven Intelligent Ingestion
// =============================================================================
// 
// The model manifest parses config.json, tokenizer.json, and other metadata
// to build an intelligent extraction plan. It knows:
// - What tensor types exist (embedding, attention, FFN, conv, etc.)
// - How to process each type (eigenmap embeddings, extract attention patterns, etc.)
// - Architectural dimensions (d_model, num_heads, num_layers) for validation
// - Parallelization strategy based on structure
//
// This metadata IS the semantic structure - it gets ingested as first-class content.
// =============================================================================

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <cstdint>

namespace hypercube {
namespace ingest {

namespace fs = std::filesystem;

// =============================================================================
// Tensor Categories - What kind of tensor is this and how to process it
// =============================================================================

enum class TensorCategory {
    UNKNOWN,
    
    // Embeddings - get eigenmap/orthonormalized projections
    TOKEN_EMBEDDING,      // language_model.model.shared.weight [vocab_size, d_model]
    POSITION_EMBEDDING,   // embed_positions.weight [max_pos, d_model]
    PATCH_EMBEDDING,      // vision patch embeddings
    
    // Attention - extract Q/K/V projections for relational patterns
    ATTENTION_QUERY,      // q_proj [d_model, d_model]
    ATTENTION_KEY,        // k_proj [d_model, d_model]
    ATTENTION_VALUE,      // v_proj [d_model, d_model]
    ATTENTION_OUTPUT,     // out_proj [d_model, d_model]
    CROSS_ATTENTION,      // encoder_attn (cross-modal attention)
    
    // FFN - extract transformation patterns
    FFN_UP,               // fc1 [ffn_dim, d_model]
    FFN_DOWN,             // fc2 [d_model, ffn_dim]
    FFN_GATE,             // gate_proj (for gated FFNs)
    
    // Normalization - layer norms, RMS norms
    LAYER_NORM,           // layernorm [d_model]
    RMS_NORM,
    
    // Convolution - for vision models
    CONV_KERNEL,          // depthwise/pointwise convs
    
    // Projection - modality projections
    MODALITY_PROJECTION,  // image_projection, text_projection
    LOGIT_HEAD,           // final_logits_bias, lm_head
    
    // MoE - Mixture of Experts (DeepSeek, Qwen)
    MOE_GATE,             // router weights
    MOE_EXPERT,           // individual expert weights
    
    // Quantization artifacts
    QUANTIZATION_SCALE,
    QUANTIZATION_ZERO_POINT
};

inline std::string category_to_string(TensorCategory cat) {
    switch (cat) {
        case TensorCategory::TOKEN_EMBEDDING: return "TOKEN_EMBEDDING";
        case TensorCategory::POSITION_EMBEDDING: return "POSITION_EMBEDDING";
        case TensorCategory::PATCH_EMBEDDING: return "PATCH_EMBEDDING";
        case TensorCategory::ATTENTION_QUERY: return "ATTENTION_QUERY";
        case TensorCategory::ATTENTION_KEY: return "ATTENTION_KEY";
        case TensorCategory::ATTENTION_VALUE: return "ATTENTION_VALUE";
        case TensorCategory::ATTENTION_OUTPUT: return "ATTENTION_OUTPUT";
        case TensorCategory::CROSS_ATTENTION: return "CROSS_ATTENTION";
        case TensorCategory::FFN_UP: return "FFN_UP";
        case TensorCategory::FFN_DOWN: return "FFN_DOWN";
        case TensorCategory::FFN_GATE: return "FFN_GATE";
        case TensorCategory::LAYER_NORM: return "LAYER_NORM";
        case TensorCategory::RMS_NORM: return "RMS_NORM";
        case TensorCategory::CONV_KERNEL: return "CONV_KERNEL";
        case TensorCategory::MODALITY_PROJECTION: return "MODALITY_PROJECTION";
        case TensorCategory::LOGIT_HEAD: return "LOGIT_HEAD";
        case TensorCategory::MOE_GATE: return "MOE_GATE";
        case TensorCategory::MOE_EXPERT: return "MOE_EXPERT";
        case TensorCategory::QUANTIZATION_SCALE: return "QUANTIZATION_SCALE";
        case TensorCategory::QUANTIZATION_ZERO_POINT: return "QUANTIZATION_ZERO_POINT";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// Model Architecture Types
// =============================================================================

enum class ModelArchitecture {
    UNKNOWN,
    
    // Encoder-Decoder
    BART,
    T5,
    DETR,           // Detection Transformer
    FLORENCE,       // Vision-Language
    
    // Decoder-Only
    GPT,
    LLAMA,
    QWEN,
    DEEPSEEK,
    
    // Encoder-Only
    BERT,
    ROBERTA,
    
    // Vision
    VIT,            // Vision Transformer
    DAVIT,          // Dual Attention ViT (Florence uses this)
    SWIN,
    RESNET,         // CNN backbone
    
    // Diffusion
    FLUX,
    STABLE_DIFFUSION,
    
    // Sentence Transformers
    SENTENCE_TRANSFORMER
};

inline std::string architecture_to_string(ModelArchitecture arch) {
    switch (arch) {
        case ModelArchitecture::BART: return "BART";
        case ModelArchitecture::T5: return "T5";
        case ModelArchitecture::DETR: return "DETR";
        case ModelArchitecture::FLORENCE: return "FLORENCE";
        case ModelArchitecture::GPT: return "GPT";
        case ModelArchitecture::LLAMA: return "LLAMA";
        case ModelArchitecture::QWEN: return "QWEN";
        case ModelArchitecture::DEEPSEEK: return "DEEPSEEK";
        case ModelArchitecture::BERT: return "BERT";
        case ModelArchitecture::ROBERTA: return "ROBERTA";
        case ModelArchitecture::VIT: return "VIT";
        case ModelArchitecture::DAVIT: return "DAVIT";
        case ModelArchitecture::SWIN: return "SWIN";
        case ModelArchitecture::RESNET: return "RESNET";
        case ModelArchitecture::FLUX: return "FLUX";
        case ModelArchitecture::STABLE_DIFFUSION: return "STABLE_DIFFUSION";
        case ModelArchitecture::SENTENCE_TRANSFORMER: return "SENTENCE_TRANSFORMER";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// Architectural Dimensions - Extracted from config
// =============================================================================

struct ArchitecturalDimensions {
    // Text/Language model dimensions
    int vocab_size = 0;
    int d_model = 0;           // hidden_size
    int num_layers = 0;        // num_hidden_layers
    int num_heads = 0;         // num_attention_heads
    int head_dim = 0;          // d_model / num_heads (or explicit)
    int ffn_dim = 0;           // intermediate_size
    int max_position = 0;      // max_position_embeddings
    
    // Vision model dimensions (if multimodal)
    std::vector<int> vision_dims;        // per-stage dims [128, 256, 512, 1024]
    int patch_size = 0;
    int image_size = 0;
    
    // MoE dimensions
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    
    // Special tokens
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    
    // Projection dimensions (for multimodal)
    int projection_dim = 0;   // shared vision-language dim
};

// =============================================================================
// Tensor Extraction Plan - What to do with each tensor
// =============================================================================

struct TensorExtractionPlan {
    std::string name;
    TensorCategory category;
    std::vector<int64_t> shape;
    std::string dtype;
    
    // Extraction flags
    bool extract_embeddings = false;    // Run eigenmap/k-NN
    bool extract_attention = false;     // Build attention relation graph
    bool extract_statistics = false;    // Just compute stats (mean, std, norm)
    bool skip = false;                  // Don't process (e.g., quantization artifacts)
    
    // Layer/head indices (for routing)
    int layer_idx = -1;
    int head_idx = -1;
    std::string module_path;            // e.g., "language_model.model.decoder"
};

// =============================================================================
// Model Manifest - Complete extraction plan for a model
// =============================================================================

struct ModelManifest {
    std::string model_name;
    std::string model_path;
    ModelArchitecture architecture = ModelArchitecture::UNKNOWN;
    ArchitecturalDimensions dims;
    
    // Raw config data (for atoms)
    std::map<std::string, std::string> config_atoms;    // key -> value string
    std::map<std::string, std::string> tokenizer_atoms;
    
    // BPE merges (for composition edges)
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    std::map<std::string, int> vocab;
    
    // Tensor extraction plans
    std::vector<TensorExtractionPlan> extraction_plans;
    
    // Summary stats
    int total_tensors = 0;
    int embedding_tensors = 0;
    int attention_tensors = 0;
    int ffn_tensors = 0;
    int norm_tensors = 0;
    int conv_tensors = 0;
    int other_tensors = 0;
    
    // Methods
    void categorize_tensor(const std::string& name, const std::vector<int64_t>& shape, 
                          const std::string& dtype);
    TensorCategory classify_tensor(const std::string& name, const std::vector<int64_t>& shape);
    void print_summary() const;
};

// =============================================================================
// Tensor Classification Logic
// =============================================================================

inline TensorCategory ModelManifest::classify_tensor(const std::string& name, 
                                                     const std::vector<int64_t>& shape) {
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // Quantization artifacts - skip these
    if (lower_name.find("scale") != std::string::npos && shape.size() == 1) {
        return TensorCategory::QUANTIZATION_SCALE;
    }
    if (lower_name.find("zero_point") != std::string::npos) {
        return TensorCategory::QUANTIZATION_ZERO_POINT;
    }
    
    // Embeddings
    if (lower_name.find("embed") != std::string::npos) {
        if (lower_name.find("position") != std::string::npos || 
            lower_name.find("pos_embed") != std::string::npos) {
            return TensorCategory::POSITION_EMBEDDING;
        }
        if (lower_name.find("patch") != std::string::npos) {
            return TensorCategory::PATCH_EMBEDDING;
        }
        if (lower_name.find("shared") != std::string::npos || 
            lower_name.find("token") != std::string::npos ||
            lower_name.find("wte") != std::string::npos) {
            return TensorCategory::TOKEN_EMBEDDING;
        }
        // Generic embedding
        if (shape.size() == 2 && shape[0] > 1000) {  // Likely vocab embedding
            return TensorCategory::TOKEN_EMBEDDING;
        }
    }
    
    // Logit head
    if (lower_name.find("lm_head") != std::string::npos ||
        lower_name.find("logits_bias") != std::string::npos ||
        lower_name.find("output_projection") != std::string::npos) {
        return TensorCategory::LOGIT_HEAD;
    }
    
    // Attention
    if (lower_name.find("attn") != std::string::npos || 
        lower_name.find("attention") != std::string::npos) {
        if (lower_name.find("encoder_attn") != std::string::npos) {
            return TensorCategory::CROSS_ATTENTION;
        }
        if (lower_name.find("q_proj") != std::string::npos || 
            lower_name.find("query") != std::string::npos) {
            return TensorCategory::ATTENTION_QUERY;
        }
        if (lower_name.find("k_proj") != std::string::npos || 
            lower_name.find("key") != std::string::npos) {
            return TensorCategory::ATTENTION_KEY;
        }
        if (lower_name.find("v_proj") != std::string::npos || 
            lower_name.find("value") != std::string::npos) {
            return TensorCategory::ATTENTION_VALUE;
        }
        if (lower_name.find("out_proj") != std::string::npos || 
            lower_name.find("o_proj") != std::string::npos) {
            return TensorCategory::ATTENTION_OUTPUT;
        }
    }
    
    // FFN
    if (lower_name.find("fc1") != std::string::npos ||
        lower_name.find("up_proj") != std::string::npos ||
        lower_name.find("gate_proj") != std::string::npos ||
        lower_name.find("wi_0") != std::string::npos) {
        if (lower_name.find("gate") != std::string::npos) {
            return TensorCategory::FFN_GATE;
        }
        return TensorCategory::FFN_UP;
    }
    if (lower_name.find("fc2") != std::string::npos ||
        lower_name.find("down_proj") != std::string::npos ||
        lower_name.find("wo") != std::string::npos) {
        return TensorCategory::FFN_DOWN;
    }
    
    // Normalization
    if (lower_name.find("norm") != std::string::npos) {
        if (lower_name.find("rms") != std::string::npos) {
            return TensorCategory::RMS_NORM;
        }
        return TensorCategory::LAYER_NORM;
    }
    if (lower_name.find("layernorm") != std::string::npos || 
        lower_name.find("layer_norm") != std::string::npos) {
        return TensorCategory::LAYER_NORM;
    }
    
    // Convolution
    if (lower_name.find("conv") != std::string::npos) {
        return TensorCategory::CONV_KERNEL;
    }
    
    // MoE
    if (lower_name.find("expert") != std::string::npos) {
        if (lower_name.find("gate") != std::string::npos || 
            lower_name.find("router") != std::string::npos) {
            return TensorCategory::MOE_GATE;
        }
        return TensorCategory::MOE_EXPERT;
    }
    
    // Projections
    if (lower_name.find("proj") != std::string::npos) {
        if (lower_name.find("image") != std::string::npos ||
            lower_name.find("text") != std::string::npos ||
            lower_name.find("visual") != std::string::npos) {
            return TensorCategory::MODALITY_PROJECTION;
        }
    }
    
    return TensorCategory::UNKNOWN;
}

inline void ModelManifest::categorize_tensor(const std::string& name, 
                                              const std::vector<int64_t>& shape,
                                              const std::string& dtype) {
    TensorExtractionPlan plan;
    plan.name = name;
    plan.shape = shape;
    plan.dtype = dtype;
    plan.category = classify_tensor(name, shape);
    
    // Determine extraction strategy based on category
    switch (plan.category) {
        case TensorCategory::TOKEN_EMBEDDING:
        case TensorCategory::POSITION_EMBEDDING:
        case TensorCategory::PATCH_EMBEDDING:
        case TensorCategory::MODALITY_PROJECTION:
            plan.extract_embeddings = true;
            embedding_tensors++;
            break;
            
        case TensorCategory::ATTENTION_QUERY:
        case TensorCategory::ATTENTION_KEY:
        case TensorCategory::ATTENTION_VALUE:
        case TensorCategory::ATTENTION_OUTPUT:
        case TensorCategory::CROSS_ATTENTION:
            plan.extract_attention = true;
            attention_tensors++;
            break;
            
        case TensorCategory::FFN_UP:
        case TensorCategory::FFN_DOWN:
        case TensorCategory::FFN_GATE:
            plan.extract_statistics = true;
            ffn_tensors++;
            break;
            
        case TensorCategory::LAYER_NORM:
        case TensorCategory::RMS_NORM:
            plan.extract_statistics = true;
            norm_tensors++;
            break;
            
        case TensorCategory::CONV_KERNEL:
            plan.extract_statistics = true;
            conv_tensors++;
            break;
            
        case TensorCategory::QUANTIZATION_SCALE:
        case TensorCategory::QUANTIZATION_ZERO_POINT:
            plan.skip = true;
            break;
            
        default:
            plan.extract_statistics = true;
            other_tensors++;
            break;
    }
    
    // Extract layer index from name
    std::regex layer_regex(R"(layers?[._](\d+))");
    std::smatch match;
    if (std::regex_search(name, match, layer_regex)) {
        plan.layer_idx = std::stoi(match[1].str());
    }
    
    // Extract module path (everything before the final component)
    size_t last_dot = name.rfind('.');
    if (last_dot != std::string::npos) {
        plan.module_path = name.substr(0, last_dot);
    }
    
    extraction_plans.push_back(std::move(plan));
    total_tensors++;
}

inline void ModelManifest::print_summary() const {
    std::cerr << "\n+--------------------------------------------------------------+\n";
    std::cerr << "|              MODEL MANIFEST SUMMARY                          |\n";
    std::cerr << "+--------------------------------------------------------------+\n";
    std::cerr << "  Model: " << model_name << "\n";
    std::cerr << "  Architecture: " << architecture_to_string(architecture) << "\n";
    std::cerr << "  Path: " << model_path << "\n";
    std::cerr << "+--------------------------------------------------------------+\n";
    std::cerr << "  DIMENSIONS:\n";
    if (dims.vocab_size > 0) 
        std::cerr << "    Vocab Size: " << dims.vocab_size << "\n";
    if (dims.d_model > 0) 
        std::cerr << "    Model Dim (d_model): " << dims.d_model << "\n";
    if (dims.num_layers > 0) 
        std::cerr << "    Layers: " << dims.num_layers << "\n";
    if (dims.num_heads > 0) 
        std::cerr << "    Attention Heads: " << dims.num_heads << "\n";
    if (dims.ffn_dim > 0) 
        std::cerr << "    FFN Dim: " << dims.ffn_dim << "\n";
    if (dims.num_experts > 0) 
        std::cerr << "    MoE Experts: " << dims.num_experts 
                  << " (top-" << dims.num_experts_per_tok << ")\n";
    if (!dims.vision_dims.empty()) {
        std::cerr << "    Vision Dims: [";
        for (size_t i = 0; i < dims.vision_dims.size(); ++i) {
            std::cerr << dims.vision_dims[i];
            if (i < dims.vision_dims.size() - 1) std::cerr << ", ";
        }
        std::cerr << "]\n";
    }
    std::cerr << "+--------------------------------------------------------------+\n";
    std::cerr << "  TENSORS BY CATEGORY:\n";
    std::cerr << "    Embeddings:    " << embedding_tensors << " (eigenmap extraction)\n";
    std::cerr << "    Attention:     " << attention_tensors << " (relation extraction)\n";
    std::cerr << "    FFN:           " << ffn_tensors << " (stats only)\n";
    std::cerr << "    Normalization: " << norm_tensors << " (stats only)\n";
    std::cerr << "    Convolution:   " << conv_tensors << " (stats only)\n";
    std::cerr << "    Other:         " << other_tensors << "\n";
    std::cerr << "    TOTAL:         " << total_tensors << "\n";
    std::cerr << "+--------------------------------------------------------------+\n";
    std::cerr << "  METADATA FOR INGESTION:\n";
    std::cerr << "    Config atoms:    " << config_atoms.size() << "\n";
    std::cerr << "    Tokenizer atoms: " << tokenizer_atoms.size() << "\n";
    std::cerr << "    BPE merges:      " << bpe_merges.size() << " (composition edges)\n";
    std::cerr << "    Vocab entries:   " << vocab.size() << "\n";
    std::cerr << "+--------------------------------------------------------------+\n\n";
}

// =============================================================================
// Config Parser - Extract architectural info from config.json
// =============================================================================

inline ModelManifest parse_model_manifest(const fs::path& model_dir) {
    ModelManifest manifest;
    manifest.model_path = model_dir.string();
    manifest.model_name = model_dir.filename().string();
    
    // Parse config.json
    fs::path config_path = model_dir / "config.json";
    if (fs::exists(config_path)) {
        std::ifstream file(config_path);
        if (file) {
            std::string content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
            
            // Simple JSON key extraction (we'll store all key-value pairs as atoms)
            auto extract_int = [&](const std::string& key) -> std::optional<int> {
                std::string search = "\"" + key + "\"";
                size_t pos = content.find(search);
                if (pos != std::string::npos) {
                    size_t colon = content.find(":", pos);
                    size_t start = content.find_first_of("-0123456789", colon);
                    size_t end = content.find_first_not_of("-0123456789", start);
                    if (start != std::string::npos) {
                        try {
                            return std::stoi(content.substr(start, end - start));
                        } catch (...) {}
                    }
                }
                return std::nullopt;
            };
            
            auto extract_string = [&](const std::string& key) -> std::optional<std::string> {
                std::string search = "\"" + key + "\"";
                size_t pos = content.find(search);
                if (pos != std::string::npos) {
                    size_t colon = content.find(":", pos);
                    size_t q1 = content.find("\"", colon);
                    size_t q2 = content.find("\"", q1 + 1);
                    if (q1 != std::string::npos && q2 != std::string::npos) {
                        return content.substr(q1 + 1, q2 - q1 - 1);
                    }
                }
                return std::nullopt;
            };
            
            // Detect architecture
            auto model_type = extract_string("model_type");
            auto arch_str = extract_string("architectures");
            
            if (model_type) {
                std::string mt = *model_type;
                if (mt.find("florence") != std::string::npos) manifest.architecture = ModelArchitecture::FLORENCE;
                else if (mt.find("detr") != std::string::npos) manifest.architecture = ModelArchitecture::DETR;
                else if (mt.find("llama") != std::string::npos) manifest.architecture = ModelArchitecture::LLAMA;
                else if (mt.find("qwen") != std::string::npos) manifest.architecture = ModelArchitecture::QWEN;
                else if (mt.find("deepseek") != std::string::npos) manifest.architecture = ModelArchitecture::DEEPSEEK;
                else if (mt.find("bert") != std::string::npos) manifest.architecture = ModelArchitecture::BERT;
                else if (mt.find("bart") != std::string::npos) manifest.architecture = ModelArchitecture::BART;
                else if (mt.find("t5") != std::string::npos) manifest.architecture = ModelArchitecture::T5;
                else if (mt.find("gpt") != std::string::npos) manifest.architecture = ModelArchitecture::GPT;
                
                manifest.config_atoms["model_type"] = mt;
            }
            
            // Extract dimensions
            if (auto v = extract_int("vocab_size")) manifest.dims.vocab_size = *v;
            if (auto v = extract_int("hidden_size")) manifest.dims.d_model = *v;
            if (auto v = extract_int("d_model")) manifest.dims.d_model = *v;
            if (auto v = extract_int("num_hidden_layers")) manifest.dims.num_layers = *v;
            if (auto v = extract_int("decoder_layers")) manifest.dims.num_layers = *v;
            if (auto v = extract_int("num_attention_heads")) manifest.dims.num_heads = *v;
            if (auto v = extract_int("decoder_attention_heads")) manifest.dims.num_heads = *v;
            if (auto v = extract_int("intermediate_size")) manifest.dims.ffn_dim = *v;
            if (auto v = extract_int("decoder_ffn_dim")) manifest.dims.ffn_dim = *v;
            if (auto v = extract_int("max_position_embeddings")) manifest.dims.max_position = *v;
            if (auto v = extract_int("projection_dim")) manifest.dims.projection_dim = *v;
            
            // MoE dimensions
            if (auto v = extract_int("num_experts")) manifest.dims.num_experts = *v;
            if (auto v = extract_int("n_routed_experts")) manifest.dims.num_experts = *v;
            if (auto v = extract_int("num_experts_per_tok")) manifest.dims.num_experts_per_tok = *v;
            if (auto v = extract_int("moe_intermediate_size")) manifest.dims.moe_intermediate_size = *v;
            
            // Special tokens
            if (auto v = extract_int("bos_token_id")) manifest.dims.bos_token_id = *v;
            if (auto v = extract_int("eos_token_id")) manifest.dims.eos_token_id = *v;
            if (auto v = extract_int("pad_token_id")) manifest.dims.pad_token_id = *v;
            
            // Store all numeric config values as atoms
            // Pattern: "key": number
            std::regex number_pattern("\"([a-z_]+)\":\\s*(-?\\d+)");
            std::smatch match;
            std::string::const_iterator search_start = content.cbegin();
            while (std::regex_search(search_start, content.cend(), match, number_pattern)) {
                manifest.config_atoms[match[1].str()] = match[2].str();
                search_start = match.suffix().first;
            }
        }
    }
    
    std::cerr << "[MANIFEST] Parsed config for " << manifest.model_name 
              << " (" << architecture_to_string(manifest.architecture) << ")\n";
    std::cerr << "[MANIFEST] Found " << manifest.config_atoms.size() << " config atoms\n";
    
    return manifest;
}

} // namespace ingest
} // namespace hypercube