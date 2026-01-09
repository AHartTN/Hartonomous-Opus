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

    // =========================================================================
    // EMBEDDINGS - semantic manifolds
    // =========================================================================
    TOKEN_EMBEDDING,      // language_model.model.shared.weight [vocab_size, d_model]
    POSITION_EMBEDDING,   // embed_positions.weight [max_pos, d_model] - 1D positional
    POSITION_EMBEDDING_2D,// row/column embeddings [grid_size, d_model] - 2D positional
    PATCH_EMBEDDING,      // vision patch embeddings [num_patches, d_model]

    // =========================================================================
    // OBJECT DETECTION - DETR/Florence visual "tokens"
    // =========================================================================
    OBJECT_QUERY,         // query_position_embeddings [num_queries, d_model] - DETR slots
    CLASS_HEAD,           // class_labels_classifier [num_classes, d_model]
    BBOX_HEAD,            // bbox_predictor layers

    // =========================================================================
    // DETECTION HEADS - Grounding-DINO, RT-DETR specific
    // =========================================================================
    DETECTION_BACKBONE,   // ResNet backbone features
    DETECTION_NECK,       // FPN/feature pyramid features
    DETECTION_HEAD,       // Detection prediction heads
    
    // =========================================================================
    // ATTENTION - extract Q/K/V projections for relational patterns
    // =========================================================================
    ATTENTION_QUERY,      // q_proj [d_model, d_model]
    ATTENTION_KEY,        // k_proj [d_model, d_model]
    ATTENTION_VALUE,      // v_proj [d_model, d_model]
    ATTENTION_OUTPUT,     // out_proj [d_model, d_model]
    CROSS_ATTENTION,      // encoder_attn (cross-modal attention)
    
    // =========================================================================
    // VISION TOWER - visual feature extractors
    // =========================================================================
    VISION_FEATURE,       // vision_tower blocks, backbone features
    VISION_PROJECTION,    // image_projection [vision_dim, lang_dim]
    
    // =========================================================================
    // FFN - transformation geometry
    // =========================================================================
    FFN_UP,               // fc1 [ffn_dim, d_model]
    FFN_DOWN,             // fc2 [d_model, ffn_dim]
    FFN_GATE,             // gate_proj (for gated FFNs)
    
    // =========================================================================
    // MOE - Mixture of Experts control flow (THE ROUTING MACHINE)
    // =========================================================================
    MOE_ROUTER,           // router.weight, gate.weight - THE BRAIN of MoE
    MOE_EXPERT_UP,        // expert.w1/gate_proj/up_proj per expert
    MOE_EXPERT_DOWN,      // expert.w2/down_proj per expert  
    MOE_EXPERT_GATE,      // expert.w3/gate_proj (gated experts)
    MOE_SHARED_EXPERT,    // shared expert (DeepSeek V2/V3)
    
    // =========================================================================
    // NORMALIZATION
    // =========================================================================
    LAYER_NORM,           // layernorm [d_model]
    RMS_NORM,
    
    // =========================================================================
    // CONVOLUTION - vision backbones
    // =========================================================================
    CONV_KERNEL,          // depthwise/pointwise convs
    
    // =========================================================================
    // PROJECTION HEADS
    // =========================================================================
    MODALITY_PROJECTION,  // image_projection, text_projection
    LOGIT_HEAD,           // final_logits_bias, lm_head
    
    // Legacy MoE categories (for backwards compat)
    MOE_GATE,             // alias for MOE_ROUTER
    MOE_EXPERT,           // alias for generic expert
    
    // Quantization artifacts
    QUANTIZATION_SCALE,
    QUANTIZATION_ZERO_POINT
};

inline std::string category_to_string(TensorCategory cat) {
    switch (cat) {
        // Embeddings
        case TensorCategory::TOKEN_EMBEDDING: return "TOKEN_EMBEDDING";
        case TensorCategory::POSITION_EMBEDDING: return "POSITION_EMBEDDING";
        case TensorCategory::POSITION_EMBEDDING_2D: return "POSITION_EMBEDDING_2D";
        case TensorCategory::PATCH_EMBEDDING: return "PATCH_EMBEDDING";
        
        // Object Detection
        case TensorCategory::OBJECT_QUERY: return "OBJECT_QUERY";
        case TensorCategory::CLASS_HEAD: return "CLASS_HEAD";
        case TensorCategory::BBOX_HEAD: return "BBOX_HEAD";

        // Detection Components
        case TensorCategory::DETECTION_BACKBONE: return "DETECTION_BACKBONE";
        case TensorCategory::DETECTION_NECK: return "DETECTION_NECK";
        case TensorCategory::DETECTION_HEAD: return "DETECTION_HEAD";
        
        // Attention
        case TensorCategory::ATTENTION_QUERY: return "ATTENTION_QUERY";
        case TensorCategory::ATTENTION_KEY: return "ATTENTION_KEY";
        case TensorCategory::ATTENTION_VALUE: return "ATTENTION_VALUE";
        case TensorCategory::ATTENTION_OUTPUT: return "ATTENTION_OUTPUT";
        case TensorCategory::CROSS_ATTENTION: return "CROSS_ATTENTION";
        
        // Vision
        case TensorCategory::VISION_FEATURE: return "VISION_FEATURE";
        case TensorCategory::VISION_PROJECTION: return "VISION_PROJECTION";
        
        // FFN
        case TensorCategory::FFN_UP: return "FFN_UP";
        case TensorCategory::FFN_DOWN: return "FFN_DOWN";
        case TensorCategory::FFN_GATE: return "FFN_GATE";
        
        // MoE - The Routing Machine
        case TensorCategory::MOE_ROUTER: return "MOE_ROUTER";
        case TensorCategory::MOE_EXPERT_UP: return "MOE_EXPERT_UP";
        case TensorCategory::MOE_EXPERT_DOWN: return "MOE_EXPERT_DOWN";
        case TensorCategory::MOE_EXPERT_GATE: return "MOE_EXPERT_GATE";
        case TensorCategory::MOE_SHARED_EXPERT: return "MOE_SHARED_EXPERT";
        case TensorCategory::MOE_GATE: return "MOE_GATE";  // legacy
        case TensorCategory::MOE_EXPERT: return "MOE_EXPERT";  // legacy
        
        // Normalization
        case TensorCategory::LAYER_NORM: return "LAYER_NORM";
        case TensorCategory::RMS_NORM: return "RMS_NORM";
        
        // Convolution
        case TensorCategory::CONV_KERNEL: return "CONV_KERNEL";
        
        // Projections
        case TensorCategory::MODALITY_PROJECTION: return "MODALITY_PROJECTION";
        case TensorCategory::LOGIT_HEAD: return "LOGIT_HEAD";
        
        // Quantization
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
    int detection_tensors = 0;
    int other_tensors = 0;
    
    // Methods
    void categorize_tensor(const std::string& name, const std::vector<int64_t>& shape, 
                          const std::string& dtype);
    TensorCategory classify_tensor(const std::string& name, const std::vector<int64_t>& shape);
    void print_summary() const;
};

// =============================================================================
// Tensor Classification Logic - Comprehensive Multimodal Support
// =============================================================================
// 
// Supports: DETR, Florence, Grounding-DINO, RT-DETR, DeepSeek, Qwen MoE, Llama-4
// 
// Key tensor patterns:
//   - Object queries: query_position_embeddings, object_queries, decoder.query_embed
//   - 2D positional: row_embeddings, column_embeddings, image_pos_embed
//   - MoE routing: router.weight, gate.weight, mlp.gate (THE ROUTING MACHINE)
//   - MoE experts: experts.*.w1/w2/w3, expert.gate_proj/up_proj/down_proj
//   - Vision: vision_tower, backbone, patch_embed, image_projection
//   - Detection: class_labels_classifier, bbox_predictor
// =============================================================================

inline TensorCategory ModelManifest::classify_tensor(const std::string& name, 
                                                     const std::vector<int64_t>& shape) {
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    // =========================================================================
    // 1. QUANTIZATION ARTIFACTS - skip
    // =========================================================================
    if (lower_name.find("scale") != std::string::npos && shape.size() == 1) {
        return TensorCategory::QUANTIZATION_SCALE;
    }
    if (lower_name.find("zero_point") != std::string::npos) {
        return TensorCategory::QUANTIZATION_ZERO_POINT;
    }

    // =========================================================================
    // 1.5. BIAS VECTORS AND RUNNING STATISTICS - skip
    // =========================================================================
    if (lower_name.find("bias") != std::string::npos && shape.size() == 1) {
        return TensorCategory::QUANTIZATION_SCALE;  // Reuse as "skip" category
    }
    if (lower_name.find("running_mean") != std::string::npos ||
        lower_name.find("running_var") != std::string::npos) {
        return TensorCategory::QUANTIZATION_SCALE;  // Reuse as "skip" category
    }
    
    // =========================================================================
    // 2. OBJECT DETECTION HEADS (DETR, Florence, RT-DETR)
    // =========================================================================
    // Object queries - THE semantic anchors of vision transformers
    if (lower_name.find("query_position_embed") != std::string::npos ||
        lower_name.find("object_queries") != std::string::npos ||
        lower_name.find("decoder.query_embed") != std::string::npos ||
        (lower_name.find("query") != std::string::npos && 
         lower_name.find("embed") != std::string::npos &&
         lower_name.find("attn") == std::string::npos)) {
        return TensorCategory::OBJECT_QUERY;
    }
    
    // Class prediction head
    if (lower_name.find("class_labels_classifier") != std::string::npos ||
        lower_name.find("class_embed") != std::string::npos ||
        (lower_name.find("class") != std::string::npos && 
         lower_name.find("weight") != std::string::npos &&
         shape.size() == 2 && shape[0] < 1000)) {  // num_classes typically < 1000
        return TensorCategory::CLASS_HEAD;
    }
    
    // Bounding box prediction head
    if (lower_name.find("bbox_predictor") != std::string::npos ||
        lower_name.find("bbox_embed") != std::string::npos ||
        lower_name.find("bbox") != std::string::npos && lower_name.find("pred") != std::string::npos) {
        return TensorCategory::BBOX_HEAD;
    }

    // General detection heads
    if (lower_name.find("pred_head") != std::string::npos ||
        lower_name.find("det_head") != std::string::npos ||
        lower_name.find("dn_embed") != std::string::npos) {
        return TensorCategory::DETECTION_HEAD;
    }
    
    // =========================================================================
    // 3. 2D POSITIONAL ENCODINGS (Florence, DETR, vision models)
    // =========================================================================
    if (lower_name.find("row_embed") != std::string::npos ||
        lower_name.find("column_embed") != std::string::npos ||
        lower_name.find("image_pos_embed") != std::string::npos ||
        (lower_name.find("pos") != std::string::npos && 
         lower_name.find("embed") != std::string::npos &&
         lower_name.find("2d") != std::string::npos)) {
        return TensorCategory::POSITION_EMBEDDING_2D;
    }
    
    // Visual/temporal positional embeddings
    if (lower_name.find("visual_temporal_embed") != std::string::npos ||
        lower_name.find("pos_idx_to_embed") != std::string::npos) {
        return TensorCategory::POSITION_EMBEDDING_2D;
    }
    
    // =========================================================================
    // 4. MOE - MIXTURE OF EXPERTS (THE ROUTING MACHINE)
    // =========================================================================
    // This is the most important MoE structure - the router/gate
    // Patterns: router.weight, gate.weight, mlp.gate, moe_gate
    bool is_moe_context = lower_name.find("expert") != std::string::npos ||
                          lower_name.find("moe") != std::string::npos;
    
    // MoE Router - THE BRAIN (must check before generic FFN)
    if (is_moe_context && 
        (lower_name.find("router") != std::string::npos ||
         (lower_name.find("gate") != std::string::npos && 
          lower_name.find("proj") == std::string::npos))) {  // gate but not gate_proj
        return TensorCategory::MOE_ROUTER;
    }
    
    // Also catch standalone router patterns
    if ((lower_name.find("mlp.gate.weight") != std::string::npos ||
         lower_name.find("moe.gate.weight") != std::string::npos) &&
        shape.size() == 2) {
        return TensorCategory::MOE_ROUTER;
    }
    
    // Shared expert (DeepSeek V2/V3)
    if (lower_name.find("shared_expert") != std::string::npos ||
        lower_name.find("shared_experts") != std::string::npos) {
        if (lower_name.find("up_proj") != std::string::npos ||
            lower_name.find("gate_proj") != std::string::npos ||
            lower_name.find("w1") != std::string::npos) {
            return TensorCategory::MOE_SHARED_EXPERT;
        }
        if (lower_name.find("down_proj") != std::string::npos ||
            lower_name.find("w2") != std::string::npos) {
            return TensorCategory::MOE_SHARED_EXPERT;
        }
        return TensorCategory::MOE_SHARED_EXPERT;
    }
    
    // MoE Expert FFN weights
    if (is_moe_context) {
        // Expert up projection (w1, gate_proj, up_proj)
        if (lower_name.find("up_proj") != std::string::npos ||
            lower_name.find("gate_proj") != std::string::npos ||
            lower_name.find(".w1") != std::string::npos ||
            lower_name.find("_w1") != std::string::npos) {
            return TensorCategory::MOE_EXPERT_UP;
        }
        
        // Expert down projection (w2, down_proj)
        if (lower_name.find("down_proj") != std::string::npos ||
            lower_name.find(".w2") != std::string::npos ||
            lower_name.find("_w2") != std::string::npos) {
            return TensorCategory::MOE_EXPERT_DOWN;
        }
        
        // Expert gate (w3 in gated FFNs)
        if (lower_name.find(".w3") != std::string::npos ||
            lower_name.find("_w3") != std::string::npos) {
            return TensorCategory::MOE_EXPERT_GATE;
        }
        
        // Generic expert weight
        return TensorCategory::MOE_EXPERT;
    }
    
    // =========================================================================
    // 5. VISION TOWER (Florence, CLIP, multimodal models)
    // =========================================================================
    if (lower_name.find("vision_tower") != std::string::npos ||
        lower_name.find("visual_encoder") != std::string::npos ||
        lower_name.find("vision_model") != std::string::npos) {
        return TensorCategory::VISION_FEATURE;
    }
    
    // Vision/image projection (cross-modal alignment)
    if (lower_name.find("image_projection") != std::string::npos ||
        lower_name.find("image_proj") != std::string::npos ||
        lower_name.find("visual_projection") != std::string::npos) {
        return TensorCategory::VISION_PROJECTION;
    }
    
    // Detection backbone (ResNet, Swin features)
    if (lower_name.find("backbone") != std::string::npos) {
        if (lower_name.find("stage") != std::string::npos ||
            lower_name.find("layer") != std::string::npos ||
            lower_name.find("stem") != std::string::npos ||
            lower_name.find("conv") != std::string::npos) {
            return TensorCategory::DETECTION_BACKBONE;
        }
    }

    // Detection neck (FPN, feature pyramid)
    if (lower_name.find("neck") != std::string::npos ||
        lower_name.find("lateral") != std::string::npos ||
        lower_name.find("fpn") != std::string::npos ||
        lower_name.find("p") != std::string::npos && lower_name.find("conv") != std::string::npos ||
        lower_name.find("reduce") != std::string::npos && lower_name.find("conv") != std::string::npos) {
        return TensorCategory::DETECTION_NECK;
    }

    // DETR decoder components
    if (lower_name.find("decoder") != std::string::npos) {
        if (lower_name.find("self_attn") != std::string::npos ||
            lower_name.find("cross_attn") != std::string::npos) {
            return TensorCategory::CROSS_ATTENTION;
        }
    }

    // CNN backbone (DETR, Grounding-DINO)
    if (lower_name.find("backbone") != std::string::npos &&
        lower_name.find("conv") != std::string::npos) {
        return TensorCategory::CONV_KERNEL;
    }
    
    // =========================================================================
    // 6. EMBEDDINGS
    // =========================================================================
    if (lower_name.find("embed") != std::string::npos) {
        // Temporal embeddings (Florence-2, video models)
        if (lower_name.find("temporal") != std::string::npos ||
            lower_name.find("pos_idx_to_embed") != std::string::npos) {
            return TensorCategory::POSITION_EMBEDDING_2D;
        }
        // 1D positional
        if (lower_name.find("position") != std::string::npos ||
            lower_name.find("pos_embed") != std::string::npos) {
            return TensorCategory::POSITION_EMBEDDING;
        }
        // Patch embeddings
        if (lower_name.find("patch") != std::string::npos) {
            return TensorCategory::PATCH_EMBEDDING;
        }
        // Token embeddings
        if (lower_name.find("shared") != std::string::npos ||
            lower_name.find("token") != std::string::npos ||
            lower_name.find("wte") != std::string::npos ||
            lower_name.find("word_embed") != std::string::npos ||
            lower_name == "embeddings.word_embeddings.weight") {
            return TensorCategory::TOKEN_EMBEDDING;
        }
        // Large embedding tables are likely vocab
        if (shape.size() == 2 && shape[0] > 1000) {
            return TensorCategory::TOKEN_EMBEDDING;
        }
    }

    // CRITICAL: Check for large 2D matrices that might be embeddings even without "embed" in name
    // This catches tensors like "language_model.model.shared.weight" [51289, 768]
    if (shape.size() == 2 && shape[0] > 10000) {
        // Likely a token embedding table
        return TensorCategory::TOKEN_EMBEDDING;
    }
    
    // =========================================================================
    // 7. LOGIT HEAD
    // =========================================================================
    if (lower_name.find("lm_head") != std::string::npos ||
        lower_name.find("logits_bias") != std::string::npos ||
        lower_name.find("output_projection") != std::string::npos) {
        return TensorCategory::LOGIT_HEAD;
    }
    
    // =========================================================================
    // 8. ATTENTION
    // =========================================================================
    if (lower_name.find("attn") != std::string::npos ||
        lower_name.find("attention") != std::string::npos) {
        // Cross-attention
        if (lower_name.find("encoder_attn") != std::string::npos ||
            lower_name.find("cross_attn") != std::string::npos) {
            return TensorCategory::CROSS_ATTENTION;
        }
        // QKV projections - MUST be 2D matrices, not biases
        if (shape.size() == 2 && (
            lower_name.find("q_proj") != std::string::npos ||
            (lower_name.find("query") != std::string::npos && lower_name.find("weight") != std::string::npos))) {
            return TensorCategory::ATTENTION_QUERY;
        }
        if (shape.size() == 2 && (
            lower_name.find("k_proj") != std::string::npos ||
            lower_name.find("key") != std::string::npos)) {
            return TensorCategory::ATTENTION_KEY;
        }
        if (shape.size() == 2 && (
            lower_name.find("v_proj") != std::string::npos ||
            lower_name.find("value") != std::string::npos)) {
            return TensorCategory::ATTENTION_VALUE;
        }
        if (shape.size() == 2 && (
            lower_name.find("out_proj") != std::string::npos ||
            lower_name.find("o_proj") != std::string::npos)) {
            return TensorCategory::ATTENTION_OUTPUT;
        }
        // QKV combined
        if (lower_name.find("qkv") != std::string::npos) {
            return TensorCategory::ATTENTION_QUERY;  // Will need special handling
        }
    }

    // =========================================================================
    // Conditional-DETR: Decomposed Content-Position Attention
    // =========================================================================
    // These projections separate content and positional information for attention
    // Pattern: {ca|sa}_{q|k|v}{content|pos}_proj where ca=cross-attn, sa=self-attn
    if (shape.size() == 2 && lower_name.find("_proj") != std::string::npos) {
        // Cross-attention decomposed components
        if (lower_name.find("ca_") != std::string::npos) {
            if (lower_name.find("qcontent_proj") != std::string::npos ||
                lower_name.find("qpos_proj") != std::string::npos ||
                lower_name.find("qpos_sine_proj") != std::string::npos) {
                return TensorCategory::ATTENTION_QUERY;
            }
            if (lower_name.find("kcontent_proj") != std::string::npos ||
                lower_name.find("kpos_proj") != std::string::npos) {
                return TensorCategory::ATTENTION_KEY;
            }
            if (lower_name.find("v_proj") != std::string::npos) {
                return TensorCategory::ATTENTION_VALUE;
            }
        }
        // Self-attention decomposed components
        if (lower_name.find("sa_") != std::string::npos) {
            if (lower_name.find("qcontent_proj") != std::string::npos ||
                lower_name.find("qpos_proj") != std::string::npos) {
                return TensorCategory::ATTENTION_QUERY;
            }
            if (lower_name.find("kcontent_proj") != std::string::npos ||
                lower_name.find("kpos_proj") != std::string::npos) {
                return TensorCategory::ATTENTION_KEY;
            }
            if (lower_name.find("v_proj") != std::string::npos) {
                return TensorCategory::ATTENTION_VALUE;
            }
        }
    }
    
    // =========================================================================
    // 9. FFN (non-MoE)
    // =========================================================================
    if (lower_name.find("fc1") != std::string::npos ||
        lower_name.find("up_proj") != std::string::npos ||
        lower_name.find("wi_0") != std::string::npos ||
        lower_name.find("wi_1") != std::string::npos) {
        return TensorCategory::FFN_UP;
    }
    if (lower_name.find("gate_proj") != std::string::npos &&
        lower_name.find("expert") == std::string::npos) {
        return TensorCategory::FFN_GATE;
    }
    if (lower_name.find("fc2") != std::string::npos ||
        lower_name.find("down_proj") != std::string::npos ||
        lower_name.find("wo") != std::string::npos) {
        return TensorCategory::FFN_DOWN;
    }
    
    // =========================================================================
    // 10. NORMALIZATION
    // =========================================================================
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
    
    // =========================================================================
    // 11. CONVOLUTION
    // =========================================================================
    if (lower_name.find("conv") != std::string::npos) {
        return TensorCategory::CONV_KERNEL;
    }
    
    // =========================================================================
    // 12. MODALITY PROJECTIONS
    // =========================================================================
    if (lower_name.find("proj") != std::string::npos) {
        if (lower_name.find("image") != std::string::npos ||
            lower_name.find("text") != std::string::npos ||
            lower_name.find("visual") != std::string::npos) {
            return TensorCategory::MODALITY_PROJECTION;
        }
    }

    // =========================================================================
    // 13. ROTARY POSITION EMBEDDING (RoPE)
    // =========================================================================
    if (lower_name.find("rotary") != std::string::npos ||
        lower_name.find("rope") != std::string::npos) {
        return TensorCategory::POSITION_EMBEDDING;
    }

    // =========================================================================
    // 14. COMMON MODEL BUFFERS AND CACHE
    // =========================================================================
    if (lower_name.find("buffer") != std::string::npos ||
        lower_name.find("cache") != std::string::npos ||
        lower_name.find("mask") != std::string::npos && shape.size() == 1) {
        return TensorCategory::QUANTIZATION_SCALE;  // Skip
    }

    // =========================================================================
    // 15. RESIDUAL CONNECTIONS AND SKIP CONNECTIONS
    // =========================================================================
    if (lower_name.find("residual") != std::string::npos ||
        lower_name.find("shortcut") != std::string::npos) {
        return TensorCategory::LAYER_NORM;  // Treat as normalization
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
        // =====================================================================
        // EMBEDDINGS - extract semantic manifolds
        // =====================================================================
        case TensorCategory::TOKEN_EMBEDDING:
        case TensorCategory::POSITION_EMBEDDING:
        case TensorCategory::POSITION_EMBEDDING_2D:
        case TensorCategory::PATCH_EMBEDDING:
        case TensorCategory::MODALITY_PROJECTION:
        case TensorCategory::VISION_PROJECTION:
            plan.extract_embeddings = true;
            embedding_tensors++;
            break;
        
        // =====================================================================
        // OBJECT DETECTION - DETR/Florence semantic anchors
        // =====================================================================
        case TensorCategory::OBJECT_QUERY:
            plan.extract_embeddings = true;  // These ARE the visual tokens!
            embedding_tensors++;
            break;
            
        case TensorCategory::CLASS_HEAD:
        case TensorCategory::BBOX_HEAD:
            plan.extract_embeddings = true;  // Class prototypes
            embedding_tensors++;
            break;

        // =====================================================================
        // DETECTION COMPONENTS
        // =====================================================================
        case TensorCategory::DETECTION_BACKBONE:
            plan.extract_embeddings = true;  // Backbone features
            detection_tensors++;
            break;

        case TensorCategory::DETECTION_NECK:
            plan.extract_embeddings = true;  // FPN features
            detection_tensors++;
            break;

        case TensorCategory::DETECTION_HEAD:
            plan.extract_embeddings = true;  // Detection predictions
            detection_tensors++;
            break;
        
        // =====================================================================
        // VISION FEATURES
        // =====================================================================
        case TensorCategory::VISION_FEATURE:
            plan.extract_embeddings = true;
            embedding_tensors++;
            break;
            
        // =====================================================================
        // ATTENTION - relational patterns
        // =====================================================================
        case TensorCategory::ATTENTION_QUERY:
        case TensorCategory::ATTENTION_KEY:
        case TensorCategory::ATTENTION_VALUE:
        case TensorCategory::ATTENTION_OUTPUT:
        case TensorCategory::CROSS_ATTENTION:
            plan.extract_attention = true;
            attention_tensors++;
            break;
        
        // =====================================================================
        // MOE - THE ROUTING MACHINE (critical structure!)
        // =====================================================================
        case TensorCategory::MOE_ROUTER:
            plan.extract_embeddings = true;  // Router IS the semantic control
            plan.extract_attention = true;   // Build router→expert graph
            embedding_tensors++;
            break;
            
        case TensorCategory::MOE_EXPERT_UP:
        case TensorCategory::MOE_EXPERT_DOWN:
        case TensorCategory::MOE_EXPERT_GATE:
        case TensorCategory::MOE_SHARED_EXPERT:
        case TensorCategory::MOE_EXPERT:
        case TensorCategory::MOE_GATE:  // legacy
            // DO NOT extract embeddings from MoE experts - hundreds of large tensors
            // Expert weights analyzed via router relations
            plan.extract_attention = true;  // Track router→expert routing patterns
            ffn_tensors++;
            break;
            
        // =====================================================================
        // FFN - transformation geometry (semantic concept mixing)
        // =====================================================================
        case TensorCategory::FFN_UP:
        case TensorCategory::FFN_DOWN:
        case TensorCategory::FFN_GATE:
            // DO NOT extract embeddings from FFN - too many tensors, too slow
            // FFN weights are better analyzed via attention relations
            plan.extract_attention = true;   // Extract transformation relationships
            ffn_tensors++;
            break;

        // =====================================================================
        // NORMALIZATION (semantic scaling and domain adaptation)
        // =====================================================================
        case TensorCategory::LAYER_NORM:
        case TensorCategory::RMS_NORM:
            // DO NOT extract embeddings from normalization - single vector per layer
            // Norm parameters don't represent semantic manifolds
            norm_tensors++;
            break;

        // =====================================================================
        // CONVOLUTION (hierarchical visual features)
        // =====================================================================
        case TensorCategory::CONV_KERNEL:
            // DO NOT extract embeddings from conv kernels - too high dimensional
            // Conv filters analyzed better via attention patterns
            conv_tensors++;
            break;
        
        // =====================================================================
        // LOGIT HEAD
        // =====================================================================
        case TensorCategory::LOGIT_HEAD:
            plan.extract_embeddings = true;  // Important for vocab→class mapping
            embedding_tensors++;
            break;
            
        // =====================================================================
        // SKIP
        // =====================================================================
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
    
    // Extract expert index for MoE
    std::regex expert_regex(R"(experts?[._](\d+))");
    if (std::regex_search(name, match, expert_regex)) {
        plan.head_idx = std::stoi(match[1].str());  // Reuse head_idx for expert_idx
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
    // Count actual extraction types dynamically
    int ffn_embeddings = 0, ffn_relations = 0, ffn_stats = 0;
    int norm_embeddings = 0, norm_relations = 0, norm_stats = 0;
    int conv_embeddings = 0, conv_relations = 0, conv_stats = 0;

    for (const auto& plan : extraction_plans) {
        if (plan.category == TensorCategory::FFN_UP ||
            plan.category == TensorCategory::FFN_DOWN ||
            plan.category == TensorCategory::FFN_GATE) {
            if (plan.extract_embeddings) ffn_embeddings++;
            else if (plan.extract_attention) ffn_relations++;
            else if (plan.extract_statistics) ffn_stats++;
        }
        else if (plan.category == TensorCategory::LAYER_NORM ||
                 plan.category == TensorCategory::RMS_NORM) {
            if (plan.extract_embeddings) norm_embeddings++;
            else if (plan.extract_attention) norm_relations++;
            else if (plan.extract_statistics) norm_stats++;
        }
        else if (plan.category == TensorCategory::CONV_KERNEL) {
            if (plan.extract_embeddings) conv_embeddings++;
            else if (plan.extract_attention) conv_relations++;
            else if (plan.extract_statistics) conv_stats++;
        }
    }

    // Determine dominant extraction method for each category
    auto get_extraction_label = [](int embeddings, int relations, int stats, [[maybe_unused]] int total) -> std::string {
        if (embeddings > 0) return "eigenmap extraction";
        if (relations > 0) return "relation extraction";
        if (stats > 0) return "stats only";
        return "skipped";
    };

    std::string ffn_label = get_extraction_label(ffn_embeddings, ffn_relations, ffn_stats, ffn_tensors);
    std::string norm_label = get_extraction_label(norm_embeddings, norm_relations, norm_stats, norm_tensors);
    std::string conv_label = get_extraction_label(conv_embeddings, conv_relations, conv_stats, conv_tensors);

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
    std::cerr << "    FFN:           " << ffn_tensors << " (" << ffn_label << ")\n";
    std::cerr << "    Normalization: " << norm_tensors << " (" << norm_label << ")\n";
    std::cerr << "    Convolution:   " << conv_tensors << " (" << conv_label << ")\n";
    std::cerr << "    Detection:     " << detection_tensors << " (eigenmap extraction)\n";
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