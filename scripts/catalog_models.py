#!/usr/bin/env python3
"""
Comprehensive Model Catalog for D:\Models
Analyzes all models, extracts config, tensor structures, and semantic patterns
"""

import sys
import json
import struct
from pathlib import Path
import os
from enum import Enum
from collections import defaultdict
import re

class TensorCategory(Enum):
    UNKNOWN = 0
    TOKEN_EMBEDDING = 1
    POSITION_EMBEDDING = 2
    POSITION_EMBEDDING_2D = 3
    PATCH_EMBEDDING = 4
    OBJECT_QUERY = 5
    CLASS_HEAD = 6
    BBOX_HEAD = 7
    DETECTION_BACKBONE = 8
    DETECTION_NECK = 9
    DETECTION_HEAD = 10
    ATTENTION_QUERY = 11
    ATTENTION_KEY = 12
    ATTENTION_VALUE = 13
    ATTENTION_OUTPUT = 14
    CROSS_ATTENTION = 15
    VISION_FEATURE = 16
    VISION_PROJECTION = 17
    FFN_UP = 18
    FFN_DOWN = 19
    FFN_GATE = 20
    MOE_ROUTER = 21
    MOE_EXPERT_UP = 22
    MOE_EXPERT_DOWN = 23
    MOE_EXPERT_GATE = 24
    MOE_SHARED_EXPERT = 25
    LAYER_NORM = 26
    RMS_NORM = 27
    CONV_KERNEL = 28
    MODALITY_PROJECTION = 29
    LOGIT_HEAD = 30
    QUANTIZATION_SCALE = 31
    QUANTIZATION_ZERO_POINT = 32

def classify_tensor(name, shape):
    lower_name = name.lower()

    # Skip artifacts
    if lower_name.find("scale") != -1 and len(shape) == 1:
        return TensorCategory.QUANTIZATION_SCALE
    if lower_name.find("zero_point") != -1:
        return TensorCategory.QUANTIZATION_ZERO_POINT
    if lower_name.find("bias") != -1 and len(shape) == 1:
        return TensorCategory.QUANTIZATION_SCALE
    if lower_name.find("running_mean") != -1 or lower_name.find("running_var") != -1:
        return TensorCategory.QUANTIZATION_SCALE

    # Object detection
    if lower_name.find("query_position_embed") != -1 or lower_name.find("object_queries") != -1 or \
       (lower_name.find("query") != -1 and lower_name.find("embed") != -1 and lower_name.find("attn") == -1):
        return TensorCategory.OBJECT_QUERY
    if lower_name.find("class_labels_classifier") != -1 or lower_name.find("class_embed") != -1 or \
       (lower_name.find("class") != -1 and lower_name.find("weight") != -1 and len(shape) == 2 and shape[0] < 1000):
        return TensorCategory.CLASS_HEAD
    if lower_name.find("bbox_predictor") != -1 or lower_name.find("bbox_embed") != -1 or \
       lower_name.find("bbox") != -1 and lower_name.find("pred") != -1:
        return TensorCategory.BBOX_HEAD

    # 2D positional
    if lower_name.find("row_embed") != -1 or lower_name.find("column_embed") != -1 or \
       lower_name.find("image_pos_embed") != -1 or \
       (lower_name.find("pos") != -1 and lower_name.find("embed") != -1 and lower_name.find("2d") != -1):
        return TensorCategory.POSITION_EMBEDDING_2D

    # MoE
    is_moe = lower_name.find("expert") != -1 or lower_name.find("moe") != -1
    if is_moe and (lower_name.find("router") != -1 or \
                   (lower_name.find("gate") != -1 and lower_name.find("proj") == -1)):
        return TensorCategory.MOE_ROUTER
    if lower_name.find("shared_expert") != -1:
        return TensorCategory.MOE_SHARED_EXPERT
    if is_moe:
        if lower_name.find("up_proj") != -1 or lower_name.find("gate_proj") != -1 or \
           lower_name.find(".w1") != -1 or lower_name.find("_w1") != -1:
            return TensorCategory.MOE_EXPERT_UP
        if lower_name.find("down_proj") != -1 or lower_name.find(".w2") != -1 or lower_name.find("_w2") != -1:
            return TensorCategory.MOE_EXPERT_DOWN
        if lower_name.find(".w3") != -1 or lower_name.find("_w3") != -1:
            return TensorCategory.MOE_EXPERT_GATE

    # Vision
    if lower_name.find("vision_tower") != -1 or lower_name.find("visual_encoder") != -1 or \
       lower_name.find("vision_model") != -1:
        return TensorCategory.VISION_FEATURE
    if lower_name.find("image_projection") != -1 or lower_name.find("image_proj") != -1 or \
       lower_name.find("visual_projection") != -1:
        return TensorCategory.VISION_PROJECTION

    # Embeddings
    if lower_name.find("embed") != -1:
        if lower_name.find("position") != -1 or lower_name.find("pos_embed") != -1:
            return TensorCategory.POSITION_EMBEDDING
        if lower_name.find("patch") != -1:
            return TensorCategory.PATCH_EMBEDDING
        if lower_name.find("shared") != -1 or lower_name.find("token") != -1 or \
           lower_name.find("wte") != -1 or lower_name.find("word_embed") != -1 or \
           lower_name == "embeddings.word_embeddings.weight":
            return TensorCategory.TOKEN_EMBEDDING
        if len(shape) == 2 and shape[0] > 10000:
            return TensorCategory.TOKEN_EMBEDDING

    # Logit head
    if lower_name.find("lm_head") != -1 or lower_name.find("logits_bias") != -1 or \
       lower_name.find("output_projection") != -1:
        return TensorCategory.LOGIT_HEAD

    # Attention - including FLUX qkv and proj patterns
    if lower_name.find("attn") != -1 or lower_name.find("attention") != -1:
        if lower_name.find("encoder_attn") != -1 or lower_name.find("cross_attn") != -1:
            return TensorCategory.CROSS_ATTENTION
        if len(shape) == 2:
            if lower_name.find("q_proj") != -1 or \
               (lower_name.find("query") != -1 and lower_name.find("weight") != -1):
                return TensorCategory.ATTENTION_QUERY
            if lower_name.find("k_proj") != -1 or lower_name.find("key") != -1:
                return TensorCategory.ATTENTION_KEY
            if lower_name.find("v_proj") != -1 or lower_name.find("value") != -1:
                return TensorCategory.ATTENTION_VALUE
            if lower_name.find("out_proj") != -1 or lower_name.find("o_proj") != -1 or \
               lower_name.find("proj") != -1:
                return TensorCategory.ATTENTION_OUTPUT
        if lower_name.find("qkv") != -1:
            return TensorCategory.ATTENTION_QUERY

    # FFN - including FLUX linear layers and MLP components
    if lower_name.find("fc1") != -1 or lower_name.find("up_proj") != -1 or \
       lower_name.find("wi_0") != -1 or lower_name.find("wi_1") != -1 or \
       lower_name.find("linear1") != -1 or \
       (lower_name.find("mlp") != -1 and lower_name.find("0") != -1 and lower_name.find("weight") != -1):
        return TensorCategory.FFN_UP
    if lower_name.find("gate_proj") != -1 and lower_name.find("expert") == -1:
        return TensorCategory.FFN_GATE
    if lower_name.find("fc2") != -1 or lower_name.find("down_proj") != -1 or \
       lower_name.find("wo") != -1 or lower_name.find("linear2") != -1 or \
       (lower_name.find("mlp") != -1 and lower_name.find("2") != -1 and lower_name.find("weight") != -1):
        return TensorCategory.FFN_DOWN

    # Normalization
    if lower_name.find("norm") != -1:
        if lower_name.find("rms") != -1:
            return TensorCategory.RMS_NORM
        return TensorCategory.LAYER_NORM
    if lower_name.find("layernorm") != -1 or lower_name.find("layer_norm") != -1:
        return TensorCategory.LAYER_NORM

    # Convolution
    if lower_name.find("conv") != -1:
        return TensorCategory.CONV_KERNEL

    # Projection
    if lower_name.find("proj") != -1:
        if lower_name.find("image") != -1 or lower_name.find("text") != -1 or \
           lower_name.find("visual") != -1:
            return TensorCategory.MODALITY_PROJECTION

    # Rotary embeddings
    if lower_name.find("rotary") != -1 or lower_name.find("rope") != -1:
        return TensorCategory.POSITION_EMBEDDING

    # Modulation layers (FLUX conditioning)
    if lower_name.find("modulation") != -1 or lower_name.find("adaLN") != -1:
        return TensorCategory.MODALITY_PROJECTION

    # Input/conditioning layers
    if lower_name.find("_in") != -1 and lower_name.find("weight") != -1:
        return TensorCategory.MODALITY_PROJECTION

    # Skip buffers
    if lower_name.find("buffer") != -1 or lower_name.find("cache") != -1 or \
       (lower_name.find("mask") != -1 and len(shape) == 1):
        return TensorCategory.QUANTIZATION_SCALE

    return TensorCategory.UNKNOWN

def generate_documentation(catalog, output_dir):
    """Generate comprehensive documentation from the catalog"""
    doc_file = output_dir / "MODEL_ARCHITECTURE_DOCUMENTATION.md"

    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write("# Model Architecture Documentation\n\n")
        f.write("Automatically generated catalog of all models in the collection.\n\n")
        f.write(f"Generated: {Path(__file__).name}\n\n")
        f.write("---\n\n")

        f.write("## Table of Contents\n\n")
        for model_name in sorted(catalog.keys()):
            f.write(f"- [{model_name}](#{model_name.lower().replace(' ', '-')})\n")
        f.write("\n---\n\n")

        for model_name in sorted(catalog.keys()):
            model_data = catalog[model_name]
            f.write(f"## {model_name}\n\n")
            f.write(f"**Path:** `{model_data['path']}`\n\n")

            # Config section
            if model_data.get('config'):
                config = model_data['config']
                f.write("### Configuration\n\n")
                f.write("```json\n")
                f.write(json.dumps(config, indent=2))
                f.write("\n```\n\n")

                # Key parameters
                f.write("#### Key Parameters\n\n")
                params = []
                if 'model_type' in config:
                    params.append(f"- **Model Type:** {config['model_type']}")
                if 'architectures' in config:
                    params.append(f"- **Architecture:** {config['architectures'][0] if config['architectures'] else 'unknown'}")
                if 'hidden_size' in config:
                    params.append(f"- **Hidden Size:** {config['hidden_size']}")
                if 'num_hidden_layers' in config:
                    params.append(f"- **Layers:** {config['num_hidden_layers']}")
                if 'num_attention_heads' in config:
                    params.append(f"- **Attention Heads:** {config['num_attention_heads']}")
                if 'intermediate_size' in config:
                    params.append(f"- **FFN Size:** {config['intermediate_size']}")
                if 'vocab_size' in config:
                    params.append(f"- **Vocab Size:** {config['vocab_size']}")
                if 'max_position_embeddings' in config:
                    params.append(f"- **Max Position:** {config['max_position_embeddings']}")
                if 'num_experts' in config:
                    params.append(f"- **MoE Experts:** {config['num_experts']}")
                if 'num_experts_per_tok' in config:
                    params.append(f"- **Experts per Token:** {config['num_experts_per_tok']}")

                for param in params:
                    f.write(f"{param}\n")
                f.write("\n")

            # Tokenizer section
            if model_data.get('tokenizer'):
                tokenizer = model_data['tokenizer']
                f.write("### Tokenizer\n\n")
                f.write(f"- **Vocab Size:** {tokenizer.get('vocab_size', 'unknown')}\n")
                f.write(f"- **Model Type:** {tokenizer.get('model_type', 'unknown')}\n")
                if tokenizer.get('special_tokens'):
                    f.write(f"- **Special Tokens:** {', '.join(tokenizer['special_tokens'])}\n")
                f.write("\n")

            # Tensor section
            if model_data.get('tensors'):
                tensors = model_data['tensors']
                f.write("### Tensor Analysis\n\n")
                f.write(f"- **Total Files:** {tensors.get('total_files', 0)}\n")
                f.write(f"- **Total Tensors:** {tensors.get('total_tensors', 0)}\n")
                f.write(f"- **Uncategorized:** {tensors.get('other_count', 0)}\n\n")

                if 'dtypes' in tensors:
                    f.write("#### Data Types\n\n")
                    for dtype, count in sorted(tensors['dtypes'].items()):
                        f.write(f"- **{dtype}:** {count}\n")
                    f.write("\n")

                if 'categories' in tensors:
                    f.write("#### Tensor Categories\n\n")
                    for cat, count in sorted(tensors['categories'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- **{cat}:** {count}\n")
                    f.write("\n")

            f.write("---\n\n")

        # Summary section
        f.write("## Summary\n\n")
        total_models = len(catalog)
        total_tensors = sum(m.get('tensors', {}).get('total_tensors', 0) for m in catalog.values())
        uncategorized = sum(m.get('tensors', {}).get('other_count', 0) for m in catalog.values())

        f.write(f"- **Total Models:** {total_models}\n")
        f.write(f"- **Total Tensors:** {total_tensors}\n")
        f.write(f"- **Uncategorized Tensors:** {uncategorized}\n")
        if total_tensors > 0:
            f.write(".1f")

        # Architecture patterns
        f.write("\n### Architecture Patterns\n\n")
        architectures = defaultdict(int)
        for model_data in catalog.values():
            if model_data.get('config'):
                arch = model_data['config'].get('model_type', 'unknown')
                architectures[arch] += 1

        for arch, count in sorted(architectures.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- **{arch}:** {count} models\n")

    print(f"Documentation generated: {doc_file}")

def analyze_config(model_path):
    """Extract config information from model directory"""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except:
        return {}

def analyze_tokenizer(model_path):
    """Extract tokenizer information"""
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.exists():
        return {}

    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer = json.load(f)
        return {
            'vocab_size': len(tokenizer.get('model', {}).get('vocab', {})),
            'special_tokens': list(tokenizer.get('added_tokens', {}).keys())[:10],  # first 10
            'model_type': tokenizer.get('model', {}).get('type', 'unknown')
        }
    except:
        return {}

def analyze_safetensors(model_path):
    """Analyze all safetensor files in model directory"""
    index_file = model_path / "model.safetensors.index.json"
    safetensor_files = list(model_path.glob("*.safetensors"))

    if index_file.exists():
        # Use index file for complete tensor list and categorization
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            if 'weight_map' in index_data:
                tensor_map = index_data['weight_map']
                total_tensors = len(tensor_map)

                # Sample a few files to get shape/dtype info for categorization
                categories = defaultdict(int)
                dtypes = defaultdict(int)
                samples_checked = 0

                for safetensor_file in safetensor_files[:3]:  # Check first 3 files
                    try:
                        with open(safetensor_file, 'rb') as f:
                            header_size_bytes = f.read(8)
                            header_size = struct.unpack('<Q', header_size_bytes)[0]
                            header_json = f.read(header_size).decode('utf-8')
                            header = json.loads(header_json)

                        for name, info in header.items():
                            if name == '__metadata__':
                                continue
                            dtype = info['dtype']
                            shape = info['shape']
                            category = classify_tensor(name, shape)

                            dtypes[dtype] += 1
                            categories[category.name] += 1

                        samples_checked += 1
                        if samples_checked >= 3:
                            break
                    except:
                        continue

                return {
                    'total_files': len(safetensor_files),
                    'total_tensors': total_tensors,
                    'categories': dict(categories),
                    'dtypes': dict(dtypes),
                    'other_count': categories.get('UNKNOWN', 0),
                    'source': 'index'
                }
        except Exception as e:
            print(f"Error reading index file: {e}")

    # Fallback to analyzing first file
    if not safetensor_files:
        return {}

    first_file = safetensor_files[0]
    try:
        with open(first_file, 'rb') as f:
            header_size_bytes = f.read(8)
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            header_json = f.read(header_size).decode('utf-8')
            header = json.loads(header_json)

        total_tensors = len(header) - 1  # exclude metadata
        categories = defaultdict(int)
        dtypes = defaultdict(int)

        for name, info in header.items():
            if name == '__metadata__':
                continue
            dtype = info['dtype']
            shape = info['shape']
            category = classify_tensor(name, shape)

            dtypes[dtype] += 1
            categories[category.name] += 1

        return {
            'total_files': len(safetensor_files),
            'total_tensors': total_tensors,
            'categories': dict(categories),
            'dtypes': dict(dtypes),
            'other_count': categories.get('UNKNOWN', 0),
            'source': 'first_file'
        }
    except Exception as e:
        return {}

def find_model_directories(base_path):
    """Recursively find all model directories"""
    models = []
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        if any(f.endswith('.safetensors') for f in files) or (root_path / 'config.json').exists():
            # Check if this is a leaf model directory (contains safetensors but parent doesn't)
            parent_has_safetensors = any((root_path.parent / f).exists() and f.endswith('.safetensors')
                                       for f in os.listdir(root_path.parent) if os.path.isfile(root_path.parent / f))
            if not parent_has_safetensors:
                models.append(root_path)
    return models

def main():
    models_dir = Path(r"D:\Models")

    print("COMPREHENSIVE MODEL CATALOG")
    print("=" * 60)

    model_dirs = find_model_directories(models_dir)
    print(f"Found {len(model_dirs)} model directories")

    catalog = {}

    for model_path in model_dirs:
        model_name = model_path.name
        print(f"\nAnalyzing: {model_name}")

        config = analyze_config(model_path)
        tokenizer = analyze_tokenizer(model_path)
        tensors = analyze_safetensors(model_path)

        catalog[model_name] = {
            'path': str(model_path),
            'config': config,
            'tokenizer': tokenizer,
            'tensors': tensors
        }

        # Print summary
        if config:
            arch = config.get('model_type', config.get('architectures', ['unknown'])[0] if config.get('architectures') else 'unknown')
            print(f"  Architecture: {arch}")
            if 'hidden_size' in config:
                print(f"  Model dim: {config['hidden_size']}")
            if 'num_hidden_layers' in config:
                print(f"  Layers: {config['num_hidden_layers']}")

        if tokenizer:
            print(f"  Vocab size: {tokenizer.get('vocab_size', 'unknown')}")

        if tensors:
            print(f"  Tensors: {tensors.get('total_tensors', 0)} ({tensors.get('other_count', 0)} uncategorized)")
            if 'categories' in tensors:
                top_cats = sorted(tensors['categories'].items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top categories: {', '.join(f'{k}:{v}' for k, v in top_cats)}")

    # Save catalog
    output_file = models_dir / "model_catalog.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"\nCatalog saved to: {output_file}")

    # Print summary stats
    print("\nSUMMARY STATISTICS")
    print("-" * 40)

    total_models = len(catalog)
    total_tensors = sum(m.get('tensors', {}).get('total_tensors', 0) for m in catalog.values())
    uncategorized = sum(m.get('tensors', {}).get('other_count', 0) for m in catalog.values())

    print(f"Total models: {total_models}")
    print(f"Total tensors: {total_tensors}")
    print(f"Uncategorized tensors: {uncategorized}")
    if total_tensors > 0:
        print(".1f")

    # Generate documentation
    generate_documentation(catalog, models_dir)

if __name__ == "__main__":
    main()