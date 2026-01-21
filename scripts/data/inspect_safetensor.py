#!/usr/bin/env python3
"""
Inspect safetensor files to understand their structure for streaming ingestion.
"""

import sys
import json
import struct
from pathlib import Path
import re
from enum import Enum

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

def inspect_safetensor(path: str):
    """Inspect a safetensor file without loading tensors into memory."""
    path = Path(path)
    
    if not path.exists():
        print(f"File not found: {path}")
        return
    
    file_size = path.stat().st_size
    print(f"\n=== Inspecting: {path.name} ===")
    print(f"File size: {file_size / (1024**3):.2f} GB ({file_size:,} bytes)")
    
    with open(path, 'rb') as f:
        # Read header size (first 8 bytes, little-endian uint64)
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        print(f"Header size: {header_size:,} bytes")
        
        # Read header JSON
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        
        # Separate metadata from tensors
        metadata = header.pop('__metadata__', {})
        
        print(f"Tensor count: {len(header)}")
        if metadata:
            print(f"Metadata keys: {list(metadata.keys())}")
        
        # Analyze tensors
        print(f"\n--- Tensor Analysis ---")

        dtypes = {}
        categories = {}
        sizes = []
        other_tensors = []

        for name, info in header.items():
            dtype = info['dtype']
            shape = info['shape']
            offsets = info['data_offsets']

            # Calculate element count
            numel = 1
            for dim in shape:
                numel *= dim

            # Calculate byte size
            byte_size = offsets[1] - offsets[0]

            # Categorize
            category = classify_tensor(name, shape)
            cat_name = category.name if category != TensorCategory.UNKNOWN else "OTHER"

            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            categories[cat_name] = categories.get(cat_name, 0) + 1
            sizes.append((name, numel, byte_size, shape, dtype, cat_name))

            # Collect OTHER tensors
            if category == TensorCategory.UNKNOWN:
                other_tensors.append((name, shape, dtype, byte_size))

        # Sort by size
        sizes.sort(key=lambda x: x[2], reverse=True)
        
        # Sort by size
        sizes.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nData types: {dtypes}")
        print(f"Categories: {categories}")

        total_data_size = sum(s[2] for s in sizes)
        print(f"Total tensor data: {total_data_size / (1024**3):.2f} GB")

        # Show OTHER tensors
        print(f"\n--- OTHER Tensors ({len(other_tensors)}) ---")
        for name, shape, dtype, byte_size in sorted(other_tensors, key=lambda x: x[3], reverse=True):
            print(f"  {name}: {shape} {dtype} ({byte_size/(1024**2):.1f} MB)")
        
        print(f"\n--- Top 20 Largest Tensors ---")
        print(f"{'Name':<60} {'Elements':>15} {'Size (MB)':>12} {'Shape':<30} {'Dtype':<8}")
        print("-" * 130)
        
        for name, numel, byte_size, shape, dtype, cat_name in sizes[:20]:
            name_short = name[:57] + "..." if len(name) > 60 else name
            shape_str = str(shape)[:27] + "..." if len(str(shape)) > 30 else str(shape)
            print(f"{name_short:<60} {numel:>15,} {byte_size/(1024**2):>12.2f} {shape_str:<30} {dtype:<8}")
        
        # Check for very large tensors
        print(f"\n--- Tensors > 100MB ---")
        large_count = 0
        for name, numel, byte_size, shape, dtype, cat_name in sizes:
            if byte_size > 100 * 1024 * 1024:
                large_count += 1
                print(f"  {name}: {byte_size/(1024**2):.1f} MB, shape={shape}")
        print(f"Total: {large_count} tensors > 100MB")
        
        # Memory requirements analysis
        print(f"\n--- Memory Requirements ---")
        element_sizes = {'F32': 4, 'F16': 2, 'BF16': 2, 'I32': 4, 'I64': 8}
        
        max_tensor_size = sizes[0][2] if sizes else 0
        print(f"Largest tensor: {max_tensor_size / (1024**3):.2f} GB ({sizes[0][0][:50]}...)")
        print(f"If loading to float32: {max_tensor_size * 2 / (1024**3):.2f} GB (for BF16/F16 -> F32)")
        
        # Streaming recommendation
        print(f"\n--- Streaming Recommendation ---")
        if max_tensor_size > 1024 * 1024 * 1024:  # > 1GB
            print("⚠️  REQUIRES STREAMING: Tensors > 1GB present")
            print("   Cannot load full tensors into memory for hashing")
            print("   Recommend: Hash directly from mmap in chunks")
        elif max_tensor_size > 100 * 1024 * 1024:  # > 100MB
            print("⚡ Large tensors present but manageable with streaming summaries")
        else:
            print("✅ All tensors < 100MB - standard processing OK")

def main():
    if len(sys.argv) < 2:
        # Default: inspect FLUX
        paths = [
            r"D:\Models\generation_models\models--black-forest-labs--FLUX.2-dev\snapshots\6aab690f8379b70adc89edfa6bb99b3537ba52a3\flux2-dev.safetensors",
            r"D:\Models\detection_models\RT-DETR-v1-R101\model.safetensors",
        ]
    else:
        paths = sys.argv[1:]
    
    for path in paths:
        try:
            inspect_safetensor(path)
        except Exception as e:
            print(f"Error inspecting {path}: {e}")

if __name__ == "__main__":
    main()
