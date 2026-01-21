#!/usr/bin/env python3
"""
Analyze uncategorized tensors from safetensor files to identify missing patterns
"""
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter

def load_safetensor_metadata(path):
    """Load tensor metadata from safetensor file header"""
    with open(path, 'rb') as f:
        # Read header size (first 8 bytes, little-endian)
        header_size = int.from_bytes(f.read(8), byteorder='little')

        # Read header JSON
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))

    # Extract tensor metadata
    tensors = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        tensors[name] = {
            'shape': meta['shape'],
            'dtype': meta['dtype']
        }

    return tensors

# Simplified classification logic (mirrors C++ logic)
def classify_tensor(name, shape):
    """Classify tensor based on name patterns (simplified from C++)"""
    lower = name.lower()

    # Quantization artifacts
    if 'scale' in lower or 'zero_point' in lower:
        return 'QUANTIZATION'

    # Object detection
    if 'query_position_embed' in lower or 'object_queries' in lower:
        return 'OBJECT_QUERY'
    if 'class_labels_classifier' in lower or 'class_embed' in lower:
        return 'CLASS_HEAD'
    if 'bbox_predictor' in lower or 'bbox_embed' in lower:
        return 'BBOX_HEAD'

    # 2D positional
    if 'row_embed' in lower or 'column_embed' in lower or 'image_pos_embed' in lower:
        return 'POSITION_EMBEDDING_2D'

    # MoE
    is_moe = 'expert' in lower or 'moe' in lower
    if is_moe and ('router' in lower or ('gate' in lower and 'proj' not in lower)):
        return 'MOE_ROUTER'
    if 'shared_expert' in lower:
        return 'MOE_SHARED_EXPERT'
    if is_moe and ('up_proj' in lower or 'gate_proj' in lower or '.w1' in lower):
        return 'MOE_EXPERT_UP'
    if is_moe and ('down_proj' in lower or '.w2' in lower):
        return 'MOE_EXPERT_DOWN'
    if is_moe and '.w3' in lower:
        return 'MOE_EXPERT_GATE'
    if is_moe:
        return 'MOE_EXPERT'

    # Vision
    if 'vision_tower' in lower or 'visual_encoder' in lower or 'vision_model' in lower:
        return 'VISION_FEATURE'
    if 'image_projection' in lower or 'image_proj' in lower or 'visual_projection' in lower:
        return 'VISION_PROJECTION'
    if 'backbone' in lower and 'conv' in lower:
        return 'CONV_KERNEL'

    # Embeddings
    if 'embed' in lower:
        if 'position' in lower or 'pos_embed' in lower:
            return 'POSITION_EMBEDDING'
        if 'patch' in lower:
            return 'PATCH_EMBEDDING'
        if 'shared' in lower or 'token' in lower or 'wte' in lower or 'word_embed' in lower:
            return 'TOKEN_EMBEDDING'
        if len(shape) == 2 and shape[0] > 1000:
            return 'TOKEN_EMBEDDING'

    # Logit head
    if 'lm_head' in lower or 'logits_bias' in lower or 'output_projection' in lower:
        return 'LOGIT_HEAD'

    # Attention
    if 'attn' in lower or 'attention' in lower:
        if 'encoder_attn' in lower or 'cross_attn' in lower:
            return 'CROSS_ATTENTION'
        if 'q_proj' in lower or ('query' in lower and 'weight' in lower):
            return 'ATTENTION_QUERY'
        if 'k_proj' in lower or 'key' in lower:
            return 'ATTENTION_KEY'
        if 'v_proj' in lower or 'value' in lower:
            return 'ATTENTION_VALUE'
        if 'out_proj' in lower or 'o_proj' in lower:
            return 'ATTENTION_OUTPUT'
        if 'qkv' in lower:
            return 'ATTENTION_QUERY'

    # FFN
    if 'fc1' in lower or ('up_proj' in lower and 'expert' not in lower) or 'wi_0' in lower or 'wi_1' in lower:
        return 'FFN_UP'
    if 'gate_proj' in lower and 'expert' not in lower:
        return 'FFN_GATE'
    if 'fc2' in lower or ('down_proj' in lower and 'expert' not in lower) or 'wo' in lower:
        return 'FFN_DOWN'

    # Normalization
    if 'norm' in lower or 'layernorm' in lower or 'layer_norm' in lower:
        if 'rms' in lower:
            return 'RMS_NORM'
        return 'LAYER_NORM'

    # Convolution
    if 'conv' in lower:
        return 'CONV_KERNEL'

    # Modality projections
    if 'proj' in lower and ('image' in lower or 'text' in lower or 'visual' in lower):
        return 'MODALITY_PROJECTION'

    return 'UNKNOWN'

def analyze_model(model_path):
    """Analyze a model and report uncategorized tensors"""
    model_path = Path(model_path)

    # Find safetensor files
    safetensor_files = list(model_path.glob('*.safetensors'))
    if not safetensor_files:
        print(f"No .safetensors files found in {model_path}")
        return

    print(f"\n=== Analyzing {model_path.name} ===")
    print(f"Found {len(safetensor_files)} safetensor file(s)\n")

    # Load all tensors
    all_tensors = {}
    for sf in safetensor_files:
        all_tensors.update(load_safetensor_metadata(sf))

    print(f"Total tensors: {len(all_tensors)}\n")

    # Classify
    categorized = defaultdict(list)
    for name, meta in all_tensors.items():
        cat = classify_tensor(name, meta['shape'])
        categorized[cat].append((name, meta))

    # Print summary
    print("CATEGORIZATION SUMMARY:")
    for cat in sorted(categorized.keys()):
        if cat != 'UNKNOWN':
            print(f"  {cat:30s}: {len(categorized[cat]):4d}")

    unknown_count = len(categorized['UNKNOWN'])
    print(f"  {'UNKNOWN':30s}: {unknown_count:4d}\n")

    if unknown_count == 0:
        print("All tensors successfully categorized!")
        return

    # Analyze unknown patterns
    print(f"=== UNKNOWN TENSORS ({unknown_count} total) ===\n")

    # Group by common patterns
    patterns = defaultdict(list)
    for name, meta in categorized['UNKNOWN']:
        # Extract pattern (everything except final weight/bias)
        parts = name.split('.')
        if len(parts) > 1 and parts[-1] in ['weight', 'bias']:
            pattern = '.'.join(parts[:-1])
        else:
            pattern = name
        patterns[pattern].append((name, meta['shape']))

    # Print patterns sorted by frequency
    print("Common patterns (grouped):")
    for pattern, tensors in sorted(patterns.items(), key=lambda x: -len(x[1]))[:20]:
        print(f"\n  Pattern: {pattern}")
        print(f"  Count: {len(tensors)}")
        for name, shape in tensors[:3]:  # Show first 3 examples
            print(f"    - {name} {shape}")
        if len(tensors) > 3:
            print(f"    ... and {len(tensors) - 3} more")

    # Analyze name components
    print("\n\nMost common keywords in unknown tensors:")
    keywords = Counter()
    for name, _ in categorized['UNKNOWN']:
        parts = name.lower().replace('_', '.').split('.')
        keywords.update(parts)

    for keyword, count in keywords.most_common(20):
        if len(keyword) > 2:  # Skip short words
            print(f"  {keyword:30s}: {count:4d}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_uncategorized.py <model_directory>")
        print("\nExample:")
        print("  python analyze_uncategorized.py D:\\Models\\detection_models\\Conditional-DETR-R50")
        sys.exit(1)

    analyze_model(sys.argv[1])
