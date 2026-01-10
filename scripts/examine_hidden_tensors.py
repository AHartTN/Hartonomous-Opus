#!/usr/bin/env python3
"""
Examine Hidden Tensors - Find tensors not mentioned in model configs
that represent undocumented architectural components.
"""

import sys
from pathlib import Path
from collections import defaultdict
from shape_based_extraction import ShapeBasedExtractor

def examine_hidden_tensors(model_path):
    """Examine tensors that might represent undocumented architecture"""

    extractor = ShapeBasedExtractor()
    extractor.model_config = extractor.analyze_model_config(model_path / "config.json")
    extractor.tensor_info = extractor.load_tensor_metadata(model_path)

    if not extractor.tensor_info:
        print(f"No tensor data found for {model_path}")
        return

    # Get all tensor names from safetensors
    tensor_names = {name for name, _, _ in extractor.tensor_info}

    # Get documented dimensions from config
    documented_dims = set()
    if extractor.model_config:
        for key, value in extractor.model_config.items():
            if isinstance(value, (int, float)) and value > 1:
                documented_dims.add(int(value))
            elif isinstance(value, list):
                documented_dims.update(int(v) for v in value if isinstance(v, (int, float)) and v > 1)

    # Classify tensors
    classifications = extractor.classify_by_shape_patterns(extractor.tensor_info, extractor.model_config)

    print(f"\n[*] EXAMINING HIDDEN TENSORS in {model_path.name}")
    print("="*60)

    print(f"[+] Total tensors: {len(tensor_names)}")
    print(f"[+] Documented dimensions in config: {sorted(documented_dims)}")

    # Find tensors with undocumented dimensions
    undocumented_dims = set()
    for name, shape, dtype in extractor.tensor_info:
        for dim in shape:
            if isinstance(dim, int) and dim > 1 and dim not in documented_dims:
                undocumented_dims.add(dim)

    print(f"[+] Undocumented dimensions found: {sorted(undocumented_dims)}")

    # Analyze remaining unknown tensors for patterns
    unknown_tensors = classifications['unknown']
    print(f"\n[?] {len(unknown_tensors)} UNKNOWN TENSORS REMAINING:")

    # Group by semantic patterns
    semantic_patterns = defaultdict(list)

    for name, shape, dtype in unknown_tensors:
        name_lower = name.lower()

        # Categorize by naming patterns
        if any(x in name_lower for x in ['gamma', 'beta', 'moving', 'variance']):
            semantic_patterns['normalization_stats'].append((name, shape, dtype))
        elif 'embedding' in name_lower or 'embed' in name_lower:
            semantic_patterns['embeddings'].append((name, shape, dtype))
        elif any(x in name_lower for x in ['scale', 'shift']):
            semantic_patterns['scale_shift'].append((name, shape, dtype))
        elif 'kernel' in name_lower or 'conv' in name_lower:
            semantic_patterns['convolutional'].append((name, shape, dtype))
        elif any(x in name_lower for x in ['attention', 'attn']):
            semantic_patterns['attention_mechanisms'].append((name, shape, dtype))
        elif 'pool' in name_lower:
            semantic_patterns['pooling'].append((name, shape, dtype))
        elif len(shape) == 1 and shape[0] > 1000:
            semantic_patterns['large_vectors'].append((name, shape, dtype))
        elif len(shape) == 2 and shape[0] > shape[1] * 3:
            semantic_patterns['expansions'].append((name, shape, dtype))
        else:
            semantic_patterns['other'].append((name, shape, dtype))

    # Report findings
    for category, tensors in semantic_patterns.items():
        if tensors:
            print(f"\n[*] {category.upper().replace('_', ' ')} ({len(tensors)} tensors):")
            for name, shape, dtype in tensors[:5]:  # Show first 5 examples
                print(f"   - {name}: {shape} ({dtype})")
            if len(tensors) > 5:
                print(f"   ... and {len(tensors) - 5} more")

    # Look for architectural patterns not in config
    print(f"\n[+] POTENTIAL HIDDEN ARCHITECTURE:")

    # Find large unexplained matrices
    large_matrices = [(name, shape, dtype) for name, shape, dtype in unknown_tensors
                     if len(shape) == 2 and shape[0] * shape[1] > 1000000]  # > 1M parameters
    if large_matrices:
        print(f"   • Large matrices (>1M params): {len(large_matrices)} found")
        for name, shape, dtype in large_matrices[:3]:
            params = shape[0] * shape[1]
            print(f"     - {name}: {shape} = {params:,} params ({dtype})")

    # Find unusual dimensionalities
    unusual_dims = set()
    for name, shape, dtype in unknown_tensors:
        for dim in shape:
            if isinstance(dim, int) and dim > 10000:  # Very large dimensions
                unusual_dims.add(dim)

    if unusual_dims:
        print(f"   • Unusual large dimensions: {sorted(unusual_dims)}")
        print("     These may represent specialized embedding spaces not documented in config")

    return {
        'undocumented_dims': undocumented_dims,
        'unknown_patterns': dict(semantic_patterns),
        'large_matrices': large_matrices
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python examine_hidden_tensors.py <model_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    examine_hidden_tensors(model_path)

if __name__ == "__main__":
    main()