#!/usr/bin/env python3
"""
Examine Actual Unknown Tensors - Read and analyze tensors that failed classification
"""

import sys
import json
from pathlib import Path
import struct

def read_safetensor_header(file_path):
    """Read safetensor header without numpy"""
    with open(file_path, 'rb') as f:
        # Read header size (8 bytes, little endian)
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]

        # Read header JSON
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)

        return header

def examine_unknown_tensors(model_name, model_path):
    """Examine tensors that were categorized as unknown"""
    print(f"\nEXAMINING UNKNOWN TENSORS IN {model_name}")
    print("=" * 60)

    # Load catalog to get unknown tensor info
    catalog_path = Path(r"D:\Models\model_catalog.json")
    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog = json.load(f)

    if model_name not in catalog:
        print(f"Model {model_name} not found in catalog")
        return

    model_data = catalog[model_name]
    if 'tensors' not in model_data or 'categories' not in model_data['tensors']:
        print("No tensor data in catalog")
        return

    categories = model_data['tensors']['categories']
    unknown_count = categories.get('UNKNOWN', 0)

    print(f"Model has {unknown_count} unknown tensors")
    print(f"Total tensors: {model_data['tensors']['total_tensors']}")
    print(f"Uncategorized: {model_data['tensors'].get('other_count', 0)}")

    # Find safetensor files
    safetensor_files = list(Path(model_path).glob("*.safetensors"))
    if not safetensor_files:
        print("No safetensor files found")
        return

    print(f"Found {len(safetensor_files)} safetensor files")

    # Read first file
    first_file = safetensor_files[0]
    print(f"\nReading: {first_file.name}")

    try:
        header = read_safetensor_header(first_file)

        # Group tensors by shape
        shape_groups = {}
        unknown_tensors = []

        for tensor_name, tensor_info in header.items():
            if tensor_name == '__metadata__':
                continue

            shape = tensor_info['shape']
            dtype = tensor_info['dtype']
            shape_tuple = tuple(shape)

            if shape_tuple not in shape_groups:
                shape_groups[shape_tuple] = []

            shape_groups[shape_tuple].append((tensor_name, dtype))

        # Find shapes that might be unknown (we can't run the classifier, but we can look at patterns)
        print(f"\nTENSOR SHAPE ANALYSIS:")
        print(f"Total unique shapes: {len(shape_groups)}")

        for shape, tensors in sorted(shape_groups.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
            count = len(tensors)
            pct = (count / len(header)) * 100 if '__metadata__' not in header else (count / (len(header) - 1)) * 100
            print(f"  {shape}: {count} tensors ({pct:.1f}%)")

            # Show some example names
            examples = tensors[:3]
            for name, dtype in examples:
                print(f"    - {name} ({dtype})")
            if len(tensors) > 3:
                print(f"    ... and {len(tensors) - 3} more")

        print("\nSHAPES THAT MIGHT BE UNKNOWN:")
        # Look for unusual patterns that might not match standard transformer rules
        for shape, tensors in shape_groups.items():
            # Skip standard shapes
            if len(shape) == 1 and shape[0] > 100:  # bias terms
                continue
            if len(shape) == 2 and shape[0] in [1024, 2048, 4096] and shape[1] in [1024, 2048, 4096]:  # attention matrices
                continue
            if len(shape) == 2 and shape[0] > shape[1] * 2:  # FFN layers
                continue

            # Show potentially unknown shapes
            if len(tensors) <= 10:  # Small groups might be unknown
                print(f"  {shape}: {len(tensors)} tensors")
                for name, dtype in tensors[:2]:
                    print(f"    - {name} ({dtype})")

    except Exception as e:
        print(f"Error reading safetensor: {e}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python examine_actual_unknown.py <model_name> <model_path>")
        print("\nExample models with unknown tensors:")
        print("  Grounding-DINO-Base D:\\Models\\hub\\Grounding-DINO-Base")
        print("  RT-DETR-v1-R101 D:\\Models\\hub\\RT-DETR-v1-R101")
        print("  6cfc37ec7edc35a0545c403f551ecdfa28133d72 D:\\Models\\hub\\models--nvidia--canary-qwen-2.5b\\snapshots\\6cfc37ec7edc35a0545c403f551ecdfa28133d72")
        sys.exit(1)

    model_name = sys.argv[1]
    model_path = sys.argv[2]

    examine_unknown_tensors(model_name, model_path)

if __name__ == "__main__":
    main()