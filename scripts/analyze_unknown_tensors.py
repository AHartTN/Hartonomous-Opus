#!/usr/bin/env python3
"""
Analyze Unknown Tensors - Detailed examination of uncategorized tensors
to identify additional mathematical patterns for semantic classification.
"""

import sys
from pathlib import Path
from collections import Counter
from shape_based_extraction import ShapeBasedExtractor

def analyze_unknown_patterns(model_path):
    """Analyze unknown tensors in a specific model"""
    extractor = ShapeBasedExtractor()
    extractor.model_config = extractor.analyze_model_config(model_path / "config.json")
    extractor.tensor_info = extractor.load_tensor_metadata(model_path)

    if not extractor.tensor_info:
        print(f"No tensor data found for {model_path}")
        return {}

    classifications = extractor.classify_by_shape_patterns(extractor.tensor_info, extractor.model_config)

    unknown_tensors = classifications['unknown']
    print(f"\nAnalyzing {model_path.name}: {len(unknown_tensors)} unknown tensors")

    # Analyze unknown patterns
    shape_counter = Counter(tuple(t[1]) for t in unknown_tensors)
    name_patterns = Counter()

    for name, shape, dtype in unknown_tensors:
        # Extract common name patterns
        name_lower = name.lower()
        if 'norm' in name_lower or 'layernorm' in name_lower:
            name_patterns['normalization'] += 1
        elif 'bias' in name_lower:
            name_patterns['bias'] += 1
        elif 'embed' in name_lower or 'pos_embed' in name_lower:
            name_patterns['embedding'] += 1
        elif 'weight' in name_lower:
            name_patterns['weight'] += 1
        elif 'scale' in name_lower or 'shift' in name_lower:
            name_patterns['scale_shift'] += 1
        else:
            name_patterns['other'] += 1

    print(f"  Name patterns: {dict(name_patterns)}")
    print(f"  Top shapes: {shape_counter.most_common(10)}")

    return {
        'total_unknown': len(unknown_tensors),
        'shapes': dict(shape_counter),
        'name_patterns': dict(name_patterns),
        'sample_tensors': [(name, shape, dtype) for name, shape, dtype in unknown_tensors[:20]]
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_unknown_tensors.py <model_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    analysis = analyze_unknown_patterns(model_path)

    print("\n" + "="*60)
    print("UNKNOWN TENSOR ANALYSIS")
    print("="*60)

    print(f"Total unknown tensors: {analysis['total_unknown']}")

    print("\nNAME PATTERNS:")
    for pattern, count in sorted(analysis['name_patterns'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / analysis['total_unknown']) * 100
        print(".1f")
    print("\nTOP SHAPES:")
    for shape, count in sorted(analysis['shapes'].items(), key=lambda x: x[1], reverse=True)[:20]:
        pct = (count / analysis['total_unknown']) * 100
        print(".1f")
    print("\nSAMPLE UNKNOWN TENSORS:")
    for i, (name, shape, dtype) in enumerate(analysis['sample_tensors'], 1):
        print(f"  {i:2d}. {name}: {shape} ({dtype})")

if __name__ == "__main__":
    main()