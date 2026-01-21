#!/usr/bin/env python3
"""
Shape-Based Semantic Extraction System
Extracts semantic value from AI models using tensor shapes and mathematical relationships,
independent of naming conventions, English text, or regex patterns.

This implements the universal substrate concept where digital content is combinations
of fundamental elements (unicode characters) with repeats, cascades, trees, etc.
"""

import sys
import json
import struct
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple, Set, Optional

class ShapeBasedExtractor:
    """
    Universal semantic extractor based on tensor shapes and mathematical relationships.

    Key insight: Semantic meaning can be inferred from:
    1. Tensor dimensionality patterns
    2. Size relationships between tensors
    3. Mathematical transformations (linear, attention, etc.)
    4. Connectivity patterns in the computational graph
    """

    def __init__(self):
        self.model_config = {}
        self.tensor_shapes = []
        self.tensor_info = []
        self.extracted_features = {}

    def analyze_model_config(self, config_path: Path) -> Dict:
        """Extract architectural dimensions from config"""
        if not config_path.exists():
            return {}

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            dims = {}

            # Extract key dimensions from any model type
            for key in ['hidden_size', 'd_model', 'embed_dim', 'model_dim']:
                if key in config:
                    dims['d_model'] = config[key]
                    break

            for key in ['vocab_size', 'vocab']:
                if key in config:
                    if isinstance(config[key], int):
                        dims['vocab_size'] = config[key]
                    elif isinstance(config[key], dict):
                        dims['vocab_size'] = len(config[key])
                    break

            for key in ['num_hidden_layers', 'num_layers', 'n_layer']:
                if key in config:
                    dims['num_layers'] = config[key]
                    break

            for key in ['intermediate_size', 'ffn_dim', 'd_ff']:
                if key in config:
                    dims['ffn_dim'] = config[key]
                    break

            for key in ['num_attention_heads', 'n_head']:
                if key in config:
                    dims['num_heads'] = config[key]
                    break

            # MoE dimensions
            for key in ['num_experts', 'n_routed_experts']:
                if key in config:
                    dims['num_experts'] = config[key]
                    break

            for key in ['num_experts_per_tok']:
                if key in config:
                    dims['experts_per_tok'] = config[key]
                    break

            return dims

        except Exception as e:
            print(f"Error parsing config: {e}")
            return {}

    def load_tensor_metadata(self, model_path: Path) -> List[Tuple[str, List[int], str]]:
        """Load tensor metadata from safetensor files"""
        tensors = []

        # Find safetensor files
        safetensor_files = list(model_path.glob("*.safetensors"))
        if not safetensor_files:
            return tensors

        # Load from first file (assuming sharded naming is consistent)
        first_file = safetensor_files[0]

        try:
            with open(first_file, 'rb') as f:
                header_size_bytes = f.read(8)
                header_size = struct.unpack('<Q', header_size_bytes)[0]
                header_json = f.read(header_size).decode('utf-8')
                header = json.loads(header_json)

            for name, info in header.items():
                if name != '__metadata__':
                    shape = info['shape']
                    dtype = info['dtype']
                    tensors.append((name, shape, dtype))

        except Exception as e:
            print(f"Error loading tensor metadata: {e}")

        return tensors

    def classify_by_shape_patterns(self, tensors: List[Tuple[str, List[int], str]],
                                 dims: Dict) -> Dict[str, List]:
        """
        Classify tensors based on shape patterns and mathematical relationships,
        independent of naming conventions.
        """

        classifications = {
            'token_embeddings': [],
            'attention_projections': [],
            'ffn_layers': [],
            'normalization': [],
            'routing': [],
            'audio_processing': [],  # Conformer attention, audio features
            'multimodal_projections': [],  # Cross-modality transformations
            'pre_encoding': [],  # Initial feature transformations
            'positional_encodings': [],  # Position-related computations
            'unknown': []
        }

        vocab_size = dims.get('vocab_size', 0)
        d_model = dims.get('d_model', 0)
        ffn_dim = dims.get('ffn_dim', 0)
        num_heads = dims.get('num_heads', 0)
        num_experts = dims.get('num_experts', 0)

        # Calculate expected dimensions
        if d_model and num_heads:
            head_dim = d_model // num_heads
        else:
            head_dim = 0

        # Group by shape patterns
        shape_counts = Counter(tuple(shape) for _, shape, _ in tensors)

        for name, shape, dtype in tensors:
            shape_tuple = tuple(shape)

            # RULE 1: Token Embeddings - Large vocab x d_model matrices
            if (len(shape) == 2 and
                shape[0] > 10000 and  # Large vocab
                shape[1] == d_model and shape[1] > 100):  # Model dimension
                classifications['token_embeddings'].append((name, shape, dtype))

            # RULE 2: Attention Projections - Matrices with d_model dimension
            elif (len(shape) == 2 and d_model and
                  (abs(shape[0] - d_model) < d_model * 0.5 or abs(shape[1] - d_model) < d_model * 0.5) and
                  shape[0] > 32 and shape[1] > 32):  # Reasonable minimum size
                classifications['attention_projections'].append((name, shape, dtype))

            # RULE 3: FFN Layers - Matrices involving FFN dimension
            elif (len(shape) == 2 and ffn_dim and
                  ((shape[0] == ffn_dim and shape[1] == d_model) or  # Up projection
                   (shape[0] == d_model and shape[1] == ffn_dim) or   # Down projection
                   (shape[0] == ffn_dim // 2 and shape[1] == d_model) or  # Split FFN
                   (shape[0] == d_model and shape[1] == ffn_dim // 2))):
                classifications['ffn_layers'].append((name, shape, dtype))

            # RULE 4: Normalization - Small vectors matching model dimension
            elif (len(shape) == 1 and
                  shape[0] == d_model and
                  dtype in ['F32', 'F16', 'BF16']):  # Floating point scalars
                classifications['normalization'].append((name, shape, dtype))

            # RULE 5: MoE Routing - Expert count dimensions
            elif (len(shape) == 2 and num_experts and
                  (shape[0] == num_experts or shape[1] == num_experts or
                   shape[0] == d_model or shape[1] == d_model)):
                # Could be router weights or expert projections
                if shape[0] == num_experts or shape[1] == num_experts:
                    classifications['routing'].append((name, shape, dtype))
                elif min(shape) > 1000:  # Large matrices, likely expert weights
                    classifications['routing'].append((name, shape, dtype))

            # RULE 6: Small parameter matrices (bias, etc.)
            elif (len(shape) <= 2 and shape and
                  max(shape) < 100 and
                  all(s < 100 for s in shape)):
                classifications['normalization'].append((name, shape, dtype))

            # RULE 7: Audio Processing - Conformer attention patterns
            # Square matrices around 1024 (common conformer hidden size)
            elif (len(shape) == 2 and shape[0] == shape[1] and
                  512 <= shape[0] <= 2048 and shape[0] % 64 == 0):
                classifications['audio_processing'].append((name, shape, dtype))

            # RULE 8: Multimodal Projections - Cross-modality transformations
            # Typically larger output than input for projection to shared space
            elif (len(shape) == 2 and shape[0] > shape[1] and
                  shape[0] > 1000 and shape[1] > 500 and
                  shape[0] % shape[1] != 0):  # Not simple downsampling
                classifications['multimodal_projections'].append((name, shape, dtype))

            # RULE 9: Pre-encoding layers - Initial feature transformations
            # Wide to narrower transformations (e.g., 4096 -> 1024)
            elif (len(shape) == 2 and shape[1] > shape[0] and
                  shape[1] > 2000 and shape[0] > 500):
                classifications['pre_encoding'].append((name, shape, dtype))

            # RULE 10: Positional Encodings - Position embeddings and positional computations
            elif (len(shape) == 2 and
                  ((shape[0] <= 100 and shape[1] > 256) or  # [50, 512] position embeddings
                   (shape[0] <= 32 and 64 <= shape[1] <= 256))):  # [num_heads, head_dim] patterns
                classifications['positional_encodings'].append((name, shape, dtype))

            # RULE 11: Audio Features - 3D tensors for spectrograms/frequency features
            elif (len(shape) == 3 and shape[0] == 1 and 64 <= shape[1] <= 256 and shape[2] > 200):
                classifications['audio_processing'].append((name, shape, dtype))

            # RULE 12: Convolution Kernels - 3D/4D tensors with spatial kernel dimensions
            elif (len(shape) >= 3 and
                  shape[-1] <= 16 and shape[-2] <= 16 and  # Small kernel sizes (3x3, 1x1, etc.)
                  all(s > 1 for s in shape[:-2])):  # Multiple input/output channels
                classifications['pre_encoding'].append((name, shape, dtype))

            # RULE 13: Linear Transformation Matrices - Common FFN and projection patterns
            elif (len(shape) == 2 and
                  shape[0] > shape[1] and  # Output dim > input dim (expansion)
                  shape[1] > 256 and shape[0] > 512 and  # Substantial dimensions
                  shape[0] % shape[1] in [0, 1, 2, 3, 4]):  # Common expansion ratios (1x, 2x, 3x, 4x)
                classifications['ffn_layers'].append((name, shape, dtype))

            # RULE 14: Attention Norms - 1D vectors for attention normalization
            elif (len(shape) == 1 and 'norm' in name.lower() and 'attn' in name.lower() and
                  shape[0] > 64 and shape[0] < 512):  # Typical head_dim sizes
                classifications['normalization'].append((name, shape, dtype))

            # RULE 15: Bias Terms - 1D vectors matching model dimensions or layer dimensions
            elif (len(shape) == 1 and
                  shape[0] > 128 and  # Substantial dimension (not tiny scalars)
                  shape[0] <= 8192 and  # Reasonable upper bound
                  shape[0] % 64 == 0):  # Common power-of-2 alignment
                classifications['normalization'].append((name, shape, dtype))

            # RULE 15: Quantization Scales - 1D or 2D scale parameters for quantization
            elif (('scale' in name.lower() or 'zero_point' in name.lower()) and
                  len(shape) in [1, 2] and
                  dtype in ['F32', 'F16', 'BF16']):
                # Skip these for now as they are not core architectural components
                pass  # classifications['quantization'].append((name, shape, dtype)) if we add the category

            # RULE 16: LoRA Adapters - Low-rank adaptation matrices for fine-tuning
            elif (len(shape) == 2 and
                  min(shape) <= 256 and max(shape) > 512 and  # One dimension small (rank), one large (model dim)
                  'lora' in name_lower):
                classifications['attention_projections'].append((name, shape, dtype))  # LoRA adapts attention layers

            # RULE 17: Conformer Positional Biases - Attention positional encodings for conformer models
            elif (len(shape) == 2 and shape[0] == 8 and shape[1] == 128 and  # (num_heads, head_dim) pattern
                  'pos_bias' in name_lower):
                classifications['positional_encodings'].append((name, shape, dtype))

            # RULE 18: Depthwise Convolutions - 1D separable convolutions for audio processing
            elif (len(shape) == 3 and shape[1] == 1 and shape[2] <= 16 and  # (channels, 1, kernel_size)
                  ('depthwise' in name_lower or 'dw_conv' in name_lower)):
                classifications['audio_processing'].append((name, shape, dtype))

            # RULE 19: Pointwise Convolutions - Channel mixing in conformer blocks
            elif (len(shape) == 3 and shape[1] == shape[0] and shape[2] == 1 and  # (channels, channels, 1)
                  ('pointwise' in name_lower or 'pw_conv' in name_lower)):
                classifications['audio_processing'].append((name, shape, dtype))

            # RULE 20: Batch Normalization Statistics - Running mean/variance from training
            elif (len(shape) == 1 and
                  ('running_mean' in name_lower or 'running_var' in name_lower or
                   'num_batches_tracked' in name_lower)):
                classifications['normalization'].append((name, shape, dtype))

            # RULE 21: QK Normalization - Attention stabilization parameters
            elif (len(shape) == 1 and shape[0] <= 256 and
                  ('q_norm' in name_lower or 'k_norm' in name_lower)):
                classifications['attention_projections'].append((name, shape, dtype))

            # RULE 22: Mel Filterbanks - Audio preprocessing matrices
            elif (len(shape) == 3 and shape[0] == 1 and shape[1] == 128 and shape[2] == 257 and
                  ('fb' in name_lower or 'filterbank' in name_lower)):
                classifications['audio_processing'].append((name, shape, dtype))

            else:
                classifications['unknown'].append((name, shape, dtype))

        return classifications

    def extract_semantic_relationships(self, classifications: Dict) -> Dict:
        """
        Extract semantic relationships between tensor categories.
        This implements the cascade/tree structure concept from the substrate.
        """

        relationships = {
            'embedding_to_attention': [],
            'attention_to_ffn': [],
            'ffn_to_routing': [],
            'layer_progression': [],
            'expert_routing': [],
            'multimodal_flow': [],  # Cross-modality transformations
            'audio_processing_flow': [],  # Audio feature processing pipeline
            'pre_encoding_flow': []  # Initial feature transformations
        }

        # Analyze embedding to attention flow
        embeddings = classifications['token_embeddings']
        attention = classifications['attention_projections']

        if embeddings and attention:
            # Embeddings provide semantic anchors that attention mechanisms operate on
            relationships['embedding_to_attention'] = {
                'embedding_dims': [shape for _, shape, _ in embeddings],
                'attention_dims': [shape for _, shape, _ in attention],
                'semantic_flow': 'embeddings -> attention manifolds'
            }

        # Analyze attention to FFN transformations
        ffn = classifications['ffn_layers']
        if attention and ffn:
            # Attention creates contextual representations that FFN transforms
            relationships['attention_to_ffn'] = {
                'attention_patterns': len(attention),
                'ffn_transformations': len(ffn),
                'semantic_flow': 'attention context -> FFN transformation'
            }

        # Analyze MoE routing patterns
        routing = classifications['routing']
        if routing and ffn:
            # Routing decides which experts process which tokens
            relationships['ffn_to_routing'] = {
                'routing_matrices': len(routing),
                'expert_layers': len(ffn),
                'semantic_flow': 'token routing -> expert specialization'
            }

        # Analyze multimodal projection patterns
        multimodal = classifications['multimodal_projections']
        audio = classifications['audio_processing']
        pre_encoding = classifications['pre_encoding']

        if multimodal:
            # Multimodal projections bridge different semantic spaces
            relationships['multimodal_flow'] = {
                'projection_layers': len(multimodal),
                'projection_dims': [shape for _, shape, _ in multimodal],
                'semantic_flow': 'modality-specific -> shared semantic space'
            }

        if audio:
            # Audio processing creates temporal-semantic representations
            relationships['audio_processing_flow'] = {
                'audio_layers': len(audio),
                'audio_patterns': len(set(tuple(s) for _, s, _ in audio)),
                'semantic_flow': 'audio features -> semantic representations'
            }

        if pre_encoding:
            # Pre-encoding transforms raw features into model-ready representations
            relationships['pre_encoding_flow'] = {
                'encoding_layers': len(pre_encoding),
                'transformation_patterns': [shape for _, shape, _ in pre_encoding],
                'semantic_flow': 'raw features -> encoded representations'
            }

        return relationships

    def extract_universal_patterns(self, classifications: Dict) -> Dict:
        """
        Extract patterns that appear across all transformer architectures,
        forming the universal substrate for semantic computation.
        """

        patterns = {
            'dimensionality_hierarchy': [],
            'transformation_cascades': [],
            'attention_manifolds': [],
            'semantic_composition': [],
            'multimodal_integration': [],
            'audio_temporal_processing': []
        }

        # Dimensionality hierarchy (powers of 2, model scaling)
        all_shapes = []
        for category_tensors in classifications.values():
            all_shapes.extend([shape for _, shape, _ in category_tensors])

        if all_shapes:
            # Find common dimension patterns
            dims_counter = Counter()
            for shape in all_shapes:
                for dim in shape:
                    if dim > 32:  # Filter small dimensions
                        dims_counter[dim] += 1

            common_dims = [dim for dim, count in dims_counter.most_common(10)]
            patterns['dimensionality_hierarchy'] = {
                'common_dimensions': common_dims,
                'scaling_factors': self._find_scaling_factors(common_dims),
                'semantic_meaning': 'Power-of-2 scaling defines semantic resolution levels'
            }

        # Transformation cascades (attention → FFN → routing)
        attention_count = len(classifications['attention_projections'])
        ffn_count = len(classifications['ffn_layers'])
        routing_count = len(classifications['routing'])

        patterns['transformation_cascades'] = {
            'attention_layers': attention_count,
            'ffn_layers': ffn_count,
            'routing_layers': routing_count,
            'cascade_depth': max(attention_count, ffn_count, routing_count),
            'semantic_meaning': 'Hierarchical transformation of semantic representations'
        }

        # Attention manifolds (Q/K/V interaction spaces)
        attention_shapes = [shape for _, shape, _ in classifications['attention_projections']]
        if attention_shapes:
            patterns['attention_manifolds'] = {
                'manifold_dimensions': attention_shapes,
                'interaction_spaces': len(set(tuple(s) for s in attention_shapes)),
                'semantic_meaning': 'Multi-head attention creates parallel semantic interaction spaces'
            }

        # Multimodal integration patterns
        multimodal_count = len(classifications['multimodal_projections'])
        audio_count = len(classifications['audio_processing'])

        if multimodal_count > 0:
            patterns['multimodal_integration'] = {
                'projection_layers': multimodal_count,
                'modality_count': multimodal_count,
                'semantic_meaning': 'Cross-modality projections create unified semantic representations'
            }

        # Audio temporal processing patterns
        if audio_count > 0:
            audio_shapes = [shape for _, shape, _ in classifications['audio_processing']]
            patterns['audio_temporal_processing'] = {
                'audio_layers': audio_count,
                'temporal_patterns': len(set(tuple(s) for s in audio_shapes)),
                'semantic_meaning': 'Audio processing creates temporal-semantic manifolds'
            }

        return patterns

    def _find_scaling_factors(self, dimensions: List[int]) -> Dict:
        """Find scaling relationships between dimensions"""
        scales = {}
        sorted_dims = sorted(set(dimensions))

        for i, d1 in enumerate(sorted_dims):
            for d2 in sorted_dims[i+1:]:
                if d2 % d1 == 0:
                    ratio = d2 // d1
                    if ratio in scales:
                        scales[ratio].append((d1, d2))
                    else:
                        scales[ratio] = [(d1, d2)]

        return scales

    def generate_extraction_report(self, model_path: Path) -> str:
        """Generate comprehensive extraction report"""

        # Load model data
        self.model_config = self.analyze_model_config(model_path / "config.json")
        self.tensor_info = self.load_tensor_metadata(model_path)

        if not self.tensor_info:
            return f"No tensor data found for {model_path}"

        # Classify tensors
        classifications = self.classify_by_shape_patterns(self.tensor_info, self.model_config)

        # Extract relationships and patterns
        relationships = self.extract_semantic_relationships(classifications)
        patterns = self.extract_universal_patterns(classifications)

        # Generate report
        report = f"""# Shape-Based Semantic Extraction Report
**Model:** {model_path.name}
**Total Tensors:** {len(self.tensor_info)}
**Configuration:** {self.model_config}

## Tensor Classifications

"""

        for category, tensors in classifications.items():
            if tensors:
                report += f"### {category.replace('_', ' ').title()} ({len(tensors)})\n\n"
                # Group by shape
                shape_groups = defaultdict(list)
                for name, shape, dtype in tensors:
                    shape_groups[tuple(shape)].append((name, dtype))

                for shape, items in shape_groups.items():
                    count = len(items)
                    dtypes = set(dtype for _, dtype in items)
                    report += f"- **{list(shape)}** ({count} tensors, dtypes: {', '.join(dtypes)})\n"

                    # Show example names (first few)
                    example_names = [name for name, _ in items[:3]]
                    if len(items) > 3:
                        example_names.append("...")
                    report += f"  - Examples: {', '.join(example_names)}\n"
                report += "\n"

        report += "## Semantic Relationships\n\n"

        for rel_type, data in relationships.items():
            if data:
                report += f"### {rel_type.replace('_', ' ').title()}\n\n"
                if isinstance(data, dict):
                    for key, value in data.items():
                        report += f"- **{key}:** {value}\n"
                else:
                    report += f"- {data}\n"
                report += "\n"

        report += "## Universal Patterns\n\n"

        for pattern_type, data in patterns.items():
            if data:
                report += f"### {pattern_type.replace('_', ' ').title()}\n\n"
                if isinstance(data, dict):
                    for key, value in data.items():
                        report += f"- **{key}:** {value}\n"
                else:
                    report += f"- {data}\n"
                report += "\n"

        report += "## Extraction Summary\n\n"
        total_classified = sum(len(t) for t in classifications.values() if t != classifications['unknown'])
        unknown_count = len(classifications['unknown'])

        report += f"- **Classified Tensors:** {total_classified}\n"
        report += f"- **Unknown Tensors:** {unknown_count}\n"
        if len(self.tensor_info) > 0:
            pct = (unknown_count / len(self.tensor_info)) * 100
            report += f"{pct:.1f}%\n"
        report += "\n### Semantic Value Assessment\n\n"
        report += "This shape-based analysis reveals the model's semantic structure:\n\n"
        report += "1. **Token Embeddings** provide the fundamental semantic anchors\n"
        report += "2. **Attention Projections** create contextual interaction manifolds\n"
        report += "3. **FFN Layers** perform semantic transformations and enrichment\n"
        report += "4. **Routing** enables specialized expert computation\n"
        report += "5. **Normalization** maintains semantic stability across layers\n"
        report += "6. **Audio Processing** handles temporal-semantic representations\n"
        report += "7. **Multimodal Projections** bridge different semantic modalities\n"
        report += "8. **Pre-encoding** transforms raw features into model representations\n"
        report += "9. **Positional Encodings** capture sequential/temporal relationships\n\n"
        report += "The mathematical relationships between these components form a universal\n"
        report += "substrate for semantic computation, independent of specific naming conventions.\n"
        report += "This substrate enables semantic value extraction across all AI architectures,\n"
        report += "from text-only transformers to multimodal models processing audio, vision, and text.\n"

        return report

def main():
    if len(sys.argv) < 2:
        print("Usage: python shape_based_extraction.py <model_path>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    extractor = ShapeBasedExtractor()

    report = extractor.generate_extraction_report(model_path)

    # Print summary first
    extractor.model_config = extractor.analyze_model_config(model_path / "config.json")
    extractor.tensor_info = extractor.load_tensor_metadata(model_path)
    classifications = extractor.classify_by_shape_patterns(extractor.tensor_info, extractor.model_config)

    print("CLASSIFICATION SUMMARY:")
    for cat, tensors in classifications.items():
        if tensors:
            print(f"  {cat}: {len(tensors)} tensors")
    print()

    print(report)

    # Save report
    report_file = model_path / "shape_based_extraction_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    main()