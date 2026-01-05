#!/usr/bin/env python3
"""
Inspect safetensor files to understand their structure for streaming ingestion.
"""

import sys
import json
import struct
from pathlib import Path

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
        sizes = []
        largest = []
        
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
            
            dtypes[dtype] = dtypes.get(dtype, 0) + 1
            sizes.append((name, numel, byte_size, shape, dtype))
        
        # Sort by size
        sizes.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nData types: {dtypes}")
        
        total_data_size = sum(s[2] for s in sizes)
        print(f"Total tensor data: {total_data_size / (1024**3):.2f} GB")
        
        print(f"\n--- Top 20 Largest Tensors ---")
        print(f"{'Name':<60} {'Elements':>15} {'Size (MB)':>12} {'Shape':<30} {'Dtype':<8}")
        print("-" * 130)
        
        for name, numel, byte_size, shape, dtype in sizes[:20]:
            name_short = name[:57] + "..." if len(name) > 60 else name
            shape_str = str(shape)[:27] + "..." if len(str(shape)) > 30 else str(shape)
            print(f"{name_short:<60} {numel:>15,} {byte_size/(1024**2):>12.2f} {shape_str:<30} {dtype:<8}")
        
        # Check for very large tensors
        print(f"\n--- Tensors > 100MB ---")
        large_count = 0
        for name, numel, byte_size, shape, dtype in sizes:
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
