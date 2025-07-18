#!/usr/bin/env python3
"""
Script to split captions.txt into 8 chunks and save them to different files.
"""

import os
import math

def split_captions(input_file="diffusionDB_enhanced_150k.txt", num_chunks=7, output_prefix="src/examples/captions/diffusionDB_enhanced_150k_chunk"):
    """
    Split captions into chunks and save to separate files.
    
    Args:
        input_file: Path to input captions file
        num_chunks: Number of chunks to create
        output_prefix: Prefix for output files
    """
    
    # Read all captions
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        captions = [line.strip() for line in f.readlines() if line.strip()]
    
    total_captions = len(captions)
    print(f"Total captions: {total_captions}")
    
    # Calculate chunk size
    chunk_size = math.ceil(total_captions / num_chunks)
    print(f"Chunk size: {chunk_size}")
    
    # Split and save chunks
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_captions)
        
        chunk_captions = captions[start_idx:end_idx]
        
        # Create output filename
        output_file = f"{output_prefix}_{i+1}.txt"
        
        # Save chunk to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for caption in chunk_captions:
                f.write(caption + '\n')
        
        print(f"Chunk {i+1}: {len(chunk_captions)} captions saved to '{output_file}' (lines {start_idx+1}-{end_idx})")
    
    print(f"\nSuccessfully split {total_captions} captions into {num_chunks} chunks!")

if __name__ == "__main__":
    split_captions() 