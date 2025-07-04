#!/usr/bin/env python3
"""
Script to run inversion on all image-prompt pairs in df_data directory.
"""

import os
import subprocess
import sys
from pathlib import Path

def get_image_prompt_pairs(df_data_dir):
    """
    Get all matching image-prompt pairs from df_data directory.
    
    Args:
        df_data_dir (str): Path to df_data directory
        
    Returns:
        list: List of tuples (image_path, prompt_content, base_name)
    """
    pairs = []
    df_data_path = Path(df_data_dir)
    
    # Get all .txt files
    txt_files = list(df_data_path.glob("*.txt"))
    
    for txt_file in txt_files:
        # Get the base name (without extension)
        base_name = txt_file.stem
        
        # Check if corresponding .png file exists
        png_file = df_data_path / f"{base_name}.png"
        
        if png_file.exists():
            # Read the prompt from txt file
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    prompt_content = f.read().strip()
                
                pairs.append((str(png_file), prompt_content, base_name))
                print(f"Found pair: {base_name}")
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
        else:
            print(f"Warning: No matching PNG file for {txt_file}")
    
    return pairs

def run_inversion_for_pair(image_path, prompt, base_name, output_base_dir="outputs"):
    """
    Run inversion for a single image-prompt pair.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Prompt text
        base_name (str): Base name for output directory
        output_base_dir (str): Base directory for outputs
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create output directory for this specific run
    output_dir = os.path.join(output_base_dir, base_name)
    
    # Prepare the command
    cmd = [
        "python", "inversion.py",
        "--source_prompt", prompt,
        "--target_prompt", prompt,  # Using same prompt for inversion
        "--guidance", "1",
        "--source_img_dir", image_path,
        "--num_steps", "30",
        "--offload",
        "--inject", "0",
        "--name", "flux-dev",
        "--output_dir", output_dir,
        "--order", "1",
        "--run_mode", "inversion"
    ]
    
    print(f"\n{'='*60}")
    print(f"Processing: {base_name}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd=".", capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully processed {base_name}")
            return True
        else:
            print(f"❌ Error processing {base_name}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception while processing {base_name}: {e}")
        return False

def main():
    """Main function to process all df_data pairs."""
    
    # Set up paths
    df_data_dir = "df_data"
    output_base_dir = "df_data_outputs_smallnoise"
    
    # Check if df_data directory exists
    if not os.path.exists(df_data_dir):
        print(f"Error: {df_data_dir} directory not found!")
        sys.exit(1)
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all image-prompt pairs
    print("Scanning for image-prompt pairs...")
    pairs = get_image_prompt_pairs(df_data_dir)
    
    if not pairs:
        print("No image-prompt pairs found!")
        sys.exit(1)
    
    print(f"\nFound {len(pairs)} image-prompt pairs to process.")
    
    # Process each pair
    successful = 0
    failed = 0
    
    for i, (image_path, prompt, base_name) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Processing {base_name}...")
        
        if run_inversion_for_pair(image_path, prompt, base_name, output_base_dir):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_base_dir}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} pairs failed to process. Check the error messages above.")
    else:
        print(f"\n🎉 All pairs processed successfully!")

if __name__ == "__main__":
    main() 