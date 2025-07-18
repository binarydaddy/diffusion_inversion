#!/usr/bin/env python3
"""
Script to enhance prompts from parquet file using PromptGenerator.
Each enhanced prompt consists of 1-3 sentences.
Uses multiprocessing with batch processing (200 prompts per API call).
"""

import os
import time
import pandas as pd
from prompt_generator import PromptGenerator
from typing import List, Tuple, Dict
import re
from multiprocessing import Pool, Manager, Lock
import signal
import sys
import json

class ParquetPromptEnhancer:
    def __init__(self):
        self.parquet_file = "dataset/metadata-large.parquet"
        self.unique_prompts_file = "unique_parquet_prompts.txt"
        self.output_file = "diffusionDB_captions.txt"
        self.temp_file_prefix = "temp_enhanced_prompts"
        self.num_workers = 10
        self.batch_size = 100  # Process 100 prompts per API call
        self.save_interval = 1000  # Save progress every 1000 prompts (10 batches)
        self.min_prompt_length = 40  # Minimum prompt length to include
        
    def extract_unique_prompts(self) -> List[str]:
        """Extract unique prompts from parquet file and save to file."""
        print(f"Loading prompts from {self.parquet_file}...")
        df = pd.read_parquet(self.parquet_file)
        
        if 'prompt' not in df.columns:
            raise ValueError("Parquet file does not contain 'prompt' column")
        
        # Get all prompts and drop NaN values
        all_prompts = df['prompt'].dropna()
        print(f"Total prompts (including duplicates): {len(all_prompts)}")
        
        # Get unique prompts
        unique_prompts = all_prompts.unique().tolist()
        print(f"Unique prompts (before filtering): {len(unique_prompts)}")
        print(f"Duplicates removed: {len(all_prompts) - len(unique_prompts)}")
        
        # Filter out prompts with less than minimum length
        filtered_prompts = [prompt for prompt in unique_prompts if len(prompt) >= self.min_prompt_length]
        filtered_count = len(unique_prompts) - len(filtered_prompts)
        
        print(f"Prompts with less than {self.min_prompt_length} characters removed: {filtered_count}")
        print(f"Final unique prompts (after filtering): {len(filtered_prompts)}")
        
        # Save filtered unique prompts to file
        with open(self.unique_prompts_file, 'w', encoding='utf-8') as f:
            for prompt in filtered_prompts:
                f.write(prompt + '\n')
        
        print(f"Saved filtered unique prompts to {self.unique_prompts_file}")
        return filtered_prompts
    
    def load_unique_prompts(self) -> List[str]:
        """Load unique prompts from file if it exists, otherwise extract from parquet."""
        if os.path.exists(self.unique_prompts_file):
            print(f"Loading unique prompts from {self.unique_prompts_file}...")
            with open(self.unique_prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} unique prompts from file")
            return prompts
        else:
            print(f"Unique prompts file not found. Extracting from parquet file...")
            return self.extract_unique_prompts()
    
    def create_batch_prompt(self, prompts: List[str]) -> str:
        """Create a batch prompt for multiple prompts."""
        prompt_list = "\n".join([f"{i+1}. {prompt}" for i, prompt in enumerate(prompts)])
        return prompt_list
    
    def parse_batch_response(self, response: str, original_prompts: List[str]) -> List[str]:
        """Parse the batch response and extract enhanced prompts."""
        enhanced_prompts = []
        
        # Try to parse the response as numbered list
        lines = response.strip().split('\n')
        current_prompt = []
        current_number = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number
            match = re.match(r'^(\d+)\.\s*(.*)', line)
            if match:
                # Save previous prompt if exists
                if current_prompt:
                    enhanced_prompts.append(' '.join(current_prompt))
                    current_prompt = []
                
                # Start new prompt
                number = int(match.group(1))
                content = match.group(2).strip()
                if content:
                    current_prompt = [content]
                current_number = number
            else:
                # Continuation of current prompt
                if line:
                    current_prompt.append(line)
        
        # Don't forget the last prompt
        if current_prompt:
            enhanced_prompts.append(' '.join(current_prompt))
        
        # If parsing failed or we got fewer prompts than expected, 
        # fall back to original prompts for missing ones
        while len(enhanced_prompts) < len(original_prompts):
            enhanced_prompts.append(original_prompts[len(enhanced_prompts)])
        
        # Ensure we don't have more enhanced prompts than originals
        enhanced_prompts = enhanced_prompts[:len(original_prompts)]
        
        return enhanced_prompts
    
    def enhance_prompt_batch_worker(self, args: Tuple[int, List[Tuple[int, str]], int]) -> List[Tuple[int, str, str]]:
        """
        Worker function to enhance a batch of prompts.
        Returns list of (index, original_prompt, enhanced_prompt)
        """
        batch_id, prompt_batch, worker_id = args
        
        # Create a new PromptGenerator instance for this worker
        prompt_generator = PromptGenerator()
        
        indices = [item[0] for item in prompt_batch]
        prompts = [item[1] for item in prompt_batch]
        
        try:
            # Create batch prompt
            batch_prompt_text = self.create_batch_prompt(prompts)
            
            chat_prompt = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a professional AI artist specializing in creating high-quality diffusion prompts for FLUX model. \
                            Your task is to enhance the given prompts into more detailed and visually rich descriptions. \
                            Each enhanced prompt should be 1-3 sentences long and describe a vivid, detailed scene. \
                            Make each prompt specific enough to generate high-quality images while maintaining the original concept. \
                            \
                            IMPORTANT: You will receive multiple prompts numbered 1, 2, 3, etc. \
                            You MUST return the enhanced prompts in the SAME numbered format, one per line. \
                            Each enhanced prompt should start with its number followed by a period and space (e.g., '1. Enhanced prompt here'). \
                            Do not add any other prefixes. Just provide the number and the enhanced prompt. \
                            Try to use diverse words and phrases to describe each scene. \
                            Do not use the same words or phrases repeatedly across different prompts."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Enhance each of these prompts into 1-3 sentences while maintaining their core concepts:\n\n{batch_prompt_text}"
                        }
                    ]
                },
            ]
            
            completion = prompt_generator.client.chat.completions.create(
                model=prompt_generator.deployment,
                messages=chat_prompt,
                max_tokens=300 * len(prompts),  # Enough for 1-3 sentences per prompt
                temperature=0.8,  # Balanced creativity
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                stop=None,
                stream=False
            )
            
            response = completion.choices[0].message.content.strip()
            
            # Parse the batch response
            enhanced_prompts = self.parse_batch_response(response, prompts)
            
            # Create results
            results = []
            for i, (idx, original_prompt) in enumerate(prompt_batch):
                if i < len(enhanced_prompts):
                    enhanced = enhanced_prompts[i]
                    # Clean up any remaining number prefixes
                    enhanced = re.sub(r'^\d+\.\s*', '', enhanced)
                    results.append((idx, original_prompt, enhanced))
                else:
                    # Fallback to original if something went wrong
                    results.append((idx, original_prompt, original_prompt))
            
            print(f"Worker {worker_id}: Successfully enhanced batch {batch_id} ({len(results)} prompts)")
            return results
            
        except Exception as e:
            print(f"Worker {worker_id}: Error enhancing batch {batch_id}: {str(e)}")
            # Return original prompts if enhancement fails
            return [(idx, prompt, prompt) for idx, prompt in prompt_batch]
    
    def save_progress(self, results: List[Tuple[int, str, str]], filename: str):
        """Save current results to file."""
        # Sort results by index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])
        
        with open(filename, 'w', encoding='utf-8') as f:
            for _, _, enhanced_prompt in sorted_results:
                f.write(enhanced_prompt + '\n')
        
        print(f"Saved {len(results)} prompts to {filename}")
    
    def enhance_all_prompts(self):
        """Enhance all prompts using multiprocessing with batch processing."""
        prompts = self.load_unique_prompts()
        total_prompts = len(prompts)
        
        print(f"\nStarting enhancement of {total_prompts} prompts")
        print(f"Using {self.num_workers} workers")
        print(f"Batch size: {self.batch_size} prompts per API call")
        print(f"Saving progress every {self.save_interval} prompts")
        
        # Prepare batches
        batches = []
        for i in range(0, total_prompts, self.batch_size):
            batch_prompts = []
            for j in range(i, min(i + self.batch_size, total_prompts)):
                batch_prompts.append((j, prompts[j]))
            batches.append(batch_prompts)
        
        print(f"Total batches: {len(batches)}")
        
        # Prepare work items
        work_items = []
        for batch_id, batch in enumerate(batches):
            worker_id = batch_id % self.num_workers
            work_items.append((batch_id, batch, worker_id))
        
        # Results storage
        all_results = []
        completed_count = 0
        
        start_time = time.time()
        
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            print('\nInterrupted! Saving current progress...')
            self.save_progress(all_results, f"{self.temp_file_prefix}_interrupted.txt")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Process batches in parallel
        with Pool(processes=self.num_workers) as pool:
            try:
                # Process with progress tracking
                for i, batch_results in enumerate(pool.imap_unordered(self.enhance_prompt_batch_worker, work_items)):
                    all_results.extend(batch_results)
                    completed_count += len(batch_results)
                    
                    # Progress update
                    elapsed = time.time() - start_time
                    rate = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_prompts - completed_count) / rate if rate > 0 else 0
                    print(f"Progress: {completed_count}/{total_prompts} ({completed_count/total_prompts*100:.1f}%) - "
                          f"Batches: {i+1}/{len(batches)} - "
                          f"Rate: {rate:.1f} prompts/sec - ETA: {eta/60:.1f} minutes")
                    
                    # Save progress periodically
                    if completed_count >= self.save_interval and completed_count % self.save_interval < self.batch_size:
                        temp_filename = f"{self.temp_file_prefix}_{completed_count}.txt"
                        self.save_progress(all_results, temp_filename)
                        
                        # Also save to main output file
                        self.save_progress(all_results, self.output_file)
                        print(f"Checkpoint saved: {completed_count} prompts processed")
                
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                print("Saving current progress...")
                self.save_progress(all_results, f"{self.temp_file_prefix}_error.txt")
                raise
        
        # Save final results
        self.save_progress(all_results, self.output_file)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*50}")
        print(f"Enhancement completed!")
        print(f"Total prompts processed: {len(all_results)}")
        print(f"Total batches processed: {len(batches)}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Average time per prompt: {total_time/len(all_results):.3f} seconds")
        print(f"Average time per batch: {total_time/len(batches):.3f} seconds")
        print(f"Results saved to: {self.output_file}")
        
        # Clean up temporary files
        self.cleanup_temp_files()
        
        return all_results
    
    def cleanup_temp_files(self):
        """Remove temporary checkpoint files."""
        import glob
        temp_files = glob.glob(f"{self.temp_file_prefix}_*.txt")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Removed temporary file: {temp_file}")
            except Exception as e:
                print(f"Error removing {temp_file}: {str(e)}")
    
    def display_samples(self, results: List[Tuple[int, str, str]], num_samples: int = 5):
        """Display sample enhanced prompts."""
        print(f"\n{'='*50}")
        print("Sample enhanced prompts:")
        print('='*50)
        
        # Take samples from different parts of the results
        sample_indices = [i * len(results) // num_samples for i in range(num_samples)]
        
        for i, idx in enumerate(sample_indices[:num_samples]):
            if idx < len(results):
                _, original, enhanced = results[idx]
                print(f"\nSample {i+1}:")
                print(f"Original: {original}")
                print(f"Enhanced: {enhanced}")

def main():
    """Main function to enhance prompts from parquet file."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance prompts from parquet file for FLUX model')
    parser.add_argument('--force-extract', action='store_true', 
                       help='Force re-extraction of unique prompts from parquet file')
    parser.add_argument('--unique-only', action='store_true',
                       help='Only extract unique prompts without enhancement')
    parser.add_argument('--min-length', type=int, default=40,
                       help='Minimum prompt length to include (default: 40 characters)')
    args = parser.parse_args()
    
    enhancer = ParquetPromptEnhancer()
    enhancer.min_prompt_length = args.min_length
    
    print("="*50)
    print("Parquet Prompt Enhancement for FLUX Model (Batch Processing)")
    print("="*50)
    
    try:
        # Check if parquet file exists
        if not os.path.exists(enhancer.parquet_file):
            print(f"Error: Parquet file not found at {enhancer.parquet_file}")
            return
        
        # Force re-extraction if requested
        if args.force_extract and os.path.exists(enhancer.unique_prompts_file):
            print(f"Removing existing unique prompts file: {enhancer.unique_prompts_file}")
            os.remove(enhancer.unique_prompts_file)
        
        # If only extracting unique prompts
        if args.unique_only:
            unique_prompts = enhancer.extract_unique_prompts()
            print(f"\nExtraction complete! {len(unique_prompts)} unique prompts saved to {enhancer.unique_prompts_file}")
            return
        
        # Enhance all prompts
        results = enhancer.enhance_all_prompts()
        
        # Display some samples
        if results:
            enhancer.display_samples(results)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 