#!/usr/bin/env python3
"""
Script to generate 10,000 random diffusion prompts for FLUX model using PromptGenerator.
Each prompt consists of 3 sentences, with 10 prompts generated per API call.
Uses multithreading with 10 workers for parallel processing.
"""

import os
import time
from prompt_generator import PromptGenerator
from typing import List
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class BulkPromptGenerator:
    def __init__(self):
        self.prompt_generator = PromptGenerator()
        self.total_prompts_needed = 10000
        self.prompts_per_call = 10  # Changed from 200 to 10
        self.sentences_per_prompt = 3
        self.max_workers = 10  # Number of parallel workers
        self.lock = threading.Lock()  # For thread-safe operations
        
    def generate_bulk_prompts(self, num_prompts: int = 10, worker_id: int = 0) -> List[str]:
        """
        Generate multiple random diffusion prompts for FLUX model.
        Each prompt should be 3 sentences long.
        """
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"You are a professional AI artist specializing in creating high-quality diffusion prompts for FLUX model. \
                        Your task is to generate {num_prompts} unique, creative, and diverse diffusion prompts. \
                        Each prompt should be exactly 3 sentences long and describe a vivid, detailed scene or concept. \
                        The prompts should cover various themes like: landscapes, portraits, abstract art, fantasy, sci-fi, nature, architecture, objects, animals, etc. \
                        Make each prompt visually rich and specific enough to generate high-quality images. \
                        Format your response as a numbered list, with each prompt on a separate line. \
                        Example format: \
                        1. A majestic dragon soaring through storm clouds above a medieval castle. Lightning illuminates its scales in brilliant blues and purples. The scene captures the raw power of nature and myth colliding. \
                        2. A serene Japanese garden with cherry blossoms falling into a koi pond. Soft morning light filters through bamboo creating dappled shadows. The composition emphasizes tranquility and natural harmony."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Generate {num_prompts} unique diffusion prompts for FLUX model. Each prompt must be exactly 3 sentences. Make them diverse, creative, and visually compelling."
                    }
                ]
            },
        ]
        
        try:
            completion = self.prompt_generator.client.chat.completions.create(
                model=self.prompt_generator.deployment,
                messages=chat_prompt,
                max_tokens=1000,  # Reduced since we're generating fewer prompts per call
                temperature=1.2,  # Higher temperature for more creativity
                top_p=0.95,
                frequency_penalty=0.3,  # Reduce repetition
                presence_penalty=0.3,   # Encourage diverse topics
                stop=None,
                stream=False
            )
            
            response_text = completion.choices[0].message.content
            prompts = self.parse_prompts_from_response(response_text)
            
            with self.lock:
                print(f"Worker {worker_id}: Generated {len(prompts)} prompts")
            
            return prompts
            
        except Exception as e:
            with self.lock:
                print(f"Worker {worker_id}: Error generating prompts: {str(e)}")
            return []
    
    def parse_prompts_from_response(self, response_text: str) -> List[str]:
        """
        Parse the numbered list response and extract individual prompts.
        """
        prompts = []
        
        # Split by lines and process each line
        lines = response_text.strip().split('\n')
        current_prompt = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (new prompt)
            if re.match(r'^\d+\.', line):
                # Save previous prompt if exists
                if current_prompt:
                    prompts.append(current_prompt.strip())
                
                # Start new prompt (remove number prefix)
                current_prompt = re.sub(r'^\d+\.\s*', '', line)
            else:
                # Continue current prompt
                if current_prompt:
                    current_prompt += " " + line
                else:
                    current_prompt = line
        
        # Add the last prompt
        if current_prompt:
            prompts.append(current_prompt.strip())
        
        # Filter out empty prompts and ensure they have content
        valid_prompts = [p for p in prompts if len(p.strip()) > 20]
        
        return valid_prompts
    
    def generate_batch_worker(self, batch_info: tuple) -> tuple:
        """
        Worker function for generating a batch of prompts.
        Returns (batch_id, prompts_list)
        """
        batch_id, num_prompts = batch_info
        prompts = self.generate_bulk_prompts(num_prompts, batch_id)
        return (batch_id, prompts)
    
    def generate_all_prompts(self, output_file: str = "0711_captions.txt"):
        """
        Generate all 10,000 prompts using multithreading.
        """
        all_prompts = []
        total_calls = (self.total_prompts_needed + self.prompts_per_call - 1) // self.prompts_per_call
        
        print(f"Generating {self.total_prompts_needed} prompts with {self.prompts_per_call} prompts per call...")
        print(f"Total API calls needed: {total_calls}")
        print(f"Using {self.max_workers} parallel workers")
        
        # Create batches for parallel processing
        batches = []
        for call_num in range(total_calls):
            remaining_prompts = self.total_prompts_needed - (call_num * self.prompts_per_call)
            prompts_this_call = min(self.prompts_per_call, remaining_prompts)
            batches.append((call_num + 1, prompts_this_call))
        
        # Process batches in parallel
        completed_batches = 0
        batch_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {executor.submit(self.generate_batch_worker, batch): batch for batch in batches}
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_info = future_to_batch[future]
                try:
                    batch_id, batch_prompts = future.result()
                    batch_results[batch_id] = batch_prompts
                    completed_batches += 1
                    
                    with self.lock:
                        print(f"Completed batch {batch_id}/{total_calls} - Got {len(batch_prompts)} prompts")
                        print(f"Progress: {completed_batches}/{total_calls} batches completed")
                        
                    # Save progress periodically
                    if completed_batches % 50 == 0:  # Save every 50 batches
                        # Reconstruct prompts in order
                        ordered_prompts = []
                        for i in range(1, completed_batches + 1):
                            if i in batch_results:
                                ordered_prompts.extend(batch_results[i])
                        
                        self.save_prompts_to_file(ordered_prompts, f"temp_{output_file}")
                        with self.lock:
                            print(f"Saved progress: {len(ordered_prompts)} prompts to temp_{output_file}")
                        
                except Exception as e:
                    with self.lock:
                        print(f"Error processing batch {batch_info[0]}: {str(e)}")
        
        # Reconstruct all prompts in order
        for batch_id in sorted(batch_results.keys()):
            all_prompts.extend(batch_results[batch_id])
        
        # Save final results
        self.save_prompts_to_file(all_prompts, output_file)
        
        print(f"\nCompleted! Generated {len(all_prompts)} total prompts.")
        print(f"Saved to {output_file}")
        
        # Clean up temp file if it exists
        temp_file = f"temp_{output_file}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Cleaned up temporary file: {temp_file}")
        
        return all_prompts
    
    def save_prompts_to_file(self, prompts: List[str], filename: str):
        """
        Save prompts to a text file, one prompt per line.
        Thread-safe version.
        """
        with self.lock:
            with open(filename, 'w', encoding='utf-8') as f:
                for prompt in prompts:
                    f.write(prompt + '\n')
    
    def validate_prompts(self, prompts: List[str]) -> List[str]:
        """
        Validate that prompts meet the 3-sentence requirement.
        """
        valid_prompts = []
        
        for prompt in prompts:
            # Count sentences (rough estimate by counting periods, exclamation marks, question marks)
            sentence_count = len(re.findall(r'[.!?]+', prompt))
            
            if sentence_count >= 2:  # Allow some flexibility
                valid_prompts.append(prompt)
            else:
                print(f"Skipping prompt with {sentence_count} sentences: {prompt[:50]}...")
        
        return valid_prompts

def main():
    """
    Main function to generate bulk prompts.
    """
    generator = BulkPromptGenerator()
    
    print("Starting multithreaded bulk prompt generation for FLUX model...")
    print(f"Target: {generator.total_prompts_needed} prompts")
    print(f"Prompts per API call: {generator.prompts_per_call}")
    print(f"Sentences per prompt: {generator.sentences_per_prompt}")
    print(f"Max workers: {generator.max_workers}")
    
    start_time = time.time()
    
    # Generate all prompts
    all_prompts = generator.generate_all_prompts("0715_captions.txt")
    
    # Validate prompts
    valid_prompts = generator.validate_prompts(all_prompts)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nValidation Results:")
    print(f"Total prompts generated: {len(all_prompts)}")
    print(f"Valid prompts: {len(valid_prompts)}")
    print(f"Total time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average time per prompt: {elapsed_time/len(all_prompts):.3f} seconds")
    
    # Save final validated prompts
    if len(valid_prompts) != len(all_prompts):
        generator.save_prompts_to_file(valid_prompts, "0715_captions_validated.txt")
        print("Saved validated prompts to 0715_captions_validated.txt")
    
    print("\nSample prompts:")
    for i, prompt in enumerate(valid_prompts[:3]):
        print(f"{i+1}. {prompt}")

if __name__ == "__main__":
    main() 