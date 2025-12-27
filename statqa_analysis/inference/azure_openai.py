# -*- coding: utf-8 -*-
"""
Azure OpenAI inference for StatQA prompts.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv

from ..datasets.registry import DatasetRegistry


# Load environment variables
load_dotenv()


class AzureOpenAIRunner:
    """
    Runs Azure OpenAI inference on prompt CSVs and writes resumable model outputs.
    """
    
    def __init__(self, output_dir: Path = Path("AnalysisOutput")):
        self.output_dir = output_dir
        self.registry = DatasetRegistry()
    
    def run_inference(
        self,
        model: str,
        dataset_id: str,
        prompt_version: str,
        trick: str,
        run_id: Optional[str] = None,
        max_retries: int = 2,
        batch_size: int = 5
    ) -> Dict[str, Any]:
        """
        Run Azure OpenAI inference on a prompt CSV.
        
        Args:
            model: Azure deployment name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            dataset_id: Dataset identifier
            prompt_version: Prompt version identifier
            trick: Prompting strategy
            run_id: Optional run identifier (default: auto-generated)
            max_retries: Max retry attempts per prompt
            batch_size: Save progress every N prompts
            
        Returns:
            Summary dict with run metadata and output paths
        """
        # Validate dataset exists
        if not self.registry.dataset_exists(dataset_id):
            raise ValueError(f"Dataset '{dataset_id}' not found")
        
        # Build run_id if not provided
        if run_id is None:
            run_id = f"{model}_{dataset_id}_{prompt_version}_{trick}"
        
        # Load prompt CSV
        prompt_csv_path = (
            self.registry.get_prompts_path(dataset_id) / 
            prompt_version / 
            f"{trick}.csv"
        )
        
        if not prompt_csv_path.exists():
            raise FileNotFoundError(
                f"Prompt CSV not found: {prompt_csv_path}\n"
                f"Build prompts first with: prompts build --dataset-id {dataset_id} "
                f"--prompt-version {prompt_version} --trick {trick}"
            )
        
        print(f"[*] Loading prompts from: {prompt_csv_path}")
        prompts_df = pd.read_csv(prompt_csv_path)
        
        if 'serial_number' not in prompts_df.columns or 'prompt' not in prompts_df.columns:
            raise ValueError("Prompt CSV must have 'serial_number' and 'prompt' columns")
        
        # Setup output directories
        run_dir = self.output_dir / "runs" / run_id
        inputs_dir = run_dir / "inputs"
        raw_dir = run_dir / "raw"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy prompt source for provenance
        prompt_copy = inputs_dir / "prompt_source.csv"
        prompts_df.to_csv(prompt_copy, index=False)
        
        # Create run manifest
        manifest = {
            "run_id": run_id,
            "model": model,
            "dataset_id": dataset_id,
            "prompt_version": prompt_version,
            "trick": trick,
            "start_time": datetime.now().isoformat(),
            "prompt_source": str(prompt_csv_path),
            "total_prompts": len(prompts_df)
        }
        
        manifest_path = inputs_dir / "run_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        # Check for existing output and resume if needed
        output_csv_path = raw_dir / "model_output.csv"
        if output_csv_path.exists():
            print(f"[i] Found existing output, resuming...")
            output_df = pd.read_csv(output_csv_path)
            start_idx = len(output_df)
        else:
            output_df = pd.DataFrame()
            start_idx = 0
        
        print(f"[*] Starting inference from row {start_idx}/{len(prompts_df)}")
        
        # Initialize Azure OpenAI client
        client = self._init_azure_client(model)
        
        # Token usage tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cached_tokens = 0
        total_tokens = 0
        
        # Run inference
        start_time = time.time()
        answers = []
        usage_records = []
        
        for idx in range(start_idx, len(prompts_df)):
            row = prompts_df.iloc[idx]
            serial_num = row['serial_number']
            prompt = row['prompt']
            
            # Call model
            answer, usage = self._call_model(client, model, prompt, max_retries)
            
            answers.append({
                'serial_number': serial_num,
                'model_answer': answer,
                'prompt_tokens': usage['prompt_tokens'],
                'completion_tokens': usage['completion_tokens'],
                'total_tokens': usage['total_tokens'],
                'prompt_cache_hit_tokens': usage['prompt_cache_hit_tokens']
            })
            
            # Accumulate usage
            total_prompt_tokens += usage['prompt_tokens']
            total_completion_tokens += usage['completion_tokens']
            total_cached_tokens += usage['prompt_cache_hit_tokens']
            total_tokens += usage['total_tokens']
            
            # Save progress periodically
            if (idx + 1) % batch_size == 0 or idx == len(prompts_df) - 1:
                batch_df = pd.DataFrame(answers)
                output_df = pd.concat([output_df, batch_df], ignore_index=True)
                output_df.to_csv(output_csv_path, index=False)
                answers = []
                
                print(f"[+] Progress: {idx + 1}/{len(prompts_df)} | "
                      f"Tokens - Input: {total_prompt_tokens}, "
                      f"Cached: {total_cached_tokens}, "
                      f"Output: {total_completion_tokens}, "
                      f"Total: {total_tokens}")
        
        end_time = time.time()
        
        # Update manifest with completion info
        manifest['end_time'] = datetime.now().isoformat()
        manifest['duration_seconds'] = end_time - start_time
        manifest['completed_prompts'] = len(prompts_df)
        manifest['total_tokens'] = total_tokens
        manifest['prompt_tokens'] = total_prompt_tokens
        manifest['completion_tokens'] = total_completion_tokens
        manifest['cached_tokens'] = total_cached_tokens
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        summary = {
            "run_id": run_id,
            "model": model,
            "total_prompts": len(prompts_df),
            "output_path": str(output_csv_path),
            "manifest_path": str(manifest_path),
            "duration_seconds": manifest['duration_seconds'],
            "total_tokens": total_tokens
        }
        
        print(f"\n{'='*70}")
        print(f"[+] Inference completed")
        print(f"    Output: {output_csv_path}")
        print(f"    Duration: {summary['duration_seconds']:.2f}s")
        print(f"    Total tokens: {total_tokens}")
        print(f"{'='*70}\n")
        
        return summary
    
    def _init_azure_client(self, model_name: str) -> AzureOpenAI:
        """Initialize Azure OpenAI client based on model name."""
        # Determine which endpoint to use
        if model_name.startswith('gpt-4'):
            endpoint = os.getenv('GPT4_ENDPOINT')
            api_key = os.getenv('GPT4_API_KEY')
        elif model_name.startswith('gpt-5'):
            endpoint = os.getenv('GPT5_ENDPOINT')
            api_key = os.getenv('GPT5_API_KEY')
        else:
            # Default to GPT4
            endpoint = os.getenv('GPT4_ENDPOINT')
            api_key = os.getenv('GPT4_API_KEY')
        
        api_version = os.getenv('API_VERSION', '2024-02-15-preview')
        
        if not endpoint or not api_key:
            raise ValueError(
                f"Azure OpenAI credentials not found for model '{model_name}'. "
                "Set GPT4_ENDPOINT/GPT4_API_KEY or GPT5_ENDPOINT/GPT5_API_KEY in environment."
            )
        
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version
        )
    
    def _call_model(
        self,
        client: AzureOpenAI,
        model_name: str,
        prompt: str,
        max_retries: int
    ) -> tuple:
        """
        Call Azure OpenAI model with retry logic.
        
        Returns:
            (answer, usage_dict)
        """
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                
                # Extract usage info
                usage = response.usage
                cached_tokens = 0
                if usage and hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                    cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                
                usage_dict = {
                    'prompt_tokens': usage.prompt_tokens if usage else 0,
                    'completion_tokens': usage.completion_tokens if usage else 0,
                    'total_tokens': usage.total_tokens if usage else 0,
                    'prompt_cache_hit_tokens': cached_tokens
                }
                
                return response.choices[0].message.content, usage_dict
                
            except Exception as e:
                print(f"[!] Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 5 * (attempt + 1)
                    print(f"[ ] Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
        
        # All retries failed
        error_usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
            'prompt_cache_hit_tokens': 0
        }
        return "Error", error_usage

