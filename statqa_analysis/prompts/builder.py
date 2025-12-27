# -*- coding: utf-8 -*-
"""
Prompt builder for StatQA questions.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from random import Random
import pandas as pd

from .templates import (
    PROMPT_TASK_DESCRIPTION,
    PROMPT_INSTRUCTION,
    PROMPT_CLASSIFICATION,
    PROMPT_RESPONSE,
    PROMPT_COT,
    PROMPT_RESPONSE_EXPLAIN,
    STATS_PROMPT,
    get_few_shot_examples,
    get_cot_examples,
)
from .metadata import get_dataset_metadata
from .template_loader import load_template_from_json, get_default_template
from ..datasets.registry import DatasetRegistry


class PromptBuilder:
    """
    Builds minimal prompt CSVs from raw dataset JSON.
    """
    
    def __init__(self, root_dir: Path = Path("StatDatasets")):
        self.registry = DatasetRegistry(root_dir)
    
    def build_prompts(
        self,
        dataset_id: str,
        prompt_version: str,
        trick: str,
        seed: Optional[int] = None,
        template_file: Optional[Path] = None,
        manual_template_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Build prompts for a dataset and write minimal CSV.
        
        Args:
            dataset_id: Dataset identifier
            prompt_version: Version identifier for prompts
            trick: Prompting strategy (zero-shot, one-shot, two-shot, zero-shot-CoT, one-shot-CoT, stats-prompt)
            seed: Random seed for deterministic few-shot selection
            
        Returns:
            Summary dict with output paths and metadata
        """
        # Validate inputs
        if not self.registry.dataset_exists(dataset_id):
            raise ValueError(f"Dataset '{dataset_id}' not found. Import it first with 'dataset import'.")
        
        valid_tricks = ['zero-shot', 'one-shot', 'two-shot', 'zero-shot-CoT', 'one-shot-CoT', 'stats-prompt', 'manual']
        if trick not in valid_tricks:
            raise ValueError(f"Invalid trick '{trick}'. Must be one of: {valid_tricks}")
        
        # Validate manual trick requirements
        if trick == 'manual':
            if not manual_template_file:
                raise ValueError("For 'manual' trick, --manual-template (file path) is required")
            if not manual_template_file.exists():
                raise FileNotFoundError(f"Manual template file not found: {manual_template_file}")
        elif manual_template_file:
            raise ValueError("--manual-template can only be used with 'manual' trick")
        
        # Load manual template if provided
        manual_template_str = None
        if manual_template_file:
            print(f"[*] Loading manual template from: {manual_template_file}")
            with open(manual_template_file, 'r', encoding='utf-8') as f:
                manual_template_str = f.read()
        
        # Load template (custom or default)
        if template_file:
            template = load_template_from_json(template_file)
            print(f"[*] Using custom template from: {template_file}")
        else:
            template = get_default_template()
            print("[*] Using default template from prompt_wording.py")
        
        # Load raw dataset
        raw_dir = self.registry.get_raw_path(dataset_id)
        data_path = raw_dir / "data.json"
        
        print(f"[*] Loading dataset: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure serial_number exists
        if not all('serial_number' in item for item in data):
            raise ValueError(
                f"Dataset '{dataset_id}' is missing serial_number field. "
                "Re-import with --ensure-serial-number."
            )
        
        # Build prompts
        print(f"[*] Building {trick} prompts...")
        prompts_data = []
        rng = Random(seed) if seed is not None else Random()
        
        for item in data:
            prompt = self._build_prompt(item, trick, rng, template, manual_template_str)
            prompts_data.append({
                'serial_number': item['serial_number'],
                'prompt': prompt
            })
        
        # Prepare output directory
        prompts_dir = self.registry.get_prompts_path(dataset_id) / prompt_version
        prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Write prompts CSV
        output_path = prompts_dir / f"{trick}.csv"
        print(f"[*] Writing prompts CSV: {output_path}")
        df = pd.DataFrame(prompts_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Save template if custom
        if template_file:
            template_copy_path = prompts_dir / "template.json"
            print(f"[*] Saving template to: {template_copy_path}")
            with open(template_file, 'r', encoding='utf-8') as src:
                template_content = json.load(src)
            with open(template_copy_path, 'w', encoding='utf-8') as dst:
                json.dump(template_content, dst, indent=2, ensure_ascii=False)
        
        # Save manual template if used
        if manual_template_str:
            manual_template_path = prompts_dir / "manual_template.txt"
            print(f"[*] Saving manual template to: {manual_template_path}")
            with open(manual_template_path, 'w', encoding='utf-8') as f:
                f.write(manual_template_str)
        
        # Create prompt manifest
        manifest = {
            "dataset_id": dataset_id,
            "prompt_version": prompt_version,
            "trick": trick,
            "seed": seed,
            "generation_time": datetime.now().isoformat(),
            "row_count": len(prompts_data),
            "prompt_template_hash": self._compute_template_hash(trick, template),
            "template_source": str(template_file) if template_file else "default",
            "has_manual_template": manual_template_str is not None,
            "manual_template_source": str(manual_template_file) if manual_template_file else None
        }
        
        manifest_path = prompts_dir / f"prompt_manifest_{trick}.json"
        print(f"[*] Writing prompt manifest: {manifest_path}")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        summary = {
            "dataset_id": dataset_id,
            "prompt_version": prompt_version,
            "trick": trick,
            "row_count": len(prompts_data),
            "output_path": str(output_path),
            "manifest_path": str(manifest_path)
        }
        
        print(f"[+] Prompts built successfully")
        print(f"    Output: {output_path}")
        print(f"    Rows: {len(prompts_data)}")
        
        return summary
    
    def _build_prompt(
        self, 
        item: Dict[str, Any], 
        trick: str, 
        rng: Random,
        template: Dict[str, Any],
        manual_template_str: Optional[str] = None
    ) -> str:
        """Build a single prompt for a question."""
        dataset_name = item['dataset']
        refined_question = item['refined_question']
        
        # Get metadata for the dataset
        try:
            meta_info_list = get_dataset_metadata(dataset_name)
        except Exception as e:
            print(f"[!] Warning: Could not get metadata for dataset '{dataset_name}': {e}")
            meta_info_list = []
        
        # Build prompt based on trick
        if trick == 'zero-shot':
            return self._build_zero_shot(refined_question, meta_info_list, template)
        elif trick == 'one-shot':
            return self._build_one_shot(refined_question, meta_info_list, rng, template)
        elif trick == 'two-shot':
            return self._build_two_shot(refined_question, meta_info_list, rng, template)
        elif trick == 'zero-shot-CoT':
            return self._build_zero_shot_cot(refined_question, meta_info_list, template)
        elif trick == 'one-shot-CoT':
            return self._build_one_shot_cot(refined_question, meta_info_list, rng, template)
        elif trick == 'stats-prompt':
            return self._build_stats_prompt(refined_question, meta_info_list, rng, template)
        elif trick == 'manual':
            return self._build_manual(refined_question, meta_info_list, manual_template_str)
        else:
            raise ValueError(f"Unsupported trick: {trick}")
    
    def _build_manual(self, question: str, meta_info: List[str], template_str: str) -> str:
        """Build manual prompt using f-string template."""
        # Format meta_info_list as a string (one per line)
        meta_info_str = '\n'.join(meta_info)
        
        # Use f-string to format the template
        try:
            prompt = template_str.format(
                meta_info_list=meta_info_str,
                refined_question=question
            )
        except KeyError as e:
            raise ValueError(
                f"Manual template contains invalid placeholder: {e}. "
                "Only '{{meta_info_list}}' and '{{refined_question}}' are allowed."
            )
        
        return prompt
    
    def _build_zero_shot(self, question: str, meta_info: List[str], template: Dict[str, Any]) -> str:
        """Build zero-shot prompt."""
        return (
            "### Task Description: " + template['task_description'] +
            "\n### Instruction: " + template['instruction'] +
            "\n### Classification List: \n" + template['classification'] +
            "\n### Column Information: \n" + '\n'.join(meta_info) +
            "\n### Statistical Question: " + question +
            "\n### Response: " + template['response']
        )
    
    def _build_one_shot(self, question: str, meta_info: List[str], rng: Random, template: Dict[str, Any]) -> str:
        """Build one-shot prompt."""
        example = rng.choice(get_few_shot_examples())
        return (
            "### Task Description: " + template['task_description'] +
            "\n### Instruction: " + template['instruction'] +
            "\n### Classification List: \n" + template['classification'] +
            "\n### Demonstration Example:\n<example start>\n" + example + "\n</example end>" +
            "\n### Column Information: \n" + '\n'.join(meta_info) +
            "\n### Statistical Question: " + question +
            "\n### Response: " + template['response']
        )
    
    def _build_two_shot(self, question: str, meta_info: List[str], rng: Random, template: Dict[str, Any]) -> str:
        """Build two-shot prompt."""
        examples = rng.sample(get_few_shot_examples(), 2)
        return (
            "### Task Description: " + template['task_description'] +
            "\n### Instruction: " + template['instruction'] +
            "\n### Classification List: \n" + template['classification'] +
            "\n### Here are two demonstration examples:\n<example start>" +
            "\n## Demonstration Example No.1:\n" + examples[0] +
            "\n## Demonstration Example No.2:\n" + examples[1] + "\n</example end>" +
            "\n### Column Information: \n" + '\n'.join(meta_info) +
            "\n### Statistical Question: " + question +
            "\n### Response: " + template['response']
        )
    
    def _build_zero_shot_cot(self, question: str, meta_info: List[str], template: Dict[str, Any]) -> str:
        """Build zero-shot Chain-of-Thought prompt."""
        cot = template.get('cot', PROMPT_COT)
        response_explain = template.get('response_explain', PROMPT_RESPONSE_EXPLAIN)
        return (
            "### Task Description: " + template['task_description'] +
            "\n### Instruction: " + template['instruction'] +
            "\n### Classification List: \n" + template['classification'] +
            "\n### Column Information: \n" + '\n'.join(meta_info) +
            "\n### Statistical Question: " + question +
            "\n### Response: " + cot + response_explain
        )
    
    def _build_one_shot_cot(self, question: str, meta_info: List[str], rng: Random, template: Dict[str, Any]) -> str:
        """Build one-shot CoT prompt."""
        example = rng.choice(get_cot_examples())
        cot = template.get('cot', PROMPT_COT)
        response_explain = template.get('response_explain', PROMPT_RESPONSE_EXPLAIN)
        return (
            "### Task Description: " + template['task_description'] +
            "\n### Instruction: " + template['instruction'] +
            "\n### Classification List: \n" + template['classification'] +
            "\n### Demonstration Example:\n<example start>\n" + example + "\n</example end>" +
            "\n### Column Information: \n" + '\n'.join(meta_info) +
            "\n### Statistical Question: " + question +
            "\n### Response: " + cot + response_explain
        )
    
    def _build_stats_prompt(self, question: str, meta_info: List[str], rng: Random, template: Dict[str, Any]) -> str:
        """Build stats-enhanced prompt."""
        example = rng.choice(get_few_shot_examples())
        stats_prompt = template.get('stats_prompt', STATS_PROMPT)
        return (
            "### Task Description: " + template['task_description'] +
            "\n### Instruction: " + template['instruction'] +
            "\n### Classification List: \n" + stats_prompt +
            "\n### Demonstration Example:\n<example start>\n" + example + "\n</example end>" +
            "\n### Column Information: \n" + '\n'.join(meta_info) +
            "\n### Statistical Question: " + question +
            "\n### Response: " + template['response']
        )
    
    def _compute_template_hash(self, trick: str, template: Optional[Dict[str, Any]] = None) -> str:
        """Compute a hash of the prompt template for versioning."""
        if template:
            # Use custom template
            template_str = f"{trick}|{template.get('task_description', '')}|{template.get('instruction', '')}|{template.get('classification', '')}"
        else:
            # Use default template
            template_str = f"{trick}|{PROMPT_TASK_DESCRIPTION}|{PROMPT_INSTRUCTION}|{PROMPT_CLASSIFICATION}"
        return hashlib.md5(template_str.encode()).hexdigest()[:8]

