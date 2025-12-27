# -*- coding: utf-8 -*-
"""
Template loader for custom prompt templates.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_template_from_json(template_path: Path) -> Dict[str, Any]:
    """
    Load a prompt template from a JSON file.
    
    Expected JSON structure:
    {
        "task_description": "...",
        "instruction": "...",
        "classification": "...",
        "response": "...",
        "cot": "...",  # optional
        "response_explain": "...",  # optional
        "stats_prompt": "..."  # optional
    }
    
    Args:
        template_path: Path to template JSON file
        
    Returns:
        Dictionary with template components
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f)
    
    # Validate required fields
    required = ['task_description', 'instruction', 'classification', 'response']
    missing = [field for field in required if field not in template]
    if missing:
        raise ValueError(f"Template missing required fields: {missing}")
    
    return template


def get_default_template() -> Dict[str, Any]:
    """
    Get the default template from prompt_wording.py constants.
    
    Returns:
        Dictionary with default template components
    """
    from .templates import (
        PROMPT_TASK_DESCRIPTION,
        PROMPT_INSTRUCTION,
        PROMPT_CLASSIFICATION,
        PROMPT_RESPONSE,
        PROMPT_COT,
        PROMPT_RESPONSE_EXPLAIN,
        STATS_PROMPT,
    )
    
    return {
        'task_description': PROMPT_TASK_DESCRIPTION,
        'instruction': PROMPT_INSTRUCTION,
        'classification': PROMPT_CLASSIFICATION,
        'response': PROMPT_RESPONSE,
        'cot': PROMPT_COT,
        'response_explain': PROMPT_RESPONSE_EXPLAIN,
        'stats_prompt': STATS_PROMPT,
    }

