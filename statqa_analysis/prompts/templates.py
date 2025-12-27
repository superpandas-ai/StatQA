# -*- coding: utf-8 -*-
"""
Prompt templates mirroring prompt_wording.py constants.
"""

from prompt_wording import (
    PROMPT_TASK_DESCRIPTION,
    PROMPT_INSTRUCTION,
    PROMPT_CLASSIFICATION,
    PROMPT_RESPONSE,
    PROMPT_COT,
    PROMPT_RESPONSE_EXPLAIN,
    CA_EG, CTT_EG, DCT_EG, VT_EG, DS_EG,
    COT_CA_EG, COT_CTT_EG, COT_DCT_EG, COT_VT_EG, COT_DS_EG,
    STATS_PROMPT
)


def get_few_shot_examples():
    """Get list of few-shot examples."""
    return [CA_EG, CTT_EG, DCT_EG, VT_EG, DS_EG]


def get_cot_examples():
    """Get list of Chain-of-Thought examples."""
    return [COT_CA_EG, COT_CTT_EG, COT_DCT_EG, COT_VT_EG, COT_DS_EG]


__all__ = [
    'PROMPT_TASK_DESCRIPTION',
    'PROMPT_INSTRUCTION',
    'PROMPT_CLASSIFICATION',
    'PROMPT_RESPONSE',
    'PROMPT_COT',
    'PROMPT_RESPONSE_EXPLAIN',
    'STATS_PROMPT',
    'get_few_shot_examples',
    'get_cot_examples',
]

