# -*- coding: utf-8 -*-
"""
Safe I/O utilities for loading and saving data.
"""

import ast
import json
import pandas as pd
from pathlib import Path
from typing import Any, Union


def safe_parse_json(s: str) -> Any:
    """
    Safely parse a JSON string.
    Returns the parsed object or None on error.
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_literal_eval(s: str) -> Any:
    """
    Safely parse a Python literal string using ast.literal_eval.
    Returns the parsed object or None on error.
    """
    try:
        # Normalize boolean strings
        normalized = s.replace("false", "False").replace("true", "True")
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError, TypeError):
        return None


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV file safely with error handling.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"[!] File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"[!] Error reading file {file_path}: {e}")


def save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False) -> None:
    """
    Save a DataFrame to CSV with error handling.
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=index)
    except Exception as e:
        raise RuntimeError(f"[!] Error saving file {file_path}: {e}")

