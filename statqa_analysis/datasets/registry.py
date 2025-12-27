# -*- coding: utf-8 -*-
"""
Dataset registry management for StatDatasets/.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class DatasetRegistry:
    """
    Manages the StatDatasets/registry.json file.
    """
    
    def __init__(self, root_dir: Path = Path("StatDatasets")):
        self.root_dir = root_dir
        self.registry_path = root_dir / "registry.json"
        self._ensure_root()
    
    def _ensure_root(self):
        """Ensure StatDatasets directory and registry exist."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write_registry({})
    
    def _read_registry(self) -> Dict[str, Any]:
        """Read the registry file."""
        if not self.registry_path.exists():
            return {}
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _write_registry(self, data: Dict[str, Any]):
        """Write the registry file."""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def register_dataset(self, dataset_id: str, metadata: Dict[str, Any]):
        """
        Register a dataset with its metadata.
        
        Args:
            dataset_id: Unique identifier for the dataset
            metadata: Dataset metadata (source_path, row_count, import_time, etc.)
        """
        registry = self._read_registry()
        registry[dataset_id] = metadata
        self._write_registry(registry)
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a dataset."""
        registry = self._read_registry()
        return registry.get(dataset_id)
    
    def list_datasets(self) -> Dict[str, Any]:
        """List all registered datasets."""
        return self._read_registry()
    
    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if a dataset is registered."""
        return dataset_id in self._read_registry()
    
    def get_raw_path(self, dataset_id: str) -> Path:
        """Get the path to the raw dataset directory."""
        return self.root_dir / "raw" / dataset_id
    
    def get_prompts_path(self, dataset_id: str) -> Path:
        """Get the path to the prompts directory for a dataset."""
        return self.root_dir / "prompts" / dataset_id

