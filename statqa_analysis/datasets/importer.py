# -*- coding: utf-8 -*-
"""
Dataset import functionality for StatQA analysis framework.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from .registry import DatasetRegistry


class DatasetImporter:
    """
    Imports datasets into StatDatasets/raw/<dataset_id>/.
    Ensures serial_number exists and creates dataset manifest.
    """
    
    def __init__(self, root_dir: Path = Path("StatDatasets")):
        self.registry = DatasetRegistry(root_dir)
    
    def import_from_json(
        self,
        dataset_id: str,
        json_path: Path,
        ensure_serial_number: bool = True,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Import a dataset from a JSON file.
        
        Args:
            dataset_id: Unique identifier for the dataset
            json_path: Path to source JSON file
            ensure_serial_number: If True, ensure serial_number exists (synthesize if missing)
            force: If True, overwrite existing dataset
            
        Returns:
            Import summary dict with row_count, serial_number_synthesized, etc.
        """
        # Check if dataset already exists
        if self.registry.dataset_exists(dataset_id) and not force:
            raise ValueError(
                f"Dataset '{dataset_id}' already exists. Use force=True to overwrite."
            )
        
        # Read source JSON
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Source JSON not found: {json_path}")
        
        print(f"[*] Loading source JSON: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of objects")
        
        # Ensure serial_number exists
        serial_synthesized = False
        if ensure_serial_number:
            data, serial_synthesized = self._ensure_serial_numbers(data)
        
        # Compute schema hash (field names, sorted)
        schema_hash = self._compute_schema_hash(data)
        
        # Prepare destination
        raw_dir = self.registry.get_raw_path(dataset_id)
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Write data.json
        data_path = raw_dir / "data.json"
        print(f"[*] Writing canonical data.json: {data_path}")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Create dataset manifest
        manifest = {
            "dataset_id": dataset_id,
            "source_path": str(json_path.absolute()),
            "import_time": datetime.now().isoformat(),
            "row_count": len(data),
            "schema_hash": schema_hash,
            "serial_number_synthesized": serial_synthesized,
            "fields": list(data[0].keys()) if data else []
        }
        
        manifest_path = raw_dir / "dataset_manifest.json"
        print(f"[*] Writing dataset manifest: {manifest_path}")
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        # Register in global registry
        self.registry.register_dataset(dataset_id, {
            "raw_dir": str(raw_dir.absolute()),
            "row_count": len(data),
            "import_time": manifest["import_time"],
            "schema_hash": schema_hash
        })
        
        summary = {
            "dataset_id": dataset_id,
            "row_count": len(data),
            "serial_number_synthesized": serial_synthesized,
            "data_path": str(data_path),
            "manifest_path": str(manifest_path)
        }
        
        print(f"[+] Dataset '{dataset_id}' imported successfully")
        print(f"    Rows: {len(data)}")
        print(f"    Serial numbers synthesized: {serial_synthesized}")
        
        return summary
    
    def _ensure_serial_numbers(self, data: List[Dict[str, Any]]) -> tuple:
        """
        Ensure all entries have serial_number.
        If missing, synthesize as 0-based index.
        
        Returns:
            (modified_data, was_synthesized)
        """
        has_serial = all('serial_number' in item for item in data)
        
        if has_serial:
            print("[i] All entries already have serial_number")
            return data, False
        
        print("[*] Synthesizing serial_number as 0-based index")
        for idx, item in enumerate(data):
            if 'serial_number' not in item:
                item['serial_number'] = idx
        
        return data, True
    
    def _compute_schema_hash(self, data: List[Dict[str, Any]]) -> str:
        """Compute a hash of the schema (sorted field names)."""
        if not data:
            return ""
        fields = sorted(data[0].keys())
        schema_str = ",".join(fields)
        return hashlib.md5(schema_str.encode()).hexdigest()[:8]

