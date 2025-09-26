from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime

class MetaStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.meta_path = base_dir / "meta.json"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self._write({})

    def _read(self) -> Dict[str, Any]:
        with open(self.meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, Any]):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_document(self, doc_id: str, meta: Dict[str, Any]):
        data = self._read()
        meta["updated_at"] = datetime.utcnow().isoformat() + "Z"
        data[doc_id] = meta
        self._write(data)

    def list_documents(self) -> Dict[str, Any]:
        return self._read()

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        return self._read().get(doc_id, {})
