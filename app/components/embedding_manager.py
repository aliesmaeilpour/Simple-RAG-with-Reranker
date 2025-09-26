from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import json
from pathlib import Path

class EmbeddingManager:
    def __init__(self, model_name: str, index_dir: Path):
        self.model = SentenceTransformer(model_name)
        self.index_dir = index_dir
        self.index_path = index_dir / "faiss.index"
        self.meta_path = index_dir / "metadata.json"
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []  # list of dicts aligned with vectors

        os.makedirs(index_dir, exist_ok=True)
        self._load_or_init()

    def _load_or_init(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine via dot on normalized vectors
            self.metadata = []
            self._persist()

    def _persist(self):
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs

    def add(self, texts: List[str], metas: List[Dict[str, Any]]):
        vecs = self.embed(texts)
        self.index.add(vecs.astype("float32"))
        self.metadata.extend(metas)
        self._persist()

    def search(self, query: str, top_k: int = 10) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        q = self.embed([query]).astype("float32")
        scores, ids = self.index.search(q, top_k)
        ids = ids[0].tolist()
        scores = scores[0].tolist()
        metas = [self.metadata[i] for i in ids if i != -1]
        # Filter -1 entries if any
        filt = [(i, s, m) for i, s, m in zip(ids, scores, metas) if i != -1]
        if not filt:
            return [], [], []
        idxs, scs, mts = zip(*filt)
        return list(idxs), list(scs), list(mts)

    def get_texts(self, ids: List[int]) -> List[str]:
        # texts are stored inside metadata to avoid a separate store
        return [self.metadata[i]["text"] for i in ids]
