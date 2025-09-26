from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import numpy as np

class SearchSystem:
    def __init__(self, embedding_manager, cross_encoder_name: str, sem_weight: float = 0.6, rerank_weight: float = 0.4):
        self.em = embedding_manager
        self.cross_encoder = CrossEncoder(cross_encoder_name)
        self.sem_weight = sem_weight
        self.rerank_weight = rerank_weight

    def retrieve_and_rerank(self, query: str, top_k: int = 12, final_k: int = 5) -> List[Dict[str, Any]]:
        ids, sem_scores, metas = self.em.search(query, top_k=top_k)
        if not ids:
            return []
        texts = [m["text"] for m in metas]
        pairs = [(query, t) for t in texts]
        rerank_scores = self.cross_encoder.predict(pairs).tolist()

        combined = []
        for i, m, s_sem, s_rer in zip(ids, metas, sem_scores, rerank_scores):
            score = self.sem_weight * s_sem + self.rerank_weight * s_rer
            m2 = dict(m)
            m2["score_sem"] = float(s_sem)
            m2["score_rerank"] = float(s_rer)
            m2["score"] = float(score)
            m2["id"] = int(i)
            combined.append(m2)

        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:final_k]
