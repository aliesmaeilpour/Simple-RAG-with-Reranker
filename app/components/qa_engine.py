from typing import Dict, Any, List
from pydantic import BaseModel
import os

class QARequest(BaseModel):
    question: str
    doc_ids: List[str] = []  # optional filter to given docs

class QAEngine:
    def __init__(self, search_system, llm=None):
        self.search = search_system
        # llm is a callable that takes prompt and returns string, can be plugged in later
        self.llm = llm

    def build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        ctx_texts = []
        for i, c in enumerate(contexts, 1):
            meta = f"(source={c.get('source')}, chunk_id={c.get('chunk_id')})"
            ctx_texts.append(f"[{i}] {meta}\n{c['text']}")
        ctx_block = "\n\n".join(ctx_texts)
        prompt = f"""You are a helpful assistant that answers in Persian when the query is in Persian.
Use the provided contexts to answer the user question. Quote sources like [1], [2] where relevant.
If the answer is not in the context, say you do not know.

Question:
{query}

Contexts:
{ctx_block}

Answer in Persian if the question is Persian, otherwise answer in the question language. Keep it concise and cite sources like [1], [2]."""
        return prompt

    def answer(self, query: str, top_k: int = 12, final_k: int = 5) -> Dict[str, Any]:
        contexts = self.search.retrieve_and_rerank(query, top_k=top_k, final_k=final_k)
        prompt = self.build_prompt(query, contexts)
        if self.llm is None:
            # return prompt and contexts for external LLM
            return {
                "prompt": prompt,
                "contexts": contexts
            }
        else:
            answer = self.llm(prompt)
            return {"answer": answer, "contexts": contexts}
