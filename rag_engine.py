import time
import logging
from typing import List
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from config import settings
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG pipeline engine for Qwen and OpenAI models"""

    def __init__(self):
        self.vector_service = VectorStore()
        if settings.llm_provider == "openai" and settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            logger.info("Initializing local Qwen model...")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.qwen_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(settings.qwen_model_name, device_map=settings.qwen_device)

    def answer_question(self, query: str, llm_provider: str = "qwen", top_k: int = 5):
        start = time.time()
        candidates = self.vector_service.search(query, top_k=top_k)
        # print('candidates:', candidates)
        context = "\n".join([c.get("content", "") for c in candidates])

        if llm_provider == "openai":
            print(f"Query: {query}")
            print(f"Context length: {len(context)}")
            print(f"Context preview: {context[:500] if context else 'EMPTY!'}")
            completion = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Persian assistant."},
                    {"role": "user", "content": f"Question: {query}\n Context:\n{context}"},
                ],
                temperature=settings.openai_temperature,
                max_completion_tokens=settings.openai_max_tokens,
            )
            print(f"Full completion object: {completion}")
            print(f"Finish reason: {completion.choices[0].finish_reason}")
            print(f"Message content: {completion.choices[0].message.content}")
            answer = completion.choices[0].message.content
        else:
            prompt = f"Question: {query}\n\n Context:\n{context}\n\n"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.qwen_max_tokens,
                temperature=settings.qwen_temperature,
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start
        return {"answer": answer, "processing_time": elapsed, "sources": candidates}