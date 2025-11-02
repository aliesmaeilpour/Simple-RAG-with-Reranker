from __future__ import annotations
from typing import List, Tuple, Dict, Any
import logging
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import settings
from vector_store import VectorIndex

log = logging.getLogger(__name__)


SYSTEM_TABLE_AWARE = (
    "You are a helpful AI assistant that answers with precision. "
    "You will cite sources using the provided document metadata. "
    "If tables are relevant, you must rely on them. Be concise and accurate."
)

USER_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "You will use the following context chunks. Each chunk has metadata with filename and page.\n\n"
    "{context}\n\n"
    "Rules:\n"
    "1. Use only the context to answer. If insufficient, say you do not know.\n"
    "2. Quote numbers and column values carefully.\n"
    "3. Cite sources as (filename p. <page>). Separate multiple citations with commas.\n"
)

def _format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        fn = d.metadata.get("filename", "unknown")
        page = d.metadata.get("page", "NA")
        ctype = d.metadata.get("content_type", "text")
        header = f"[{ctype}] {fn}, page {page}"
        parts.append(f"{header}\n{d.page_content}\n")
    return "\n---\n".join(parts)


class QwenLLM:
    """Thin wrapper that supports local HF or DashScope API."""

    def __init__(self) -> None:
        self.backend = settings.llm_backend
        if self.backend == "local":
            device = 0 if torch.cuda.is_available() and settings.qwen_device != "cpu" else -1
            log.info("Loading Qwen local model %s", settings.qwen_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(settings.qwen_model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.qwen_model_name,
                dtype="auto" if settings.qwen_dtype == "auto" else getattr(torch, settings.qwen_dtype),
                device_map="auto" if device == -1 else None,
                trust_remote_code=True,
            )
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                model_kwargs={"trust_remote_code": True},
            )
            self.dashscope = None
        else:
            try:
                import dashscope  # type: ignore
            except Exception as e:
                raise RuntimeError("dashscope package is required for API backend") from e
            if not settings.dashscope_api_key:
                raise RuntimeError("Missing dashscope_api_key in environment")
            dashscope.api_key = settings.dashscope_api_key
            self.dashscope = dashscope
            self.pipeline = None
            self.tokenizer = None
            self.model = None

    def generate(self, prompt: str, params: Dict[str, Any] | None = None) -> str:
        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens", settings.qwen_max_new_tokens))
        temperature = float(params.get("temperature", settings.qwen_temperature))
        top_p = float(params.get("top_p", settings.qwen_top_p))
        repetition_penalty = float(params.get("repetition_penalty", settings.qwen_repetition_penalty))

        if self.backend == "local":
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
            )
            return outputs[0]["generated_text"]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_TABLE_AWARE},
                {"role": "user", "content": prompt},
            ]
            rsp = self.dashscope.Generation.call(  # type: ignore
                model="qwen-plus",
                input=messages,
                result_format="message",
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            return rsp.output.choices[0].message.content  # type: ignore


class RAGEngine:
    def __init__(self, index: VectorIndex):
        self.index = index
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        self.llm = QwenLLM()

    def ingest_documents(self, docs: List[Document]) -> None:
        split_docs: List[Document] = []
        for d in docs:
            chunks = self.splitter.split_text(d.page_content)
            for ch in chunks:
                meta = dict(d.metadata)
                split_docs.append(Document(page_content=ch, metadata=meta))
        self.index.add_documents(split_docs)

    def _retrieve(self, query: str, table_only: bool, top_k: int, filters: Dict[str, Any] | None) -> List[Document]:
        k = max(settings.retrieval_k, top_k)
        if table_only:
            docs = self.index.search_table_only(query, k=k)
        else:
            docs = self.index.search(query, k=k, filters=filters)
        reranked = self.index.rerank(query, docs, top_k=min(settings.rerank_k, top_k))
        return reranked or docs[:top_k]

    def _build_prompt(self, question: str, docs: List[Document]) -> str:
        ctx = _format_context(docs)
        return USER_PROMPT_TEMPLATE.format(question=question, context=ctx)

    @staticmethod
    def _to_sources(docs: List[Document]) -> List[Dict[str, Any]]:
        out = []
        for d in docs:
            out.append({
                "doc_id": d.metadata.get("doc_id", ""),
                "filename": d.metadata.get("filename", ""),
                "page": int(d.metadata.get("page", 0)),
                "score": float(d.metadata.get("rerank_score", 0.0)),
                "content_type": d.metadata.get("content_type", "text"),
                "snippet": d.page_content[:400],
            })
        return out

    def answer(self, question: str, top_k: int = 5, table_only: bool = False,
               filters: Dict[str, Any] | None = None, gen_params: Dict[str, Any] | None = None):
        docs = self._retrieve(question, table_only, top_k, filters)
        if not docs:
            return "I do not have enough information to answer that.", []
        prompt = self._build_prompt(question, docs)
        raw = self.llm.generate(prompt, gen_params)
        answer = raw.strip()
        return answer, self._to_sources(docs)
