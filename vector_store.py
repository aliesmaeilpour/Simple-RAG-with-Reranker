import faiss
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from config import settings
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store with reranking support"""

    def __init__(self):

        if settings.llm_provider == 'openai':
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
            self.embedding_model = None
            self.openai_embedding_model = settings.openai_embedding_model
            self.dimension = 1536
        else:
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.openai_client = None

        # Initialize reranker if enabled
        self.reranker = None
        if settings.use_reranking:
            self.reranker = CrossEncoder(settings.rerank_model)

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []

        # Load existing index if available
        self.load()

    def _get_openai_embeddings(self, texts: List[str]) -> np.ndarray:
        """"get embeddings using OpenAI API key"""
        texts = [text.replace('\n', ' ') for text in texts]
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=self.openai_embedding_model
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            except Exception as e:
                logger.error(f"Error getting OpenAI embeddings: {e}")
                raise
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def _get_embeddings(self, texts: List[str]):
        """"Get emgeddings using the defined provider"""
        if settings.llm_provider == 'openai':
            logger.info(f'Generating OpenAI embeddings for {len(texts)} texts.')
            return self._get_openai_embeddings(texts)
        else:
            logger.info(f'Generating local embeddings for {len(texts)} texst.')
            return self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            ).astype(np.float32)

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            return

        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents.")
        embeddings = self._get_embeddings(texts)

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Save the index
        self.save()
        logger.info(f"Added {len(documents)} documents to vector store.")

    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Generate query embedding
        query_embedding = self._get_embeddings([query])

        # Search in FAISS
        k = min(top_k * 2 if self.reranker else top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                result = {
                    "content": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(distance)
                }

                # Apply metadata filter if provided
                if filter_metadata:
                    match = all(
                        result["metadata"].get(key) == value
                        for key, value in filter_metadata.items()
                    )
                    if not match:
                        continue

                results.append(result)

        # Rerank if enabled
        if self.reranker and len(results) > 0:
            logger.info("Reranking results...")
            pairs = [[query, r["content"]] for r in results]
            scores = self.reranker.predict(pairs)

            for i, score in enumerate(scores):
                results[i]["rerank_score"] = float(score)

            # Sort by rerank score and take top k
            results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
            results = results[:settings.rerank_top_k]

        return results[:top_k]

    def save(self):
        """Save the vector store to disk"""
        index_path = os.path.join(settings.vector_store_path, "faiss.index")
        meta_path = os.path.join(settings.vector_store_path, "metadata.pkl")

        faiss.write_index(self.index, index_path)

        with open(meta_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)

        logger.info("Vector store saved")

    def load(self):
        """Load the vector store from disk"""
        index_path = os.path.join(settings.vector_store_path, "faiss.index")
        meta_path = os.path.join(settings.vector_store_path, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)

            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]

            logger.info(f"Loaded vector store with {len(self.documents)} documents")