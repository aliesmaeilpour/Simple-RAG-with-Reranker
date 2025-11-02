import faiss
import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.schema import Document
from config import settings
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS-based vector store with reranking support"""

    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

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

    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        if not documents:
            return

        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Save the index
        self.save()
        logger.info(f"Added {len(documents)} documents to vector store")

    def search(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)

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
            results = results[:self.config.rerank_top_k]

        return results[:top_k]

    def save(self):
        """Save the vector store to disk"""
        index_path = os.path.join(self.config.vector_store_path, "faiss.index")
        meta_path = os.path.join(self.config.vector_store_path, "metadata.pkl")

        faiss.write_index(self.index, index_path)

        with open(meta_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadatas": self.metadatas
            }, f)

        logger.info("Vector store saved")

    def load(self):
        """Load the vector store from disk"""
        index_path = os.path.join(self.config.vector_store_path, "faiss.index")
        meta_path = os.path.join(self.config.vector_store_path, "metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(index_path)

            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadatas = data["metadatas"]

            logger.info(f"Loaded vector store with {len(self.documents)} documents")