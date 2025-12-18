from sentence_transformers import SentenceTransformer
import logging
from config import Config
from typing import List, Dict, Optional, Any
from uuid import uuid4
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from langchain_core.documents import Document
from config import settings
from openai import OpenAI


logger = logging.getLogger(__name__)


class ChromaChatHistory:
    """
    Stores chat messages inside a ChromaDB collection per session.
    Each message stored as:
        - id: UUID
        - text: user or assistant message
        - embedding: OpenAI embedding
        - metadata: {role, session_id, index}
    """

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
        # self.session_id = session_id
        self.messages_collection_name = "chat_messages"
        self.documents_collection_name = "chat_documents"
        self.collection_name = "chat_content"

        self.chroma = chromadb.PersistentClient(path="./chromadb")

        self._initialized = False

    def initialize(self) -> None:
        """Initialize both collections for messages and documents."""
        if self._initialized:
            return

        try:
            # Messages collection
            self.messages_collection = self.chroma.get_or_create_collection(
                name=self.messages_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Chat messages for all sessions",
                    "created_at": datetime.now().isoformat(),
                },
            )

            # Documents collection
            self.documents_collection = self.chroma.get_or_create_collection(
                name=self.documents_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "description": "Documents and knowledge base",
                    "created_at": datetime.now().isoformat(),
                },
            )

            self._initialized = True
            logger.info(
                f"ChromaDB initialized - Messages: {self.messages_collection.count()}, "
                f"Documents: {self.documents_collection.count()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

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
                show_progress_bar=False,
                convert_to_numpy=True
            ).astype(np.float32)

    def add_message(
        self,
        role: str,
        content: str,
        session_id: str
    ) -> None:
        """
        Store a new message into ChromaDB.
        """

        message_id = f"{session_id}_{uuid4().hex[:12]}"
        timestamp = datetime.now().isoformat()
        
        # Build metadata with session_id for filtering
        metadata = {
            "session_id": session_id,
            "role": role,
            "timestamp": timestamp,
            "message_index": self._get_next_index(session_id),
        }
        
        embedding = self._get_embeddings(content)

        self.messages_collection.add(
            ids=[message_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata],
        )

    def add_conversation(
            self,
            user_message: str,
            assistant_message: str,
            session_id: str,
    ) -> None:
        

        self.add_message(
            role="user",
            content=user_message,
            session_id=session_id
        )

        self.add_message(
            role="assistant",
            content=assistant_message,
            session_id=session_id
        )

    def add_document_chunks(
        self,
        document_chunks: List[Document],
        session_id: str,
        doc_type: str = "text"
    ) -> List[str]:
        """
        Add pre-chunked document content.

        Args:
            chunks: List of text chunks
            session_id: Session identifier
            source: Document source/filename
            doc_type: Type of document
            metadata: Additional metadata

        Returns:
            List of chunk IDs
        """
        self._ensure_initialized()

        if not document_chunks:
            return []

        texts = [doc.page_content for doc in document_chunks]
        metadatas = [doc.metadata for doc in document_chunks]

        chunk_ids = []
        timestamp = datetime.now().isoformat()
        doc_group_id = str(uuid4())

        embeddings = self._get_embeddings(texts)

        for i, (chunk, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            chunk_id = f"doc_{session_id}_{uuid4().hex[:12]}"
            chunk_ids.append(chunk_id)
            source = metadata.get("source")

            chunk_metadata = {
                "session_id": session_id,
                "doc_group_id": doc_group_id,
                "doc_type": doc_type,
                "chunk_index": i,
                "total_chunks": len(texts),
                "timestamp": timestamp,
                "content_type": "document",
                "char_count": len(chunk),
            }

            chunk_metadata.update(metadata)

            self.documents_collection.add(
                ids=[chunk_id],
                embeddings=[embedding.tolist()],
                documents=[chunk],
                metadatas=[chunk_metadata],
            )

        logger.info(
            f"Added {len(document_chunks)} pre-chunked document pieces from '{source}' "
            f"to session {session_id}"
        )
        return chunk_ids
    
    def has_uploaded_documents(self, session_id: str) -> bool:
        """
        Checks if there is at least one document associated with the session_id.
        """
        result = self.documents_collection.get(
            where={"session_id": session_id},
            limit=1,
            include=[]
        )
        
        return len(result['ids']) > 0

    def search_documents(
        self,
        query: str,
        session_id: str,
        n_results: int = 5,
        score_threshold: float = 0.5,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across stored documents.

        Args:
            query: Search query
            session_id: Optional session filter
            n_results: Number of results
            score_threshold: Minimum similarity score
            source: Optional source filter

        Returns:
            List of matching document chunks
        """
        self._ensure_initialized()

        query_embedding = self._get_embeddings(query)

        where_filter: Dict[str, Any] = {
            "session_id": session_id,
            "content_type": "document"
        }

        if source:
            where_filter["source"] = source

        results = self.documents_collection.query(
            query_embeddings=[query_embedding],
            where=where_filter,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        matches = []
        for doc, meta, distance in zip(docs, metas, distances):
            similarity = 1 - distance
            if similarity < score_threshold:
                continue

            matches.append({
                "content": doc,
                "score": float(similarity),
                "source": meta.get("source"),
                "doc_type": meta.get("doc_type"),
                "chunk_index": meta.get("chunk_index"),
                "session_id": meta.get("session_id"),
                "timestamp": meta.get("timestamp"),
            })

        if self.reranker and matches:
            matches = self._rerank_results(query, matches, "content")

        return matches
    
    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        content_key: str,
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.
        """
        texts = [r[content_key] for r in results]
        pairs = [(query, text) for text in texts]

        scores = self.reranker.predict(pairs)

        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results

    def _get_next_index(self, session_id: str) -> int:
        """
        Get the next message index for a session.
        """
        existing = self.messages_collection.get(
            where={"session_id": session_id,
                   "content_type": "chat"
            },
            include=["metadatas"],
        )
        
        if not existing["ids"]:
            return 0
        
        indices = [
            m.get("message_index", 0) 
            for m in existing["metadatas"]
        ]
        return max(indices) + 1

    def search(self, query: str, session_id: str) -> List[Dict]:

        embedding = self._get_embeddings(query)

        results = self.messages_collection.query(
            query_embeddings=[embedding],
            where={"session_id": session_id},
            n_results=5,
        )

        if not results["documents"]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        scores = results["distances"][0]

        messages = []
        for doc, meta, score in zip(docs, metas, scores):
            # use cosine similarity thresholds
            if score < 0.5:
                continue

            messages.append(
                {
                    "role": meta["role"],
                    "content": doc,
                    "score": float(score),
                    "index": meta["message_index"],
                }
            )

        messages.sort(key=lambda x: x["index"])

        return messages

    def get_formatted_history_for_context(self, session_id: str, user_message: str) -> List[Dict]:

        recent = self.get_recent_messages(session_id, limit=6)
        relevant = self.search(user_message, session_id)

        # Merge but avoid duplicates
        combined = {m["index"]: m for m in recent}
        for m in relevant:
            combined[m["index"]] = m

        messages = list(combined.values())
        messages.sort(key=lambda x: x["index"])

        return [{"role": m["role"], "content": m["content"]} for m in messages]
    
    def delete_session(self, session_id: str) -> int:
        """
        Delete all messages and documents for a session.
        """
        self._ensure_initialized()

        self.messages_collection.delete(where={"session_id": session_id})
        self.documents_collection.delete(where={"session_id": session_id})

        logger.info(f"Deleted session data for session_id={session_id}")

    def clear_session(self):
        """
        Deletes the entire Chroma collection.
        """
        try:
            self.chroma.delete_collection(self.collection_name)
        except Exception:
            pass

    def get_recent_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Retrieve messages for a session, ordered by index.
        
        Args:
            session_id: Session identifier
            limit: Maximum messages to return (newest if limited)
            
        Returns:
            List of message dictionaries with role and content
        """
        if not self._initialized:
            self.initialize()
        
        # Query by session_id metadata
        results = self.messages_collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        
        if not results["ids"]:
            return []
        
        # Combine and sort by message_index
        messages = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            messages.append({
                "role": meta.get("role", "user"),
                "content": doc,
                "timestamp": meta.get("timestamp"),
                "index": meta.get("message_index", 0),
            })
        
        # Sort by index
        messages.sort(key=lambda x: x["index"])
        
        # Apply limit (keep most recent)
        if limit and len(messages) > limit:
            messages = messages[-limit:]
        
        # Return simplified format
        return messages
    
    def get_message_collection_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics dictionary
        """
        if not self._initialized:
            self.initialize()
        
        results = self.messages_collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )
        
        if not results["ids"]:
            return {
                "session_id": session_id,
                "exists": False,
                "message_count": 0,
            }
        
        user_count = 0
        assistant_count = 0
        total_chars = 0
        timestamps = []
        
        for doc, meta in zip(results["documents"], results["metadatas"]):
            role = meta.get("role", "")
            if role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1
            
            total_chars += len(doc) if doc else 0
            
            ts = meta.get("timestamp")
            if ts:
                timestamps.append(ts)
        
        timestamps.sort()
        
        return {
            "session_id": session_id,
            "exists": True,
            "message_count": len(results["ids"]),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "total_characters": total_chars,
            "first_message_at": timestamps[0] if timestamps else None,
            "last_message_at": timestamps[-1] if timestamps else None,
        }
    
    def list_sessions(
        self,
    ) -> List[Dict[str, Any]]:
        """
        List all unique sessions with message counts.
        
        Returns:
            List of session info dictionaries
        """
        if not self._initialized:
            self.initialize()
        
        # Get all documents to extract unique session_ids
        all_docs = self.messages_collection.get(
            include=["metadatas"],
        )
        
        if not all_docs["ids"]:
            return []
        
        # Aggregate by session_id
        sessions: Dict[str, Dict[str, Any]] = {}
        
        for meta in all_docs["metadatas"]:
            sid = meta.get("session_id")
            if not sid:
                continue
            
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "message_count": 0,
                    "first_message": meta.get("timestamp"),
                    "last_message": meta.get("timestamp"),
                }
            
            sessions[sid]["message_count"] += 1
            
            # Track first/last timestamps
            ts = meta.get("timestamp")
            if ts:
                if ts < sessions[sid]["first_message"]:
                    sessions[sid]["first_message"] = ts
                if ts > sessions[sid]["last_message"]:
                    sessions[sid]["last_message"] = ts
        
        # Convert to list and sort by last_message (newest first)
        session_list = list(sessions.values())
        session_list.sort(key=lambda x: x["last_message"] or "", reverse=True)
        
        # Apply pagination
        return session_list