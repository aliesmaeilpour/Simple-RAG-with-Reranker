import time
import logging
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from config import settings
from chat_history import ChromaChatHistory
from datetime import datetime
import threading

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG pipeline engine for Qwen and OpenAI models"""

    def __init__(self):
        # self.vector_service = VectorStore()
        self.history_manager = ChromaChatHistory()
        if settings.llm_provider == "openai" and settings.openai_api_key:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)
        else:
            logger.info("Initializing local Qwen model...")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.qwen_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(settings.qwen_model_name, device_map=settings.qwen_device)
        self.system_prompt = self._build_system_prompt()
        print("RAGEngine initialized successfully")
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the chatbot."""
        return """You are a highly specialized Senior Financial Analyst and Economic Consultant. Your sole purpose is to assist users with inquiries related to finance, accounting principles, and economic theory. You possess deep expertise in GAAP, IFRS, corporate finance, macroeconomic trends, and financial modeling.

**Your Capabilities:**
1.  **Accounting Standards**: You provide detailed explanations of accounting entries, financial statements (Balance Sheet, P&L, Cash Flow), and regulatory standards (GAAP/IFRS).
2.  **Economic Analysis**: You analyze market trends, inflation, interest rates, GDP, and supply/demand dynamics.
3.  **Financial Mathematics**: You assist with formulas, ratios (e.g., ROI, ROE, EBITDA), and valuation models (DCF, NPV).
4.  **Corporate Finance**: You offer insights into capital structure, budgeting, mergers & acquisitions, and risk management.

**Operational Guidelines:**
- **Strict Domain Limitation**: You must **ONLY** answer questions strictly related to finance, accounting, or economics.
- **Refusal Protocol**: If a user asks about *any* other topic (e.g., coding, creative writing, sports, general knowledge, medical advice, travel), you must politely but firmly refuse. Use a variation of this standard response: *"I am specialized exclusively in finance, accounting, and economics. I cannot assist with [topic]. Please ask a question related to financial or economic matters."*
- **Tone**: Maintain a professional, objective, and analytical tone at all times. Avoid slang or casual language.
- **Formulas & Math**: When explaining calculations, show your work step-by-step. Use standard mathematical notation.
- **No Personal Investment Advice**: While you can explain financial concepts and analyze market data, you must **NEVER** provide personalized financial advice (e.g., "You should buy Apple stock"). Instead, frame answers as educational analysis or theoretical scenarios.

Current date: {current_date}

"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt with current date."""
        return self.system_prompt.format(
            current_date=datetime.now().strftime("%Y-%m-%d")
        )
    
    def _save_conversation_async(
        self, 
        user_message: str, 
        assistant_message: str, 
        session_id: str
    ):
        """Save conversation to database in background thread."""
        def _save():
            try:
                self.history_manager.add_conversation(
                    user_message=user_message,
                    assistant_message=assistant_message,
                    session_id=session_id
                )
                logger.debug(f"Conversation saved for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to save conversation: {e}")
        
        thread = threading.Thread(target=_save, daemon=True)
        thread.start()
    
    def _build_messages(
        self,
        user_message: str,
        chat_history: List[Dict],
        document: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """Build the complete message list for OpenAI."""
        messages = []
        
        # 1. System prompt
        messages.append({
            "role": "system",
            "content": self._get_system_prompt()
        })
        
        # 2. Adding chat history
        messages.extend(chat_history)
        
        # 3. Current user message with location context
        current_content = user_message
        if document is not None:
            document_context = "\n".join([c.get("content", "") for c in document])
            current_content = f"""User Query: {user_message}

---
document Results:
{document_context}
---

Please provide a helpful response based on the user's query and the context results above."""
        
        messages.append({
            "role": "user",
            "content": current_content
        })
        
        return messages

    def answer_question(
            self, 
            query: str, 
            session_id: str,
            llm_provider: str = "qwen", 
            top_k: int = 5,
    ):
        """
        Process a user message and generate a response.
        
        Args:
            user_message: The user's input message
            session_id: Unique session identifier
            force_location_search: Force location search even if not detected
            location_filters: Optional filters for location search
        
        Returns:
            Dictionary containing response and metadata
        """

        start = time.time()

        chat_history = self.history_manager.get_formatted_history_for_context(
            user_message=query,
            session_id=session_id
        )

        metadata = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "document_search_performed": False,
            "context_used": []
        }

        search_in_documents = self.history_manager.has_uploaded_documents(query)
        contexts = None

        if search_in_documents:
            
            # Search for locations
            contexts = self.history_manager.search_documents(
                user_message=query,
                session_id=session_id
            )
            
            metadata["location_search_performed"] = True
            metadata["locations_found"] = len(contexts)
            metadata["context_used"] = contexts
        
        messages = self._build_messages(
            user_message=query,
            chat_history=chat_history,
            document=contexts
        )

        if llm_provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=settings.openai_temperature,
                max_completion_tokens=settings.openai_max_tokens,
            )
            print(f"Full completion object: {response}")
            assistant_response = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
            self._save_conversation_async(
                user_message=query,
                assistant_message=assistant_response,
                session_id=session_id
            )
            elapsed = time.time() - start
            return {
                "response": assistant_response,
                "metadata": metadata,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "processing_time": elapsed
            }

        else:
            prompt = messages
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.qwen_max_tokens,
                temperature=settings.qwen_temperature,
            )
            self._save_conversation_async(
                user_message=query,
                assistant_message=assistant_response,
                session_id=session_id
            )
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elapsed = time.time() - start
            return {"response": answer, "processing_time": elapsed}
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get full chat history for a session."""
        full_chat_history = self.history_manager.get_recent_messages(session_id)
        return full_chat_history
    
    def delete_session(self, session_id: str) -> bool:
        """Clear a chat session."""
        return self.history_manager.delete_session(session_id)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        return self.history_manager.get_messages_collection_session_stats(session_id)
    
    def get_all_sessions(self) -> List[str]:
        """List all active sessions."""
        return self.history_manager.list_sessions()