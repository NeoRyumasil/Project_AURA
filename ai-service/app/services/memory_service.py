"""
Memory service using Supabase pgvector for semantic search.
Replaces the previous Qdrant-based implementation — zero Docker containers needed.
"""
from __future__ import annotations
from supabase import create_client
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import logging
import threading

logger = logging.getLogger(__name__)

session_ttl_minutes = 120

@dataclass
class Interaction:
    role : str
    content : str
    emotion : str
    timestamp : datetime = field(default_factory=datetime.now)

    def inject_to_llm(self) -> dict:
        return {"role" : self.role, "content" : self.content}


@dataclass
class SessionStore:
    interactions: list[Interaction] = field(default_factory=list)
    last_active: datetime = field(default_factory=datetime.now)

    def set_last_active(self):
        self.last_active = datetime.now()

    def is_expired(self, ttl: int) -> bool:
        return datetime.now() - self.last_active > timedelta(minutes=ttl)

class MemoryService:
    def __init__(self, max_session_interaction: int = 200):
        self.session: dict[str, SessionStore] = {}
        self.lock = threading.Lock()
        self.max_session = max_session_interaction
        self.client = None
        self.embeddings = None

        # Initialize Supabase client
        if settings.SUPABASE_URL and settings.SUPABASE_SERVICE_KEY:
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
            logger.info("Memory Service connected to Supabase")
        else:
            logger.warning("Supabase credentials not set. Memory service disabled.")

        # Initialize embeddings model via OpenRouter
        api_key = settings.OPENROUTER_API_KEY
        if api_key:
            self.embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model="openai/text-embedding-3-small",
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            logger.warning("OPENROUTER_API_KEY not set. Memory embedding disabled.")

    def get_create_session(self, session_id:str) -> SessionStore:
        if session_id not in self.session:
            self.session[session_id] = SessionStore()
            logger.info(f"MemoryService: new session: {session_id[:8]}")
        
        store = self.session[session_id]
        store.set_last_active()
        return store
    
    def clean_expired(self):
        expired = [session_id for session_id, store in self.session.items()
                   if store.is_expired(session_ttl_minutes)]

        for session_id in expired:
            del self.session[session_id]
            logger.info(f"MemoryService: expired session '{session_id[:8]}…' removed.")

    def add_interaction(self, session_id: str, user_text : str, assistant_text : str, user_emotion: str = "neutral", assistant_emotion: str = "neutral") -> None:
        with self.lock:
            self.clean_expired()
            store = self.get_create_session(session_id)
            store.interactions.append(Interaction("user", user_text, user_emotion))
            store.interactions.append(Interaction("assistant", assistant_text, assistant_emotion))

            if len(store.interactions) > self.max_session:
                store.interactions = store.interactions[-self.max_session:]
    
    def get_messages_in_session(self, session_id: str, n: int = 20) -> list[dict]:
        if n <= 0 :
            return []
        
        with self.lock:
            store = self.get_create_session(session_id)
            return [message.inject_to_llm() for message in store.interactions[-n:]]
        
    def get_last_n(self, session_id: str, n: int) -> list[Interaction]:
        if n <= 0:
            return []
        
        with self.lock:
            store = self._get_or_create(session_id)
            return list(store.interactions[-n:])
    
    def count_session(self, session_id: str) -> int:
        with self.lock:
            if session_id not in self._sessions:
                return 0
            
            return len(self.session[session_id].interactions)
    
    def clear_session(self, session_id: str) -> None:
       with self.lock:
            if session_id in self._sessions:
                del self.session[session_id]

            logger.info(f"MemoryService: session '{session_id[:8]}…' cleared.")
    
    def active_session(self) -> int:
        with self.lock:
            return len(self.session)

    async def store(self, text: str, metadata: dict = None):
        """Embed and store a memory in Supabase pgvector."""
        if not self.client or not self.embeddings or not text.strip():
            return

        try:
            vector = await self.embeddings.aembed_query(text)

            self.client.table("memories").insert({
                "content": text,
                "embedding": vector,
                "metadata": metadata or {},
            }).execute()

            logger.info(f"Stored memory: {text[:40]}...")
        except Exception as e:
            logger.error(f"Memory store error: {e}")

    async def search(self, query: str, limit: int = 3) -> list[str]:
        """Retrieve relevant memories via cosine similarity."""
        if not self.client or not self.embeddings:
            return []

        try:
            vector = await self.embeddings.aembed_query(query)

            # Use Supabase RPC for pgvector similarity search
            result = self.client.rpc("match_memories", {
                "query_embedding": vector,
                "match_count": limit,
            }).execute()

            return [row["content"] for row in (result.data or [])]
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            return []


memory_service = MemoryService()
