import asyncio
import logging
from uuid import UUID

from core.services.memory import memory_service
from core.services.llm import llm_service

logger = logging.getLogger(__name__)

class Summarizer:
    async def summarize_interaction(self, conversation_id: UUID) -> str:

        if not conversation_id:
            return "Invalid ID"

        try:
            history = await memory_service.get_history(conversation_id, n=50)
            if not history:
                logger.info(f"No History on : {conversation_id}")
                return "No History."

            chat_text = ""
            for m in history:
                role = "User" if m["role"] == "user" else "AURA"
                chat_text += f"{role}: {m['content']}\n"

            messages = [
                {
                    "role": "system", 
                    "content": "You are a summarizing assistant. Your task is to summarize the conversation clearly and concisely. This summary will be used as knowledge for an AI companion."
                },
                {
                    "role": "user", 
                    "content": f"Summarize this:\n\n{chat_text}"
                }
            ]

            response = await asyncio.to_thread(llm_service.generate, messages)
            summary = response.get("text", "").strip()

            if not summary:
                logger.warning(f"Fail to summarize on {conversation_id}")
                return "Summarization Fail"

            logger.info(f"Ringkasan berhasil dibuat untuk sesi {conversation_id}")
            
            await memory_service.store(
                text=f"Summary of conversation: {summary}", 
                metadata={"type": "conversation_summary", "conversation_id": str(conversation_id)}
            )

            return summary

        except Exception as e:
            logger.error(f"Error pada SummarizerModule: {e}", exc_info=True)
            return f"Terjadi kesalahan saat meringkas: {str(e)}"

summarizer_service = Summarizer()