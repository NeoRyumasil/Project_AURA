import re
import logging

from fastapi import APIRouter, HTTPException
from app.services.memory_service import memory_service
from app.models.chat import ChatRequest, ChatResponse
from app.services.brain.graph import brain
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Run Graph
    try:
        conversation_id = request.conversation_id
        
        if not conversation_id:
            new_id = await memory_service.create_conversation()
            conversation_id = str(new_id) if new_id else "default"
        
        initial_state = {
            "messages":   [HumanMessage(content=request.message)],
            "emotion":    "neutral",
            "conversation_id": conversation_id,
        }

        config = {"configurable": {"thread_id": conversation_id}}
        result = brain.invoke(initial_state, config=config)
        
        # Extract response
        last_msg = result["messages"][-1].content
        emotion = result.get("emotion", "neutral")
        
        # Look for tool calls in the last turn
        tools_used = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append({
                        "name": tc.get("name"),
                        "args": tc.get("args", {})
                    })
                    
        # Clean tags
        text = last_msg
        if text.startswith("["):
             match = re.match(r'^\[(.*?)\]', text)
             if match:
                 text = text[match.end():].strip()

        return ChatResponse(
            text=text,
            emotion=emotion,
            conversation_id=conversation_id,
            tools_used=tools_used if tools_used else None
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")

        return ChatResponse(
            text=f"Brain Freeze: {str(e)}",
            emotion="confused",
            conversation_id=request.conversation_id or "default",
        )
