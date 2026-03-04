from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    text: str
    emotion: str = "neutral"
    tools_used: list[dict] | None = None
