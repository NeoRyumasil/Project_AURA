from app.services.brain.state import BrainState
from app.services.llm import llm_service
from app.services.prompter import prompter
from app.services.memory_service import memory_service
from langchain_core.messages import AIMessage, HumanMessage

session_history_window = 20

# Node to generate response based on persona, conversation history and detected emotion (convesation history not being tested yet)
def generate_response(state: BrainState) -> dict:

    # BrainState contains conversation history and detected emotion
    messages = state["messages"]
    detected_emotion = state.get("emotion", "neutral")
    session_id = state.get("session_id", "default")

    system_message = prompter.build("",context=None)[0]
    history = memory_service.get_messages_in_session(session_id, session_history_window)
    
    # Reformat messages to LLM format
    current_message = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            current_message.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            current_message.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict):
            current_message.append(msg)
    
    # Add system prompt with persona and current time
    messages_format = [system_message] + history + current_message

    # Generate response from LLM
    response = llm_service.generate(messages_format)

    user_message = current_message[-1]["content"] if current_message else ""

    memory_service.add_interaction(
        session_id=session_id,
        user_text=user_message,
        assistant_text=response["text"],
        user_emotion=detected_emotion,
        assistant_emotion=response["emotion"]
    )

    # Return response
    return {"messages": [AIMessage(content=response["text"])], "emotion": response["emotion"]}