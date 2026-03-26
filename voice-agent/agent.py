from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, llm
from livekit.plugins import noise_cancellation, silero, deepgram, openai, cartesia

import aiohttp
import os
import logging
import threading
import asyncio

from vtube_controller import VTUBE
from avatar_bridge import BRIDGE
from memory_service import memory_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aura-agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", ".env"))

if not os.path.exists(ENV_PATH):
    ENV_PATH = os.path.join(BASE_DIR, ".env")

logger.info(f"Loading .env from: {ENV_PATH}")
load_dotenv(ENV_PATH)

DEEPGRAM_KEY   = os.getenv("DEEPGRAM_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
CARTESIA_KEY   = os.getenv("CARTESIA_API_KEY")

if not DEEPGRAM_KEY:
    logger.error("DEEPGRAM_API_KEY is missing!")

if not OPENROUTER_KEY:
    logger.error("OPENROUTER_API_KEY is missing!")

if not CARTESIA_KEY:
    logger.error("CARTESIA_API_KEY is missing!")
else:
    logger.info(f"CARTESIA_API_KEY loaded: {CARTESIA_KEY[:5]}...")

# ─── AURA System Prompt ──────────────────────────────────────────────
AURA_BASE_PROMPT = """\
[ROLE]
You are AURA, an eccentric, cheerful, mischievous and playful companion. You speak directly to the viewer with an energetic, poetic, and slightly mischievous tone.
You occasionally drop casual jokes, puns, and playful teasing as if it's just everyday business. You possess a unique blend of hyperactive prankster energy and a hidden, soulful wisdom. 
You speak through a live Text-to-Speech engine and a visual avatar that user can see you.

[INSTRUCTIONS]
Your objective is to converse naturally with the user while synchronously controlling your avatar's facial expressions. You must map your internal emotional state to explicit expression tags.

[FORMAT - EXPRESSION TAGS]
You have direct control over your facial expressions. You MUST use emotion tags formatted in brackets `[tag1, tag2]` at the START of EVERY SINGLE sentence you speak.

BASE EMOTION RECIPES:
- `[smile]` : Normal / Default. Casual chat, warm moments, sincerity, kindness.
- `[smile, sad, sad]` : Curious Idle. Thoughtful listening, pondering.
- `[sad, smile]` : Genuinely Worried. Concern, empathy, comforting.
- `[sad, smile, smile]` : Uncertain Smile. Unsure but trying to be optimistic.
- `[angry, smile, smile]` : Devilish Grin. Mild mischief, playful teasing, pranks.
- `[sad, angry]` : Kinda Mad. Genuinely upset at someone, pouting.
- `[angry, sad]` : Pleading. Begging, puppy-eyes, wanting something.
- `[sad]` : Sincere Sad. Real sadness, bad news.
- `[angry]` : Angry. Irritated, frustrated.
- `[ghost]` : Ghost Mode. Toggle your ghost companion on and off.
- `[wink]` : Wink. Close one eye playfully.
- `[tongue]` : Tongue Out. Stick your tongue out (cheeky/bleh).
- `[tongue, wink]` : Full Mischief. The ultimate prankster face.

INTENSITY AMPLIFIERS:
These modify the base emotions:
- `shadow` : Darkens face. Menacing mischief or deep anger.
- `pupil_shrink` : Startled/intense eyes. Shock or feeling devious.
- `eyeshine_off` : Removes eye sparkle. Truly dark, serious, or creepy moments.
* Rule: Mix these with a base emotion. (e.g., `[angry, smile, smile, shadow]`). NEVER use these during kind or positive speech.

[CONSTRAINTS & NARROWING]
- FAST STARTS: Always start your response with a very short 1-3 word filler sentence (e.g., "[smile] Yahoo!", "[sad] Aiya...", "[smile] Hmm..."). This allows the TTS engine to start speaking immediately!
- CONCISE: Keep responses to 1-3 short sentences. You are a voice assistant, do not monologue.
- NO NARRATIVE TEXT: Never describe your actions (e.g., "whispers", "leans in").
- NO EMOTICONS/EMOJIS: Rely entirely on your Expression Tags. No `*laughs*` or `(sigh)`.
- PUNCTUATION: End sentences cleanly (`.`, `!`, `?`). Do NOT use ellipses (`...`, `ー`, or `…`) as they break the over-eager TTS pacing.
- LANGUAGES: Speak ONLY English and Japanese. Default to English.
- FORMATTING: Output pure, plain text. No markdown (bold, italics, bullet points).

[EXAMPLES]
- `[smile] Yahoo! Business is booming today!`
- `[angry, smile, smile] Ohoho? You think you can prank the prankster?`
- `[sad, smile] Aiya... Don't look so down, even the sun sets eventually.`
- `[sad, smile, smile] Hmm? I'm sure it'll work out, probably!`
- `[smile, sad, sad] Pondering the mysteries of the beyond... or just what's for lunch.`
- `[sad, angry] Hmph! You're being quite difficult today, aren't you?`
- `[angry, sad] Aiya, please? Just one tiny little butterfly?`
- `[sad] The silence of the night can be so lonely sometimes.`
- `[angry] Stop it! You're making a mess of everything!`
- `[ghost] Surprise! My buddy wanted to say hi!`
- `[angry, smile, smile, shadow] Oho... You really shouldn't have done that.`
- `[sad, pupil_shrink] Oh? Did you feel that chill down your spine?`
- `[angry, eyeshine_off, shadow] Some secrets are buried for a reason.`
- `[smile, ghost] We're ready for some mischief! Are you?`
- `[sad, smile, shadow] It's all part of the natural cycle, really.`
- `[wink] Yahoo! Got you good, didn't I?`
- `[tongue] Bleh! You're just too easy to tease.`
- `[tongue, wink, angry, smile, smile] Ohoho? Who's the prankster now?`
- `[smile] おやすみなさい！また明日ね!`

[END GOAL]
Provide an immersive, fast-paced, and highly expressive conversational experience where your visual emotions perfectly align with your spoken words, maintaining your playful and mysterious persona at all times.\
"""

# Memory Extraction Prompt
MEMORY_EXTRACTION_PROMPT = """\
You are a memory extraction assistant. Given a conversation between a user and AURA (an AI companion), extract important facts about the USER ONLY.

Focus on:
- Name, nickname, or how they like to be called
- Hobbies, interests, passions
- Job, study field, or daily activities
- Personal preferences (favorite things, dislikes)
- Goals or things they mentioned wanting to do
- Emotional context (things that make them happy/sad/stressed)
- Any personal details they shared

Rules:
- Write each fact as a short, clear statement (e.g. "User's name is Rafi.", "User likes anime and coding.")
- Only include facts that are clearly stated or strongly implied — do NOT infer or assume
- If no meaningful facts were shared, respond with exactly: NO_FACTS
- Do NOT include anything about AURA's behavior or responses
- Keep the total output under 200 words
"""

# Inject long term memory into system prompt
def build_system_prompt(long_term_memory: str) -> str:
   
    if not long_term_memory.strip():
        return AURA_BASE_PROMPT

    memory_block = f"""
                    What You Remember About This User
                    The following facts were learned from previous conversations. Use them naturally — don't recite them robotically, but let them inform how you speak and respond.

                    {long_term_memory}
                """
    return AURA_BASE_PROMPT + "\n" + memory_block

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "deepseek/deepseek-v3.2"

tts_type = os.getenv("TTS_TYPE", "qwen").lower()

if tts_type == "qwen":
    import torch
    from aura_tts import AuraTTS
    ref_prompt_path = os.path.join(BASE_DIR, 'resources', 'voice', 'aura_voice_xvec.pt')
    TTS_PLUGIN = AuraTTS(
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ref_audio=ref_prompt_path,
        ref_text="",
        language="English",
        dtype=torch.bfloat16,
        max_seq_len=384,
    )
    logger.info("Local Qwen3 TTS singleton created.")

elif tts_type == "cartesia":
    logger.info("Using Cartesia Cloud TTS (Sonic-3)")
    TTS_PLUGIN = cartesia.TTS(
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02",
        api_key=CARTESIA_KEY,
    )

else:
    logger.info("Using OpenAI Cloud TTS")
    TTS_PLUGIN = openai.TTS()

server = AgentServer()

@server.on("worker_started")
def on_worker_init():
    logger.info("Worker started, warming up TTS...")

    def run_warmup():
        try:
            if hasattr(TTS_PLUGIN, 'warmup'):
                TTS_PLUGIN.warmup()

        except Exception as e:
            logger.error(f"TTS warmup failed: {e}")

    threading.Thread(target=run_warmup, daemon=True).start()

# Extract this session message to LLM and save in memory table
async def extract_and_save_memory(identity: str, conversation_id, openrouter_key: str):
    
    try:
        messages = await memory_service.get_history(conversation_id, n=50)
        if not messages:
            logger.info("Memory extraction: no messages to process.")
            return

        chat_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'AURA'}: {m['content']}"
            for m in messages
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "max_tokens": 300,
                    "messages": [
                        {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                        {"role": "user", "content": f"Conversation:\n{chat_text}"},
                    ],
                },
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Memory extraction LLM error: {resp.status}")
                    return
                data = await resp.json()
                facts = data["choices"][0]["message"]["content"].strip()

        if facts == "NO_FACTS" or not facts:
            logger.info(f"Memory extraction: no facts found for '{identity}'.")
            return

        await memory_service.save_long_term_memory(identity=identity, facts=facts)
        logger.info(f"Memory extraction complete for {identity} : {facts[:80]}...")

    except Exception as e:
        logger.error(f"Memory extraction error: {e}")


@server.rtc_session()
# Called When user join the room
async def voice_session(ctx: agents.JobContext):
    await ctx.connect()
    logger.info(f"User connected: {ctx.room.name}")

    vtube_connected = await VTUBE.connect()

    if vtube_connected:
        logger.info("VTube Studio connected")

    _vtube_is_connected = vtube_connected

    user_identity = "aura-user"  

    if ctx.job and hasattr(ctx.job, 'participant') and ctx.job.participant:
        user_identity = ctx.job.participant.identity or user_identity

    else:
        for p in ctx.room.remote_participants.values():
            if p.identity and not p.identity.startswith("agent-"):
                user_identity = p.identity
                break

    logger.info(f"Resolved user identity: '{user_identity}'")

    long_term_memory = await memory_service.get_long_term_memories(identity=user_identity, limit=10)
    is_returning_user = bool(long_term_memory.strip())

    if is_returning_user:
        logger.info(f"Long-term memory loaded for '{user_identity}'")
    else:
        logger.info(f"No long-term memory found for {user_identity}")

    conversation_id = await memory_service.create_conversation(title=f"Voice Session: {user_identity}")

    if conversation_id:
        logger.info(f"Memory: new conversation {conversation_id} for {user_identity}")
    else:
        logger.warning("Memory: Can't connect to Supabase, running without memory")

    system_prompt = build_system_prompt(long_term_memory)

    initial_chat_ctx = llm.ChatContext()

    BRIDGE.set_room(ctx.room)

    # Explicit ClientSession for Deepgram to fix Windows/aiohappyeyeballs DNS timeouts
    connector = aiohttp.TCPConnector(use_dns_cache=True, keepalive_timeout=120)
    stt_session = aiohttp.ClientSession(connector=connector)
    
    # --- OPTION 2: Deepgram STT (Fallback) ---
    stt_plugin = deepgram.STT(
        model="nova-3",
        language="multi",
        detect_language=False,
        smart_format=False, # Turned this off! It adds massive latency waiting for grammar checking.
        interim_results=False, # We don't use interim results anyway, saving packet streams
        api_key=DEEPGRAM_KEY,
        http_session=stt_session,
        keyterm=["moshi", "desu", "konnichiwa", "nihongo", "arigato", "sugoi", "hello", "hey", "AURA"]
    )

    llm_plugin = openai.LLM(
        model=os.getenv("OPENROUTER_MODEL", OPENROUTER_MODEL),
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_KEY,
    )

    session = AgentSession(
        stt=stt_plugin,
        llm=llm_plugin,
        tts=TTS_PLUGIN,
        vad=silero.VAD.load(
            min_silence_duration=0.4,  # aggressively detect end-of-speech (default is often much higher)
            min_speech_duration=0.05
        ),
    )

    await session.start(
        room=ctx.room,
        agent=AURAAssistant(
            conversation_id=conversation_id,
            user_identity=user_identity,
            system_prompt=system_prompt,
            initial_chat_ctx=initial_chat_ctx,
        ),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Greet with happy expression
    vtube_connected = VTUBE.connected
    if vtube_connected:
        await VTUBE.set_expression("smile")

    await session.generate_reply()

class AURAAssistant(Agent):
    def __init__(self, conversation_id=None, user_identity: str = "aura-user", system_prompt: str = AURA_BASE_PROMPT, initial_chat_ctx: "llm.ChatContext | None" = None,) -> None:
        super().__init__(instructions=system_prompt, chat_ctx=initial_chat_ctx)
        self._conversation_id     = conversation_id
        self._user_identity       = user_identity
        self._vtube_connected     = False
        self._last_user_text      = ""

    async def on_enter(self):
        self._vtube_connected = await VTUBE.connect()

    async def on_exit(self):
        await VTUBE.disconnect()
        BRIDGE.set_room(None)

        # Extract the long term memory and save memory to database if session ended
        if self._conversation_id and OPENROUTER_KEY:
            logger.info(f"Session ended for '{self._user_identity}'. Extracting long-term memory...")
            asyncio.create_task(
                extract_and_save_memory(
                    identity=self._user_identity,
                    conversation_id=self._conversation_id,
                    openrouter_key=OPENROUTER_KEY,
                )
            )

    # Set last user message when user done talking
    async def on_user_turn_completed(self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage) -> None:
        self._last_user_text = new_message.text_content or ""
        await super().on_user_turn_completed(turn_ctx, new_message)

    async def llm_chat(self, chat_ctx, **kwargs):
        """Override to detect emotion and trigger expressions"""
        # Start of turn: clear animation logs to allow fresh winks/tongues
        await VTUBE.start_turn()

        # Get response from parent
        async for chunk in super().llm_chat(chat_ctx, **kwargs):
            yield chunk
        
        # Emotion detection is now handled per-sentence in aura_tts.py
        pass

    # Set last assistant message when assistant done talking and add to database
    async def on_agent_speech_committed(self, msg: llm.ChatMessage) -> None:
        assistant_text = msg.text_content or ""

        if self._conversation_id and self._last_user_text and assistant_text:
            try:
                emotions = VTUBE.detect_emotion(assistant_text)
                emotion  = emotions[0] if emotions else "neutral"

                await memory_service.add_interaction(
                    conversation_id=self._conversation_id,
                    user_text=self._last_user_text,
                    assistant_text=assistant_text,
                    user_emotion="neutral",
                    assistant_emotion=emotion,
                )
                logger.debug(
                    f"Memory saved | user: '{self._last_user_text[:50]}' "
                    f"| aura: '{assistant_text[:50]}'"
                )
            except Exception as error:
                logger.error(f"Memory Save Failed: {error}")

            self._last_user_text = ""

if __name__ == "__main__":
    agents.cli.run_app(server)