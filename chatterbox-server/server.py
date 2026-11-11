# ~/Desktop/OmniBot/chatterbox-server/server.py
import io
import re
import edge_tts
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Omni-Bot Instant Neural TTS Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Studio-grade Indian Male Neural Voices (Free, Unlimited, ~0.3s latency)
# - hi-IN-MadhurNeural: Native Indian Male voice (speaks both Hindi & English/Hinglish fluently!)
# - en-IN-PrabhatNeural: Crisp Indian English Male voice
VOICE_HINDI_HINGLISH = "hi-IN-MadhurNeural"
VOICE_ENGLISH = "en-IN-PrabhatNeural"

class TTSRequest(BaseModel):
    text: str
    language_id: str = "en"       # "hi" or "en"
    exaggeration: float = 0.55
    cfg_weight: float = 0.4

def clean_speech_text(raw: str) -> str:
    # Convert any paralinguistic tags into natural spoken equivalents
    text = re.sub(r"\[chuckle\]", " heheh, ", raw, flags=re.IGNORECASE)
    text = re.sub(r"\[laugh\]", " haha, ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[sigh\]", " phew... ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[.*?\]", "", text)
    return text.strip()

@app.post("/tts")
async def synthesize(req: TTSRequest):
    clean_text = clean_speech_text(req.text)
    if not clean_text:
        clean_text = "Hello!"

    # Use MadhurNeural for Hindi/Hinglish (it reads both Devanagari and English words with a natural Indian male voice)
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", clean_text))
    voice = VOICE_HINDI_HINGLISH if (req.language_id == "hi" or has_devanagari) else VOICE_ENGLISH

    communicate = edge_tts.Communicate(
        text=clean_text,
        voice=voice,
        rate="+6%",   # Snappy, energetic conversational pace
        pitch="-2Hz"  # Warm, grounded male tone
    )

    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])

    return Response(content=audio_buffer.getvalue(), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    print("[Neural TTS] READY! Instant Hindi & English Male Voice on http://127.0.0.1:4123")
    uvicorn.run(app, host="127.0.0.1", port=4123)