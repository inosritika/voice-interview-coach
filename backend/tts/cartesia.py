import httpx

from config import CARTESIA_API_KEY, CARTESIA_VOICE_ID
from tts.base import TTSEngine

CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2024-06-10"


class CartesiaTTS(TTSEngine):
    """Hosted TTS via Cartesia. Needs CARTESIA_API_KEY and CARTESIA_VOICE_ID.
    Same interface as the local piper engine, so it's a drop-in flag swap.
    Requests WAV back so the browser plays it the same way as the local path."""

    def __init__(self) -> None:
        if not CARTESIA_API_KEY:
            raise RuntimeError("TTS_ENGINE=cartesia but CARTESIA_API_KEY is not set")
        if not CARTESIA_VOICE_ID:
            raise RuntimeError("TTS_ENGINE=cartesia but CARTESIA_VOICE_ID is not set")

    async def synthesize(self, text: str) -> bytes:
        headers = {
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": CARTESIA_VERSION,
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": "sonic-2",
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 22050,
            },
            "language": "en",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(CARTESIA_URL, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.content
