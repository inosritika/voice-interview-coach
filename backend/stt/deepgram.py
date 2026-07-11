import httpx

from config import DEEPGRAM_API_KEY
from stt.base import STTEngine

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


class DeepgramSTT(STTEngine):
    """Hosted STT via Deepgram's pre-recorded API. Lighter install, faster,
    but needs DEEPGRAM_API_KEY. Kept behind the same interface as the local
    engine so pipeline code doesn't care which is active."""

    def __init__(self) -> None:
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("STT_ENGINE=deepgram but DEEPGRAM_API_KEY is not set")

    async def transcribe(self, audio_bytes: bytes) -> str:
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/raw",
        }
        # Raw PCM now (step 2), so we must tell Deepgram how to interpret it.
        params = {
            "model": "nova-2",
            "smart_format": "true",
            "encoding": "linear16",
            "sample_rate": "16000",
            "channels": "1",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                DEEPGRAM_URL, headers=headers, params=params, content=audio_bytes
            )
            resp.raise_for_status()
            data = resp.json()
        return data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
