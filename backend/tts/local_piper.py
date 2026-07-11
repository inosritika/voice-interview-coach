import asyncio
import io
import wave

from piper import PiperVoice

from config import PIPER_VOICE
from tts.base import TTSEngine


class LocalPiperTTS(TTSEngine):
    """piper running locally. No API key, no network.

    Needs a downloaded voice: a `<name>.onnx` file with its `<name>.onnx.json`
    config beside it. Set the path via PIPER_VOICE. Voices:
    https://github.com/rhasspy/piper/blob/master/VOICES.md
    """

    def __init__(self) -> None:
        self._voice: PiperVoice | None = None

    def _get_voice(self) -> PiperVoice:
        if self._voice is None:
            self._voice = PiperVoice.load(PIPER_VOICE)
        return self._voice

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        voice = self._get_voice()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            # synthesize_wav sets the WAV header (rate/width/channels) itself.
            voice.synthesize_wav(text, wav_file)
        return buf.getvalue()
