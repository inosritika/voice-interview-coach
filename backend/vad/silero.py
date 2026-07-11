import numpy as np
import torch
from silero_vad import load_silero_vad

from vad.base import VADEngine


class SileroVAD(VADEngine):
    """Silero VAD — the standard small, fast speech detector. Runs locally in a
    few milliseconds per 32 ms frame, so it comfortably keeps up with real-time
    audio on CPU.

    We load the ONNX build (onnxruntime is already installed for piper), which
    avoids a second heavy runtime. The model is recurrent: each call updates
    hidden state from the previous frame, which is *why* it beats a naive
    "is it loud?" energy check — it has a sense of what speech sounds like over
    time, not just this instant.
    """

    def __init__(self) -> None:
        self._model = load_silero_vad(onnx=True)

    def speech_prob(self, frame: np.ndarray) -> float:
        # The Silero wrapper validates a torch tensor even in ONNX mode.
        tensor = torch.from_numpy(frame)
        return float(self._model(tensor, 16000))

    def reset(self) -> None:
        self._model.reset_states()
