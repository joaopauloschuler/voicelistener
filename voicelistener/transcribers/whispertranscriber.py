"""Whisper-based speech-to-text transcriber."""

import numpy as np
from faster_whisper import WhisperModel


class WhisperTranscriber:
    """Transcribes audio using faster-whisper."""

    def __init__(self, model="base.en", device="cpu", compute_type="int8"):
        self._whisper = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 numpy audio array to text."""
        segments, _ = self._whisper.transcribe(audio, language="en", beam_size=3)
        return " ".join(seg.text for seg in segments).strip()
