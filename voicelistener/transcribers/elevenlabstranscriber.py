"""ElevenLabs speech-to-text transcriber."""

import io
import os
import wave

import numpy as np
from elevenlabs import ElevenLabs


class ElevenLabsTranscriber:
    """Transcribes audio using ElevenLabs Scribe API."""

    def __init__(self, model_id="scribe_v2"):
        api_key = os.environ["ELEVENLABS_API_KEY"]
        self._client = ElevenLabs(api_key=api_key)
        self._model_id = model_id

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 numpy audio array to text."""
        buf = io.BytesIO()
        pcm = (audio * 32767).astype(np.int16)
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())
        buf.seek(0)

        response = self._client.speech_to_text.convert(
            file=buf,
            model_id=self._model_id,
        )
        return response.text.strip()
