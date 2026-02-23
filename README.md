# voicelistener

Real-time voice recognition using Whisper or ElevenLabs. Speech detection uses fast RMS energy gating by default; Silero VAD is available as an optional higher-accuracy mode.

## Structure

```
voicelistener/
├── __init__.py
├── __main__.py              # CLI entry point
├── voicelistener.py         # VoiceListener class (audio + VAD + threading)
├── requirements.txt
└── transcribers/
    ├── __init__.py
    ├── whispertranscriber.py      # WhisperTranscriber class
    └── elevenlabstranscriber.py   # ElevenLabsTranscriber class
```

## Installation

```bash
pip install voicelistener
```

## CLI usage

```bash
# Default (local Whisper)
python -m voicelistener

# ElevenLabs (requires ELEVENLABS_API_KEY env var)
python -m voicelistener --transcriber elevenlabs
```

Listens to your microphone, detects speech, and prints transcriptions to stdout. Press Ctrl+C to stop.

| Flag | Default | Description |
|---|---|---|
| `--transcriber` | `whisper` | Speech-to-text backend (`whisper` or `elevenlabs`) |

## Library usage

```python
from voicelistener import VoiceListener, WhisperTranscriber, ElevenLabsTranscriber

# Local Whisper
transcriber = WhisperTranscriber(model_id="base.en")

# Or ElevenLabs (set ELEVENLABS_API_KEY env var)
# transcriber = ElevenLabsTranscriber(model_id="scribe_v2")

listener = VoiceListener(transcriber=transcriber)

for text in listener:
    print(text)
```

### Callback style

```python
def handle(text):
    print(f"Heard: {text}")

listener = VoiceListener(
    transcriber=WhisperTranscriber(),
    on_transcription=handle,
)
listener.start()
```

### VoiceListener options

| Parameter | Default | Description |
|---|---|---|
| `transcriber` | (required) | Object with a `transcribe(audio) -> str` method |
| `silence_timeout_ms` | `2000` | Silence duration (ms) to finalize an utterance |
| `min_utterance_ms` | `250` | Minimum speech length to transcribe |
| `pre_buffer_ms` | `150` | Audio kept before VAD triggers |
| `energy_only` | `True` | Use RMS energy for speech detection (no torch required); set `False` to enable Silero VAD |
| `vad_threshold` | `0.5` | Silero VAD confidence threshold (used only when `energy_only=False`) |
| `energy_threshold` | `0.005` | RMS energy threshold; frames below this are treated as silence |
| `on_transcription` | `None` | Callback invoked with each transcription |
| `on_speech_start` | `None` | Callback invoked when speech is detected |
| `on_speech_end` | `None` | Callback invoked when speech ends (silence timeout) |

### Custom transcriber

Implement a class with a `transcribe` method:

```python
class MyTranscriber:
    def transcribe(self, audio):
        # audio is a float32 numpy array at 16kHz
        return "transcribed text"

listener = VoiceListener(transcriber=MyTranscriber())
```
