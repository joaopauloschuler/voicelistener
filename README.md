# voicelistener

Real-time voice recognition using Silero VAD and Whisper.

## Structure

```
voicelistener/
├── __init__.py
├── __main__.py              # CLI entry point
├── voicelistener.py         # VoiceListener class (audio + VAD + threading)
├── requirements.txt
└── transcribers/
    ├── __init__.py
    └── whispertranscriber.py  # WhisperTranscriber class
```

## Setup

```bash
pip install -r voicelistener/requirements.txt
```

## CLI usage

```bash
python -m voicelistener
```

Listens to your microphone, detects speech, and prints transcriptions to stdout. Press Ctrl+C to stop.

## Library usage

```python
from voicelistener import VoiceListener, WhisperTranscriber

transcriber = WhisperTranscriber(model="base.en")
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
| `vad_threshold` | `0.5` | Silero VAD confidence threshold |
| `on_transcription` | `None` | Callback invoked with each transcription |

### Custom transcriber

Implement a class with a `transcribe` method:

```python
class MyTranscriber:
    def transcribe(self, audio):
        # audio is a float32 numpy array at 16kHz
        return "transcribed text"

listener = VoiceListener(transcriber=MyTranscriber())
```
