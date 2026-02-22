"""CLI entry point: python -m voicelistener"""

import argparse

from voicelistener.voicelistener import VoiceListener

TRANSCRIBER_CHOICES = ["whisper", "elevenlabs"]


def _make_transcriber(name: str):
    if name == "whisper":
        from voicelistener.transcribers.whispertranscriber import WhisperTranscriber
        return WhisperTranscriber()
    elif name == "elevenlabs":
        from voicelistener.transcribers.elevenlabstranscriber import ElevenLabsTranscriber
        return ElevenLabsTranscriber()


def main():
    parser = argparse.ArgumentParser(description="Real-time voice listener")
    parser.add_argument(
        "--transcriber",
        choices=TRANSCRIBER_CHOICES,
        default="whisper",
        help="Speech-to-text backend (default: whisper)",
    )
    args = parser.parse_args()

    print("Loading models...", flush=True)
    transcriber = _make_transcriber(args.transcriber)
    listener = VoiceListener(
        transcriber=transcriber,
        on_speech_start=lambda: print("[speaking...]", flush=True),
        on_speech_end=lambda: print("[silent]", flush=True),
    )
    print("Listening...\n", flush=True)

    try:
        for text in listener:
            print(text, flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
