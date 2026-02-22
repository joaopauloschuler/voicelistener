"""CLI entry point: python -m voicelistener"""

from voicelistener.transcribers import WhisperTranscriber
from voicelistener.voicelistener import VoiceListener


def main():
    print("Loading models...", flush=True)
    transcriber = WhisperTranscriber()
    listener = VoiceListener(transcriber=transcriber)
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
